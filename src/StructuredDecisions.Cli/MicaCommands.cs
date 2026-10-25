using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using StructuredDecisions.Diagnostics;
using StructuredDecisions.Io;
using StructuredDecisions.Laya.Io;
using StructuredDecisions.Mica;

namespace StructuredDecisions.Cli;

/// <summary>The <c>mica*</c> commands: Mica v0.5 4B, the Qwen3.5 decision model.</summary>
internal static class MicaCommands
{
    private static readonly JsonSerializerOptions Indented = new()
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary><c>--model-dir</c>, or the checkpoint downloaded from <c>--source hub|host</c> (hub by default).</summary>
    public static string Directory(CommandLine options)
    {
        if (options.Value("model-dir") is string directory) return directory;
        var source = string.Equals(options.Value("source"), "host", StringComparison.OrdinalIgnoreCase) ? MicaSource.Host : MicaSource.Hub;
        Console.Error.WriteLine(source == MicaSource.Hub
            ? $"Downloading {MicaCheckpoint.HubRepo}@{MicaCheckpoint.HubRevision[..12]} (about 9.3 GB) …"
            : $"Downloading {MicaCheckpoint.HostUri} (about 8.5 GB) …");
        var progress = new Program.ConsoleProgress();
        string prepared = MicaCheckpoint.Prepare(source, options.Value("cache"), progress, options.Value("token"));
        progress.Finish();
        return prepared;
    }

    /// <summary>
    /// The request: <c>--request file.json</c>, <c>--request-json '{…}'</c>, or a state (<c>--text</c>,
    /// <c>--state-json</c>, <c>--state-file</c>) plus <c>--questions-json '{…}'</c> or <c>--questions file.json</c>.
    /// </summary>
    private static JsonNode Request(CommandLine options)
    {
        if (options.Value("request") is string file) return MicaContract.Parse(File.ReadAllText(file))!;
        if (options.Value("request-json") is string json) return MicaContract.Parse(json)!;

        JsonNode? state = options.Value("state-file") is string path
            ? (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ? MicaContract.Parse(File.ReadAllText(path)) : JsonValue.Create(File.ReadAllText(path)))
            : options.Value("state-json") is string stateJson ? MicaContract.Parse(stateJson)
            : options.Value("text") is string text ? JsonValue.Create(text)
            : throw new ArgumentException("provide --request, --request-json, or a state (--text, --state-json, --state-file) with questions.");
        JsonNode questions = options.Value("questions") is string questionFile ? MicaContract.Parse(File.ReadAllText(questionFile))!
            : options.Value("questions-json") is string questionJson ? MicaContract.Parse(questionJson)!
            : throw new ArgumentException("provide --questions file.json or --questions-json '{…}'.");
        return new JsonObject { ["state"] = state, ["questions"] = questions };
    }

    private static MicaJudge Open(CommandLine options)
    {
        string directory = Directory(options);
        var stopwatch = Stopwatch.StartNew();
        var judge = MicaJudge.FromDirectory(directory);
        Console.Error.WriteLine(string.Format(CultureInfo.InvariantCulture, "loaded {0} ({1:F2} GiB of weights) in {2:F1} s",
            directory, judge.Model.WeightBytes / (1024.0 * 1024 * 1024), stopwatch.Elapsed.TotalSeconds));
        return judge;
    }

    /// <summary><c>laya mica</c>: answer one request and print Mica's response.</summary>
    public static int Answer(CommandLine options)
    {
        var request = Request(options);
        MicaContract.RowsFromRequest(request.DeepClone());       // reject a bad request before loading 8 GB
        var judge = Open(options);
        var stopwatch = Stopwatch.StartNew();
        var response = judge.SystemOne(request);
        Console.WriteLine(response.ToJsonString(Indented));
        Console.Error.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} input tokens, {1:F1} s",
            response["usage"]!["input_tokens"], stopwatch.Elapsed.TotalSeconds));
        return 0;
    }

    /// <summary>
    /// <c>laya mica-dump</c>: every layer's output for one question row, as full fp32 tensors in a
    /// safetensors file — the C# side of <c>tools/dump_mica_reference.py checkpoint</c>, which
    /// <c>tools/compare_mica_dumps.py</c> compares it with.
    /// </summary>
    public static int Dump(CommandLine options)
    {
        var request = Request(options);
        int rowIndex = int.Parse(options.Value("row") ?? "0", CultureInfo.InvariantCulture);
        var rows = MicaContract.RowsFromRequest(request);
        var plain = rows.Where(r => !r.Multi).Concat(rows.Where(r => r.Multi).SelectMany(MicaContract.InclusionRows)).ToList();
        var row = plain[rowIndex];
        var judge = Open(options);

        var recorder = new StateRecorder();
        var stopwatch = Stopwatch.StartNew();
        var (distributions, tokens) = judge.Predict([row], recorder);
        Console.Error.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} tokens in {1:F1} s", tokens, stopwatch.Elapsed.TotalSeconds));

        string output = options.Value("out") ?? "artifacts/dumps/mica-dotnet";
        System.IO.Directory.CreateDirectory(output);
        // A single row has no shared prefix, so every tensor is the whole prompt's.
        var tensors = recorder.States.Select(s => new SafetensorsTensor(s.Name.Replace("row0.", "", StringComparison.Ordinal), s.Shape, s.Values, "F32")).ToList();
        SafetensorsWriter.Write(Path.Combine(output, "dotnet.safetensors"), tensors);
        var meta = new JsonObject
        {
            ["prompt"] = judge.Prompt.Render(row),
            ["input_ids"] = new JsonArray([.. judge.PromptTokens(row).Select(i => (JsonNode)i)]),
            ["labels"] = new JsonArray([.. MicaCodebook.LabelsFor(row).Select(l => (JsonNode)l)]),
            ["probabilities"] = new JsonArray([.. distributions[0].Select(p => (JsonNode)p)]),
        };
        File.WriteAllText(Path.Combine(output, "dotnet.json"), meta.ToJsonString(Indented));
        Console.WriteLine($"wrote {tensors.Count} tensors to {output}");
        Console.WriteLine($"probabilities: {string.Join(", ", distributions[0].Select(p => p.ToString("F6", CultureInfo.InvariantCulture)))}");
        return 0;
    }

    /// <summary>
    /// <c>laya mica-bench</c>: the decoder's forward pass over the token ids of every dump in <c>--dumps</c>
    /// (default <c>artifacts/dumps</c>) — the same sequences <c>tools/bench_mica_python.py</c> times — and,
    /// when that script's timings are there, the speed-up over them.
    /// </summary>
    public static int Bench(CommandLine options)
    {
        if (options.Values("requests").Count > 0) return BenchRequests(options);
        string dumps = options.Value("dumps") ?? "artifacts/dumps";
        int repeat = int.Parse(options.Value("repeat") ?? "3", CultureInfo.InvariantCulture);
        var judge = Open(options);
        var cases = System.IO.Directory.GetDirectories(dumps).Order(StringComparer.Ordinal)
            .Where(d => File.Exists(Path.Combine(d, "reference.json")))
            .Select(d => (Name: Path.GetFileName(d), Ids: JsonNode.Parse(File.ReadAllText(Path.Combine(d, "reference.json")))!["input_ids"]!
                .AsArray().Select(n => n!.GetValue<int>()).ToArray()))
            .ToList();
        if (cases.Count == 0) throw new ArgumentException($"no reference.json under {dumps}; run tools/dump_mica_reference.py checkpoint first.");

        var python = new Dictionary<string, JsonNode?>();
        foreach (string mode in new[] { "fp32", "bf16" })
        {
            string path = Path.Combine(dumps, $"python-{mode}-timing.json");
            if (File.Exists(path)) python[mode] = JsonNode.Parse(File.ReadAllText(path))!["cases"];
        }

        // JIT warm-up, which also fills the model's state-buffer pool: from here on a state costs no page faults,
        // as in a long-running service.
        using (var warm = judge.Model.NewState()) judge.Model.Forward(cases[0].Ids, warm);
        if (options.Has("stages"))
        {
            using var scope = ForwardTiming.Collect();
            using var state = judge.Model.NewState();
            judge.Model.Forward(cases[0].Ids, state);
            Console.WriteLine(scope.Timing.Describe());
        }
        foreach (var (name, ids) in cases)
        {
            var times = new List<double>();
            for (int r = 0; r < repeat; ++r)
            {
                var stopwatch = Stopwatch.StartNew();
                using (var state = judge.Model.NewState()) judge.Model.Forward(ids, state);
                times.Add(stopwatch.Elapsed.TotalSeconds);
            }
            double median = times.Order().ElementAt(times.Count / 2);
            string line = string.Format(CultureInfo.InvariantCulture, "{0,-12} {1,5} tokens  C# {2,6:F2} s  ({3:F0} tok/s)",
                name, ids.Length, median, ids.Length / median);
            foreach (var (mode, timings) in python)
            {
                if (timings?[name]?["median"] is JsonNode reference)
                {
                    double seconds = reference.GetValue<double>();
                    line += string.Format(CultureInfo.InvariantCulture, "   python {0} {1:F2} s → {2:F2}x", mode, seconds, seconds / median);
                }
            }
            Console.WriteLine(line);
        }
        return 0;
    }

    /// <summary>
    /// <c>laya mica-bench --requests a.json b.json …</c>: whole requests through <see cref="MicaJudge.SystemOne(JsonNode?, IStateRecorder?, ParallelOptions?)"/>,
    /// each timed with the caches cold (cleared before it) and warm (after the other requests have run, so the
    /// prompt header is cached but not this request's state), against <c>python-fp32-requests-timing.json</c>:
    /// the Python reference reading every question's prompt in full.
    /// </summary>
    private static int BenchRequests(CommandLine options)
    {
        string dumps = options.Value("dumps") ?? "artifacts/dumps";
        int repeat = int.Parse(options.Value("repeat") ?? "3", CultureInfo.InvariantCulture);
        var requests = options.Values("requests").Concat(options.Positional).Select(p => (Name: Path.GetFileNameWithoutExtension(p), Json: File.ReadAllText(p))).ToList();
        string pythonPath = Path.Combine(dumps, "python-fp32-requests-timing.json");
        var python = File.Exists(pythonPath) ? JsonNode.Parse(File.ReadAllText(pythonPath))!["cases"]!.AsObject() : null;
        var judge = Open(options);
        judge.SystemOne(requests[0].Json);       // JIT and state-buffer warm-up

        foreach (var (name, json) in requests)
        {
            var cold = new List<double>();
            var warm = new List<double>();
            int tokens = 0;
            for (int r = 0; r < repeat; ++r)
            {
                // Cold: nothing cached at all (a judge loaded with warm: false, before its first request).
                judge.PrefixCache.Clear();
                var stopwatch = Stopwatch.StartNew();
                var response = judge.SystemOne(json);
                cold.Add(stopwatch.Elapsed.TotalSeconds);
                tokens = response["usage"]!["input_tokens"]!.GetValue<int>();

                // Warm: the header is cached (as after the load-time warm-up), this request's state is not.
                judge.PrefixCache.Clear();
                judge.Warm();
                stopwatch.Restart();
                judge.SystemOne(json);
                warm.Add(stopwatch.Elapsed.TotalSeconds);
            }
            double coldMedian = cold.Order().ElementAt(cold.Count / 2), warmMedian = warm.Order().ElementAt(warm.Count / 2);
            string line = string.Format(CultureInfo.InvariantCulture, "{0,-16} {1,5} tokens  C# cold {2,6:F2} s  warm {3,6:F2} s", name, tokens, coldMedian, warmMedian);
            if (python is not null)
            {
                double seconds = python.Where(p => p.Key.StartsWith(name + "#", StringComparison.Ordinal)).Sum(p => p.Value!["median"]!.GetValue<double>());
                if (seconds > 0)
                {
                    line += string.Format(CultureInfo.InvariantCulture, "   python fp32 {0:F2} s → {1:F2}x cold, {2:F2}x warm", seconds, seconds / coldMedian, seconds / warmMedian);
                }
            }
            Console.WriteLine(line);
        }
        return 0;
    }

    /// <summary><c>laya mica-convert</c>: the hub checkpoint → the host layout, with sizes and SHA-256 for the upload.</summary>
    public static int Convert(CommandLine options)
    {
        string source = Directory(options);
        string output = options.Value("out") ?? throw new ArgumentException("mica-convert needs --out <dir>.");
        var files = MicaConverter.Convert(source, output, message => Console.Error.WriteLine(message));
        Console.WriteLine($"Upload these to {MicaCheckpoint.HostUri}:");
        foreach (string file in files)
        {
            string path = Path.Combine(output, file);
            using var stream = File.OpenRead(path);
            string hash = System.Convert.ToHexStringLower(SHA256.HashData(stream));
            Console.WriteLine($"  {file,-22} {new FileInfo(path).Length,14:N0} bytes  sha256 {hash}");
        }
        return 0;
    }
}

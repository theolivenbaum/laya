using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Laya;
using Laya.Diagnostics;
using Laya.Io;
using Laya.Runtime;
using Laya.Tokenizers;

namespace Laya.Cli;

internal static class Program
{
    private static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = true,
        // Match the Python payload: a choice answer has no "score" key at all, rather than null.
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        var options = CommandLine.Parse(args.AsSpan(1));
        if (options.Value("threads") is string threads
            && int.TryParse(threads, CultureInfo.InvariantCulture, out int degree))
        {
            LayaRuntime.MaxDegreeOfParallelism = degree;
        }

        try
        {
            return args[0] switch
            {
                "download" => Download(options),
                "predict" => Predict(options),
                "route" => Route(options),
                "presets" => ListPresets(),
                "tokenize" => Tokenize(options),
                "dump-states" => DumpStates(options),
                "bench" => Bench(options),
                "profile" => ProfileCommand.Run(options, OpenAgent, ReadState, ReadQuestions),
                "dataset" => TrainCommands.Dataset(options),
                "train" => TrainCommands.Train(options, ModelDirectory),
                "evaluate" => TrainCommands.Evaluate(options, ModelDirectory),
                _ => Unknown(args[0]),
            };
        }
        catch (Exception exception) when (exception is ArgumentException or IOException
            or InvalidDataException or KeyNotFoundException or HttpRequestException or NotSupportedException)
        {
            Console.Error.WriteLine("error: " + exception.Message);
            return 1;
        }
    }

    private static int Unknown(string command)
    {
        Console.Error.WriteLine($"error: unknown command '{command}'.");
        PrintUsage();
        return 1;
    }

    private static void PrintUsage() => Console.WriteLine("""
        laya — System 1 decision engine (.NET port)

        USAGE
          laya <command> [options]

        COMMANDS
          download      Fetch a checkpoint from the model host (or Hugging Face)
          predict       Answer a question set about some state
          route         Show which checkpoint a state would route to (loads nothing)
          presets       List the built-in question sets
          tokenize      Tokenize text with a checkpoint's tokenizer
          dump-states   Write per-layer activations for parity checking
          bench         Time the forward pass
          profile       Stage timings, allocations and a sampling profile of a forward pass
          dataset       Download a typed-decisions split to JSON lines
          train         Fine-tune a checkpoint (RLCD objective, AdamW, temperature calibration)
          evaluate      Score a checkpoint on typed-decisions cases (accuracy, Brier, ECE, ...)

        COMMON OPTIONS
          --model <name>        english | multilingual | typed-decisions   (default: english)
          --source <where>      host | hub   (default: host — https://models.curiosity.ai/laya)
          --standalone          Take --model from its own repo rather than the bundle repo
          --repo <id>           A specific Hugging Face repo id, instead of --model
          --subfolder <path>    Checkpoint subfolder inside --repo
          --model-dir <path>    Use a checkpoint already on disk instead of downloading
          --cache <path>        Download cache root (default: ~/.cache/laya or $LAYA_HOME)
          --token <token>       Hugging Face token (default: $HF_TOKEN)
          --threads <n>         Kernel threads (default: $LAYA_THREADS, else every core)
          --lid <name>          route: none | catalyst — a language classifier for text the
                                built-in heuristic cannot identify (default: none)

        TRAINING OPTIONS
          --data <file.jsonl>   Training cases (default: download --dataset's train split)
          --dataset <id>        Hugging Face dataset (default: LocalLLaMA/typed-decisions)
          --config <name>       Dataset config (default: all)
          --out <dir>           Where the fine-tuned checkpoint is written
          --epochs <n>          (default 4)       --micro-batch <n>   (default 8)
          --grad-accum <n>      (default 4)       --group-size <n>    (default 4)
          --lr-encoder <x>      (default 2.5e-5)  --lr-head <x>       (default 1e-4)
          --train-layers <n>    Train only the top n encoder layers (0 = head only; default all)
          --max-steps <n>       Stop after n optimizer steps
          --max-len <n>         Sequence budget for training and the saved config
          --head-max-len <n>    Instructions + options budget
          --limit <n>           Use only the first n cases
          --eval-data <file>    Evaluate the result on these cases (or --eval for the test split)

        EXAMPLES
          laya download --model english --cache ./artifacts/models
          laya download --model multilingual --source hub
          laya download --repo convaiinnovations/laya-typed-decisions
          laya predict --model-dir ./artifacts/models/english --preset triage \
                       --text "I was charged twice and nobody answers"
          laya route --text "Mein Konto wurde zweimal belastet"
          laya route --lid catalyst --text "Saya ditagih dua kali untuk langganan saya"
          laya profile --model-dir ./artifacts/models/english --preset triage \
                       --text "…" --threads 1 --no-trace
          laya dataset --split train
          laya train --model english --data artifacts/data/LocalLLaMA_typed-decisions.all.train.jsonl \
                     --out artifacts/models/my-typed-decisions --train-layers 4 --eval
          laya evaluate --model-dir artifacts/models/my-typed-decisions --split test
          laya dump-states --model-dir ./artifacts/models/english \
                           --text "hello" --question noul:"Is this a greeting?" \
                           --out artifacts/dumps/dotnet.json
        """);

    private static int Download(CommandLine options)
    {
        Console.WriteLine($"Ready: {DownloadCheckpoint(options)}");
        return 0;
    }

    private static int Predict(CommandLine options)
    {
        using var agent = OpenAgent(options);
        object? state = ReadState(options);
        var questions = ReadQuestions(options);

        var stopwatch = Stopwatch.StartNew();
        var result = agent.SystemOne(state, questions);
        stopwatch.Stop();

        Console.WriteLine(JsonSerializer.Serialize(ToPayload(result), Json));
        Console.Error.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0} question(s), {1} input tokens, {2:F0} ms", questions.Count, result.Usage.InputTokens,
            stopwatch.Elapsed.TotalMilliseconds));
        return 0;
    }

    private static int Route(CommandLine options)
    {
        object? state = ReadState(options);
        var questions = options.Has("preset") || options.Has("question") ? ReadQuestions(options) : null;
        var router = new Router(autoTaskDetection: options.Has("auto-task"),
            standaloneRepos: options.Has("standalone"))
        {
            LanguageClassifier = LanguageClassifier(options),
        };
        var decision = router.Route(state, questions, options.Value("force-model"), options.Value("task"),
            options.Value("lang"));
        Console.WriteLine(JsonSerializer.Serialize(decision, Json));
        return 0;
    }

    /// <summary><c>--lid catalyst</c> adds a statistical language identifier to routing; the default is none.</summary>
    private static ILanguageClassifier? LanguageClassifier(CommandLine options) => options.Value("lid") switch
    {
        null or "none" or "heuristic" => null,
        "catalyst" => Laya.Catalyst.CatalystLanguageClassifier.CreateAsync().GetAwaiter().GetResult(),
        string other => throw new ArgumentException($"unknown --lid '{other}'; expected none or catalyst."),
    };

    private static int ListPresets()
    {
        foreach (string name in Presets.Names)
        {
            var preset = Presets.ByName(name);
            Console.WriteLine($"{name} ({preset.Count} questions)");
            foreach (var (id, question) in preset)
            {
                Console.WriteLine($"  {id,-18} {QuestionTypes.Name(question.Type),-6} {question.InstructionText()}");
            }
            Console.WriteLine();
        }
        return 0;
    }

    private static int Tokenize(CommandLine options)
    {
        string directory = ModelDirectory(options);
        string tokenizerDirectory = Path.Combine(directory, "tokenizer");
        var tokenizer = HuggingFaceTokenizer.FromDirectory(
            Directory.Exists(tokenizerDirectory) ? tokenizerDirectory : directory);

        // --jsonl treats each line as a JSON string, so a corpus can carry newlines and quotes.
        bool jsonLines = options.Has("jsonl");
        IEnumerable<string> inputs = options.Value("file") is string file
            ? File.ReadLines(file).Select(line => jsonLines ? JsonSerializer.Deserialize<string>(line)! : line)
            : [options.Value("text") ?? throw new ArgumentException("tokenize needs --text or --file.")];

        foreach (string line in inputs)
        {
            var ids = tokenizer.Encode(line);
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                text = line,
                ids,
                tokens = options.Has("with-tokens") ? ids.Select(tokenizer.TokenOf).ToArray() : null,
            }, Json.WriteIndented ? new JsonSerializerOptions(Json) { WriteIndented = false } : Json));
        }
        return 0;
    }

    private static int DumpStates(CommandLine options)
    {
        using var agent = OpenAgent(options);
        object? state = ReadState(options);
        var questions = ReadQuestions(options);

        var recorder = new StateRecorder();
        var result = agent.SystemOne(state, questions, recorder);

        string output = options.Value("out") ?? "artifacts/dumps/dotnet.json";
        string answersJson = JsonSerializer.Serialize(
            result.Answers.ToDictionary(a => a.Key, a => ToAnswerPayload(a.Value)), Json);
        recorder.Save(output, sampleSize: int.Parse(options.Value("sample") ?? "64", CultureInfo.InvariantCulture),
            full: options.Has("full"), answersJson: answersJson);
        Console.WriteLine(recorder.Describe());
        Console.WriteLine();
        Console.WriteLine($"wrote {recorder.States.Count} tensors to {output}");
        Console.WriteLine(JsonSerializer.Serialize(ToPayload(result), Json));
        return 0;
    }

    private static int Bench(CommandLine options)
    {
        using var agent = OpenAgent(options);
        object? state = ReadState(options);
        var questions = ReadQuestions(options);
        int iterations = int.Parse(options.Value("iterations") ?? "3", CultureInfo.InvariantCulture);

        agent.SystemOne(state, questions);   // warm up the JIT and the page cache
        var timings = new List<double>();
        for (int i = 0; i < iterations; ++i)
        {
            var stopwatch = Stopwatch.StartNew();
            var result = agent.SystemOne(state, questions);
            stopwatch.Stop();
            timings.Add(stopwatch.Elapsed.TotalMilliseconds);
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "run {0}: {1:F1} ms for {2} questions / {3} tokens",
                i + 1, timings[^1], questions.Count, result.Usage.InputTokens));
        }
        timings.Sort();
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "median {0:F1} ms, {1:F1} ms/question  ({2})",
            timings[timings.Count / 2], timings[timings.Count / 2] / Math.Max(1, questions.Count),
            LayaRuntime.Describe()));
        return 0;
    }

    /// <summary>
    /// Which checkpoint the command should use.
    ///
    /// <para>The three checkpoints are published twice: bundled in <c>convaiinnovations/laya</c>,
    /// where two of them sit in a subfolder, and each in its own repo, where the same files sit at
    /// the root. <c>--repo</c> names a repository directly; <c>--standalone</c> picks the separate
    /// repo for a named checkpoint; otherwise the bundle is used and only the requested subfolder
    /// is downloaded.</para>
    /// </summary>
    private static ModelSpec ResolveSpec(CommandLine options)
    {
        if (options.Value("repo") is string repo) return new ModelSpec(repo, options.Value("subfolder"));

        string name = Router.Normalise(options.Value("model") ?? "english");
        var catalogue = options.Has("standalone") ? Router.StandaloneModels : Router.DefaultModels;
        return catalogue[name];
    }

    private static Agent OpenAgent(CommandLine options) => Agent.FromDirectory(ModelDirectory(options));

    private static string ModelDirectory(CommandLine options)
        => options.Value("model-dir") is string directory ? directory : DownloadCheckpoint(options);

    /// <summary>
    /// Fetches the requested checkpoint if it is not cached and returns its directory.
    ///
    /// <para>The model host is the default source; <c>--source hub</c> — and any of the
    /// hub-specific flags — takes the checkpoint from Hugging Face instead.</para>
    /// </summary>
    private static string DownloadCheckpoint(CommandLine options)
    {
        var progress = new ConsoleProgress();
        if (!UsesHub(options))
        {
            var checkpoint = RemoteCheckpoint.FromName(Router.Normalise(options.Value("model") ?? "english"));
            Console.WriteLine($"Downloading {checkpoint.Name} from {checkpoint.BaseUri} …");
            string cached = checkpoint.Prepare(options.Value("cache"), progress);
            progress.Finish();
            return cached;
        }

        var spec = ResolveSpec(options);
        Console.WriteLine($"Downloading {spec} …");
        using var downloader = new HuggingFaceDownloader(options.Value("token"));
        string root = downloader.SnapshotAsync(spec.Repo, options.Value("cache"),
            include: HuggingFaceDownloader.CheckpointFilter(spec.Subfolder), progress: progress)
            .GetAwaiter().GetResult();
        progress.Finish();
        return spec.Subfolder is null ? root : Path.Combine(root, spec.Subfolder);
    }

    /// <summary>The hub is used on request, and whenever a hub-only flag names what to fetch.</summary>
    private static bool UsesHub(CommandLine options)
        => string.Equals(options.Value("source"), "hub", StringComparison.OrdinalIgnoreCase)
            || options.Has("standalone")
            || options.Value("repo") is not null
            || options.Value("subfolder") is not null;

    private static object? ReadState(CommandLine options)
    {
        if (options.Value("state-file") is string path)
        {
            string content = File.ReadAllText(path);
            return path.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
                ? JsonDocument.Parse(content).RootElement.Clone()
                : content;
        }
        if (options.Value("state-json") is string json) return JsonDocument.Parse(json).RootElement.Clone();
        return options.Value("text") ?? throw new ArgumentException("provide --text, --state-json or --state-file.");
    }

    /// <summary>
    /// Questions come from a preset, a JSON file, or repeated <c>--question</c> flags in the
    /// compact <c>id=type:instructions[|option,option]</c> form.
    /// </summary>
    private static QuestionSet ReadQuestions(CommandLine options)
    {
        if (options.Value("preset") is string preset) return Presets.ByName(preset);
        if (options.Value("questions-file") is string file) return QuestionJson.Parse(File.ReadAllText(file));

        var inline = options.Values("question");
        if (inline.Count == 0)
        {
            throw new ArgumentException("provide --preset, --questions-file or one or more --question flags.");
        }

        var questions = new QuestionSet();
        int index = 0;
        foreach (string spec in inline)
        {
            var (id, question) = QuestionJson.ParseInline(spec, index++);
            questions.Add(id, question);
        }
        return questions;
    }

    private static object ToAnswerPayload(Answer answer) => new
    {
        type = answer.Type,
        choice = answer.Choice,
        score = answer.Score,
        noul = answer.Noul,
        probabilities = answer.Probabilities?.ToDictionary(p => p.Key, p => p.Value),
        legend = answer.Legend?.ToDictionary(l => l.Key, l => l.Value),
        confidence = answer.Confidence,
        action = new { act_probability = answer.Action.ActProbability },
    };

    private static object ToPayload(DecisionResult result) => new
    {
        model = result.Model,
        answers = result.Answers.ToDictionary(a => a.Key, a => ToAnswerPayload(a.Value)),
        usage = new { input_tokens = result.Usage.InputTokens, output_tokens = result.Usage.OutputTokens },
        routing = result.Routing,
    };

    /// <summary>
    /// A progress line that redraws in place on a terminal. When stderr is redirected there is no
    /// carriage return to redraw with, so it falls back to one line per 10% — otherwise a log file
    /// collects a line per megabyte.
    /// </summary>
    private sealed class ConsoleProgress : IProgress<DownloadProgress>
    {
        private static readonly bool Interactive = !Console.IsErrorRedirected;

        private string _current = string.Empty;
        private int _lastPercent = -1;
        private bool _wrote;

        public void Report(DownloadProgress value)
        {
            if (value.File != _current)
            {
                Finish();
                _current = value.File;
                _lastPercent = -1;
            }

            int percent = value.TotalBytes > 0 ? (int)(100 * value.BytesRead / value.TotalBytes) : 0;
            int step = Interactive ? 1 : 10;
            bool complete = value.TotalBytes > 0 && value.BytesRead >= value.TotalBytes;
            if (!complete && percent / step == _lastPercent / step) return;
            _lastPercent = percent;

            string line = string.Format(CultureInfo.InvariantCulture,
                "  {0,-44} {1,6:F1} / {2,6:F1} MiB  {3,3}%", Shorten(value.File),
                value.BytesRead / 1048576.0, value.TotalBytes / 1048576.0, percent);

            if (Interactive)
            {
                Console.Error.Write('\r' + line);
                _wrote = true;
            }
            else
            {
                Console.Error.WriteLine(line);
            }
        }

        public void Finish()
        {
            if (_wrote) Console.Error.WriteLine();
            _wrote = false;
        }

        private static string Shorten(string path) => path.Length <= 44 ? path : "…" + path[^43..];
    }
}

using System.Diagnostics;
using System.Text.Json.Nodes;
using StructuredDecisions.Diagnostics;
using StructuredDecisions.Io;
using StructuredDecisions.Mica.Models;
using StructuredDecisions.Tokenizers;

namespace StructuredDecisions.Mica;

/// <summary>
/// Mica (<c>sky7350/Mica-v0.5-4B</c>): a Qwen3.5-4B fine-tune that answers typed questions about a state —
/// yes/no, choice, score — with a probability for every allowed answer, read from the next-token logits of
/// one prefill. Nothing is generated.
///
/// <para>The port of Mica's runtime: requests are validated and expanded by <see cref="MicaContract"/>, each
/// row is rendered by <see cref="MicaPrompt"/> and tokenized, and the logits of the label tokens
/// (<see cref="MicaCodebook"/>) at the last prompt position become the distribution, at the shipped
/// calibration (temperature 1, no noul bias). Questions about one state share their prompt up to the
/// question, so that prefix is read once and every question continues from a rewound copy of the decoder
/// state — exactly the activations of reading each prompt on its own (see <see cref="Qwen35State"/>).</para>
/// </summary>
public sealed partial class MicaJudge : IDecisionModel
{
    /// <summary>The longest prompt the runtime accepts, as Mica's server; longer requests are rejected, never cut.</summary>
    public const int DefaultMaxPromptTokens = 32768;

    private readonly Dictionary<string, int[]> _labelIds = new(StringComparer.Ordinal);

    public Qwen35Model Model { get; }
    public HuggingFaceTokenizer Tokenizer { get; }
    public MicaPrompt Prompt { get; }
    public string ModelName { get; init; } = "mica-v0.5-4b";
    public int MaxPromptTokens { get; init; } = DefaultMaxPromptTokens;

    /// <summary>Softmax temperature per kind (<c>calibration.json</c>); Mica v0.5 ships 1.0 for all.</summary>
    public double Temperature { get; init; } = 1.0;

    /// <summary>Added to the Yes logit of a noul before the softmax; Mica v0.5 ships 0.</summary>
    public double NoulLogitBias { get; init; }

    public MicaJudge(Qwen35Model model, HuggingFaceTokenizer tokenizer)
    {
        Model = model;
        Tokenizer = tokenizer;
        Prompt = new MicaPrompt(tokenizer.AddedTokens.Select(t => t.Content));
        // Fail up front, as Mica's codebook.verify does, if a label is not one token of its own.
        LabelIds(MicaCodebook.NoulLabels);
        LabelIds(MicaCodebook.OptionLabels);
    }

    /// <summary>
    /// Loads a Mica checkpoint directory: <c>config.json</c>, the safetensors (one file, or shards with
    /// their index) and <c>tokenizer.json</c> + <c>tokenizer_config.json</c>, at the root or under
    /// <c>tokenizer/</c>.
    /// </summary>
    /// <param name="warm">
    /// Read the prompt header (the system turn every prompt starts with) once now, so that even the first request
    /// starts from its snapshot — as Mica's server does with <c>start_warm</c>. About a second at load.
    /// </param>
    public static MicaJudge FromDirectory(string directory, bool warm = true)
    {
        var config = Qwen35Config.Load(Path.Combine(directory, "config.json"));
        string tokenizerDirectory = File.Exists(Path.Combine(directory, "tokenizer.json")) ? directory : Path.Combine(directory, "tokenizer");
        var tokenizer = HuggingFaceTokenizer.FromDirectory(tokenizerDirectory);
        Qwen35Model model;
        using (var weights = new SafetensorsCheckpoint(directory)) model = new Qwen35Model(config, weights);
        var (temperature, bias) = ReadCalibration(Path.Combine(directory, "calibration.json"));
        var judge = new MicaJudge(model, tokenizer) { Temperature = temperature, NoulLogitBias = bias };
        if (warm) judge.Warm();
        return judge;
    }

    /// <summary>Puts the prompt header in <see cref="PrefixCache"/>, unless it is there already.</summary>
    public void Warm(ParallelOptions? parallel = null)
    {
        var row = new MicaRow { Id = "warm", Kind = "noul", State = "", Question = "?", Candidates = [new("false", "No"), new("true", "Yes")] };
        int[] prompt = PromptTokens(row);
        int header = HeaderLength(row);
        if (header == 0) return;
        using var state = PrefixCache.Restore(Model, prompt.AsSpan(0, header), out int read);
        if (read >= header) return;
        var sequence = new Qwen35Sequence { Tokens = prompt.AsMemory(read, header - read), State = state, Readout = false, Checkpoints = [header - read] };
        Model.Forward([sequence], parallel: parallel);
        PrefixCache.RememberHeader(prompt.AsSpan(0, header), sequence.Snapshots![0]);
    }

    private static (double Temperature, double Bias) ReadCalibration(string path)
    {
        if (!File.Exists(path)) return (1.0, 0.0);
        var root = JsonNode.Parse(File.ReadAllText(path)) as JsonObject
            ?? throw new InvalidDataException($"{path}: calibration must be an object");
        if (root.ContainsKey("schema") || !root.ContainsKey("temperature")) throw new InvalidDataException($"{path}: unsupported calibration schema");
        double temperature = root["temperature"]!.GetValue<double>();
        double bias = root["noul_logit_bias"]?.GetValue<double>() ?? 0.0;
        if (!double.IsFinite(temperature) || temperature <= 0 || !double.IsFinite(bias)) throw new InvalidDataException($"{path}: invalid calibration");
        return (temperature, bias);
    }

    private int[] LabelIds(IReadOnlyList<string> labels)
    {
        string key = string.Join('\u0001', labels);
        if (_labelIds.TryGetValue(key, out var cached)) return cached;
        var ids = new int[labels.Count];
        for (int i = 0; i < labels.Count; ++i)
        {
            var encoded = Tokenizer.Encode(labels[i]);
            if (encoded.Count != 1) throw new InvalidDataException($"label '{labels[i]}' is not a single token in this tokenizer");
            ids[i] = encoded[0];
        }
        if (ids.Distinct().Count() != ids.Length) throw new InvalidDataException("output labels collide on one token");
        _labelIds[key] = ids;
        return ids;
    }

    /// <summary>The prompt tokens of a row, exactly as they are read.</summary>
    public int[] PromptTokens(MicaRow row) => [.. Tokenizer.Encode(Prompt.Render(row))];

    /// <summary>Answers a <c>/v1/systemone</c> request given as JSON text.</summary>
    public JsonObject SystemOne(string requestJson, IStateRecorder? recorder = null, ParallelOptions? parallel = null)
        => SystemOne(MicaContract.Parse(requestJson), recorder, parallel);

    /// <summary>Answers a parsed <c>/v1/systemone</c> request; the response has Mica's shape.</summary>
    public JsonObject SystemOne(JsonNode? request, IStateRecorder? recorder = null, ParallelOptions? parallel = null)
    {
        int inputTokens = 0;
        var response = MicaContract.AnswerRequest(request, rows =>
        {
            var (distributions, tokens) = Predict(rows, recorder, parallel);
            inputTokens = tokens;
            return distributions;
        }, ModelName);
        response["usage"] = new JsonObject { ["input_tokens"] = inputTokens, ["output_tokens"] = 0 };
        return response;
    }

    /// <summary>
    /// One distribution per row over its labels — [false, true] for a noul — and the total prompt tokens.
    ///
    /// <para>Nothing is read twice that need not be, as in Mica's own runtime: the prompt header (the system
    /// turn, the same for every request) and the most recent state are kept as decoder snapshots across calls
    /// (<see cref="PrefixCache"/>), the rows of one request read their common prefix once, and each row then reads
    /// only its own remainder from a rewound copy. Reading a prompt in pieces gives bit-for-bit the activations of
    /// reading it at once, so none of this changes a result. A <paramref name="recorder"/> turns the caches off,
    /// so that a parity dump covers every token.</para>
    /// </summary>
    public (IReadOnlyList<IReadOnlyList<double>> Distributions, int InputTokens) Predict(IReadOnlyList<MicaRow> rows,
        IStateRecorder? recorder = null, ParallelOptions? parallel = null)
    {
        var prompts = rows.Select(PromptTokens).ToList();
        foreach (var tokens in prompts)
        {
            if (tokens.Length > MaxPromptTokens)
            {
                throw new MicaRequestException($"input is {tokens.Length} tokens; the limit is {MaxPromptTokens}");
            }
        }
        if (recorder is not null) return (PredictUncached(rows, prompts, recorder, parallel), prompts.Sum(p => p.Length));

        // Every row keeps at least its last token to read, since that position produces the logits.
        int shared = prompts.Min(p => p.Length) - 1;
        for (int i = 1; i < prompts.Count && shared > 0; ++i)
        {
            int n = 0;
            while (n < shared && prompts[0][n] == prompts[i][n]) n++;
            shared = n;
        }

        // Snapshot points inside the shared prefix: the end of the header, and the end of the state.
        int header = Math.Min(HeaderLength(rows[0]), shared);
        int stateEnd = Math.Min(StateEndLength(rows[0], prompts[0]), shared);
        var prefix = prompts[0].AsMemory(0, shared);

        // Start from the longest snapshot this prompt begins with; take the others during the pass that reads past them.
        var state = PrefixCache.Restore(Model, prefix.Span, out int read);
        var points = new List<(int Length, bool Header)>();
        if (header > read) points.Add((header, true));
        if (stateEnd > read && stateEnd > header) points.Add((stateEnd, false));

        var distributions = new IReadOnlyList<double>[rows.Count];
        try
        {
            if (rows.Count == 1)
            {
                // One question: one pass over the rest of its prompt, snapshots taken on the way.
                var only = new Qwen35Sequence
                {
                    Tokens = prompts[0].AsMemory(read),
                    State = state,
                    Checkpoints = [.. points.Select(p => p.Length - read)],
                };
                Model.Forward([only], parallel: parallel);
                Remember(prompts[0], points, only.Snapshots!);
                distributions[0] = Readout(rows[0], only.Hidden!, recorder: null, 0);
                return (distributions, prompts[0].Length);
            }

            // Several: one pass over the rest of the common prefix, then every question's remainder in one batched
            // pass, each from its own copy of the prefix state.
            if (read < shared)
            {
                var common = new Qwen35Sequence
                {
                    Tokens = prefix[read..],
                    State = state,
                    Readout = false,
                    Checkpoints = [.. points.Select(p => p.Length - read)],
                };
                Model.Forward([common], parallel: parallel);
                Remember(prompts[0], points, common.Snapshots!);
            }
            var branches = new Qwen35Sequence[rows.Count];
            for (int i = 0; i < rows.Count; ++i)
            {
                branches[i] = new Qwen35Sequence { Tokens = prompts[i].AsMemory(shared), State = i == rows.Count - 1 ? state : state.Clone() };
            }
            try
            {
                Model.Forward(branches, parallel: parallel);
            }
            finally
            {
                for (int i = 0; i < rows.Count - 1; ++i) branches[i].State.Dispose();
            }
            for (int i = 0; i < rows.Count; ++i) distributions[i] = Readout(rows[i], branches[i].Hidden!, recorder: null, i);
            return (distributions, prompts.Sum(p => p.Length));
        }
        finally
        {
            state.Dispose();
        }
    }

    /// <summary>Hands the pass's snapshots to the cache, which keeps them (or disposes them when caching is off).</summary>
    private void Remember(int[] prompt, List<(int Length, bool Header)> points, Qwen35State[] snapshots)
    {
        for (int k = 0; k < points.Count; ++k)
        {
            var tokens = prompt.AsSpan(0, points[k].Length);
            if (points[k].Header) PrefixCache.RememberHeader(tokens, snapshots[k]);
            else PrefixCache.RememberState(tokens, snapshots[k]);
        }
    }

    /// <summary>Every row read from the first token, each on its own: what a parity dump records.</summary>
    private List<IReadOnlyList<double>> PredictUncached(IReadOnlyList<MicaRow> rows, List<int[]> prompts, IStateRecorder recorder,
        ParallelOptions? parallel)
    {
        var distributions = new List<IReadOnlyList<double>>(rows.Count);
        for (int i = 0; i < rows.Count; ++i)
        {
            using var state = Model.NewState();
            var hidden = Model.Forward(prompts[i], state, Prefixed(recorder, $"row{i}."), parallel);
            distributions.Add(Readout(rows[i], hidden, recorder, i));
        }
        return distributions;
    }

    private double[] Readout(MicaRow row, float[] hidden, IStateRecorder? recorder, int index)
    {
        var labels = LabelIds(MicaCodebook.LabelsFor(row));
        var logits = new float[labels.Length];
        Model.Logits(hidden, labels, logits);
        recorder?.Record($"row{index}.label_logits", logits, [logits.Length]);
        return Distribution(logits, row.Kind);
    }

    private int _headerLength = -1;

    /// <summary>
    /// Tokens every prompt starts with, as Mica's runtime finds them (<c>_header_tokens</c>): two renderings that
    /// differ only in the state agree up to it; one token less in case the first state token merges with the text
    /// before it.
    /// </summary>
    private int HeaderLength(MicaRow row)
    {
        if (_headerLength >= 0) return _headerLength;
        int[] a = PromptTokens(new MicaRow { Id = row.Id, Kind = row.Kind, Question = row.Question, Candidates = row.Candidates, AllowCannotGrade = row.AllowCannotGrade, State = "alpha" });
        int[] b = PromptTokens(new MicaRow { Id = row.Id, Kind = row.Kind, Question = row.Question, Candidates = row.Candidates, AllowCannotGrade = row.AllowCannotGrade, State = "omega" });
        int n = 0;
        while (n < Math.Min(a.Length, b.Length) && a[n] == b[n]) n++;
        return _headerLength = Math.Max(0, n - 1);
    }

    /// <summary>
    /// Tokens of this prompt up to the end of its state (<c>_state_cut</c> in Mica's runtime): the prompt text up
    /// to <c>\n&lt;/state&gt;\n</c>, tokenized, agrees with the prompt that far; two tokens less in case the
    /// boundary merges.
    /// </summary>
    private int StateEndLength(MicaRow row, int[] prompt)
    {
        const string StateEnd = "\n</state>\n";
        string text = Prompt.Render(row);
        int at = text.IndexOf(StateEnd, StringComparison.Ordinal);
        if (at < 0) return 0;
        var head = Tokenizer.Encode(text[..(at + StateEnd.Length)]);
        int n = 0;
        while (n < Math.Min(head.Count, prompt.Length) && head[n] == prompt[n]) n++;
        return Math.Max(0, n - 2);
    }

    /// <summary>The decoder snapshots kept across calls: the prompt header, and the most recent state.</summary>
    public MicaPrefixCache PrefixCache { get; } = new();

    /// <summary><c>calibration.distribution</c>: <c>softmax(logits / T)</c>, plus the noul bias on Yes, in double.</summary>
    private double[] Distribution(float[] logits, string kind)
    {
        var scaled = logits.Select(v => v / Temperature).ToArray();
        if (kind == "noul" && NoulLogitBias != 0) scaled[1] += NoulLogitBias;
        double peak = scaled.Max();
        var weights = scaled.Select(v => Math.Exp(v - peak)).ToArray();
        double total = PythonText.Sum(weights);
        return [.. weights.Select(w => w / total)];
    }

    private static IStateRecorder? Prefixed(IStateRecorder? recorder, string prefix)
        => recorder is null ? null : new PrefixedRecorder(recorder, prefix);

    private sealed class PrefixedRecorder(IStateRecorder inner, string prefix) : IStateRecorder
    {
        public void Record(string name, ReadOnlySpan<float> values, ReadOnlySpan<int> shape) => inner.Record(prefix + name, values, shape);
    }
}

/// <summary>
/// Decoder snapshots that outlive a call, like the header and state caches of Mica's server: the state after the
/// prompt header (the same for every request), and the state after the most recent request's state, so a later
/// question about the same state reads only the question. Each is an independent copy; a hit is copied again, so
/// concurrent calls never share a state. <see cref="Clear"/> drops both.
/// </summary>
public sealed class MicaPrefixCache
{
    private readonly Lock _gate = new();
    private (int[] Tokens, Qwen35State State)? _header;
    private (int[] Tokens, Qwen35State State)? _state;

    /// <summary>When false, nothing is kept or restored.</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>Calls that restored the header or the state snapshot, for diagnostics.</summary>
    public int HeaderHits { get; private set; }

    public int StateHits { get; private set; }

    /// <summary>A state holding the longest cached snapshot that <paramref name="prompt"/> starts with, or an empty one.</summary>
    internal Qwen35State Restore(Models.Qwen35Model model, ReadOnlySpan<int> prompt, out int length)
    {
        lock (_gate)
        {
            if (Enabled)
            {
                // The state snapshot extends the header one, so it is the better hit when both match.
                if (_state is { } state && Matches(state.Tokens, prompt))
                {
                    StateHits++;
                    length = state.Tokens.Length;
                    return state.State.Clone();
                }
                if (_header is { } header && Matches(header.Tokens, prompt))
                {
                    HeaderHits++;
                    length = header.Tokens.Length;
                    return header.State.Clone();
                }
            }
        }
        length = 0;
        return model.NewState();
    }

    private static bool Matches(int[] cached, ReadOnlySpan<int> prompt)
        => cached.Length <= prompt.Length && prompt[..cached.Length].SequenceEqual(cached);

    internal void RememberHeader(ReadOnlySpan<int> tokens, Qwen35State state) => Remember(ref _header, tokens, state);

    internal void RememberState(ReadOnlySpan<int> tokens, Qwen35State state) => Remember(ref _state, tokens, state);

    /// <summary>Keeps <paramref name="state"/> (the cache owns it from here) as the snapshot after <paramref name="tokens"/>.</summary>
    private void Remember(ref (int[] Tokens, Qwen35State State)? slot, ReadOnlySpan<int> tokens, Qwen35State state)
    {
        if (!Enabled || tokens.Length == 0)
        {
            state.Dispose();
            return;
        }
        (int[] Tokens, Qwen35State State)? previous;
        lock (_gate)
        {
            previous = slot;
            slot = (tokens.ToArray(), state);
        }
        previous?.State.Dispose();
    }

    public void Clear()
    {
        lock (_gate)
        {
            _header?.State.Dispose();
            _state?.State.Dispose();
            _header = _state = null;
        }
    }
}

using System.Diagnostics;
using System.Text.Json.Nodes;
using Laya.Diagnostics;
using Laya.Io;
using Laya.Models;
using Laya.Tokenizers;

namespace Laya.Mica;

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
public sealed class MicaJudge
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
    public static MicaJudge FromDirectory(string directory)
    {
        var config = Qwen35Config.Load(Path.Combine(directory, "config.json"));
        string tokenizerDirectory = File.Exists(Path.Combine(directory, "tokenizer.json")) ? directory : Path.Combine(directory, "tokenizer");
        var tokenizer = HuggingFaceTokenizer.FromDirectory(tokenizerDirectory);
        Qwen35Model model;
        using (var weights = new SafetensorsCheckpoint(directory)) model = new Qwen35Model(config, weights);
        var (temperature, bias) = ReadCalibration(Path.Combine(directory, "calibration.json"));
        return new MicaJudge(model, tokenizer) { Temperature = temperature, NoulLogitBias = bias };
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
    /// One distribution per row over its labels — [false, true] for a noul. Rows share a decoder state up to
    /// their longest common token prefix; each then reads only its own remainder. Returns the distributions
    /// and the total prompt tokens of the rows.
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

        // Every row keeps at least its last token to read, since that position produces the logits. A lone
        // row is read in one piece: there is nothing to share.
        int shared = rows.Count > 1 ? prompts.Min(p => p.Length) - 1 : 0;
        for (int i = 1; i < prompts.Count && shared > 0; ++i)
        {
            int n = 0;
            while (n < shared && prompts[0][n] == prompts[i][n]) n++;
            shared = n;
        }

        var state = Model.NewState();
        Qwen35State.Mark? mark = null;
        if (shared > 0)
        {
            Model.Forward(prompts[0].AsSpan(0, shared), state, Prefixed(recorder, "prefix."), parallel);
            mark = state.Save();
        }

        var distributions = new List<IReadOnlyList<double>>(rows.Count);
        for (int i = 0; i < rows.Count; ++i)
        {
            if (mark is not null && i > 0) state.Rewind(mark);
            var hidden = Model.Forward(prompts[i].AsSpan(shared), state, Prefixed(recorder, $"row{i}."), parallel);
            var labels = LabelIds(MicaCodebook.LabelsFor(rows[i]));
            var logits = new float[labels.Length];
            Model.Logits(hidden, labels, logits);
            recorder?.Record($"row{i}.label_logits", logits, [logits.Length]);
            distributions.Add(Distribution(logits, rows[i].Kind));
        }
        return (distributions, prompts.Sum(p => p.Length));
    }

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

using System.Globalization;
using System.Text.Json;
using Laya.Diagnostics;
using Laya.Io;
using Laya.Models;
using Laya.Numerics;
using Laya.Runtime;
using Laya.Tokenizers;

namespace Laya.D1;

/// <summary>
/// LiquidAI's d1 decision model (<c>LiquidAI/d1-3B</c>): typed questions over a state, answered in
/// one causal forward pass each with zero output tokens.
///
/// <para>d1 is LFM2.5-VL-3B post-trained for System One decisions. Unlike a laya checkpoint there is
/// no decision head: the state and the question are rendered as a chat prompt
/// (<see cref="D1Prompt"/>), the LFM2 backbone runs to the answer slot, and the answer is a softmax
/// over the option codes' LM-head logits. Several questions over one state share the state's
/// tokens as the trunk of a <see cref="Lfm2Tree"/>, so the state is read once.</para>
///
/// <code>
/// using var d1 = D1Agent.FromDirectory("artifacts/models/d1-3B");
/// var result = d1.SystemOne("I was charged twice", Presets.Triage());
/// </code>
///
/// <para>Pictures are part of the state: <see cref="SystemOne(object?, QuestionSet, IReadOnlyList{RgbImage}, IStateRecorder?, ParallelOptions?)"/>
/// runs them through the SigLIP2 tower and the projector (<see cref="D1Vision"/>, loaded on first
/// use) and scatters the result into the prompt's <c>&lt;image&gt;</c> slots.</para>
/// </summary>
public sealed class D1Agent : IDecisionEngine
{
    /// <summary>The Hugging Face repository the checkpoint is published in.</summary>
    public const string DefaultRepository = "LiquidAI/d1-3B";

    /// <summary>The files a text-only d1 run needs; the vision tower lives in the same safetensors.</summary>
    public static readonly string[] CheckpointFiles =
        ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "processor_config.json"];

    /// <summary>Rows of branches packed into one pass, as <c>SystemOne(token_budget=65536)</c>.</summary>
    public int TokenBudget { get; init; } = 65536;

    public Lfm2Config Config { get; }
    public HuggingFaceTokenizer Tokenizer { get; }
    public Lfm2LanguageModel Model { get; }
    public D1Prompt Prompt { get; }
    public string ModelDirectory { get; }
    public string ModelName { get; }

    public D1ImageProcessor ImageProcessor { get; }

    /// <summary>The <c>&lt;image&gt;</c> token whose embeddings image features replace.</summary>
    public int ImageTokenId { get; }

    private readonly Lazy<D1Vision> _vision;

    /// <summary>The vision tower and projector, loaded from the checkpoint on first use.</summary>
    public D1Vision Vision => _vision.Value;

    private D1Agent(string directory, Lfm2Config config, HuggingFaceTokenizer tokenizer, Lfm2LanguageModel model, string bos)
    {
        ModelDirectory = directory;
        Config = config;
        Tokenizer = tokenizer;
        Model = model;
        Prompt = new D1Prompt(tokenizer, bos);
        ModelName = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory)));
        ImageProcessor = D1ImageProcessor.FromDirectory(directory);

        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "config.json")));
        var root = document.RootElement.Clone();
        ImageTokenId = root.TryGetProperty("image_token_id", out var image) ? image.GetInt32() : tokenizer.IdOf(D1ImageProcessor.ImageToken);
        _vision = new Lazy<D1Vision>(() =>
        {
            if (!root.TryGetProperty("vision_config", out _))
            {
                throw new NotSupportedException($"'{directory}' is a text-only checkpoint; it has no vision tower.");
            }
            using var weights = new SafetensorsFile(Path.Combine(directory, "model.safetensors"));
            return new D1Vision(root, weights);
        }, LazyThreadSafetyMode.ExecutionAndPublication);
    }

    /// <summary>True when <paramref name="directory"/> holds a d1 / LFM2(-VL) checkpoint rather than a laya one.</summary>
    public static bool IsCheckpoint(string directory)
    {
        string path = Path.Combine(directory, "config.json");
        if (!File.Exists(path) || File.Exists(Path.Combine(directory, "rl_agent_config.json"))) return false;
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(path));
            return document.RootElement.TryGetProperty("model_type", out var type)
                && type.GetString() is "lfm2_vl" or "lfm2";
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>Loads a d1 checkpoint that is already on disk.</summary>
    public static D1Agent FromDirectory(string directory)
    {
        string configPath = Path.Combine(directory, "config.json");
        if (!File.Exists(configPath)) throw new FileNotFoundException($"'{directory}' has no config.json.", configPath);
        string weightsPath = Path.Combine(directory, "model.safetensors");
        if (!File.Exists(weightsPath)) throw new FileNotFoundException($"'{directory}' has no model.safetensors.", weightsPath);

        var config = Lfm2Config.Load(configPath);
        var tokenizer = HuggingFaceTokenizer.FromDirectory(directory);
        string bos = ReadBos(directory) ?? (config.BosTokenId >= 0 ? tokenizer.TokenOf(config.BosTokenId) : "");

        using var weights = new SafetensorsFile(weightsPath);
        string prefix = weights.Contains("model.language_model.embed_tokens.weight") ? "model.language_model."
            : weights.Contains("model.embed_tokens.weight") ? "model."
            : throw new InvalidDataException($"'{directory}': no LFM2 language model in model.safetensors.");
        var model = new Lfm2LanguageModel(config, weights, prefix);
        return new D1Agent(directory, config, tokenizer, model, bos);
    }

    /// <summary>
    /// Loads <paramref name="modelIdOrPath"/>, downloading the checkpoint from Hugging Face first when
    /// it is a repository id that is not on disk.
    /// </summary>
    public static D1Agent Load(string modelIdOrPath = DefaultRepository, string? token = null, string? cacheRoot = null,
        IProgress<DownloadProgress>? progress = null)
    {
        if (Directory.Exists(modelIdOrPath)) return FromDirectory(modelIdOrPath);
        using var downloader = new HuggingFaceDownloader(token);
        string root = downloader.SnapshotAsync(modelIdOrPath, cacheRoot,
            include: path => CheckpointFiles.Contains(path, StringComparer.Ordinal), progress: progress)
            .GetAwaiter().GetResult();
        return FromDirectory(root);
    }

    private static string? ReadBos(string directory)
    {
        string path = Path.Combine(directory, "tokenizer_config.json");
        if (!File.Exists(path)) return null;
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        return document.RootElement.TryGetProperty("bos_token", out var bos) && bos.ValueKind == JsonValueKind.String
            ? bos.GetString()
            : null;
    }

    public DecisionResult SystemOne(object? state, QuestionSet questions) => SystemOne(state, questions, recorder: null);

    public DecisionResult SystemOne(object? state, QuestionSet questions, ParallelOptions? parallel)
        => SystemOne(state, questions, recorder: null, parallel);

    /// <summary>
    /// Answers every question about <paramref name="state"/> (a string, any JSON value, or null).
    /// One question is its whole prompt in one causal pass; several are one pass over a tree whose
    /// trunk is the state, exactly as <c>SystemOne.run</c> arranges them.
    /// </summary>
    /// <param name="recorder">Receives every intermediate tensor, for parity checking.</param>
    public DecisionResult SystemOne(object? state, QuestionSet questions, IStateRecorder? recorder,
        ParallelOptions? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(questions);
        var list = questions.ToArray();
        var answers = new List<KeyValuePair<string, Answer>>(list.Length);
        if (list.Length == 0) return Result(answers, 0);

        var probabilities = Probabilities(state, [.. list.Select(q => q.Value)], recorder, parallel, out int read);
        for (int i = 0; i < list.Length; ++i)
        {
            answers.Add(new(list[i].Key, BuildAnswer(list[i].Value, probabilities[i])));
        }
        return Result(answers, read);
    }

    /// <summary>
    /// Answers every question about <paramref name="state"/> and <paramref name="images"/> together;
    /// the state may be null when the pictures are the whole state.
    /// </summary>
    public DecisionResult SystemOne(object? state, QuestionSet questions, IReadOnlyList<RgbImage> images,
        IStateRecorder? recorder = null, ParallelOptions? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(questions);
        ArgumentNullException.ThrowIfNull(images);
        var list = questions.ToArray();
        var answers = new List<KeyValuePair<string, Answer>>(list.Length);
        if (list.Length == 0) return Result(answers, 0);

        var probabilities = Probabilities(state, [.. list.Select(q => q.Value)], images, recorder, parallel, out int read);
        for (int i = 0; i < list.Length; ++i)
        {
            answers.Add(new(list[i].Key, BuildAnswer(list[i].Value, probabilities[i])));
        }
        return Result(answers, read);
    }

    /// <inheritdoc cref="SystemOne(object?, QuestionSet, IStateRecorder?, ParallelOptions?)"/>
    public DecisionResult Predict(object? state, QuestionSet questions, ParallelOptions? parallel = null)
        => SystemOne(state, questions, recorder: null, parallel);

    private DecisionResult Result(List<KeyValuePair<string, Answer>> answers, int read) => new()
    {
        Model = ModelName,
        Answers = answers,
        Usage = new Usage(read, 0),
    };

    /// <summary>
    /// Each question's distribution over its options (<c>yes</c>, <c>no</c> for a noul), and the
    /// number of tokens read.
    /// </summary>
    public double[][] Probabilities(object? state, IReadOnlyList<Question> questions, IStateRecorder? recorder,
        ParallelOptions? parallel, out int tokensRead)
        => Probabilities(state, questions, [], recorder, parallel, out tokensRead);

    /// <inheritdoc cref="Probabilities(object?, IReadOnlyList{Question}, IStateRecorder?, ParallelOptions?, out int)"/>
    public double[][] Probabilities(object? state, IReadOnlyList<Question> questions, IReadOnlyList<RgbImage> images,
        IStateRecorder? recorder, ParallelOptions? parallel, out int tokensRead)
    {
        var result = new double[questions.Count][];

        // Pictures: the chat template writes one <image> each at the head of the user turn, and the
        // processor expands each into its tiles' placeholder tokens.
        string markup = string.Concat(Enumerable.Repeat(D1ImageProcessor.ImageToken, images.Count));
        var processed = images.Select(ImageProcessor.Process).ToArray();
        string Expand(string text)
        {
            if (processed.Length == 0) return text;
            var builder = new System.Text.StringBuilder();
            int from = 0, next = 0;
            int at;
            while (next < processed.Length && (at = text.IndexOf(D1ImageProcessor.ImageToken, from, StringComparison.Ordinal)) >= 0)
            {
                builder.Append(text, from, at - from).Append(processed[next++].Placeholder);
                from = at + D1ImageProcessor.ImageToken.Length;
            }
            return builder.Append(text, from, text.Length - from).ToString();
        }
        float[]? features = null;
        if (processed.Length > 0)
        {
            using (ForwardTiming.Measure("vision"))
            {
                features = [.. processed.SelectMany((image, i) => Vision.Embed(image, recorder, $"image{i}", parallel))];
            }
        }

        string prefix = Prompt.Prefix(state, markup);
        if (questions.Count == 1)
        {
            var ids = Tokenizer.Encode(Expand(prefix + Prompt.Suffix(questions[0])));
            var tree = Lfm2Tree.Chain(ids.Count);
            var hidden = Run(ids, tree, features, recorder, parallel);
            result[0] = Readout(questions[0], hidden.AsSpan((ids.Count - 1) * Config.HiddenSize, Config.HiddenSize), recorder, 0, parallel);
            tokensRead = ids.Count;
            return result;
        }

        var trunk = Tokenizer.Encode(Expand(prefix));
        var branches = questions.Select(q => Tokenizer.Encode(Prompt.Suffix(q))).ToArray();
        tokensRead = trunk.Count + branches.Sum(b => b.Count);

        foreach (var chunk in Plan(branches.Select(b => b.Count).ToArray()))
        {
            var ids = new List<int>(trunk.Count + chunk.Sum(i => branches[i].Count));
            ids.AddRange(trunk);
            foreach (int i in chunk) ids.AddRange(branches[i]);
            var tree = Lfm2Tree.Branched(trunk.Count, [.. chunk.Select(i => branches[i].Count)]);
            var hidden = Run(ids, tree, features, recorder, parallel);
            for (int j = 0; j < chunk.Count; ++j)
            {
                int q = chunk[j];
                result[q] = Readout(questions[q], hidden.AsSpan(tree.Leaves[j] * Config.HiddenSize, Config.HiddenSize),
                    recorder, q, parallel);
            }
        }
        return result;
    }

    /// <summary>
    /// Embeds <paramref name="ids"/>, scatters the image features into the <c>&lt;image&gt;</c> rows
    /// in order (<c>masked_scatter</c>), and runs the stack.
    /// </summary>
    private float[] Run(List<int> ids, Lfm2Tree tree, float[]? features, IStateRecorder? recorder, ParallelOptions? parallel)
    {
        recorder?.Record("input_ids", [.. ids.Select(i => (float)i)], [ids.Count]);
        int d = Config.HiddenSize;
        int slots = ids.Count(id => id == ImageTokenId);
        int available = features is null ? 0 : features.Length / d;
        if (slots != available)
        {
            throw new InvalidOperationException(
                $"Image features and image tokens do not match, tokens: {slots}, features: {available}");
        }
        if (features is null) return Model.Forward([.. ids], tree, recorder, parallel);

        var hidden = new float[ids.Count * d];
        for (int t = 0; t < ids.Count; ++t) Model.Embedding(ids[t], hidden.AsSpan(t * d, d));
        recorder?.Record("embeddings", hidden, [ids.Count, d]);
        int next = 0;
        for (int t = 0; t < ids.Count; ++t)
        {
            if (ids[t] == ImageTokenId) features.AsSpan(next++ * d, d).CopyTo(hidden.AsSpan(t * d, d));
        }
        return Model.Run(hidden, tree, recorder, parallel);
    }

    /// <summary><c>_plan</c>: consecutive groups of at most <see cref="TokenBudget"/> branch tokens.</summary>
    private List<List<int>> Plan(int[] lengths)
    {
        var plan = new List<List<int>> { new() };
        int used = 0;
        for (int i = 0; i < lengths.Length; ++i)
        {
            if (plan[^1].Count > 0 && used + lengths[i] > TokenBudget)
            {
                plan.Add([]);
                used = 0;
            }
            plan[^1].Add(i);
            used += lengths[i];
        }
        return plan;
    }

    /// <summary>
    /// <c>readout</c>: each option scores its best form, and the answer is a softmax over the scores.
    ///
    /// <para>The reference takes the scores from the log-softmax over the whole vocabulary, but the
    /// log-normalizer is the same for every option and cancels in the softmax, so only the option
    /// tokens' logits are computed — a dozen dot products instead of a 128000 x 2048 product. With a
    /// recorder attached the full log-softmax is computed as well, for the parity dump.</para>
    /// </summary>
    private double[] Readout(Question question, ReadOnlySpan<float> hidden, IStateRecorder? recorder, int index,
        ParallelOptions? parallel)
    {
        var groups = Prompt.ReadoutIds(question);
        var scores = new double[groups.Count];
        float[]? logz = null;
        if (recorder is not null)
        {
            logz = new float[Config.VocabSize];
            Model.Logits(hidden, logz, parallel);
            double max = logz.Max();
            double total = 0;
            foreach (float v in logz) total += Math.Exp(v - max);
            float lse = (float)(max + Math.Log(total));
            for (int v = 0; v < logz.Length; ++v) logz[v] -= lse;
            recorder.Record($"logz.{index}", logz, [logz.Length]);
        }

        for (int g = 0; g < groups.Count; ++g)
        {
            double best = double.NegativeInfinity;
            foreach (int id in groups[g]) best = Math.Max(best, logz is null ? Model.Logit(hidden, id) : logz[id]);
            scores[g] = best;
        }

        double top = scores.Max();
        double sum = 0;
        var probabilities = new double[scores.Length];
        for (int i = 0; i < scores.Length; ++i) sum += probabilities[i] = Math.Exp(scores[i] - top);
        for (int i = 0; i < scores.Length; ++i) probabilities[i] /= sum;
        recorder?.Record($"probabilities.{index}", [.. probabilities.Select(p => (float)p)], [probabilities.Length]);
        return probabilities;
    }

    /// <summary><c>api.answer</c>: a noul's P(yes); a choice's pick; a score's expected level.</summary>
    private static Answer BuildAnswer(Question question, double[] probabilities)
    {
        if (question.Type == QuestionType.Noul)
        {
            double yes = probabilities[0];
            return new Answer { Type = "noul", Noul = yes, Confidence = Math.Max(yes, 1 - yes) };
        }

        int best = 0;
        for (int i = 1; i < probabilities.Length; ++i)
        {
            if (probabilities[i] > probabilities[best]) best = i;
        }

        if (question.Type == QuestionType.Choice)
        {
            var names = question.OptionKeys;
            return new Answer
            {
                Type = "choice",
                Choice = names[best],
                Confidence = probabilities[best],
                Probabilities = [.. names.Select((name, i) => new KeyValuePair<string, double>(name, probabilities[i]))],
            };
        }

        double expected = 0;
        for (int i = 0; i < probabilities.Length; ++i) expected += i * probabilities[i];
        var levels = question.Levels ?? [];
        return new Answer
        {
            Type = "score",
            Score = expected,
            Confidence = probabilities[best],
            Probabilities = [.. probabilities.Select((p, i) => new KeyValuePair<string, double>(i.ToString(CultureInfo.InvariantCulture), p))],
            Legend = [.. levels.Select((level, i) => new KeyValuePair<string, string>(
                i.ToString(CultureInfo.InvariantCulture), level as string ?? Question.RenderCriterion(level)))],
        };
    }

    /// <summary>Nothing unmanaged is held; the weights are managed arrays.</summary>
    public void Dispose() => GC.SuppressFinalize(this);
}

using Laya.Diagnostics;
using Laya.Io;
using Laya.Models;
using Laya.Numerics;
using Laya.Runtime;
using Laya.Tokenizers;

namespace Laya;

/// <summary>
/// The System 1 decision runtime: load a checkpoint, then answer every question about a state in
/// one forward pass each, with calibrated probabilities.
///
/// <code>
/// using var agent = Agent.Load();                       // English checkpoint, downloaded on demand
/// var result = agent.SystemOne("I was charged twice", Presets.Triage());
/// Console.WriteLine(result["intent"].Choice);
/// </code>
/// </summary>
public sealed class Agent : IDecisionEngine
{
    private readonly SafetensorsFile _weights;

    public LayaConfig Config { get; }
    public ModernBertConfig EncoderConfig { get; }
    public HuggingFaceTokenizer Tokenizer { get; }
    public DecisionModel Model { get; }
    public string ModelDirectory { get; }

    private Agent(string modelDirectory, LayaConfig config, ModernBertConfig encoderConfig,
        HuggingFaceTokenizer tokenizer, SafetensorsFile weights)
    {
        ModelDirectory = modelDirectory;
        Config = config;
        EncoderConfig = encoderConfig;
        Tokenizer = tokenizer;
        _weights = weights;
        Model = new DecisionModel(encoderConfig, config, weights);
    }

    /// <summary>
    /// Loads a checkpoint from a local directory, downloading it from the hub first when
    /// <paramref name="modelIdOrPath"/> is a repository id that is not already on disk.
    /// </summary>
    /// <param name="subfolder">
    /// Selects one checkpoint from the bundle repo, e.g. <c>"multilingual"</c>. Only that subfolder
    /// is downloaded, so bundling does not cost every user the whole family.
    /// </param>
    public static Agent Load(string modelIdOrPath = "convaiinnovations/laya", string? subfolder = null,
        string? token = null, string? cacheRoot = null, IProgress<DownloadProgress>? progress = null)
    {
        string directory = Resolve(modelIdOrPath, subfolder, token, cacheRoot, progress);
        return FromDirectory(directory);
    }

    /// <summary>Loads a checkpoint that is already on disk.</summary>
    public static Agent FromDirectory(string directory)
    {
        string configPath = Path.Combine(directory, "rl_agent_config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException(
                $"Incompatible model: '{directory}' does not contain 'rl_agent_config.json'. " +
                "Make sure this is a laya decision checkpoint.", configPath);
        }

        string weightsPath = Path.Combine(directory, "model.safetensors");
        if (!File.Exists(weightsPath))
        {
            throw new FileNotFoundException($"Incompatible model: no 'model.safetensors' in '{directory}'.", weightsPath);
        }

        var config = LayaConfig.Load(configPath);

        string encoderConfigPath = Path.Combine(directory, "encoder", "config.json");
        if (!File.Exists(encoderConfigPath))
        {
            throw new FileNotFoundException(
                $"'{directory}' has no encoder/config.json. The .NET port builds the encoder from that " +
                "file rather than downloading the base model from the hub.", encoderConfigPath);
        }
        var encoderConfig = ModernBertConfig.Load(encoderConfigPath);

        string tokenizerDirectory = Path.Combine(directory, "tokenizer");
        var tokenizer = HuggingFaceTokenizer.FromDirectory(
            Directory.Exists(tokenizerDirectory) ? tokenizerDirectory : directory);
        if (tokenizer.MaskTokenId < 0 || tokenizer.ClsTokenId < 0 || tokenizer.SepTokenId < 0)
        {
            throw new InvalidDataException(
                $"'{directory}': the tokenizer is missing one of [CLS]/[SEP]/[MASK]; sequence building needs all three.");
        }

        var weights = new SafetensorsFile(weightsPath);
        VerifyCompatibility(weights, directory);
        return new Agent(directory, config, encoderConfig, tokenizer, weights);
    }

    private static void VerifyCompatibility(SafetensorsFile weights, string modelId)
    {
        foreach (string prefix in (string[])["encoder.", "type_emb.", "scorer.", "act_head."])
        {
            if (!weights.Entries.Keys.Any(k => k.StartsWith(prefix, StringComparison.Ordinal)))
            {
                throw new InvalidDataException(
                    $"Incompatible model weights for '{modelId}': the checkpoint has no '{prefix}' parameters. " +
                    "Expected a laya decision model with an encoder and decision heads.");
            }
        }
    }

    private static string Resolve(string modelIdOrPath, string? subfolder, string? token, string? cacheRoot,
        IProgress<DownloadProgress>? progress)
    {
        if (Directory.Exists(modelIdOrPath))
        {
            return string.IsNullOrEmpty(subfolder) ? modelIdOrPath : Path.Combine(modelIdOrPath, subfolder);
        }

        bool looksLocal = modelIdOrPath.StartsWith('/') || modelIdOrPath.StartsWith("./", StringComparison.Ordinal)
            || modelIdOrPath.StartsWith("../", StringComparison.Ordinal) || Path.IsPathRooted(modelIdOrPath);
        if (looksLocal)
        {
            throw new DirectoryNotFoundException(
                $"Local model path not found: '{modelIdOrPath}'.");
        }

        using var downloader = new HuggingFaceDownloader(token);
        string root = downloader.SnapshotAsync(modelIdOrPath, cacheRoot,
            include: HuggingFaceDownloader.CheckpointFilter(subfolder), progress: progress)
            .GetAwaiter().GetResult();

        string directory = string.IsNullOrEmpty(subfolder) ? root : Path.Combine(root, subfolder);
        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"Subfolder '{subfolder}' not found in '{modelIdOrPath}'.");
        }
        return directory;
    }

    /// <summary>
    /// Evaluates every question about <paramref name="state"/>.
    ///
    /// <para>All the questions go through the encoder together, as one concatenated batch, so the
    /// 421M parameters are read from memory once rather than once per question. Nothing crosses
    /// between questions: attention, the type embedding and the marker read-out are all per
    /// sequence, exactly as PyTorch's padding mask arranges — but unlike a padded batch, a short
    /// question does not pay for the longest one in the set.</para>
    /// </summary>
    public DecisionResult SystemOne(object? state, QuestionSet questions)
        => SystemOne(state, questions, recorder: null);

    /// <inheritdoc cref="SystemOne(object?, QuestionSet)"/>
    /// <param name="recorder">Receives every intermediate tensor, for parity checking.</param>
    public DecisionResult SystemOne(object? state, QuestionSet questions, IStateRecorder? recorder)
    {
        ArgumentNullException.ThrowIfNull(questions);
        if (questions.Count == 0)
        {
            return new DecisionResult
            {
                Model = "laya-rl-agent",
                Answers = [],
                Usage = new Usage(0, 0),
            };
        }

        var questionList = questions.ToArray();
        var sequences = new BuiltSequence[questionList.Length];
        var labels = new string[questionList.Length];
        var items = new List<DecisionModel.BatchItem>(questionList.Length);
        int inputTokens = 0;

        for (int i = 0; i < questionList.Length; ++i)
        {
            var (id, question) = questionList[i];
            var sequence = SequenceBuilder.Build(Tokenizer, state, question, Config.MaxLength, Config.HeadMaxLength);
            if (sequence.MarkerPositions.Length != question.RenderOptions().Count)
            {
                throw new ArgumentException(
                    $"question '{id}' has options that exceed head_max_len={Config.HeadMaxLength}.", nameof(questions));
            }

            sequences[i] = sequence;
            labels[i] = id + ".";
            inputTokens += sequence.TokenIds.Length;
            items.Add(new DecisionModel.BatchItem(sequence.TokenIds, sequence.MarkerPositions, (int)question.Type));

            recorder?.Record($"input_ids.{id}", [.. sequence.TokenIds.Select(t => (float)t)],
                [sequence.TokenIds.Length]);
            recorder?.Record($"marker_pos.{id}", [.. sequence.MarkerPositions.Select(m => (float)m)],
                [sequence.MarkerPositions.Length]);
        }

        var outputs = Model.Forward(items, recorder is null ? null : new BatchRecorder(recorder, labels));

        var answers = new List<KeyValuePair<string, Answer>>(questionList.Length);
        for (int i = 0; i < questionList.Length; ++i)
        {
            var (id, question) = questionList[i];
            answers.Add(new KeyValuePair<string, Answer>(id,
                BuildAnswer(question, outputs[i], sequences[i].MarkerPositions.Length)));
        }

        return new DecisionResult
        {
            Model = "laya-rl-agent",
            Answers = answers,
            Usage = new Usage(inputTokens, 0),
        };
    }

    /// <summary>Alias matching the Python <c>predict</c>.</summary>
    public DecisionResult Predict(object? state, QuestionSet questions) => SystemOne(state, questions);

    private Answer BuildAnswer(Question question, DecisionOutput output, int options)
    {
        float temperature = TemperatureFor(question.Type, options);
        var probabilities = new float[options];
        for (int i = 0; i < options; ++i) probabilities[i] = output.OptionLogits[i] / MathF.Max(1e-3f, temperature);
        SimdOps.Softmax(probabilities);

        double confidence = Math.Round(Calibration.ConfidenceFromProbabilities(probabilities, options), 4);
        var action = new ActionEstimate(Math.Round(output.ActionProbabilities[0], 4));

        switch (question.Type)
        {
            case QuestionType.Choice:
            {
                var keys = question.OptionKeys;
                return new Answer
                {
                    Type = "choice",
                    Choice = keys[SimdOps.ArgMax(probabilities)],
                    Probabilities = [.. keys.Select((key, i) => new KeyValuePair<string, double>(key, Math.Round(probabilities[i], 4)))],
                    Confidence = confidence,
                    Action = action,
                };
            }
            case QuestionType.Score:
            {
                double expected = 0d;
                for (int i = 0; i < options; ++i) expected += i * probabilities[i];
                var levels = question.Levels ?? [];
                return new Answer
                {
                    Type = "score",
                    Score = Math.Round(expected, 4),
                    Legend = [.. levels.Select((level, i) => new KeyValuePair<string, string>(
                        i.ToString(System.Globalization.CultureInfo.InvariantCulture), Question.RenderCriterion(level)))],
                    Probabilities = [.. Enumerable.Range(0, options).Select(i => new KeyValuePair<string, double>(
                        i.ToString(System.Globalization.CultureInfo.InvariantCulture), Math.Round(probabilities[i], 4)))],
                    Confidence = confidence,
                    Action = action,
                };
            }
            default:
            {
                double trueProbability = probabilities[1];
                return new Answer
                {
                    Type = "noul",
                    Noul = Math.Round(trueProbability, 4),
                    Confidence = Math.Round(Math.Max(trueProbability, 1d - trueProbability), 4),
                    Action = action,
                };
            }
        }
    }

    /// <summary>
    /// The calibration temperature for a question: the per-bucket value when one was fitted for
    /// this <c>type:option-count</c> combination, otherwise the per-type fallback.
    /// </summary>
    public float TemperatureFor(QuestionType type, int options)
    {
        string bucket = QuestionTypes.TemperatureBucket(type, options);
        if (Config.TemperatureByOptions.TryGetValue(bucket, out float scoped)) return scoped;
        int index = (int)type;
        return index < Config.Temperature.Count ? Config.Temperature[index] : 1f;
    }

    public void Dispose() => _weights.Dispose();

    private sealed class PrefixedRecorder(IStateRecorder inner, string prefix) : IStateRecorder
    {
        public void Record(string name, ReadOnlySpan<float> values, ReadOnlySpan<int> shape)
            => inner.Record(prefix + name, values, shape);
    }
}

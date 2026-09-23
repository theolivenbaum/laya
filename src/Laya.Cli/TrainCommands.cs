using System.Globalization;
using System.Text.Json;
using Laya.Models;
using Laya.Tokenizers;
using Laya.Training;

namespace Laya.Cli;

/// <summary><c>laya dataset</c>, <c>laya train</c> and <c>laya evaluate</c>.</summary>
internal static class TrainCommands
{
    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    /// <summary>Downloads a typed-decisions split to JSON lines.</summary>
    public static int Dataset(CommandLine options)
    {
        string path = Download(options, options.Value("split") ?? "train");
        Console.WriteLine($"Ready: {path}");
        return 0;
    }

    private static string Download(CommandLine options, string split)
    {
        string dataset = options.Value("dataset") ?? "LocalLLaMA/typed-decisions";
        string config = options.Value("config") ?? "all";
        string directory = options.Value("data-dir") ?? Path.Combine("artifacts", "data");
        var progress = new Progress<string>(line => Console.Error.WriteLine(line));
        return DecisionDataset.DownloadAsync(dataset, config, split, directory, options.Value("token"), progress)
            .GetAwaiter().GetResult();
    }

    /// <summary>A JSON-lines file given by <c>--flag</c>, or the named split of <c>--dataset</c>.</summary>
    private static List<DecisionCase> ReadCases(CommandLine options, string flag, string split)
    {
        string path = options.Value(flag) ?? Download(options, options.Value("split") ?? split);
        var cases = DecisionDataset.ReadJsonLines(path);
        if (options.Value("limit") is string limit) cases = [.. cases.Take(Int(limit))];
        return cases;
    }

    public static int Train(CommandLine options, Func<CommandLine, string> modelDirectory)
    {
        string source = modelDirectory(options);
        string output = options.Value("out") ?? throw new ArgumentException("train needs --out <directory>.");
        var config = LayaConfig.Load(Path.Combine(source, "rl_agent_config.json"));
        int maxLength = options.Value("max-len") is string ml ? Int(ml) : config.MaxLength;
        int headMaxLength = options.Value("head-max-len") is string hml ? Int(hml) : config.HeadMaxLength;

        string tokenizerDirectory = Path.Combine(source, "tokenizer");
        var tokenizer = HuggingFaceTokenizer.FromDirectory(Directory.Exists(tokenizerDirectory) ? tokenizerDirectory : source);
        var cases = ReadCases(options, "data", "train");
        var items = DecisionDataset.BuildItems(tokenizer, cases, maxLength, headMaxLength).Select(i => i.Item).ToList();
        Console.Error.WriteLine($"{cases.Count} cases -> {items.Count} training sequences (max_len {maxLength}, head_max_len {headMaxLength})");

        List<TrainingItem>? calibration = null;
        if (options.Value("calibration-data") is string calibrationPath)
        {
            calibration = [.. DecisionDataset.BuildItems(tokenizer, DecisionDataset.ReadJsonLines(calibrationPath), maxLength, headMaxLength)
                .Select(i => i.Item)];
        }

        var defaults = new TrainerOptions();
        var trainerOptions = defaults with
        {
            Epochs = Int(options.Value("epochs"), defaults.Epochs),
            MicroBatch = Int(options.Value("micro-batch"), defaults.MicroBatch),
            GradientAccumulation = Int(options.Value("grad-accum"), defaults.GradientAccumulation),
            GroupSize = Int(options.Value("group-size"), defaults.GroupSize),
            EncoderLearningRate = Double(options.Value("lr-encoder"), defaults.EncoderLearningRate),
            HeadLearningRate = Double(options.Value("lr-head"), defaults.HeadLearningRate),
            SigmaStart = Double(options.Value("sigma-start"), defaults.SigmaStart),
            SigmaEnd = Double(options.Value("sigma-end"), defaults.SigmaEnd),
            TrainableEncoderLayers = options.Value("train-layers") is string layers ? Int(layers) : null,
            MaxSteps = options.Value("max-steps") is string steps ? Int(steps) : null,
            Seed = Int(options.Value("seed"), defaults.Seed),
            LogEvery = Int(options.Value("log-every"), 10),
            RollingCheckpoint = !options.Has("no-rolling-checkpoint"),
            ModelName = options.Value("name") ?? "laya-fine-tuned",
        };

        var trainer = new Trainer(trainerOptions) { Progress = new SynchronousProgress(Print) };
        var report = trainer.Train(source, items, output, calibration, maxLength, headMaxLength);
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            output = report.OutputDirectory,
            optimizer_steps = report.OptimizerSteps,
            micro_steps = report.MicroSteps,
            epoch_loss = report.EpochLoss,
            temperature = report.Temperatures.ByType,
            temperature_by_options = report.Temperatures.ByBucket,
            trainable_parameters = report.TrainableParameters,
            hours = Math.Round(report.Elapsed.TotalHours, 3),
        }, Indented));

        if (options.Has("eval") || options.Value("eval-data") is not null)
        {
            return EvaluateDirectory(options, output);
        }
        return 0;
    }

    /// <summary>Refits a checkpoint's temperatures on labelled cases, leaving its weights alone.</summary>
    public static int Calibrate(CommandLine options, Func<CommandLine, string> modelDirectory)
    {
        string directory = modelDirectory(options);
        var config = LayaConfig.Load(Path.Combine(directory, "rl_agent_config.json"));
        var tokenizer = HuggingFaceTokenizer.FromDirectory(Path.Combine(directory, "tokenizer"));
        var items = DecisionDataset.BuildItems(tokenizer, ReadCases(options, "data", "train"), config.MaxLength, config.HeadMaxLength)
            .Select(i => i.Item).ToList();
        var sample = Trainer.CalibrationSample(items, new TrainerOptions
        {
            CalibrationStride = Int(options.Value("stride"), 1),
            CalibrationMaximum = Int(options.Value("max-items"), 400),
        });
        var fit = Trainer.Recalibrate(directory, sample);
        Console.WriteLine($"calibrated {directory} on {sample.Count} items: {Trainer.Describe(fit)}");
        return 0;
    }

    public static int Evaluate(CommandLine options, Func<CommandLine, string> modelDirectory)
        => EvaluateDirectory(options, modelDirectory(options));

    private static int EvaluateDirectory(CommandLine options, string directory)
    {
        using var agent = Agent.FromDirectory(directory);
        var cases = ReadCases(options, "eval-data", "test");
        var progress = new SynchronousProgress<string>(line => Console.Error.Write("\r" + line));
        var report = Evaluation.Evaluate(agent, cases, progress);
        Console.Error.WriteLine();
        string json = JsonSerializer.Serialize(report, Indented);
        Console.WriteLine(json);
        if (options.Value("report") is string path) File.WriteAllText(path, json);
        return 0;
    }

    private static void Print(TrainingProgress p)
    {
        if (p.Message is not null)
        {
            Console.Error.WriteLine($"[{p.Elapsed:hh\\:mm\\:ss}] {p.Message}");
            return;
        }
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"[{p.Elapsed:hh\\:mm\\:ss}] epoch {p.Epoch}/{p.Epochs} | micro-step {p.MicroStep} | step {p.OptimizerStep} | " +
            $"loss {p.Loss:F4} | reward {p.MeanReward:F3} | lr {p.LearningRate:E2} | sigma {p.Sigma:F2}"));
    }

    private static int Int(string? value, int fallback) => value is null ? fallback : Int(value);

    private static int Int(string value) => int.Parse(value, CultureInfo.InvariantCulture);

    private static double Double(string? value, double fallback)
        => value is null ? fallback : double.Parse(value, CultureInfo.InvariantCulture);

    /// <summary>
    /// Reports on the caller's thread. <see cref="Progress{T}"/> posts to the thread pool, which lets
    /// a progress line arrive after the result it describes.
    /// </summary>
    private sealed class SynchronousProgress<T>(Action<T> handler) : IProgress<T>
    {
        public void Report(T value) => handler(value);
    }

    private sealed class SynchronousProgress(Action<TrainingProgress> handler) : IProgress<TrainingProgress>
    {
        public void Report(TrainingProgress value) => handler(value);
    }
}

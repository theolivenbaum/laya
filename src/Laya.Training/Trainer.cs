using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Laya.Models;

namespace Laya.Training;

/// <summary>
/// Hyperparameters. The defaults are the reference notebook's (<c>train_ddp.py</c>): four epochs,
/// micro-batches of eight with four accumulation steps, a GRPO group of four, AdamW at 2.5e-5 for
/// the encoder and 1e-4 for everything else with cosine annealing, σ annealed 0.4 → 0.1, and
/// gradient clipping at 1.0. The notebook runs two GPUs, so its effective batch is 64; one process
/// here sees 32 per step unless <see cref="GradientAccumulation"/> is doubled.
/// </summary>
public sealed record TrainerOptions
{
    public int Epochs { get; init; } = 4;
    public int MicroBatch { get; init; } = 8;
    public int GradientAccumulation { get; init; } = 4;
    public int GroupSize { get; init; } = 4;
    public double EncoderLearningRate { get; init; } = 2.5e-5;
    public double HeadLearningRate { get; init; } = 1.0e-4;
    public double MinimumLearningRate { get; init; } = 1e-6;
    public double WeightDecay { get; init; } = 0.01;
    public double MaxGradientNorm { get; init; } = 1.0;
    public double SigmaStart { get; init; } = 0.4;
    public double SigmaEnd { get; init; } = 0.1;
    public double SphericalWeight { get; init; } = 0.75;
    public double RpsWeight { get; init; } = 1.0;
    public double CrossEntropyWeight { get; init; } = 1.0;
    public float HeadDropout { get; init; } = 0.1f;

    /// <summary>Null trains the whole encoder (the reference); <c>n</c> trains its top <c>n</c> layers; 0 freezes it.</summary>
    public int? TrainableEncoderLayers { get; init; }

    public int Seed { get; init; } = 42;

    /// <summary>Stop after this many optimizer steps (smoke tests, time-boxed runs).</summary>
    public int? MaxSteps { get; init; }

    /// <summary>
    /// Calibration uses one training item in <c>n</c>, at most <see cref="CalibrationMaximum"/> (the
    /// notebook: 15 and 400), drawn at random — see <see cref="Trainer.CalibrationSample"/>.
    /// </summary>
    public int CalibrationStride { get; init; } = 15;
    public int CalibrationMaximum { get; init; } = 400;

    /// <summary>Fit a <c>type:option-count</c> bucket once it has this many calibration samples.</summary>
    public int MinimumBucketSamples { get; init; } = 30;

    /// <summary>Write <c>checkpoint_latest</c> after every epoch, as the notebook does.</summary>
    public bool RollingCheckpoint { get; init; } = true;

    public string ModelName { get; init; } = "laya-fine-tuned";
    public int LogEvery { get; init; } = 50;
    public ParallelOptions? Parallel { get; init; }
}

/// <summary>A progress line: where training is and how it is going.</summary>
public sealed record TrainingProgress(int Epoch, int Epochs, int MicroStep, int OptimizerStep, double Loss,
    double MeanReward, double LearningRate, double Sigma, TimeSpan Elapsed, string? Message = null);

/// <summary>What a finished run reports.</summary>
public sealed record TrainingReport(int OptimizerSteps, int MicroSteps, double[] EpochLoss, TemperatureFit Temperatures,
    TimeSpan Elapsed, string OutputDirectory, long TrainableParameters);

/// <summary>
/// Fine-tunes a Laya checkpoint with the RLCD objective and writes a checkpoint
/// <c>Agent.FromDirectory</c> loads — the C# counterpart of the reference notebook's
/// <c>train_ddp.py</c>, single-process on the CPU.
/// </summary>
public sealed class Trainer(TrainerOptions? options = null)
{
    public TrainerOptions Options { get; } = options ?? new TrainerOptions();

    public IProgress<TrainingProgress>? Progress { get; init; }

    public TrainingReport Train(string sourceDirectory, IReadOnlyList<TrainingItem> items, string outputDirectory,
        IReadOnlyList<TrainingItem>? calibrationItems = null, int? maxLength = null, int? headMaxLength = null,
        CancellationToken cancellationToken = default)
    {
        if (items.Count == 0) throw new ArgumentException("no training items.", nameof(items));
        var o = Options;
        var parallel = LayaRuntime.Resolve(o.Parallel);
        var clock = Stopwatch.StartNew();

        var model = TrainableDecisionModel.FromDirectory(sourceDirectory);
        model.SetTrainableEncoderLayers(o.TrainableEncoderLayers);
        model.HeadDropout = o.HeadDropout;
        Report(new TrainingProgress(0, o.Epochs, 0, 0, 0, 0, 0, 0, clock.Elapsed,
            $"loaded {model.ParameterCount / 1e6:F0}M parameters, {model.TrainableParameterCount / 1e6:F1}M trainable; " +
            $"{items.Count} items"));

        var trainable = model.Parameters.Where(p => p.Trainable).ToList();
        var optimizer = new AdamW(
        [
            new ParameterGroup([.. trainable.Where(p => p.Name.Contains("encoder.", StringComparison.Ordinal))], o.EncoderLearningRate),
            new ParameterGroup([.. trainable.Where(p => !p.Name.Contains("encoder.", StringComparison.Ordinal))], o.HeadLearningRate),
        ], weightDecay: o.WeightDecay);
        int totalUpdates = items.Count / (o.MicroBatch * o.GradientAccumulation) * o.Epochs;
        var schedule = new CosineSchedule(optimizer, Math.Max(1, totalUpdates), o.MinimumLearningRate);
        var objective = new RlcdObjective
        {
            GroupSize = o.GroupSize,
            SphericalWeight = o.SphericalWeight,
            RpsWeight = o.RpsWeight,
            CrossEntropyWeight = o.CrossEntropyWeight,
        };

        var order = items.ToList();
        var epochLoss = new List<double>();
        int microStep = 0;
        bool stop = false;
        for (int epoch = 0; epoch < o.Epochs && !stop; ++epoch)
        {
            Shuffle(order, new Random(o.Seed + epoch));
            var noise = new Random(unchecked(o.Seed * 7919 + epoch));
            double progress = (double)epoch / Math.Max(1, o.Epochs - 1);
            double sigma = o.SigmaStart + (o.SigmaEnd - o.SigmaStart) * progress;
            double lossSum = 0d;
            int batches = 0, accumulated = 0;
            model.ZeroGradients();

            for (int start = 0; start < order.Count; start += o.MicroBatch)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var chunk = order.GetRange(start, Math.Min(o.MicroBatch, order.Count - start));
                ulong seed = Ops.Mix((ulong)o.Seed, (ulong)microStep + 1);

                var (logits, tape) = model.Forward(chunk, training: true, keepTape: true, seed, parallel);
                var result = objective.Compute(chunk, logits, sigma, noise, 1.0 / o.GradientAccumulation);
                model.Backward(tape!, result.LogitGradients, parallel);
                accumulated++;
                microStep++;

                if (accumulated % o.GradientAccumulation == 0 || start + o.MicroBatch >= order.Count)
                {
                    Gradients.ClipByGlobalNorm(trainable, o.MaxGradientNorm);
                    optimizer.Step(parallel);
                    schedule.Step();
                    model.ZeroGradients();
                    if (o.MaxSteps is int max && optimizer.Steps >= max) stop = true;
                }

                lossSum += result.Loss;
                batches++;
                if (batches % o.LogEvery == 0 || stop)
                {
                    Report(new TrainingProgress(epoch + 1, o.Epochs, batches, optimizer.Steps, result.Loss,
                        result.MeanReward, optimizer.Groups[^1].LearningRate, sigma, clock.Elapsed));
                }
                if (stop) break;
            }

            double average = lossSum / Math.Max(1, batches);
            epochLoss.Add(average);
            Report(new TrainingProgress(epoch + 1, o.Epochs, batches, optimizer.Steps, average, 0,
                optimizer.Groups[^1].LearningRate, sigma, clock.Elapsed, $"epoch {epoch + 1}/{o.Epochs} done, average loss {average:F4}"));

            if (o.RollingCheckpoint && !stop && epoch + 1 < o.Epochs)
            {
                string rolling = Path.Combine(outputDirectory, "checkpoint_latest");
                model.Save(rolling, sourceDirectory, BuildConfig(sourceDirectory, o, null, maxLength, headMaxLength, optimizer.Steps, epoch + 1, clock.Elapsed));
                File.WriteAllText(Path.Combine(rolling, "checkpoint_meta.json"), JsonSerializer.Serialize(
                    new { epoch = epoch + 1, total_epochs = o.Epochs, avg_loss = average }, new JsonSerializerOptions { WriteIndented = true }));
            }
        }

        // Post-training temperature calibration, on a sample of the training items unless given a set.
        var calibration = calibrationItems ?? CalibrationSample(items, o);
        var fit = Calibrate(model, calibration, o.MinimumBucketSamples, parallel);
        Report(new TrainingProgress(o.Epochs, o.Epochs, 0, optimizer.Steps, 0, 0, 0, 0, clock.Elapsed,
            $"calibrated on {calibration.Count} items: {Describe(fit)}"));

        model.Save(outputDirectory, sourceDirectory,
            BuildConfig(sourceDirectory, o, fit, maxLength, headMaxLength, optimizer.Steps, epochLoss.Count, clock.Elapsed));
        return new TrainingReport(optimizer.Steps, microStep, [.. epochLoss], fit, clock.Elapsed, outputDirectory,
            model.TrainableParameterCount);
    }

    private void Report(TrainingProgress progress) => Progress?.Report(progress);

    /// <summary>
    /// The notebook calibrates on <c>all_items[::15][:400]</c>. Items arrive grouped by case, and a
    /// typed-decisions case asks five questions in a fixed order, so a stride of 15 picks the same
    /// question of every third case: every sample is a <c>choice</c>, and <c>score</c> and <c>noul</c>
    /// are never fitted. The same number of items is drawn at random instead (seeded, so a run is
    /// reproducible), which samples every question of every workflow.
    /// </summary>
    public static List<TrainingItem> CalibrationSample(IReadOnlyList<TrainingItem> items, TrainerOptions options)
    {
        int count = Math.Min(options.CalibrationMaximum, (items.Count + options.CalibrationStride - 1) / Math.Max(1, options.CalibrationStride));
        var indices = Enumerable.Range(0, items.Count).ToArray();
        new Random(options.Seed).Shuffle(indices);
        return [.. indices.Take(count).Order().Select(i => items[i])];
    }

    /// <summary>Runs the calibration items through the model (no dropout) and fits the temperatures.</summary>
    public static TemperatureFit Calibrate(TrainableDecisionModel model, IReadOnlyList<TrainingItem> items,
        int minimumBucketSamples = 30, ParallelOptions? parallel = null)
    {
        var samples = new List<CalibrationSample>(items.Count);
        for (int start = 0; start < items.Count; start += 16)
        {
            var chunk = items.Skip(start).Take(16).ToList();
            var (logits, _) = model.Forward(chunk, training: false, keepTape: false, parallel: parallel);
            for (int i = 0; i < chunk.Count; ++i) samples.Add(new CalibrationSample(chunk[i].QuestionType, logits[i], chunk[i].Target));
        }
        return TemperatureFitting.Fit(samples, minimumBucketSamples);
    }

    public static string Describe(TemperatureFit fit) => string.Create(CultureInfo.InvariantCulture,
        $"choice {fit.ByType[0]:F3}, score {fit.ByType[1]:F3}, noul {fit.ByType[2]:F3}; " +
        $"buckets {string.Join(", ", fit.ByBucket.Select(b => $"{b.Key}={b.Value:F3} (n={fit.SamplesPerBucket[b.Key]})"))}");

    /// <summary>
    /// Refits the temperatures of an existing checkpoint in place: the weights are untouched, only
    /// <c>temperature</c> and <c>temperature_by_options</c> in <c>rl_agent_config.json</c> change.
    /// </summary>
    public static TemperatureFit Recalibrate(string directory, IReadOnlyList<TrainingItem> items,
        int minimumBucketSamples = 30, ParallelOptions? parallel = null)
    {
        var model = TrainableDecisionModel.FromDirectory(directory);
        var fit = Calibrate(model, items, minimumBucketSamples, parallel);
        string path = Path.Combine(directory, "rl_agent_config.json");
        var config = JsonNode.Parse(File.ReadAllText(path))!.AsObject();
        WriteTemperatures(config, fit);
        File.WriteAllText(path, config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        return fit;
    }

    private static void WriteTemperatures(JsonObject config, TemperatureFit fit)
    {
        config["temperature"] = new JsonArray([.. fit.ByType.Select(t => (JsonNode)JsonValue.Create((double)t))]);
        // Replace, never merge: a bucket left over from the base checkpoint would shadow the new fit.
        var buckets = new JsonObject();
        foreach (var (bucket, value) in fit.ByBucket) buckets[bucket] = (double)value;
        config["temperature_by_options"] = buckets;
    }

    private static void Shuffle<T>(List<T> list, Random random)
    {
        for (int i = list.Count - 1; i > 0; --i)
        {
            int j = random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    /// <summary>
    /// The source checkpoint's <c>rl_agent_config.json</c>, with what training changed. Keys this code
    /// does not know about are kept, so a config written by a newer Python package survives a C# run.
    /// </summary>
    private static JsonObject BuildConfig(string sourceDirectory, TrainerOptions o, TemperatureFit? fit, int? maxLength,
        int? headMaxLength, int steps, int epochs, TimeSpan elapsed)
    {
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(sourceDirectory, "rl_agent_config.json")))!.AsObject();
        if (maxLength is int ml) config["max_len"] = ml;
        if (headMaxLength is int hml) config["head_max_len"] = hml;
        config["fine_tuned"] = true;
        config["model_name"] = o.ModelName;
        if (fit is not null) WriteTemperatures(config, fit);
        config["training"] = new JsonObject
        {
            ["updates"] = steps,
            ["epochs_completed"] = epochs,
            ["hours"] = Math.Round(elapsed.TotalHours, 2),
            ["world_size"] = 1,
            ["fine_tuned_from_checkpoint"] = true,
            ["trainer"] = "Laya.Training (.NET)",
            ["trainable_encoder_layers"] = o.TrainableEncoderLayers is int n ? n : null,
        };
        return config;
    }
}

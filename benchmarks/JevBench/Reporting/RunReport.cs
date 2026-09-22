using Laya.JevBench.Dataset;
using Laya.JevBench.Scoring;

namespace Laya.JevBench.Reporting;

public sealed record TierSummary
{
    public required int N { get; init; }
    public required int Correct { get; init; }
    public required int Wrong { get; init; }
    public required int Failed { get; init; }
    public required int NoGroundTruth { get; init; }
    public required double Accuracy { get; init; }
    public required IReadOnlyDictionary<string, (int Correct, int N)> ByFamily { get; init; }

    /// <summary>
    /// Mean Brier score over items with a <em>valid</em> distribution only (matching
    /// <c>jevbench/summarize.py:metric</c> — invalid/failed answers are excluded from calibration
    /// entirely, unlike accuracy, which counts them as wrong).
    /// </summary>
    public double? BrierMean { get; init; }
}

public sealed record CalibrationSummary
{
    public double? Ece { get; init; }
    public int EceN { get; init; }
    public double? ProbabilityFidelity { get; init; }
    public int FidelityN { get; init; }
    public double? Score { get; init; }
    // "What if this system reported hard 0/1 confidence instead of a calibrated distribution?" —
    // the same simulated-one-hot comparator the baseline row publishes.
    public double? OnehotEce { get; init; }
    public double? OnehotProbabilityFidelity { get; init; }
    public double? OnehotScore { get; init; }
}

/// <summary>
/// Aggregates a completed run into the same shape the copied baseline (<c>data/baseline/laya.json</c>,
/// <c>laya-per-task.json</c>) publishes, so the two can be compared field for field. See
/// <see cref="Scoring.Composite"/> for why the composite score here is informative for
/// regression-tracking rather than a reproduction of the ranked JevBench Score.
/// </summary>
public sealed class RunReport
{
    public required IReadOnlyList<TaskResult> Results { get; init; }
    public required IReadOnlyDictionary<string, JevTask> TasksById { get; init; }
    public required string EndpointCondition { get; init; }
    public required string EndpointKind { get; init; }

    public IReadOnlyDictionary<string, TierSummary> TierSummaries { get; private set; } = null!;
    public IReadOnlyDictionary<string, double> TierChances { get; private set; } = null!;
    public CalibrationSummary Calibration { get; private set; } = null!;
    public LatencySummary Latency { get; private set; } = null!;
    public double MeanInputTokens { get; private set; }
    public double? Intelligence { get; private set; }
    public double? CalibrationScore { get; private set; }
    public double? Speed { get; private set; }
    public double? CostUsdPer1000 { get; private set; }
    public double? Cost { get; private set; }
    public double? JevBenchScore { get; private set; }
    public ParaphraseConsistencyResult Paraphrase { get; private set; } = null!;

    public static RunReport Build(IReadOnlyList<TaskResult> results, IReadOnlyDictionary<string, JevTask> tasksById,
        string endpointCondition, string endpointKind)
    {
        var report = new RunReport
        {
            Results = results, TasksById = tasksById, EndpointCondition = endpointCondition, EndpointKind = endpointKind,
        };
        report.Compute();
        return report;
    }

    private void Compute()
    {
        var tiers = new Dictionary<string, TierSummary>(StringComparer.Ordinal);
        var tierChances = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var tierGroup in Results.GroupBy(r => r.Tier))
        {
            int correct = 0, wrong = 0, failed = 0, noGt = 0;
            var byFamily = new Dictionary<string, (int Correct, int N)>(StringComparer.Ordinal);
            var chances = new List<int>();
            var briers = new List<double>();
            foreach (var r in tierGroup)
            {
                switch (r.Outcome)
                {
                    case "c": correct++; break;
                    case "w": wrong++; break;
                    case "f": failed++; break;
                    default: noGt++; break;
                }
                if (r.Outcome is "c" or "w")
                {
                    var (c, n) = byFamily.GetValueOrDefault(r.Family, (0, 0));
                    byFamily[r.Family] = (c + (r.Outcome == "c" ? 1 : 0), n + 1);
                }
                chances.Add(TasksById[r.Id].Labels.Count);
                if (r.Ok && r.Scored.Valid && r.Scored.Correct is not null)
                {
                    var task = TasksById[r.Id];
                    briers.Add(Metrics.BrierScore(r.Scored.Probs!, task.ExpectedLabel!, task.Labels));
                }
            }
            int scorable = correct + wrong + failed; // items with ground truth (excludes "n")
            tiers[tierGroup.Key] = new TierSummary
            {
                N = tierGroup.Count(), Correct = correct, Wrong = wrong, Failed = failed, NoGroundTruth = noGt,
                Accuracy = scorable > 0 ? (double)correct / scorable : 0.0,
                ByFamily = byFamily,
                BrierMean = briers.Count > 0 ? briers.Average() : null,
            };
            tierChances[tierGroup.Key] = Composite.Chance(chances);
        }
        TierSummaries = tiers;
        TierChances = tierChances;

        var accuracyByTier = tiers.ToDictionary(kv => kv.Key, kv => kv.Value.Accuracy, StringComparer.Ordinal);
        Intelligence = Composite.Intelligence(accuracyByTier, tierChances);

        Latency = Metrics.LatencySummaryOf([.. Results.Where(r => r.Ok).Select(r => r.LatencyS)]);
        MeanInputTokens = Results.Count(r => r.Ok) > 0 ? Results.Where(r => r.Ok).Average(r => r.InputTokens) : 0.0;

        Calibration = ComputeCalibration();
        CalibrationScore = Calibration.Score;

        Speed = Composite.Speed(Latency.P50S, Latency.P95S, EndpointKind);

        // Same estimate basis the baseline row uses (no local-weights provider tariff exists):
        // a deepinfra encoder of the same size class, $0.01 / M input tokens, nothing generated.
        CostUsdPer1000 = MeanInputTokens * 1000 * 0.01 / 1_000_000;
        Cost = CostUsdPer1000 is > 0 ? Composite.Cost(CostUsdPer1000.Value) : null;

        var axes = new Dictionary<string, double?>
        {
            ["intelligence"] = Intelligence, ["calibration"] = CalibrationScore, ["speed"] = Speed, ["cost"] = Cost,
        };
        JevBenchScore = Composite.JevBenchScore(axes);

        var outcomesById = Results.ToDictionary(r => r.Id, r => r.Scored, StringComparer.Ordinal);
        Paraphrase = Metrics.ParaphraseConsistency(outcomesById, [.. TasksById.Values]);
    }

    private CalibrationSummary ComputeCalibration()
    {
        var hard = Results.Where(r => r.Tier == "hard" && r.Ok && r.Scored.Valid && r.Scored.Correct is not null).ToList();
        if (hard.Count == 0) return new CalibrationSummary();

        var pairs = hard.Select(r => (Confidence: Scoring.Scoring.TopLabelConfidence(r.Scored.Probs!), Correct: r.Scored.Correct!.Value)).ToList();
        var ece = Metrics.EceTopLabel(pairs);
        var onehotPairs = hard.Select(r => (Confidence: 1.0, Correct: r.Scored.Correct!.Value)).ToList();
        var onehotEce = Metrics.EceTopLabel(onehotPairs);

        var fidelityItems = hard
            .Select(r => (Result: r, Task: TasksById[r.Id]))
            .Where(t => t.Task.GoldProbabilities is not null)
            .ToList();

        double? meanTvd = null, onehotMeanTvd = null;
        if (fidelityItems.Count > 0)
        {
            var tvds = fidelityItems.Select(t =>
                Metrics.Tvd(t.Result.Scored.Probs!, t.Task.GoldProbabilities!, t.Task.Labels)).ToList();
            meanTvd = tvds.Average();
            var onehotTvds = fidelityItems.Select(t =>
            {
                var onehot = t.Task.Labels.ToDictionary(l => l, l => l == t.Result.Scored.Predicted ? 1.0 : 0.0, StringComparer.Ordinal);
                return Metrics.Tvd(onehot, t.Task.GoldProbabilities!, t.Task.Labels);
            }).ToList();
            onehotMeanTvd = onehotTvds.Average();
        }

        return new CalibrationSummary
        {
            Ece = ece.Ece, EceN = ece.N,
            ProbabilityFidelity = meanTvd is null ? null : 100 * (1 - meanTvd.Value), FidelityN = fidelityItems.Count,
            Score = Composite.Calibration(ece.Ece, meanTvd),
            OnehotEce = onehotEce.Ece,
            OnehotProbabilityFidelity = onehotMeanTvd is null ? null : 100 * (1 - onehotMeanTvd.Value),
            OnehotScore = Composite.Calibration(onehotEce.Ece, onehotMeanTvd),
        };
    }
}

using System.Text.Json;
using System.Text.Json.Serialization;

namespace Laya.JevBench.Reporting;

/// <summary>Writes a completed run in (close to) the shape of the copied baseline files, for diffing.</summary>
public static class JsonOutput
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public static void WriteSummary(string path, RunReport report, string modelDirectory, string note)
    {
        var tiers = report.TierSummaries.ToDictionary(kv => kv.Key, kv => (object)new
        {
            n = kv.Value.N,
            correct = kv.Value.Correct,
            wrong = kv.Value.Wrong,
            failed = kv.Value.Failed,
            accuracy = kv.Value.Accuracy,
            chance = report.TierChances[kv.Key],
            brier_mean = kv.Value.BrierMean,
            by_family = kv.Value.ByFamily.ToDictionary(f => f.Key, f => (object)new { correct = f.Value.Correct, n = f.Value.N, accuracy = (double)f.Value.Correct / f.Value.N }),
        });

        var payload = new
        {
            key = "laya-dotnet",
            display = "Laya (.NET port, this repo)",
            note,
            model_directory = modelDirectory,
            endpoint_condition = report.EndpointCondition,
            coverage = new
            {
                total_public_items = report.Results.Count,
                easy = report.TierSummaries.GetValueOrDefault("easy")?.N ?? 0,
                standard_original_only = report.TierSummaries.GetValueOrDefault("standard")?.N ?? 0,
                standard_full_tier_n = 96,
                hard_public_only = report.TierSummaries.GetValueOrDefault("hard")?.N ?? 0,
                hard_full_tier_n = 220,
                judge_public_items = 0,
                judge_full_tier_n = 146,
                caveat = "judge tier has zero public items (both its cohorts are held out); standard and hard are public-only subsets of their full tiers. Tier accuracies below are computed over exactly the items this run scored, not the full v1.2 tiers.",
            },
            tiers,
            latency = new { n = report.Latency.N, p50_s = report.Latency.P50S, p95_s = report.Latency.P95S },
            mean_input_tokens = report.MeanInputTokens,
            calibration = new
            {
                hard_tier_only = true,
                ece = report.Calibration.Ece,
                ece_n = report.Calibration.EceN,
                probability_fidelity = report.Calibration.ProbabilityFidelity,
                probability_fidelity_n_of_20 = report.Calibration.FidelityN,
                score = report.Calibration.Score,
                onehot = new
                {
                    note = "what the calibration score would be if this system reported hard 0/1 confidence instead of a calibrated distribution",
                    ece = report.Calibration.OnehotEce,
                    probability_fidelity = report.Calibration.OnehotProbabilityFidelity,
                    score = report.Calibration.OnehotScore,
                },
            },
            paraphrase_consistency = new
            {
                pairs = report.Paraphrase.Pairs,
                both_answered_valid = report.Paraphrase.BothAnsweredValid,
                both_correct = report.Paraphrase.BothCorrect,
                consistency = report.Paraphrase.Consistency,
            },
            cost = new
            {
                kind = "estimate",
                usd_per_1000 = report.CostUsdPer1000,
                basis = "ESTIMATE, same basis as the published baseline row: a hosted deepinfra encoder of the same size class, $0.01/M input tokens, $0.0/M output x this run's own measured mean input tokens per decision. Local weights have no provider tariff of their own.",
            },
            axes = new
            {
                intelligence = report.Intelligence,
                calibration = report.CalibrationScore,
                speed = report.Speed,
                cost = report.Cost,
            },
            jevbench_score = new
            {
                value = report.JevBenchScore,
                caveat = "Not the ranked JevBench Score: chance is recomputed from the public subset this tool actually ran (see Scoring/Composite.cs), and only easy/standard(partial)/hard(partial) tiers have any coverage. Useful for tracking laya's own public-item accuracy/calibration/speed over time, not for comparing against the published leaderboard.",
            },
        };

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(payload, Options));
    }

    public static void WritePerTask(string path, RunReport report)
    {
        var byTier = report.TierSummaries.ToDictionary(kv => kv.Key, kv => (object)new
        {
            c = kv.Value.Correct, w = kv.Value.Wrong, f = kv.Value.Failed, n = kv.Value.NoGroundTruth,
            accuracy = Math.Round(kv.Value.Accuracy, 4),
        });

        var publicTasks = report.Results.ToDictionary(r => r.Id,
            r => (object)new object[] { r.Outcome, Math.Round(r.LatencyS, 4) });

        var payload = new
        {
            display = "Laya (.NET port, this repo)",
            by_tier = byTier,
            public_tasks = publicTasks,
        };

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(payload, Options));
    }

    public static void WriteComparison(string path, Baseline baseline, RunReport report, IReadOnlyList<TaskDelta> deltas)
    {
        var payload = new
        {
            baseline_display = baseline.Display,
            baseline_source = "results/v1.2/additions/laya.json + laya-per-task.json (JevBench v1.2.2, python `laya` package 0.3.3, our CPU, 4 threads, Ryzen 5 3600), restricted to the 231 public items",
            tier_accuracy = report.TierSummaries.Keys.ToDictionary(t => t, t => (object)new
            {
                baseline = baseline.TierAccuracy.GetValueOrDefault(t),
                current = report.TierSummaries[t].Accuracy,
                delta = report.TierSummaries[t].Accuracy - baseline.TierAccuracy.GetValueOrDefault(t),
            }),
            n_flipped = deltas.Count,
            n_regressed = deltas.Count(d => d.Regressed),
            n_improved = deltas.Count(d => d.Improved),
            flipped_tasks = deltas.Select(d => new { d.Id, baseline = d.Baseline, current = d.Current, d.Regressed, d.Improved }),
        };

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(payload, Options));
    }
}

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Serialization;
using Laya.Runtime;

namespace Laya.Training;

/// <summary>The benchmark metrics of the reference notebook (cells 14 and 18).</summary>
public sealed record EvaluationReport
{
    [JsonPropertyName("n_cases")] public int Cases { get; init; }
    [JsonPropertyName("n_decisions")] public int Decisions { get; init; }
    [JsonPropertyName("accuracy")] public double Accuracy { get; init; }
    [JsonPropertyName("soft_accuracy")] public double SoftAccuracy { get; init; }
    [JsonPropertyName("brier_score")] public double Brier { get; init; }
    [JsonPropertyName("ece")] public double Ece { get; init; }
    [JsonPropertyName("score_mae")] public double ScoreMae { get; init; }
    [JsonPropertyName("within_1_level")] public double WithinOneLevel { get; init; }
    [JsonPropertyName("kl_divergence")] public double KlDivergence { get; init; }
    [JsonPropertyName("total_variation")] public double TotalVariation { get; init; }
    [JsonPropertyName("latency_p50_ms")] public double LatencyP50 { get; init; }
    [JsonPropertyName("latency_p95_ms")] public double LatencyP95 { get; init; }
    [JsonPropertyName("per_workflow")] public IReadOnlyDictionary<string, WorkflowAccuracy> PerWorkflow { get; init; } =
        new Dictionary<string, WorkflowAccuracy>();
}

public sealed record WorkflowAccuracy(
    [property: JsonPropertyName("n_decisions")] int Decisions,
    [property: JsonPropertyName("accuracy")] double Accuracy);

/// <summary>Runs a checkpoint over typed-decisions cases and scores it exactly as the notebook does.</summary>
public static class Evaluation
{
    public static EvaluationReport Evaluate(IDecisionEngine engine, IReadOnlyList<DecisionCase> cases,
        IProgress<string>? progress = null, ParallelOptions? parallel = null)
    {
        var accuracies = new List<double>();
        var soft = new List<double>();
        var brier = new List<double>();
        var kl = new List<double>();
        var tv = new List<double>();
        var mae = new List<double>();
        var withinOne = new List<double>();
        var confidences = new List<double>();
        var corrects = new List<bool>();
        var latencies = new List<double>();
        var perWorkflow = new Dictionary<string, List<double>>(StringComparer.Ordinal);

        for (int c = 0; c < cases.Count; ++c)
        {
            var @case = cases[c];
            var stopwatch = Stopwatch.StartNew();
            var result = engine.SystemOne(@case.State, @case.Questions, parallel);
            latencies.Add(stopwatch.Elapsed.TotalMilliseconds);

            foreach (var (id, question) in @case.Questions)
            {
                if (!@case.Gold.TryGetValue(id, out var gold) || !result.TryGet(id, out var answer)) continue;
                double correct;
                switch (question.Type)
                {
                    case QuestionType.Choice:
                    {
                        var keys = question.OptionKeys;
                        correct = answer.Choice == gold.Label ? 1 : 0;
                        var p = Normalise([.. keys.Select(k => ProbabilityOrDefault(answer, k))]);
                        var g = Normalise([.. keys.Select(k => gold.Probabilities.TryGetValue(k, out double v) ? v : 1e-6)]);
                        confidences.Add(p.Max());
                        Distribution(p, g, soft, brier, tv, kl);
                        break;
                    }
                    case QuestionType.Noul:
                    {
                        double pv = answer.Noul ?? 0.5;
                        double gv = gold.Noul ?? gold.Probabilities.GetValueOrDefault("true", 0.5);
                        string predicted = pv >= 0.5 ? "true" : "false";
                        correct = string.Equals(predicted, gold.Label, StringComparison.OrdinalIgnoreCase) ? 1 : 0;
                        confidences.Add(Math.Max(pv, 1 - pv));
                        Distribution([1 - pv, pv], [1 - gv, gv], soft, brier, tv, kl);
                        break;
                    }
                    default:
                    {
                        double predictedScore = answer.Score ?? 0;
                        double goldScore = gold.Score ?? 0;
                        mae.Add(Math.Abs(predictedScore - goldScore));
                        withinOne.Add(Math.Abs(predictedScore - goldScore) <= 1.0 ? 1 : 0);
                        int levels = question.Levels?.Count ?? 0;
                        var probabilities = Enumerable.Range(0, levels)
                            .Select(i => answer.ProbabilityOf(i.ToString(CultureInfo.InvariantCulture))).ToArray();
                        int predictedLevel;
                        if (probabilities.Sum() > 0)
                        {
                            var p = Normalise(probabilities);
                            predictedLevel = Array.IndexOf(p, p.Max());
                            confidences.Add(p.Max());
                        }
                        else
                        {
                            predictedLevel = (int)Math.Round(predictedScore, MidpointRounding.ToEven);
                            confidences.Add(0.5);
                        }
                        int goldLevel = int.TryParse(gold.Label, NumberStyles.Integer, CultureInfo.InvariantCulture, out int l)
                            ? l
                            : (int)Math.Round(goldScore, MidpointRounding.ToEven);
                        correct = predictedLevel == goldLevel ? 1 : 0;
                        break;
                    }
                }
                accuracies.Add(correct);
                corrects.Add(correct > 0);
                if (@case.Workflow is string workflow)
                {
                    if (!perWorkflow.TryGetValue(workflow, out var list)) perWorkflow[workflow] = list = [];
                    list.Add(correct);
                }
            }
            progress?.Report($"{c + 1}/{cases.Count} cases, accuracy so far {Mean(accuracies):F3}");
        }

        latencies.Sort();
        return new EvaluationReport
        {
            Cases = cases.Count,
            Decisions = accuracies.Count,
            Accuracy = Mean(accuracies),
            SoftAccuracy = Mean(soft),
            Brier = Mean(brier),
            Ece = Calibration.ExpectedCalibrationError([.. confidences], [.. corrects]),
            ScoreMae = Mean(mae),
            WithinOneLevel = Mean(withinOne),
            KlDivergence = Mean(kl),
            TotalVariation = Mean(tv),
            LatencyP50 = Percentile(latencies, 50),
            LatencyP95 = Percentile(latencies, 95),
            PerWorkflow = perWorkflow.OrderBy(p => p.Key, StringComparer.Ordinal)
                .ToDictionary(p => p.Key, p => new WorkflowAccuracy(p.Value.Count, Math.Round(Mean(p.Value), 4))),
        };
    }

    /// <summary><c>p_ans["probabilities"].get(k, 1e-6)</c>: a reported 0 stays 0, only a missing key defaults.</summary>
    private static double ProbabilityOrDefault(Answer answer, string key)
    {
        foreach (var (name, value) in answer.Probabilities ?? []) if (name == key) return value;
        return 1e-6;
    }

    private static double[] Normalise(double[] values)
    {
        double sum = values.Sum();
        return sum > 0 ? [.. values.Select(v => v / sum)] : values;
    }

    private static void Distribution(double[] p, double[] g, List<double> soft, List<double> brier, List<double> tv, List<double> kl)
    {
        double s = 0, b = 0, t = 0, k = 0;
        for (int i = 0; i < p.Length; ++i)
        {
            s += p[i] * g[i];
            b += (p[i] - g[i]) * (p[i] - g[i]);
            t += Math.Abs(p[i] - g[i]);
            k += g[i] * Math.Log(Math.Clamp(g[i] / p[i], 1e-12, 1e4));
        }
        soft.Add(s);
        brier.Add(b);
        tv.Add(0.5 * t);
        kl.Add(k);
    }

    private static double Mean(List<double> values) => values.Count == 0 ? 0 : values.Average();

    /// <summary><c>numpy.percentile</c> with its default linear interpolation, over sorted values.</summary>
    private static double Percentile(List<double> sorted, double percentile)
    {
        if (sorted.Count == 0) return 0;
        double rank = percentile / 100 * (sorted.Count - 1);
        int low = (int)Math.Floor(rank), high = (int)Math.Ceiling(rank);
        return sorted[low] + (sorted[high] - sorted[low]) * (rank - low);
    }
}

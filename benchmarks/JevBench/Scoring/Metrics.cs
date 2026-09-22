using Laya.JevBench.Dataset;

namespace Laya.JevBench.Scoring;

public sealed record EceResult(double Ece, int N);

public sealed record LatencySummary(int N, double? P50S, double? P95S);

public sealed record ParaphraseConsistencyResult(
    int Pairs, int BothAnsweredValid, int BothCorrect, int UnscorableExpectedNone, double? Consistency);

/// <summary>Port of <c>jevbench/metrics.py</c>'s pure scoring functions.</summary>
public static class Metrics
{
    /// <summary>
    /// Multi-class Brier sum over the exact label set. Binary (noul) questions use the 2-class
    /// convention <c>(p_yes - y)^2 + (p_no - (1-y))^2</c>, so binary and multi-class Brier are
    /// directly comparable.
    /// </summary>
    public static double BrierScore(IReadOnlyDictionary<string, double> probs, string expectedLabel, IReadOnlyList<string> labels)
    {
        if (labels.Count == 2)
        {
            if (!probs.TryGetValue(expectedLabel, out double pYes))
            {
                throw new KeyNotFoundException($"expected label '{expectedLabel}' missing from probs");
            }
            string noLabel = labels[1] == expectedLabel ? labels[0] : labels[1];
            double pNo = probs.TryGetValue(noLabel, out double v) ? v : 1.0 - pYes;
            return (pYes - 1.0) * (pYes - 1.0) + (pNo - 0.0) * (pNo - 0.0);
        }
        double total = 0.0;
        foreach (string label in labels)
        {
            double target = label == expectedLabel ? 1.0 : 0.0;
            double p = probs.TryGetValue(label, out double v) ? v : 0.0;
            total += (p - target) * (p - target);
        }
        return total;
    }

    /// <summary>Expected calibration error, top-label confidence, 10 equal-width bins.</summary>
    public static EceResult EceTopLabel(IReadOnlyList<(double Confidence, bool Correct)> pairs, int nBins = 10)
    {
        var n = new int[nBins];
        var confSum = new double[nBins];
        var correct = new int[nBins];
        foreach (var (confidenceRaw, isCorrect) in pairs)
        {
            double confidence = Math.Min(Math.Max(confidenceRaw, 0.0), 1.0);
            int idx = Math.Min((int)(confidence * nBins), nBins - 1);
            n[idx]++;
            confSum[idx] += confidence;
            if (isCorrect) correct[idx]++;
        }
        int total = n.Sum();
        double ece = 0.0;
        for (int i = 0; i < nBins; ++i)
        {
            if (n[i] == 0) continue;
            double acc = (double)correct[i] / n[i];
            double meanConf = confSum[i] / n[i];
            ece += (double)n[i] / total * Math.Abs(acc - meanConf);
        }
        return new EceResult(ece, total);
    }

    /// <summary>Mean absolute error over (expected level, predicted expected value) pairs.</summary>
    public static double? OrdinalMae(IReadOnlyList<(int Expected, double PredictedEv)> pairs)
        => pairs.Count == 0 ? null : pairs.Average(p => Math.Abs(p.Expected - p.PredictedEv));

    /// <summary>Linear-interpolation percentile, matching Python's <c>metrics.percentile</c>.</summary>
    public static double? Percentile(IReadOnlyList<double> values, double q)
    {
        if (values.Count == 0) return null;
        var sorted = values.OrderBy(v => v).ToArray();
        double k = (sorted.Length - 1) * q;
        int f = (int)Math.Floor(k);
        int c = (int)Math.Ceiling(k);
        if (f == c) return sorted[(int)k];
        return sorted[f] * (c - k) + sorted[c] * (k - f);
    }

    public static LatencySummary LatencySummaryOf(IReadOnlyList<double> seconds)
        => new(seconds.Count, Percentile(seconds, 0.5), Percentile(seconds, 0.95));

    /// <summary>
    /// Total-variation distance between two label distributions: <c>0.5 * sum |p(l) - q(l)|</c>.
    /// Port of <c>composite_v13.tvd</c>.
    /// </summary>
    public static double Tvd(IReadOnlyDictionary<string, double> p, IReadOnlyDictionary<string, double> q, IReadOnlyList<string> labels)
        => 0.5 * labels.Sum(label =>
            Math.Abs((p.TryGetValue(label, out double pv) ? pv : 0.0) - (q.TryGetValue(label, out double qv) ? qv : 0.0)));

    /// <summary>
    /// Both-correct consistency over paraphrase pairs (<c>task.Group</c>). A pair counts once both
    /// members have a valid schema outcome; expected=null members make a pair unscorable.
    /// </summary>
    public static ParaphraseConsistencyResult ParaphraseConsistency(
        IReadOnlyDictionary<string, ScoredOutcome> outcomesById, IReadOnlyList<JevTask> tasks)
    {
        var groups = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (var task in tasks)
        {
            if (task.Group is null) continue;
            if (!groups.TryGetValue(task.Group, out var list)) groups[task.Group] = list = [];
            list.Add(task.Id);
        }

        int pairs = 0, bothAnswered = 0, bothCorrect = 0, unscorable = 0;
        foreach (var (_, ids) in groups.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            if (ids.Count < 2) continue;
            pairs++;
            var outcomes = ids.Select(id => outcomesById.GetValueOrDefault(id)).ToArray();
            if (outcomes.Any(o => o is null || !o.Valid)) continue;
            bothAnswered++;
            if (outcomes.Any(o => o!.Correct is null)) { unscorable++; continue; }
            if (outcomes.All(o => o!.Correct == true)) bothCorrect++;
        }
        return new ParaphraseConsistencyResult(pairs, bothAnswered, bothCorrect, unscorable,
            bothAnswered > 0 ? (double)bothCorrect / bothAnswered : null);
    }
}

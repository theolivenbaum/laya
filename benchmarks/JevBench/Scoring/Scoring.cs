using System.Globalization;
using Laya.JevBench.Dataset;

namespace Laya.JevBench.Scoring;

/// <summary>Thrown by <see cref="Scoring.ValidateProbs"/> — port of JevBench's <c>InvalidDistribution</c>.</summary>
public sealed class InvalidDistributionException(string message) : Exception(message);

/// <summary>
/// The outcome of scoring one distribution against a task — port of <c>jevbench/scoring.py:score_task</c>.
/// </summary>
public sealed record ScoredOutcome
{
    public required bool Valid { get; init; }
    public required bool StrictValid { get; init; }
    public required bool Renormalized { get; init; }
    public IReadOnlyDictionary<string, double>? Probs { get; init; }
    public bool? Correct { get; init; }
    public string? Predicted { get; init; }
    public double? OrdinalEv { get; init; }
    public string? Error { get; init; }
}

/// <summary>
/// Distribution validation, argmax and ordinal expected-value scoring — a line-for-line port of
/// <c>jevbench/scoring.py</c>. Malformed distributions fail closed: invalid and counted wrong,
/// never repaired into a probability.
/// </summary>
public static class Scoring
{
    public const double SumTol = 1e-3;

    // v1 froze SUM_TOL = 1e-3 before the run; RENORM_TOL is the band inside which a distribution
    // that misses SUM_TOL (three-decimal rounding on a many-option answer) is rescaled and scored
    // normally instead of thrown out for arithmetic. Outside it, still invalid, still wrong.
    public const double RenormTol = 2e-2;

    public static IReadOnlyDictionary<string, double> ValidateProbs(
        IReadOnlyDictionary<string, double> probs, IReadOnlyList<string> labels, double sumTol = SumTol)
    {
        var want = new HashSet<string>(labels, StringComparer.Ordinal);
        var got = new HashSet<string>(probs.Keys, StringComparer.Ordinal);
        if (!want.SetEquals(got))
        {
            var missing = want.Except(got).OrderBy(x => x, StringComparer.Ordinal);
            var extra = got.Except(want).OrderBy(x => x, StringComparer.Ordinal);
            throw new InvalidDistributionException(
                $"label keys mismatch: missing=[{string.Join(", ", missing)}] extra=[{string.Join(", ", extra)}]");
        }
        var clean = new Dictionary<string, double>(StringComparer.Ordinal);
        double total = 0.0;
        foreach (var (key, value) in probs)
        {
            if (!double.IsFinite(value)) throw new InvalidDistributionException($"prob['{key}'] is not finite");
            if (value is < 0.0 or > 1.0) throw new InvalidDistributionException($"prob['{key}'] out of [0,1]: {value}");
            clean[key] = value;
            total += value;
        }
        if (Math.Abs(total - 1.0) > sumTol)
        {
            throw new InvalidDistributionException($"probs sum to {total}, tolerance {sumTol}");
        }
        return clean;
    }

    /// <summary>Deterministic argmax: ties broken by the lexicographically smallest label.</summary>
    public static string ArgmaxLabel(IReadOnlyDictionary<string, double> probs)
    {
        string? best = null;
        double bestP = -1.0;
        foreach (string key in probs.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            if (probs[key] > bestP) { best = key; bestP = probs[key]; }
        }
        return best ?? throw new InvalidOperationException("argmax over an empty distribution");
    }

    public static double ExpectedValue(IReadOnlyDictionary<string, double> probs)
        => probs.Sum(kv => double.Parse(kv.Key, CultureInfo.InvariantCulture) * kv.Value);

    public static double TopLabelConfidence(IReadOnlyDictionary<string, double> probs)
        => probs.Count == 0 ? 0.0 : probs.Values.Max();

    /// <summary>Port of <c>score_task</c>: validates, renormalizes inside the rounding band if needed, scores.</summary>
    public static ScoredOutcome ScoreTask(IReadOnlyDictionary<string, double> probs, JevTask task)
    {
        IReadOnlyDictionary<string, double> clean;
        bool strictValid, renormalized;
        try
        {
            clean = ValidateProbs(probs, task.Labels);
            strictValid = true;
            renormalized = false;
        }
        catch (InvalidDistributionException strictError)
        {
            try
            {
                clean = ValidateProbs(probs, task.Labels, RenormTol);
            }
            catch (InvalidDistributionException)
            {
                return new ScoredOutcome
                {
                    Valid = false, StrictValid = false, Renormalized = false,
                    Error = strictError.Message, Correct = false, Predicted = null,
                };
            }
            double total = clean.Values.Sum();
            if (total <= 0)
            {
                return new ScoredOutcome
                {
                    Valid = false, StrictValid = false, Renormalized = false,
                    Error = "probabilities sum to zero", Correct = false, Predicted = null,
                };
            }
            clean = clean.ToDictionary(kv => kv.Key, kv => kv.Value / total, StringComparer.Ordinal);
            strictValid = false;
            renormalized = true;
        }

        string? expected = task.ExpectedLabel;
        if (expected is null)
        {
            string? pred = task.Question.Type != "score" ? ArgmaxLabel(clean) : null;
            return new ScoredOutcome
            {
                Valid = true, StrictValid = strictValid, Renormalized = renormalized, Probs = clean,
                Correct = null, Predicted = pred,
                OrdinalEv = task.Question.Type == "score" ? ExpectedValue(clean) : null,
            };
        }

        if (task.Question.Type == "score")
        {
            double ev = ExpectedValue(clean);
            string predicted = ArgmaxLabel(clean);
            return new ScoredOutcome
            {
                Valid = true, StrictValid = strictValid, Renormalized = renormalized, Probs = clean,
                OrdinalEv = ev, Predicted = predicted, Correct = predicted == expected,
            };
        }
        else
        {
            string predicted = ArgmaxLabel(clean);
            return new ScoredOutcome
            {
                Valid = true, StrictValid = strictValid, Renormalized = renormalized, Probs = clean,
                Predicted = predicted, Correct = predicted == expected,
            };
        }
    }
}

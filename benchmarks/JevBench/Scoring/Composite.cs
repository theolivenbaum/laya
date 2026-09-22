namespace Laya.JevBench.Scoring;

/// <summary>
/// Port of <c>jevbench/composite_v13.py</c> — the JevBench Score: chance-corrected Intelligence,
/// Calibration, Speed and Cost, 25% each, geometric mean, with a growing penalty below 50
/// Intelligence.
///
/// <para><b>One deliberate difference from the original:</b> the official per-tier "chance" figures
/// (<c>TIER_OPTION_COUNTS</c> in the Python source) are option-count histograms over all 534 items
/// in the full v1.2 set, most of which are private. This tool only ever sees the public subset it
/// actually runs (231 of 534 items — no judge-tier item is public at all), so chance is instead
/// computed directly from the public items this run scored (see <c>Composite.Chance</c>). That
/// keeps the score internally consistent for a public-only run, at the cost of not being bit-for-bit
/// the published constant; both are "1/options averaged over the item set", just a different item
/// set. This score is therefore informative for regression-tracking laya against its own past runs,
/// not a reproduction of the ranked JevBench Score.</para>
/// </summary>
public static class Composite
{
    public static readonly IReadOnlyDictionary<string, double> TierWeights = new Dictionary<string, double>
    {
        ["easy"] = 0.14,
        ["standard"] = 0.28,
        ["judge"] = 0.28,
        ["hard"] = 0.30,
    };

    public const double SpeedBestS = 0.1;
    public const double SpeedPerDecade = 20.0;
    public const double CostBestUsd = 0.001;
    public const double CostPerDecade = 30.0;
    public const double LoadFactor = 2.0;
    public const double OwnServerAddS = 0.15;

    public static double Clamp(double x, double lo = 0.0, double hi = 100.0) => Math.Max(lo, Math.Min(hi, x));

    /// <summary>Chance for one item: 1/options (1/levels for score items).</summary>
    public static double ItemChance(int optionCount) => 1.0 / optionCount;

    /// <summary>Mean per-item chance over a tier's items — the public-subset analogue of <c>TIER_CHANCES</c>.</summary>
    public static double Chance(IEnumerable<int> optionCounts)
    {
        var counts = optionCounts.ToArray();
        if (counts.Length == 0) throw new ArgumentException("a tier needs at least one item");
        return counts.Average(ItemChance);
    }

    public static double ChanceCorrectedAccuracy(double accuracy, double chance)
        => Clamp(100 * (accuracy - chance) / (1 - chance));

    /// <summary>Weighted chance-corrected tier accuracy; tiers absent from <paramref name="tiers"/> are renormalised away.</summary>
    public static double? Intelligence(IReadOnlyDictionary<string, double> tiers, IReadOnlyDictionary<string, double> tierChances)
    {
        double score = 0.0, weight = 0.0;
        foreach (var (tier, tierWeight) in TierWeights)
        {
            if (tiers.TryGetValue(tier, out double accuracy))
            {
                score += tierWeight * ChanceCorrectedAccuracy(accuracy, tierChances[tier]);
                weight += tierWeight;
            }
        }
        return weight > 0 ? score / weight : null;
    }

    /// <summary>"cpu"/"gpu" endpoints (self-hosted) get the ×2 + 0.15s production-load approximation; "api" gets none.</summary>
    public static double? AdjustedLatency(double? seconds, string endpointKind)
    {
        if (seconds is null) return null;
        if (endpointKind == "api") return seconds;
        return seconds * LoadFactor + (endpointKind is "gpu" or "cpu" ? OwnServerAddS : 0.0);
    }

    public static double SpeedPoint(double seconds) => Clamp(100 - SpeedPerDecade * Math.Log10(seconds / SpeedBestS));

    public static double? Speed(double? p50S, double? p95S, string endpointKind)
    {
        double? a = AdjustedLatency(p50S, endpointKind);
        double? b = AdjustedLatency(p95S, endpointKind);
        return a is null || b is null ? null : (SpeedPoint(a.Value) + SpeedPoint(b.Value)) / 2;
    }

    public static double Cost(double usdPer1000)
    {
        if (!(usdPer1000 > 0)) throw new ArgumentException("every system needs a positive price; a missing price is never scored as 100");
        return Clamp(100 - CostPerDecade * Math.Log10(usdPer1000 / CostBestUsd));
    }

    /// <summary>Mean of the ECE score and probability fidelity (both 0-100); null when no distribution exists.</summary>
    public static double? Calibration(double? ece, double? meanTvd)
    {
        if (ece is null) return null;
        double eceScore = Math.Max(0.0, 100 * (1 - ece.Value / 0.5));
        return meanTvd is null ? eceScore : (eceScore + 100 * (1 - meanTvd.Value)) / 2;
    }

    public static double NearChanceMultiplier(double? intelligenceScore)
        => intelligenceScore is null || intelligenceScore >= 50 ? 1.0 : Math.Pow(Math.Max(intelligenceScore.Value, 0.0) / 50, 2);

    /// <summary>Geometric mean of the weighted axes present, times the near-chance penalty.</summary>
    public static double Geometric(IReadOnlyDictionary<string, double?> axes, IReadOnlyDictionary<string, double> weights)
    {
        double totalWeight = weights.Values.Sum();
        double logSum = 0.0;
        foreach (var (axis, weight) in weights)
        {
            if (weight <= 0) continue;
            double value = axes.TryGetValue(axis, out double? v) ? v ?? 0.0 : 0.0;
            logSum += weight / totalWeight * Math.Log(Math.Max(value, 1.0));
        }
        return Math.Exp(logSum) * NearChanceMultiplier(axes.TryGetValue("intelligence", out double? intel) ? intel : null);
    }

    public static readonly IReadOnlyDictionary<string, double> EqualWeights = new Dictionary<string, double>
    {
        ["intelligence"] = 0.25, ["calibration"] = 0.25, ["speed"] = 0.25, ["cost"] = 0.25,
    };

    public static double JevBenchScore(IReadOnlyDictionary<string, double?> axes) => Geometric(axes, EqualWeights);
}

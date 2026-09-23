using Laya.Runtime;

namespace Laya.Training;

/// <summary>One calibration sample: a question's untempered option logits and its soft target.</summary>
public sealed record CalibrationSample(int QuestionType, float[] Logits, float[] Target);

/// <summary>What calibration produced.</summary>
public sealed record TemperatureFit(float[] ByType, IReadOnlyDictionary<string, float> ByBucket,
    IReadOnlyDictionary<string, int> SamplesPerBucket);

/// <summary>
/// Post-training temperature scaling: the temperature <c>T</c> minimising the soft cross-entropy
/// <c>−mean Σ t · log softmax(z / T)</c>, fitted per question type as <c>fit_one_temp</c> does.
///
/// <para>The reference fits <c>log T</c> with L-BFGS; the objective is one-dimensional and smooth, so a
/// golden-section search over <c>log T</c> reaches the same minimum without a gradient.</para>
///
/// <para>Two deliberate differences from the notebook. It also fits the <c>type:option-count</c>
/// buckets inference looks up <em>first</em>, where there are enough samples: the notebook rewrites
/// only the per-type values and leaves the base checkpoint's buckets in the config, so the stale
/// buckets shadow the new fit for every question that falls into one. And results are clamped to the
/// range inference applies (<see cref="Calibration.TemperatureMin"/>–<see cref="Calibration.TemperatureMax"/>),
/// rather than to the notebook's [0.1, 10], so what is saved is what will be used.</para>
/// </summary>
public static class TemperatureFitting
{
    /// <summary>The notebook's fallback when a type has no calibration samples at all.</summary>
    public const float MissingTypeTemperature = 1.2f;

    /// <summary>Below this many samples a fit is not attempted (the reference returns 1.0).</summary>
    public const int MinimumSamples = 10;

    public static float FitOne(IReadOnlyList<CalibrationSample> samples)
    {
        if (samples.Count < MinimumSamples) return 1f;

        double Objective(double logT)
        {
            double t = Math.Exp(logT), total = 0d;
            foreach (var sample in samples)
            {
                int k = sample.Logits.Length;
                double max = double.NegativeInfinity;
                for (int i = 0; i < k; ++i) max = Math.Max(max, sample.Logits[i] / t);
                double sum = 0d;
                for (int i = 0; i < k; ++i) sum += Math.Exp(sample.Logits[i] / t - max);
                double logSum = max + Math.Log(sum);
                for (int i = 0; i < k; ++i) total -= sample.Target[i] * (sample.Logits[i] / t - logSum);
            }
            return total / samples.Count;
        }

        double low = Math.Log(0.02), high = Math.Log(50);
        double ratio = (Math.Sqrt(5) - 1) / 2;
        double a = high - ratio * (high - low), b = low + ratio * (high - low);
        double fa = Objective(a), fb = Objective(b);
        for (int iteration = 0; iteration < 80 && high - low > 1e-7; ++iteration)
        {
            if (fa < fb)
            {
                high = b; b = a; fb = fa;
                a = high - ratio * (high - low);
                fa = Objective(a);
            }
            else
            {
                low = a; a = b; fa = fb;
                b = low + ratio * (high - low);
                fb = Objective(b);
            }
        }
        return (float)Math.Exp((low + high) / 2);
    }

    /// <summary>
    /// Fits every type, and every bucket with at least <paramref name="minimumBucketSamples"/> samples
    /// (pass <c>int.MaxValue</c> to fit types only).
    /// </summary>
    public static TemperatureFit Fit(IReadOnlyList<CalibrationSample> samples, int minimumBucketSamples = 30)
    {
        var byType = new float[3];
        for (int type = 0; type < 3; ++type)
        {
            var selected = samples.Where(s => s.QuestionType == type).ToList();
            byType[type] = selected.Count == 0
                ? MissingTypeTemperature
                : Calibration.ClampTemperature(FitOne(selected));
        }

        var byBucket = new Dictionary<string, float>(StringComparer.Ordinal);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var group in samples.GroupBy(s => QuestionTypes.TemperatureBucket((QuestionType)s.QuestionType, s.Logits.Length)))
        {
            var list = group.ToList();
            counts[group.Key] = list.Count;
            if (list.Count >= minimumBucketSamples) byBucket[group.Key] = Calibration.ClampTemperature(FitOne(list));
        }
        return new TemperatureFit(byType, byBucket, counts);
    }
}

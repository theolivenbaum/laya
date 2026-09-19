using System.Collections.Concurrent;

namespace Laya.Numerics;

/// <summary>
/// Temporary investigation hook: per-projection activation statistics, to decide whether dynamic
/// per-row int8 activation quantization is viable for this model. Enabled with LAYA_QSTATS=1.
/// </summary>
public static class QuantStats
{
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("LAYA_QSTATS") == "1";

    private sealed class Acc
    {
        public long Rows;
        public double SumAmax, SumRms, SumRatio, MaxRatio;
        public double WorstAmax;
        public double SumEffLevels;   // how many int8 levels the typical value actually occupies
        public double SumZeroFrac;    // fraction of values that quantize to 0 at amax/127
        public int[]? ArgmaxCount;    // how often each input channel holds the row's largest magnitude
        public double[]? ChannelMean; // mean |value| per input channel
    }

    private static readonly ConcurrentDictionary<string, Acc> Table = new();

    public static void Observe(string name, ReadOnlySpan<float> data, int rows, int features)
    {
        var acc = Table.GetOrAdd(name, _ => new Acc());
        lock (acc)
        {
            for (int r = 0; r < rows; ++r)
            {
                var row = data.Slice(r * features, features);
                float amax = 0f;
                double sumSq = 0;
                for (int i = 0; i < features; ++i)
                {
                    float v = MathF.Abs(row[i]);
                    if (v > amax) amax = v;
                    sumSq += (double)row[i] * row[i];
                }
                double rms = Math.Sqrt(sumSq / features);
                double ratio = rms > 0 ? amax / rms : 0;
                double scale = amax / 127.0;
                int zeros = 0;
                if (scale > 0)
                {
                    for (int i = 0; i < features; ++i)
                    {
                        if (Math.Round(row[i] / scale) == 0) zeros++;
                    }
                }
                acc.ArgmaxCount ??= new int[features];
                acc.ChannelMean ??= new double[features];
                int argmax = 0;
                for (int i = 0; i < features; ++i)
                {
                    float v = MathF.Abs(row[i]);
                    acc.ChannelMean[i] += v;
                    if (v > MathF.Abs(row[argmax])) argmax = i;
                }
                acc.ArgmaxCount[argmax]++;

                acc.Rows++;
                acc.SumAmax += amax;
                acc.SumRms += rms;
                acc.SumRatio += ratio;
                acc.MaxRatio = Math.Max(acc.MaxRatio, ratio);
                acc.WorstAmax = Math.Max(acc.WorstAmax, amax);
                acc.SumEffLevels += scale > 0 ? rms / scale : 0;
                acc.SumZeroFrac += (double)zeros / features;
            }
        }
    }

    public static void Report(TextWriter w)
    {
        w.WriteLine();
        w.WriteLine("activation statistics per projection input (mean over token rows)");
        w.WriteLine($"{"site",-28} {"rows",8} {"amax",12} {"rms",12} {"amax/rms",10} {"worst",10} {"levels",8} {"zeroed",8}");
        foreach (var (name, a) in Table.OrderBy(kv => kv.Key))
        {
            lock (a)
            {
                w.WriteLine($"{name,-28} {a.Rows,8} {a.SumAmax / a.Rows,12:F2} {a.SumRms / a.Rows,12:F4} " +
                            $"{a.SumRatio / a.Rows,10:F1} {a.MaxRatio,10:F1} {a.SumEffLevels / a.Rows,8:F2} {a.SumZeroFrac / a.Rows,8:P1}");
            }
        }
        w.WriteLine();
        w.WriteLine("'levels' = rms / (amax/127): how many int8 steps a typical value spans.");
        w.WriteLine();
        w.WriteLine("where the outliers live: the channels that most often hold the row's maximum");
        foreach (var (name, a) in Table.OrderBy(kv => kv.Key))
        {
            lock (a)
            {
                if (a.ArgmaxCount is null || a.ChannelMean is null) continue;
                var top = a.ArgmaxCount
                    .Select((count, channel) => (channel, count))
                    .OrderByDescending(t => t.count).Take(8).Where(t => t.count > 0).ToList();
                double covered = top.Sum(t => (double)t.count) / a.Rows;
                int distinct = a.ArgmaxCount.Count(c => c > 0);
                w.WriteLine($"  {name,-28} {distinct,5} distinct channels ever max; top 8 cover {covered,6:P1}");
                w.WriteLine($"  {"",-28} {string.Join(", ", top.Select(t => $"ch{t.channel}:{(double)t.count / a.Rows:P0}"))}");
                // How much bigger the worst channel is than the median channel, on average.
                var sorted = a.ChannelMean.OrderBy(v => v).ToArray();
                double median = sorted[sorted.Length / 2];
                w.WriteLine($"  {"",-28} largest channel mean / median channel mean = {sorted[^1] / median,8:F1}");
            }
        }
        w.WriteLine("'zeroed' = share of values that round to zero under per-row amax/127 scaling.");
    }
}

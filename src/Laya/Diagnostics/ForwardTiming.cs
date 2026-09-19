using System.Diagnostics;
using System.Globalization;

namespace Laya.Diagnostics;

/// <summary>
/// Wall-clock and allocation accounting per stage of a forward pass.
///
/// <para>A sampling profiler says which method is hot; this says which <em>stage</em> is, which is
/// what you need to decide whether to attack the GEMM, attention, or the plumbing around them.
/// Collection is opt-in and off by default: the timers are cheap but not free, and a stopwatch per
/// layer would show up in its own numbers.</para>
/// </summary>
public sealed class ForwardTiming
{
    private readonly Dictionary<string, (long Ticks, long Bytes, int Count)> _stages = new(StringComparer.Ordinal);
    private readonly Lock _gate = new();

    /// <summary>The collector in force, or null when nothing is being measured.</summary>
    public static ForwardTiming? Current { get; private set; }

    /// <summary>Starts collecting into a fresh set of counters until the returned scope is disposed.</summary>
    public static Scope Collect()
    {
        var timing = new ForwardTiming();
        Current = timing;
        return new Scope(timing);
    }

    /// <summary>Times one stage. A no-op unless <see cref="Collect"/> is active.</summary>
    public static Stage Measure(string name) => new(Current, name);

    private void Add(string name, long ticks, long bytes)
    {
        lock (_gate)
        {
            _stages.TryGetValue(name, out var existing);
            _stages[name] = (existing.Ticks + ticks, existing.Bytes + bytes, existing.Count + 1);
        }
    }

    /// <summary>Stages ordered by the time they took, longest first.</summary>
    public IReadOnlyList<(string Name, TimeSpan Elapsed, long Bytes, int Count)> Stages
    {
        get
        {
            lock (_gate)
            {
                return [.. _stages
                    .Select(e => (e.Key, TimeSpan.FromTicks(e.Value.Ticks), e.Value.Bytes, e.Value.Count))
                    .OrderByDescending(e => e.Item2)];
            }
        }
    }

    public string Describe()
    {
        var stages = Stages;
        double total = stages.Sum(s => s.Elapsed.TotalMilliseconds);
        var lines = new List<string>
        {
            string.Format(CultureInfo.InvariantCulture, "{0,-26} {1,10} {2,7} {3,12} {4,8}",
                "stage", "ms", "%", "alloc", "calls"),
        };
        foreach (var (name, elapsed, bytes, count) in stages)
        {
            lines.Add(string.Format(CultureInfo.InvariantCulture, "{0,-26} {1,10:F1} {2,6:F1}% {3,12} {4,8}",
                name, elapsed.TotalMilliseconds, total > 0 ? 100 * elapsed.TotalMilliseconds / total : 0,
                Bytes(bytes), count));
        }
        lines.Add(string.Format(CultureInfo.InvariantCulture, "{0,-26} {1,10:F1}", "total (sum of stages)", total));
        return string.Join(Environment.NewLine, lines);
    }

    private static string Bytes(long value) => value switch
    {
        > 1 << 30 => $"{value / (double)(1 << 30):F2} GiB",
        > 1 << 20 => $"{value / (double)(1 << 20):F1} MiB",
        > 1 << 10 => $"{value / (double)(1 << 10):F1} KiB",
        _ => $"{value} B",
    };

    /// <summary>The scope that keeps <see cref="Current"/> set.</summary>
    public readonly struct Scope(ForwardTiming timing) : IDisposable
    {
        public ForwardTiming Timing => timing;

        public void Dispose() => Current = null;
    }

    /// <summary>
    /// One timed stage; dispose to record it.
    ///
    /// <para>Recording is idempotent. A <c>using var</c> plus an explicit <c>Dispose()</c> is an
    /// easy thing to write, and it silently recorded the stage twice — the second time spanning
    /// everything up to the end of the enclosing method, which made one stage look six times more
    /// expensive than it was.</para>
    /// </summary>
    public ref struct Stage
    {
        private readonly ForwardTiming? _timing;
        private readonly string _name;
        private readonly long _timestamp;
        private readonly long _allocated;
        private bool _recorded;

        internal Stage(ForwardTiming? timing, string name)
        {
            _timing = timing;
            _name = name;
            _recorded = false;
            _timestamp = timing is null ? 0 : Stopwatch.GetTimestamp();
            // Only the calling thread's allocations are attributed; the parallel kernels allocate
            // on worker threads, which is itself worth seeing as a gap between this and the GC's
            // total.
            _allocated = timing is null ? 0 : GC.GetAllocatedBytesForCurrentThread();
        }

        public void Dispose()
        {
            if (_timing is null || _recorded) return;
            _recorded = true;
            _timing.Add(_name, Stopwatch.GetElapsedTime(_timestamp).Ticks,
                GC.GetAllocatedBytesForCurrentThread() - _allocated);
        }
    }
}

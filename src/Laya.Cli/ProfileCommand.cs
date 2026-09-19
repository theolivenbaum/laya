using System.Globalization;
using Laya.Diagnostics;
using Laya.Runtime;
using Memory.Introspect;
using Memory.Introspect.Trace;

namespace Laya.Cli;

/// <summary>
/// Measures where a forward pass spends its time and what it allocates.
///
/// <para>Three views, because they answer different questions. Stage timings say whether the cost
/// is in the projections, in attention, or in the plumbing. A sampling profile says which method is
/// actually on the CPU. An allocation report says what the pass is asking the GC for — which for a
/// tensor workload is usually the thing you did not mean to do.</para>
/// </summary>
internal static class ProfileCommand
{
    public static int Run(CommandLine options, Func<CommandLine, Agent> openAgent,
        Func<CommandLine, object?> readState, Func<CommandLine, QuestionSet> readQuestions)
    {
        int threads = int.Parse(options.Value("threads") ?? "1", CultureInfo.InvariantCulture);
        LayaRuntime.MaxDegreeOfParallelism = threads;

        using var agent = openAgent(options);
        object? state = readState(options);
        var questions = readQuestions(options);
        int iterations = int.Parse(options.Value("iterations") ?? "3", CultureInfo.InvariantCulture);

        Console.WriteLine(LayaRuntime.Describe());
        Console.WriteLine();

        agent.SystemOne(state, questions);      // warm the JIT, the page cache and the weights

        // ---------------------------------------------------------------- stage timings
        using (var scope = ForwardTiming.Collect())
        {
            for (int i = 0; i < iterations; ++i) agent.SystemOne(state, questions);
            Console.WriteLine($"Stage timings over {iterations} iteration(s)");
            Console.WriteLine(scope.Timing.Describe());
        }
        Console.WriteLine();

        // ---------------------------------------------------------------- allocation totals
        long before = GC.GetTotalAllocatedBytes(precise: true);
        var collections = new[] { GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2) };
        for (int i = 0; i < iterations; ++i) agent.SystemOne(state, questions);
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;

        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "Allocated {0} per call ({1} over {2} calls); gen0 {3}, gen1 {4}, gen2 {5}",
            Bytes(allocated / iterations), Bytes(allocated), iterations,
            GC.CollectionCount(0) - collections[0], GC.CollectionCount(1) - collections[1],
            GC.CollectionCount(2) - collections[2]));
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "Managed heap {0}, weights {1}, working set {2}",
            Bytes(GC.GetTotalMemory(forceFullCollection: false)),
            Bytes(agent.Model.WeightBytes),
            Bytes(Environment.WorkingSet)));

        if (options.Has("no-trace")) return 0;

        // ---------------------------------------------------------------- sampling + allocations
        var duration = TimeSpan.FromSeconds(double.Parse(options.Value("seconds") ?? "10",
            CultureInfo.InvariantCulture));
        var introspector = MemoryIntrospector.Create(new MemoryIntrospectorOptions());
        int self = Environment.ProcessId;

        Console.WriteLine();
        Console.WriteLine($"Sampling the forward pass for {duration.TotalSeconds:F0}s …");
        using var stop = new CancellationTokenSource();
        var load = Task.Run(() =>
        {
            while (!stop.IsCancellationRequested) agent.SystemOne(state, questions);
        });

        try
        {
            var sample = introspector.CollectSamplingProfileAsync(self, duration).GetAwaiter().GetResult();
            if (sample.Success)
            {
                Console.WriteLine();
                Console.WriteLine("Top methods by exclusive time (which method is running)");
                foreach (var method in sample.TopMethods(count: 12, inclusive: false))
                {
                    Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "  {0,6:F2}%  {1}", method.ExclusiveMetricPercent, Shorten(method.Name)));
                }

                Console.WriteLine();
                Console.WriteLine("Top call trees by inclusive time (which tree is expensive)");
                foreach (var method in sample.TopMethods(count: 12, inclusive: true))
                {
                    Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "  {0,6:F2}%  {1}", method.InclusiveMetricPercent, Shorten(method.Name)));
                }

                if (options.Value("save-trace") is string tracePath)
                {
                    sample.SaveToDisk(tracePath);
                    Console.WriteLine($"\nwrote {tracePath}");
                }
            }
            else
            {
                Console.Error.WriteLine($"sampling failed: {sample.Exception?.Message ?? "no data"}");
            }

            var report = introspector.CollectAllocationReportAsync(self, duration, count: 12)
                .GetAwaiter().GetResult();
            Console.WriteLine();
            if (report.IsEmpty)
            {
                Console.WriteLine("Allocation report: nothing sampled.");
            }
            else
            {
                AllocationTracing.Write(Console.Out, report);
            }
        }
        finally
        {
            stop.Cancel();
            load.GetAwaiter().GetResult();
        }
        return 0;
    }

    /// <summary>Sampled names are "Module!Namespace.Type.Method(args)"; the args are noise here.</summary>
    private static string Shorten(string name)
    {
        int parenthesis = name.IndexOf('(', StringComparison.Ordinal);
        if (parenthesis > 0) name = name[..parenthesis];
        return name.Length <= 96 ? name : "…" + name[^95..];
    }

    private static string Bytes(long value) => value switch
    {
        > 1L << 30 => $"{value / (double)(1L << 30):F2} GiB",
        > 1 << 20 => $"{value / (double)(1 << 20):F1} MiB",
        > 1 << 10 => $"{value / (double)(1 << 10):F1} KiB",
        _ => $"{value} B",
    };
}

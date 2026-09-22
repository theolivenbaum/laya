using System.Globalization;
using Laya;
using Laya.JevBench.Dataset;
using Laya.JevBench.Reporting;
using Laya.Runtime;

namespace Laya.JevBench;

/// <summary>
/// JevBench, run natively against laya's own C# <c>Agent</c> — no Python, no network calls to
/// jevbench's own code. This is the .NET analogue of running
/// <c>jevbench/adapters/laya_local.py</c> against the public JevBench items
/// (<c>datasets/public/{easy,original,hard}.jsonl</c>, copied into <c>data/</c>), scored with a
/// line-for-line port of <c>jevbench/scoring.py</c>, <c>metrics.py</c> and <c>composite_v13.py</c>
/// (see <see cref="Scoring.Composite"/> for the one deliberate scoring difference: chance is
/// recomputed from the public subset, since the private 303 items this repo cannot see are not
/// available to compute the official per-tier baselines from).
/// </summary>
internal static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        var options = CommandLine.Parse(args.AsSpan(1));
        try
        {
            return args[0] switch
            {
                "run" => Run(options),
                _ => Unknown(args[0]),
            };
        }
        catch (Exception e) when (e is ArgumentException or IOException or InvalidDataException or KeyNotFoundException)
        {
            Console.Error.WriteLine("error: " + e.Message);
            return 1;
        }
    }

    private static int Unknown(string command)
    {
        Console.Error.WriteLine($"error: unknown command '{command}'.");
        PrintUsage();
        return 1;
    }

    private static void PrintUsage() => Console.WriteLine("""
        laya-jevbench — JevBench's public items, run against laya's own .NET Agent

        USAGE
          laya-jevbench run --model-dir <path> [options]

        OPTIONS
          --model-dir <path>     A laya checkpoint directory (required; see `laya download --model english`)
          --tiers <list>         Comma-separated subset of easy,standard,hard (default: all three)
          --limit <n>            Only run the first <n> items of each tier (smoke-testing)
          --threads <n>          Kernel threads per forward pass (default: runtime default)
          --out <dir>            Where to write the run's summary/per-task/comparison JSON (default: artifacts/jevbench)
          --progress-every <n>   Print progress every n items (default: 20)
          --data <dir>           Where the copied JevBench datasets live (default: alongside this tool's data/)

        EXAMPLE
          dotnet run --project benchmarks/JevBench -c Release -- run \
              --model-dir artifacts/models-cache/models.curiosity.ai/english/main --threads 4
        """);

    private static int Run(CommandLine options)
    {
        string modelDirectory = options.Value("model-dir")
            ?? throw new ArgumentException("provide --model-dir (a laya checkpoint directory).");
        string dataDirectory = options.Value("data") ?? DefaultDataDirectory();
        string outDirectory = options.Value("out") ?? "artifacts/jevbench";
        int? limit = options.Value("limit") is string l ? int.Parse(l, CultureInfo.InvariantCulture) : null;
        int progressEvery = options.Value("progress-every") is string p ? int.Parse(p, CultureInfo.InvariantCulture) : 20;
        var tiers = (options.Value("tiers") ?? "easy,standard,hard").Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        ParallelOptions? parallel = options.Value("threads") is string t
            ? new ParallelOptions { MaxDegreeOfParallelism = int.Parse(t, CultureInfo.InvariantCulture) }
            : null;

        var tasks = LoadTasks(dataDirectory, tiers, limit);
        Console.WriteLine($"Loaded {tasks.Count} public JevBench items ({string.Join(", ", tiers)}) from {dataDirectory}");

        using var agent = Agent.FromDirectory(modelDirectory);
        Console.WriteLine($"Agent loaded from {modelDirectory}");

        int threadsUsed = parallel?.MaxDegreeOfParallelism ?? Environment.ProcessorCount;
        string endpointCondition = $"this run, local CPU ({threadsUsed} thread(s))";

        var results = new List<Reporting.TaskResult>(tasks.Count);
        var overall = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < tasks.Count; ++i)
        {
            results.Add(Runner.Run(agent, tasks[i], parallel));
            if ((i + 1) % progressEvery == 0 || i + 1 == tasks.Count)
            {
                Console.WriteLine($"{i + 1}/{tasks.Count} completed ({overall.Elapsed.TotalSeconds:F1}s elapsed)");
            }
        }

        var tasksById = tasks.ToDictionary(t => t.Id, StringComparer.Ordinal);
        var report = RunReport.Build(results, tasksById, endpointCondition, "cpu");

        PrintReport(report);

        Directory.CreateDirectory(outDirectory);
        string summaryPath = Path.Combine(outDirectory, "laya-dotnet.json");
        string perTaskPath = Path.Combine(outDirectory, "laya-dotnet-per-task.json");
        JsonOutput.WriteSummary(summaryPath, report, modelDirectory,
            $"JevBench public items run natively through laya's .NET Agent, {DateTime.UtcNow:yyyy-MM-dd}.");
        JsonOutput.WritePerTask(perTaskPath, report);
        Console.WriteLine($"\nWrote {summaryPath}");
        Console.WriteLine($"Wrote {perTaskPath}");

        string baselinePerTask = Path.Combine(dataDirectory, "baseline", "laya-per-task.json");
        string baselineSummary = Path.Combine(dataDirectory, "baseline", "laya.json");
        if (File.Exists(baselinePerTask) && File.Exists(baselineSummary))
        {
            var baseline = Baseline.Load(baselinePerTask, baselineSummary, tasksById);
            var deltas = BaselineComparison.Diff(baseline, results);
            string comparisonPath = Path.Combine(outDirectory, "comparison-vs-baseline.json");
            JsonOutput.WriteComparison(comparisonPath, baseline, report, deltas);
            PrintComparison(baseline, report, deltas);
            Console.WriteLine($"Wrote {comparisonPath}");
        }

        return 0;
    }

    private static List<JevTask> LoadTasks(string dataDirectory, string[] tiers, int? limit)
    {
        var tasks = new List<JevTask>();
        void LoadTier(string file, string tier)
        {
            if (!tiers.Contains(tier, StringComparer.Ordinal)) return;
            var loaded = JevTaskLoader.Load(Path.Combine(dataDirectory, file), tier);
            if (limit is { } n) loaded = loaded.Take(n).ToList();
            tasks.AddRange(loaded);
        }
        LoadTier("easy.jsonl", "easy");
        LoadTier("original.jsonl", "standard");
        LoadTier("hard.jsonl", "hard");
        return tasks;
    }

    private static string DefaultDataDirectory()
    {
        string candidate = Path.Combine(AppContext.BaseDirectory, "data");
        if (Directory.Exists(candidate)) return candidate;
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string maybe = Path.Combine(directory.FullName, "benchmarks", "JevBench", "data");
            if (Directory.Exists(maybe)) return maybe;
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException("could not find the JevBench data/ directory; pass --data explicitly.");
    }

    private static void PrintReport(RunReport report)
    {
        Console.WriteLine();
        Console.WriteLine("Tier accuracy (public-subset; judge tier has no public items):");
        foreach (var (tier, summary) in report.TierSummaries.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "  {0,-10} n={1,4}  correct={2,4}  wrong={3,4}  failed={4,3}  accuracy={5:P1}  chance={6:P1}  brier={7}",
                tier, summary.N, summary.Correct, summary.Wrong, summary.Failed, summary.Accuracy, report.TierChances[tier], Fmt(summary.BrierMean)));
        }
        Console.WriteLine();
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "Calibration (hard tier): ece={0}  fidelity={1} (n={2}/20 public)  score={3}",
            Fmt(report.Calibration.Ece), Fmt(report.Calibration.ProbabilityFidelity), report.Calibration.FidelityN, Fmt(report.Calibration.Score)));
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "Latency: n={0}  p50={1:F3}s  p95={2:F3}s   mean input tokens={3:F1}",
            report.Latency.N, report.Latency.P50S, report.Latency.P95S, report.MeanInputTokens));
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "Axes: intelligence={0}  calibration={1}  speed={2}  cost={3}  -> JevBench Score (public-subset, informal)={4}",
            Fmt(report.Intelligence), Fmt(report.CalibrationScore), Fmt(report.Speed), Fmt(report.Cost), Fmt(report.JevBenchScore)));
    }

    private static void PrintComparison(Baseline baseline, RunReport report, IReadOnlyList<Reporting.TaskDelta> deltas)
    {
        Console.WriteLine();
        Console.WriteLine($"Comparison vs baseline ({baseline.Display}, JevBench v1.2.2, python laya 0.3.3):");
        foreach (var tier in report.TierSummaries.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            double b = baseline.TierAccuracy.GetValueOrDefault(tier);
            double c = report.TierSummaries[tier].Accuracy;
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "  {0,-10} baseline={1:P1}  current={2:P1}  delta={3:+0.0%;-0.0%;+0.0%}", tier, b, c, c - b));
        }
        Console.WriteLine($"  {deltas.Count} item(s) flipped outcome ({deltas.Count(d => d.Regressed)} regressed, {deltas.Count(d => d.Improved)} improved)");
        foreach (var d in deltas.Take(20))
        {
            Console.WriteLine($"    {(d.Regressed ? "REGRESSED" : d.Improved ? "improved " : "changed  ")} {d.Id}: {d.Baseline} -> {d.Current}");
        }
        if (deltas.Count > 20) Console.WriteLine($"    … and {deltas.Count - 20} more (see comparison-vs-baseline.json)");
    }

    private static string Fmt(double? v) => v is { } x ? x.ToString("F2", CultureInfo.InvariantCulture) : "n/a";
}

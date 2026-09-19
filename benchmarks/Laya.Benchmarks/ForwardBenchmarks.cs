using BenchmarkDotNet.Attributes;
using Laya.Runtime;

namespace Laya.Benchmarks;

/// <summary>
/// The whole forward pass on real weights. Skipped unless a checkpoint is on disk, because it
/// needs 840 MB of them: point <c>LAYA_TEST_MODELS</c> at a directory holding <c>english/</c>, or
/// run <c>laya download</c> into <c>artifacts/models</c>.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class ForwardBenchmarks
{
    private Agent? _agent;
    private QuestionSet _questions = null!;
    private const string State =
        "I was charged twice for my subscription last month and nobody from support has answered " +
        "my three emails. If this is not fixed by Friday I am cancelling.";

    [Params(1, 4)]
    public int Threads { get; set; }

    public static string? ModelDirectory()
    {
        string root = Environment.GetEnvironmentVariable("LAYA_TEST_MODELS")
            ?? Path.Combine(FindRepositoryRoot(), "artifacts", "models");
        string english = Path.Combine(root, "english");
        return File.Exists(Path.Combine(english, "model.safetensors")) ? english : null;
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, ".reference"))) return directory.FullName;
            directory = directory.Parent;
        }
        return Environment.CurrentDirectory;
    }

    [GlobalSetup]
    public void Setup()
    {
        string? directory = ModelDirectory();
        if (directory is null) return;
        _agent = Agent.FromDirectory(directory);
        _questions = Presets.Triage();
    }

    [GlobalCleanup]
    public void Cleanup() => _agent?.Dispose();

    [IterationSetup]
    public void SetThreads() => LayaRuntime.MaxDegreeOfParallelism = Threads;

    /// <summary>Five typed questions over one paragraph of state — 404 tokens in total.</summary>
    [Benchmark]
    public object? TriagePreset() => _agent?.SystemOne(State, _questions);

    /// <summary>One <c>noul</c> question, the smallest useful unit of work.</summary>
    [Benchmark]
    public object? SingleQuestion() => _agent?.SystemOne(State,
        new QuestionSet().Add("urgent", Question.Noul("Is this urgent?")));
}

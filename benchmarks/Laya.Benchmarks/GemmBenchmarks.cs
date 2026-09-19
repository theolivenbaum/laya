using BenchmarkDotNet.Attributes;
using Laya.Numerics;

namespace Laya.Benchmarks;

/// <summary>
/// The projection kernel at the shapes the encoder actually asks for.
///
/// <para>These four shapes are ~95% of a forward pass, so this is the benchmark that matters for
/// throughput. <c>Threads</c> is a parameter rather than a fixture because the single-threaded
/// number is the one that says whether the kernel itself is good; the parallel number mostly
/// measures how well the panels partition.</para>
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class GemmBenchmarks
{
    private PackedMatrix _weights = null!;
    private float[] _input = null!;
    private float[] _output = null!;

    /// <summary>Tokens per call. 404 is the triage preset over a paragraph of state.</summary>
    [Params(404)]
    public int Rows { get; set; }

    [Params(1, 4)]
    public int Threads { get; set; }

    /// <summary>in × out, named after the projection it stands for.</summary>
    [Params("qkv:1024x3072", "attn_out:1024x1024", "mlp_in:1024x5248", "mlp_out:2624x1024")]
    public string Shape { get; set; } = "mlp_in:1024x5248";

    private int _inFeatures;
    private int _outFeatures;

    [GlobalSetup]
    public void Setup()
    {
        string[] parts = Shape.Split(':')[1].Split('x');
        _inFeatures = int.Parse(parts[0]);
        _outFeatures = int.Parse(parts[1]);

        var random = new Random(1);
        var rowMajor = new float[(long)_outFeatures * _inFeatures];
        for (int i = 0; i < rowMajor.Length; ++i) rowMajor[i] = (float)random.NextDouble() - 0.5f;

        _weights = new PackedMatrix(rowMajor, _outFeatures, _inFeatures);
        _input = new float[(long)Rows * _inFeatures];
        for (int i = 0; i < _input.Length; ++i) _input[i] = (float)random.NextDouble() - 0.5f;
        _output = new float[(long)Rows * _outFeatures];
    }

    [IterationSetup]
    public void SetThreads() => LayaRuntime.MaxDegreeOfParallelism = Threads;

    /// <summary>Floating-point operations per call, for converting the timing into GFLOP/s.</summary>
    public double FlopsPerCall => 2.0 * Rows * _outFeatures * _inFeatures;

    [Benchmark]
    public void Multiply() => _weights.Multiply(_input, Rows, [], _output);
}

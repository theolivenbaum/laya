using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Laya.Numerics;

namespace Laya.Benchmarks;

/// <summary>
/// The elementwise and reduction kernels. Individually small, but the encoder runs them over
/// millions of values per call — <c>Gelu</c> was once 14% of a forward pass on its own.
/// </summary>
// InvocationCount/UnrollFactor of 1 so [IterationSetup] runs before every single measured call.
// With the defaults BDN batches dozens of invocations per iteration, the input is restored only
// once, and every kernel after the first runs on its own output.
[SimpleJob(invocationCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser(displayGenColumns: false)]
public class KernelBenchmarks
{
    private float[] _values = null!;
    private float[] _pristine = null!;
    private float[] _other = null!;
    private float[] _weight = null!;
    private float[] _bias = null!;

    /// <summary>One MLP intermediate row block: 404 tokens x 2624 channels.</summary>
    [Params(1_060_096)]
    public int Length { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(2);
        _pristine = new float[Length];
        _values = new float[Length];
        _other = new float[Length];
        for (int i = 0; i < Length; ++i)
        {
            _pristine[i] = (float)(random.NextDouble() * 8 - 4);
            _other[i] = (float)(random.NextDouble() * 2 - 1);
        }
        _weight = new float[1024];
        _bias = new float[1024];
        for (int i = 0; i < 1024; ++i)
        {
            _weight[i] = (float)random.NextDouble();
            _bias[i] = (float)random.NextDouble();
        }
    }

    /// <summary>
    /// Every kernel here writes in place, so the input has to be restored between iterations.
    /// Without this the benchmark feeds its own output back: repeated GELU drives the values
    /// toward zero, they become denormal, and the "kernel" then measures forty times slower than
    /// it is because denormal arithmetic traps to microcode.
    /// </summary>
    [IterationSetup]
    public void RestoreInput() => Array.Copy(_pristine, _values, Length);

    [Benchmark]
    public void Gelu() => SimdOps.Gelu(_values);

    [Benchmark]
    public void Relu() => SimdOps.Relu(_values);

    [Benchmark]
    public void Add() => SimdOps.Add(_values, _other);

    [Benchmark]
    public void AddScaled() => SimdOps.AddScaled(_values, _other, 0.25f);

    [Benchmark]
    public float Dot() => SimdOps.Dot(_values, _other);

    [Benchmark]
    public void LayerNormRows()
    {
        // 404 token rows of 1024 channels, the shape every norm in the encoder sees.
        for (int row = 0; row < 404; ++row)
        {
            var slice = _values.AsSpan(row * 1024, 1024);
            SimdOps.LayerNorm(slice, _weight, _bias, 1e-5f, slice);
        }
    }
}

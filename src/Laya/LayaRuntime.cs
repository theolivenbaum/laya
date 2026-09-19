using System.Globalization;
using Laya.Numerics;

namespace Laya;

/// <summary>
/// Process-wide execution settings for the kernels.
///
/// <para>Separate from any one model so a benchmark, a test, or a server that already owns its
/// threads can pin the degree of parallelism. Defaults come from the environment, so nothing has to
/// be threaded through the API to profile a single-threaded run.</para>
/// </summary>
public static class LayaRuntime
{
    private static int _maxDegreeOfParallelism = ReadThreads();

    /// <summary>
    /// How many threads the GEMM and attention kernels may use. Defaults to
    /// <c>$LAYA_THREADS</c>, or <see cref="Environment.ProcessorCount"/>. Set to 1 for a
    /// single-threaded run.
    /// </summary>
    public static int MaxDegreeOfParallelism
    {
        get => _maxDegreeOfParallelism;
        set => _maxDegreeOfParallelism = value <= 0
            ? Environment.ProcessorCount
            : Math.Min(value, Environment.ProcessorCount);
    }

    /// <summary>
    /// The precision the projection weights are packed at, for callers that do not pass one
    /// explicitly. Defaults to <c>$LAYA_QUANTIZATION</c> (<c>none</c> or <c>int8</c>), and to
    /// <see cref="Quantization.None"/> — the only mode parity is measured against.
    /// </summary>
    public static Quantization Quantization { get; set; } = ReadQuantization();

    private static Quantization ReadQuantization()
        => Environment.GetEnvironmentVariable("LAYA_QUANTIZATION")?.Trim().ToLowerInvariant() switch
        {
            "int8" => Quantization.Int8,
            _ => Quantization.None,
        };

    /// <summary>True when work should run inline on the calling thread.</summary>
    public static bool SingleThreaded => _maxDegreeOfParallelism <= 1;

    /// <summary>The parallel options the kernels share, rebuilt only when the setting changes.</summary>
    public static ParallelOptions ParallelOptions => new() { MaxDegreeOfParallelism = _maxDegreeOfParallelism };

    private static int ReadThreads()
    {
        string? configured = Environment.GetEnvironmentVariable("LAYA_THREADS");
        if (configured is not null
            && int.TryParse(configured, NumberStyles.Integer, CultureInfo.InvariantCulture, out int threads)
            && threads > 0)
        {
            return Math.Min(threads, Environment.ProcessorCount);
        }
        return Environment.ProcessorCount;
    }

    /// <summary>One line describing how the kernels are configured, for benchmarks and bug reports.</summary>
    public static string Describe()
        => $"threads={MaxDegreeOfParallelism}/{Environment.ProcessorCount}, {SimdOps.Capabilities}, "
         + $"weights={Quantization.ToString().ToLowerInvariant()}"
         + (Quantization == Quantization.Int8 ? $" ({PackedInt8Matrix.Kernel})" : string.Empty);
}

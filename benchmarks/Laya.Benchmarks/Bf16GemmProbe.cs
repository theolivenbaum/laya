using System.Diagnostics;
using System.Globalization;
using Laya.Numerics;

namespace Laya.Benchmarks;

/// <summary>
/// A quick, stopwatch-level probe of <see cref="BFloat16Matrix"/> on d1's projection shapes, single
/// threaded: <c>dotnet run -c Release -- --bf16-gemm [rows…]</c>. It prints GFLOP/s per shape, the
/// number to set against <c>tools/bench_gemm.py</c> (PyTorch's fp32 matmul on the same shapes).
/// </summary>
internal static class Bf16GemmProbe
{
    public static readonly (string Name, int Out, int In)[] Shapes =
    [
        ("conv.in_proj", 6144, 2048), ("conv.out_proj", 2048, 2048), ("attn.qkv", 3072, 2048),
        ("mlp.w1|w3", 21504, 2048), ("mlp.w2", 2048, 10752),
    ];

    public static void Run(string[] args)
    {
        LayaRuntime.MaxDegreeOfParallelism = 1;
        int[] rows = args.Length > 0 ? [.. args.Select(a => int.Parse(a, CultureInfo.InvariantCulture))] : [61, 116, 600];
        var random = new Random(1);
        foreach (var (name, outFeatures, inFeatures) in Shapes)
        {
            var bits = new ushort[(long)outFeatures * inFeatures];
            for (long i = 0; i < bits.Length; ++i) bits[i] = (ushort)(BitConverter.SingleToUInt32Bits((float)(random.NextDouble() - 0.5) * 0.05f) >> 16);
            var matrix = new BFloat16Matrix(bits, outFeatures, inFeatures);
            foreach (int t in rows)
            {
                var input = new float[t * inFeatures];
                for (int i = 0; i < input.Length; ++i) input[i] = (float)(random.NextDouble() - 0.5);
                var output = new float[t * outFeatures];
                matrix.Multiply(input, t, output);
                int repeats = Math.Max(2, (int)(4e9 / (2.0 * t * outFeatures * inFeatures)));
                var stopwatch = Stopwatch.StartNew();
                for (int r = 0; r < repeats; ++r) matrix.Multiply(input, t, output);
                double seconds = stopwatch.Elapsed.TotalSeconds / repeats;
                Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,-14} rows {1,4}  [{2}x{3}]  {4,7:F2} ms  {5,6:F1} GFLOP/s",
                    name, t, outFeatures, inFeatures, seconds * 1e3, 2.0 * t * outFeatures * inFeatures / seconds / 1e9));
            }
        }
    }
}

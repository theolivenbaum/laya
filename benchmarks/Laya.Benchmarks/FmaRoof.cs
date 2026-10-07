using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Laya.Benchmarks;

/// <summary>
/// The machine's FMA ceiling: twelve independent register-to-register FMA chains, no memory.
/// <c>dotnet run -c Release -- --fma-roof</c>. Measure this before reading any GFLOP/s figure.
/// The loops sit in their own call-free methods: every zmm register is caller-saved, so
/// accumulators live across a Stopwatch call get spilled to the stack on every iteration.
/// </summary>
internal static class FmaRoof
{
    private const int Iterations = 50_000_000;

    public static void Run()
    {
        for (int round = 0; round < 3; ++round)
        {
            var sw = Stopwatch.StartNew();
            Sink += Loop512(Iterations);
            double g512 = Iterations * 12.0 * 32 / sw.Elapsed.TotalSeconds / 1e9;
            sw.Restart();
            Sink += Loop256(Iterations);
            double g256 = Iterations * 12.0 * 16 / sw.Elapsed.TotalSeconds / 1e9;
            Console.WriteLine($"512-bit: {g512:F1} GFLOP/s   256-bit: {g256:F1} GFLOP/s");
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float Loop512(int n)
    {
        var a = Vector512.Create(1.0000001f); var b = Vector512.Create(0.9999999f);
        var c0 = Vector512.Create(0f); var c1 = c0; var c2 = c0; var c3 = c0; var c4 = c0; var c5 = c0;
        var c6 = c0; var c7 = c0; var c8 = c0; var c9 = c0; var ca = c0; var cb = c0;
        for (int i = 0; i < n; ++i)
        {
            c0 = Vector512.FusedMultiplyAdd(a, b, c0); c1 = Vector512.FusedMultiplyAdd(a, b, c1);
            c2 = Vector512.FusedMultiplyAdd(a, b, c2); c3 = Vector512.FusedMultiplyAdd(a, b, c3);
            c4 = Vector512.FusedMultiplyAdd(a, b, c4); c5 = Vector512.FusedMultiplyAdd(a, b, c5);
            c6 = Vector512.FusedMultiplyAdd(a, b, c6); c7 = Vector512.FusedMultiplyAdd(a, b, c7);
            c8 = Vector512.FusedMultiplyAdd(a, b, c8); c9 = Vector512.FusedMultiplyAdd(a, b, c9);
            ca = Vector512.FusedMultiplyAdd(a, b, ca); cb = Vector512.FusedMultiplyAdd(a, b, cb);
        }
        return (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + ca + cb).ToScalar();
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float Loop256(int n)
    {
        var a = Vector256.Create(1.0000001f); var b = Vector256.Create(0.9999999f);
        var c0 = Vector256.Create(0f); var c1 = c0; var c2 = c0; var c3 = c0; var c4 = c0; var c5 = c0;
        var c6 = c0; var c7 = c0; var c8 = c0; var c9 = c0; var ca = c0; var cb = c0;
        for (int i = 0; i < n; ++i)
        {
            c0 = Vector256.FusedMultiplyAdd(a, b, c0); c1 = Vector256.FusedMultiplyAdd(a, b, c1);
            c2 = Vector256.FusedMultiplyAdd(a, b, c2); c3 = Vector256.FusedMultiplyAdd(a, b, c3);
            c4 = Vector256.FusedMultiplyAdd(a, b, c4); c5 = Vector256.FusedMultiplyAdd(a, b, c5);
            c6 = Vector256.FusedMultiplyAdd(a, b, c6); c7 = Vector256.FusedMultiplyAdd(a, b, c7);
            c8 = Vector256.FusedMultiplyAdd(a, b, c8); c9 = Vector256.FusedMultiplyAdd(a, b, c9);
            ca = Vector256.FusedMultiplyAdd(a, b, ca); cb = Vector256.FusedMultiplyAdd(a, b, cb);
        }
        return (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + ca + cb).ToScalar();
    }

    public static float Sink;
}

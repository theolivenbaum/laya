using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;

namespace Laya.Benchmarks;

/// <summary>The 32 x 12 register tile alone, over L1-resident operands: the instruction mix's own ceiling.</summary>
internal static unsafe class TileProbe
{
    public static void Run(int steps, int panels, int groups)
    {
        float* a = (float*)NativeMemory.AlignedAlloc((nuint)(groups * 2 * steps * 16 * 4), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)(panels * steps * 12 * 4), 64);
        float* c = (float*)NativeMemory.AlignedAlloc((nuint)(12 * 32 * 4), 64);
        for (int i = 0; i < groups * 2 * steps * 16; ++i) a[i] = 0.001f * (i % 7);
        for (int i = 0; i < panels * steps * 12; ++i) b[i] = 0.001f * (i % 5);
        int repeats = 2_000_000_000 / (steps * 24 * 32);
        for (int round = 0; round < 3; ++round)
        {
            var sw = Stopwatch.StartNew();
            for (int r = 0; r < repeats; ++r)
            {
                float* ag = a + (r % groups) * 2 * steps * 16;
                Tile32x12(ag, ag + steps * 16, b + (r % panels) * steps * 12, steps, c, 32, r > 0);
            }
            Console.WriteLine($"tile 32x12, {steps} steps, {panels} weight panels, {groups} row tiles: {repeats * (double)steps * 32 * 12 * 2 / sw.Elapsed.TotalSeconds / 1e9:F1} GFLOP/s");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void Tile32x12(float* a0, float* a1, float* b, int steps, float* c, int columnStride, bool accumulate)
    {
        Vector512<float> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51,
            c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1;
        if (accumulate)
        {
            float* e = c;
            c00 = Vector512.Load(e); c01 = Vector512.Load(e + 16); e += columnStride;
            c10 = Vector512.Load(e); c11 = Vector512.Load(e + 16); e += columnStride;
            c20 = Vector512.Load(e); c21 = Vector512.Load(e + 16); e += columnStride;
            c30 = Vector512.Load(e); c31 = Vector512.Load(e + 16); e += columnStride;
            c40 = Vector512.Load(e); c41 = Vector512.Load(e + 16); e += columnStride;
            c50 = Vector512.Load(e); c51 = Vector512.Load(e + 16); e += columnStride;
            c60 = Vector512.Load(e); c61 = Vector512.Load(e + 16); e += columnStride;
            c70 = Vector512.Load(e); c71 = Vector512.Load(e + 16); e += columnStride;
            c80 = Vector512.Load(e); c81 = Vector512.Load(e + 16); e += columnStride;
            c90 = Vector512.Load(e); c91 = Vector512.Load(e + 16); e += columnStride;
            ca0 = Vector512.Load(e); ca1 = Vector512.Load(e + 16); e += columnStride;
            cb0 = Vector512.Load(e); cb1 = Vector512.Load(e + 16);
        }
        else
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector512<float>.Zero;
            c60 = c61 = c70 = c71 = c80 = c81 = c90 = c91 = ca0 = ca1 = cb0 = cb1 = Vector512<float>.Zero;
        }

        for (int i = 0; i < steps; ++i, a0 += 16, a1 += 16, b += 12)
        {
            var x0 = Vector512.Load(a0);
            var x1 = Vector512.Load(a1);
            var w = Vector512.Create(b[0]);
            c00 = Vector512.FusedMultiplyAdd(x0, w, c00); c01 = Vector512.FusedMultiplyAdd(x1, w, c01);
            w = Vector512.Create(b[1]);
            c10 = Vector512.FusedMultiplyAdd(x0, w, c10); c11 = Vector512.FusedMultiplyAdd(x1, w, c11);
            w = Vector512.Create(b[2]);
            c20 = Vector512.FusedMultiplyAdd(x0, w, c20); c21 = Vector512.FusedMultiplyAdd(x1, w, c21);
            w = Vector512.Create(b[3]);
            c30 = Vector512.FusedMultiplyAdd(x0, w, c30); c31 = Vector512.FusedMultiplyAdd(x1, w, c31);
            w = Vector512.Create(b[4]);
            c40 = Vector512.FusedMultiplyAdd(x0, w, c40); c41 = Vector512.FusedMultiplyAdd(x1, w, c41);
            w = Vector512.Create(b[5]);
            c50 = Vector512.FusedMultiplyAdd(x0, w, c50); c51 = Vector512.FusedMultiplyAdd(x1, w, c51);
            w = Vector512.Create(b[6]);
            c60 = Vector512.FusedMultiplyAdd(x0, w, c60); c61 = Vector512.FusedMultiplyAdd(x1, w, c61);
            w = Vector512.Create(b[7]);
            c70 = Vector512.FusedMultiplyAdd(x0, w, c70); c71 = Vector512.FusedMultiplyAdd(x1, w, c71);
            w = Vector512.Create(b[8]);
            c80 = Vector512.FusedMultiplyAdd(x0, w, c80); c81 = Vector512.FusedMultiplyAdd(x1, w, c81);
            w = Vector512.Create(b[9]);
            c90 = Vector512.FusedMultiplyAdd(x0, w, c90); c91 = Vector512.FusedMultiplyAdd(x1, w, c91);
            w = Vector512.Create(b[10]);
            ca0 = Vector512.FusedMultiplyAdd(x0, w, ca0); ca1 = Vector512.FusedMultiplyAdd(x1, w, ca1);
            w = Vector512.Create(b[11]);
            cb0 = Vector512.FusedMultiplyAdd(x0, w, cb0); cb1 = Vector512.FusedMultiplyAdd(x1, w, cb1);
        }

        float* s = c;
        c00.Store(s); c01.Store(s + 16); s += columnStride;
        c10.Store(s); c11.Store(s + 16); s += columnStride;
        c20.Store(s); c21.Store(s + 16); s += columnStride;
        c30.Store(s); c31.Store(s + 16); s += columnStride;
        c40.Store(s); c41.Store(s + 16); s += columnStride;
        c50.Store(s); c51.Store(s + 16); s += columnStride;
        c60.Store(s); c61.Store(s + 16); s += columnStride;
        c70.Store(s); c71.Store(s + 16); s += columnStride;
        c80.Store(s); c81.Store(s + 16); s += columnStride;
        c90.Store(s); c91.Store(s + 16); s += columnStride;
        ca0.Store(s); ca1.Store(s + 16); s += columnStride;
        cb0.Store(s); cb1.Store(s + 16);
    }

}

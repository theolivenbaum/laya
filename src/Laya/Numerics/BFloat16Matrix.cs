using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Laya.Numerics;

/// <summary>
/// A projection weight kept in bfloat16, in the panel layout of <see cref="PackedMatrix"/>, and
/// multiplied in fp32.
///
/// <para>LFM2 checkpoints are stored in bf16, and bf16 is exactly the top half of an fp32, so the
/// weights PyTorch computes with in fp32 are recovered bit for bit by a 16-bit shift. Keeping them
/// in bf16 halves the resident size (d1-3B's text stack is 5 GB instead of 10) and the bytes every
/// product streams from memory, without moving a single result: the arithmetic is fp32 FMA on the
/// widened values, as in PyTorch.</para>
///
/// <para>The widening is not done in the inner loop. For each panel of output columns a block of
/// <see cref="BlockDepth"/> reduction steps is widened once into a small fp32 scratch that stays in
/// L2, and the same register-tiled kernel as <see cref="PackedMatrix"/> then streams every token row
/// against it. The conversion costs one pass over the block per call, which a prompt of more than a
/// handful of tokens amortizes to nothing; converting inside the kernel instead would put two
/// shuffle uops on the FMA ports for every weight vector of every row.</para>
///
/// <para>Blocking the reduction also keeps the scratch small: the MLP's down projection reduces over
/// 10752 inputs, and a whole fp32 panel of it (2.75 MB) would no longer fit in L2. Blocks after the
/// first accumulate into the output rows; the order of the additions is unchanged, so the result is
/// bit-identical to an unblocked reduction.</para>
/// </summary>
public sealed class BFloat16Matrix
{
    private readonly ushort[] _data;          // [panel][inFeatures][panelWidth]
    private readonly int _panelWidth;
    private readonly int _panels;
    private readonly long _panelStride;

    public int InFeatures { get; }
    public int OutFeatures { get; }

    /// <summary>Reduction steps widened at a time. 512 x 64 lanes x 4 bytes = 128 KiB of scratch.</summary>
    public static int BlockDepth { get; set; } = ReadBlockDepth();

    private static int ReadBlockDepth()
    {
        string? configured = Environment.GetEnvironmentVariable("LAYA_GEMM_KC");
        return int.TryParse(configured, out int value) && value >= 16 ? value : 512;
    }

    [ThreadStatic] private static float[]? t_scratch;

    /// <summary>Repacks a PyTorch <c>[out, in]</c> bf16 weight (raw bit patterns).</summary>
    public BFloat16Matrix(ReadOnlySpan<ushort> rowMajor, int outFeatures, int inFeatures)
    {
        if (rowMajor.Length < (long)outFeatures * inFeatures)
        {
            throw new ArgumentException($"expected {(long)outFeatures * inFeatures} weights for [{outFeatures}, {inFeatures}].",
                nameof(rowMajor));
        }

        OutFeatures = outFeatures;
        InFeatures = inFeatures;
        _panelWidth = PanelWidth;
        _panels = (outFeatures + _panelWidth - 1) / _panelWidth;
        _panelStride = (long)inFeatures * _panelWidth;
        _data = new ushort[_panels * _panelStride];

        for (int panel = 0; panel < _panels; ++panel)
        {
            int first = panel * _panelWidth;
            int columns = Math.Min(_panelWidth, outFeatures - first);
            long destination = panel * _panelStride;
            for (int j = 0; j < columns; ++j)
            {
                var source = rowMajor.Slice((int)((long)(first + j) * inFeatures), inFeatures);
                for (int i = 0; i < inFeatures; ++i) _data[destination + (long)i * _panelWidth + j] = source[i];
            }
        }
    }

    /// <summary>Output columns per panel on this machine: four 512-bit vectors, or two portable ones.</summary>
    public static int PanelWidth => SimdOps.UseVector512 ? 4 * 16 : 2 * Vector<float>.Count;

    public long Bytes => (long)_data.Length * sizeof(ushort);

    /// <summary>Row <paramref name="column"/> of the original <c>[out, in]</c> matrix, widened.</summary>
    public void ReadRow(int column, Span<float> destination)
    {
        int panel = column / _panelWidth;
        int lane = column % _panelWidth;
        long start = panel * _panelStride + lane;
        var bits = MemoryMarshal.Cast<float, uint>(destination);
        for (int i = 0; i < InFeatures; ++i) bits[i] = (uint)_data[start + (long)i * _panelWidth] << 16;
    }

    /// <summary><c>output = input · weightᵀ</c> over <paramref name="rows"/> contiguous rows.</summary>
    public void Multiply(ReadOnlySpan<float> input, int rows, Span<float> output, ParallelOptions? parallel = null)
        => Multiply(input, rows, InFeatures, output, OutFeatures, parallel);

    /// <summary>
    /// <c>output = input · weightᵀ</c> with strided rows: <paramref name="inputStride"/> and
    /// <paramref name="outputStride"/> are the distances between consecutive token rows.
    /// </summary>
    public unsafe void Multiply(ReadOnlySpan<float> input, int rows, int inputStride, Span<float> output,
        int outputStride, ParallelOptions? parallel = null)
    {
        if (rows == 0) return;
        if (input.Length < (long)(rows - 1) * inputStride + InFeatures) throw new ArgumentException("input is too small", nameof(input));
        if (output.Length < (long)(rows - 1) * outputStride + OutFeatures) throw new ArgumentException("output is too small", nameof(output));

        fixed (ushort* weights = _data)
        fixed (float* inputPointer = input, outputPointer = output)
        {
            int workers = LayaRuntime.WorkersOf(parallel);
            if (workers <= 1 || (long)rows * OutFeatures * InFeatures <= 2_000_000)
            {
                float* scratch = Scratch();
                for (int panel = 0; panel < _panels; ++panel)
                {
                    Panel(weights, inputPointer, rows, inputStride, outputPointer, outputStride, panel, scratch);
                }
                return;
            }

            nint weightAddress = (nint)weights;
            nint inputAddress = (nint)inputPointer;
            nint outputAddress = (nint)outputPointer;
            int chunk = Math.Max(1, _panels / (workers * 4));
            int chunks = (_panels + chunk - 1) / chunk;
            Parallel.For(0, chunks, LayaRuntime.Resolve(parallel), index =>
            {
                float* scratch = Scratch();
                int first = index * chunk;
                int last = Math.Min(_panels, first + chunk);
                for (int panel = first; panel < last; ++panel)
                {
                    Panel((ushort*)weightAddress, (float*)inputAddress, rows, inputStride, (float*)outputAddress,
                        outputStride, panel, scratch);
                }
            });
        }
    }

    /// <summary>
    /// A thread's widening scratch, pinned for the life of the thread. It is allocated on the pinned
    /// object heap, so the pointer handed to the kernels stays valid without a <c>fixed</c> block.
    /// </summary>
    private unsafe float* Scratch()
    {
        int length = BlockDepth * PanelWidth;
        var scratch = t_scratch;
        if (scratch is null || scratch.Length < length)
        {
            scratch = GC.AllocateUninitializedArray<float>(length, pinned: true);
            t_scratch = scratch;
        }
        return (float*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(scratch));
    }

    private unsafe void Panel(ushort* weights, float* input, int rows, int inputStride, float* output,
        int outputStride, int panel, float* scratch)
    {
        int width = _panelWidth;
        int columns = Math.Min(width, OutFeatures - panel * width);
        ushort* panelBase = weights + panel * _panelStride;
        int depth = BlockDepth;

        for (int k0 = 0; k0 < InFeatures; k0 += depth)
        {
            int steps = Math.Min(depth, InFeatures - k0);
            Widen(panelBase + (long)k0 * width, scratch, steps * width);
            bool accumulate = k0 > 0;
            if (columns != width)
            {
                Ragged(scratch, input + k0, rows, inputStride, output + panel * width, outputStride, steps, columns, accumulate);
            }
            else if (SimdOps.UseVector512)
            {
                Kernel512(scratch, input + k0, rows, inputStride, output + panel * width, outputStride, steps, accumulate);
            }
            else
            {
                KernelPortable(scratch, input + k0, rows, inputStride, output + panel * width, outputStride, steps, accumulate);
            }
        }
    }

    /// <summary>bf16 bit patterns to fp32: each value moves to the top half of a 32-bit lane.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Widen(ushort* source, float* destination, int count)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (; i + 32 <= count; i += 32)
            {
                var packed = Vector512.Load(source + i);
                var (low, high) = Vector512.Widen(packed);
                Vector512.ShiftLeft(low, 16).AsSingle().Store(destination + i);
                Vector512.ShiftLeft(high, 16).AsSingle().Store(destination + i + 16);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (; i + 16 <= count; i += 16)
            {
                var packed = Vector256.Load(source + i);
                var (low, high) = Vector256.Widen(packed);
                Vector256.ShiftLeft(low, 16).AsSingle().Store(destination + i);
                Vector256.ShiftLeft(high, 16).AsSingle().Store(destination + i + 8);
            }
        }
        uint* bits = (uint*)destination;
        for (; i < count; ++i) bits[i] = (uint)source[i] << 16;
    }

    /// <summary>
    /// The 6 x 64 register tile of <see cref="PackedMatrix"/>'s 512-bit kernel over one widened
    /// block: twenty-four accumulators, four weight vectors and one reused broadcast.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel512(float* panel, float* input, int rows, int inputStride, float* output,
        int outputStride, int steps, bool accumulate)
    {
        const int width = 16;
        const int panelWidth = 4 * width;
        const int rowBlock = 6;

        int row = 0;
        for (; row + rowBlock <= rows; row += rowBlock)
        {
            float* d = output + (long)row * outputStride;
            Vector512<float> c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23,
                c30, c31, c32, c33, c40, c41, c42, c43, c50, c51, c52, c53;
            if (accumulate)
            {
                float* e = d;
                c00 = Vector512.Load(e); c01 = Vector512.Load(e + width); c02 = Vector512.Load(e + 2 * width); c03 = Vector512.Load(e + 3 * width);
                e += outputStride;
                c10 = Vector512.Load(e); c11 = Vector512.Load(e + width); c12 = Vector512.Load(e + 2 * width); c13 = Vector512.Load(e + 3 * width);
                e += outputStride;
                c20 = Vector512.Load(e); c21 = Vector512.Load(e + width); c22 = Vector512.Load(e + 2 * width); c23 = Vector512.Load(e + 3 * width);
                e += outputStride;
                c30 = Vector512.Load(e); c31 = Vector512.Load(e + width); c32 = Vector512.Load(e + 2 * width); c33 = Vector512.Load(e + 3 * width);
                e += outputStride;
                c40 = Vector512.Load(e); c41 = Vector512.Load(e + width); c42 = Vector512.Load(e + 2 * width); c43 = Vector512.Load(e + 3 * width);
                e += outputStride;
                c50 = Vector512.Load(e); c51 = Vector512.Load(e + width); c52 = Vector512.Load(e + 2 * width); c53 = Vector512.Load(e + 3 * width);
            }
            else
            {
                c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = c20 = c21 = c22 = c23 = Vector512<float>.Zero;
                c30 = c31 = c32 = c33 = c40 = c41 = c42 = c43 = c50 = c51 = c52 = c53 = Vector512<float>.Zero;
            }

            float* a0 = input + (long)(row + 0) * inputStride;
            float* a1 = input + (long)(row + 1) * inputStride;
            float* a2 = input + (long)(row + 2) * inputStride;
            float* a3 = input + (long)(row + 3) * inputStride;
            float* a4 = input + (long)(row + 4) * inputStride;
            float* a5 = input + (long)(row + 5) * inputStride;
            float* w = panel;

            for (int i = 0; i < steps; ++i, w += panelWidth)
            {
                var b0 = Vector512.Load(w + 0 * width);
                var b1 = Vector512.Load(w + 1 * width);
                var b2 = Vector512.Load(w + 2 * width);
                var b3 = Vector512.Load(w + 3 * width);
                var x = Vector512.Create(a0[i]);
                c00 = Vector512.FusedMultiplyAdd(x, b0, c00);
                c01 = Vector512.FusedMultiplyAdd(x, b1, c01);
                c02 = Vector512.FusedMultiplyAdd(x, b2, c02);
                c03 = Vector512.FusedMultiplyAdd(x, b3, c03);
                x = Vector512.Create(a1[i]);
                c10 = Vector512.FusedMultiplyAdd(x, b0, c10);
                c11 = Vector512.FusedMultiplyAdd(x, b1, c11);
                c12 = Vector512.FusedMultiplyAdd(x, b2, c12);
                c13 = Vector512.FusedMultiplyAdd(x, b3, c13);
                x = Vector512.Create(a2[i]);
                c20 = Vector512.FusedMultiplyAdd(x, b0, c20);
                c21 = Vector512.FusedMultiplyAdd(x, b1, c21);
                c22 = Vector512.FusedMultiplyAdd(x, b2, c22);
                c23 = Vector512.FusedMultiplyAdd(x, b3, c23);
                x = Vector512.Create(a3[i]);
                c30 = Vector512.FusedMultiplyAdd(x, b0, c30);
                c31 = Vector512.FusedMultiplyAdd(x, b1, c31);
                c32 = Vector512.FusedMultiplyAdd(x, b2, c32);
                c33 = Vector512.FusedMultiplyAdd(x, b3, c33);
                x = Vector512.Create(a4[i]);
                c40 = Vector512.FusedMultiplyAdd(x, b0, c40);
                c41 = Vector512.FusedMultiplyAdd(x, b1, c41);
                c42 = Vector512.FusedMultiplyAdd(x, b2, c42);
                c43 = Vector512.FusedMultiplyAdd(x, b3, c43);
                x = Vector512.Create(a5[i]);
                c50 = Vector512.FusedMultiplyAdd(x, b0, c50);
                c51 = Vector512.FusedMultiplyAdd(x, b1, c51);
                c52 = Vector512.FusedMultiplyAdd(x, b2, c52);
                c53 = Vector512.FusedMultiplyAdd(x, b3, c53);
            }

            c00.Store(d + 0 * width); c01.Store(d + 1 * width); c02.Store(d + 2 * width); c03.Store(d + 3 * width);
            d += outputStride;
            c10.Store(d + 0 * width); c11.Store(d + 1 * width); c12.Store(d + 2 * width); c13.Store(d + 3 * width);
            d += outputStride;
            c20.Store(d + 0 * width); c21.Store(d + 1 * width); c22.Store(d + 2 * width); c23.Store(d + 3 * width);
            d += outputStride;
            c30.Store(d + 0 * width); c31.Store(d + 1 * width); c32.Store(d + 2 * width); c33.Store(d + 3 * width);
            d += outputStride;
            c40.Store(d + 0 * width); c41.Store(d + 1 * width); c42.Store(d + 2 * width); c43.Store(d + 3 * width);
            d += outputStride;
            c50.Store(d + 0 * width); c51.Store(d + 1 * width); c52.Store(d + 2 * width); c53.Store(d + 3 * width);
        }

        for (; row < rows; ++row)
        {
            float* d = output + (long)row * outputStride;
            Vector512<float> t0, t1, t2, t3;
            if (accumulate)
            {
                t0 = Vector512.Load(d); t1 = Vector512.Load(d + width); t2 = Vector512.Load(d + 2 * width); t3 = Vector512.Load(d + 3 * width);
            }
            else
            {
                t0 = t1 = t2 = t3 = Vector512<float>.Zero;
            }
            float* a0 = input + (long)row * inputStride;
            float* w = panel;
            for (int i = 0; i < steps; ++i, w += panelWidth)
            {
                var x = Vector512.Create(a0[i]);
                t0 = Vector512.FusedMultiplyAdd(x, Vector512.Load(w + 0 * width), t0);
                t1 = Vector512.FusedMultiplyAdd(x, Vector512.Load(w + 1 * width), t1);
                t2 = Vector512.FusedMultiplyAdd(x, Vector512.Load(w + 2 * width), t2);
                t3 = Vector512.FusedMultiplyAdd(x, Vector512.Load(w + 3 * width), t3);
            }
            t0.Store(d + 0 * width); t1.Store(d + 1 * width); t2.Store(d + 2 * width); t3.Store(d + 3 * width);
        }
    }

    /// <summary>The same over the portable vector width: a 6 x 2-vector tile.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void KernelPortable(float* panel, float* input, int rows, int inputStride, float* output,
        int outputStride, int steps, bool accumulate)
    {
        int width = Vector<float>.Count;
        int panelWidth = 2 * width;
        const int rowBlock = 6;

        int row = 0;
        for (; row + rowBlock <= rows; row += rowBlock)
        {
            float* d = output + (long)row * outputStride;
            Vector<float> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
            if (accumulate)
            {
                float* e = d;
                c00 = Vector.Load(e); c01 = Vector.Load(e + width); e += outputStride;
                c10 = Vector.Load(e); c11 = Vector.Load(e + width); e += outputStride;
                c20 = Vector.Load(e); c21 = Vector.Load(e + width); e += outputStride;
                c30 = Vector.Load(e); c31 = Vector.Load(e + width); e += outputStride;
                c40 = Vector.Load(e); c41 = Vector.Load(e + width); e += outputStride;
                c50 = Vector.Load(e); c51 = Vector.Load(e + width);
            }
            else
            {
                c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector<float>.Zero;
            }

            float* a0 = input + (long)(row + 0) * inputStride;
            float* a1 = input + (long)(row + 1) * inputStride;
            float* a2 = input + (long)(row + 2) * inputStride;
            float* a3 = input + (long)(row + 3) * inputStride;
            float* a4 = input + (long)(row + 4) * inputStride;
            float* a5 = input + (long)(row + 5) * inputStride;
            float* w = panel;
            for (int i = 0; i < steps; ++i, w += panelWidth)
            {
                var b0 = Vector.Load(w);
                var b1 = Vector.Load(w + width);
                var x = new Vector<float>(a0[i]);
                c00 = Vector.FusedMultiplyAdd(x, b0, c00); c01 = Vector.FusedMultiplyAdd(x, b1, c01);
                x = new Vector<float>(a1[i]);
                c10 = Vector.FusedMultiplyAdd(x, b0, c10); c11 = Vector.FusedMultiplyAdd(x, b1, c11);
                x = new Vector<float>(a2[i]);
                c20 = Vector.FusedMultiplyAdd(x, b0, c20); c21 = Vector.FusedMultiplyAdd(x, b1, c21);
                x = new Vector<float>(a3[i]);
                c30 = Vector.FusedMultiplyAdd(x, b0, c30); c31 = Vector.FusedMultiplyAdd(x, b1, c31);
                x = new Vector<float>(a4[i]);
                c40 = Vector.FusedMultiplyAdd(x, b0, c40); c41 = Vector.FusedMultiplyAdd(x, b1, c41);
                x = new Vector<float>(a5[i]);
                c50 = Vector.FusedMultiplyAdd(x, b0, c50); c51 = Vector.FusedMultiplyAdd(x, b1, c51);
            }

            c00.Store(d); c01.Store(d + width); d += outputStride;
            c10.Store(d); c11.Store(d + width); d += outputStride;
            c20.Store(d); c21.Store(d + width); d += outputStride;
            c30.Store(d); c31.Store(d + width); d += outputStride;
            c40.Store(d); c41.Store(d + width); d += outputStride;
            c50.Store(d); c51.Store(d + width);
        }

        for (; row < rows; ++row)
        {
            float* d = output + (long)row * outputStride;
            var t0 = accumulate ? Vector.Load(d) : Vector<float>.Zero;
            var t1 = accumulate ? Vector.Load(d + width) : Vector<float>.Zero;
            float* a0 = input + (long)row * inputStride;
            float* w = panel;
            for (int i = 0; i < steps; ++i, w += panelWidth)
            {
                var x = new Vector<float>(a0[i]);
                t0 = Vector.FusedMultiplyAdd(x, Vector.Load(w), t0);
                t1 = Vector.FusedMultiplyAdd(x, Vector.Load(w + width), t1);
            }
            t0.Store(d);
            t1.Store(d + width);
        }
    }

    /// <summary>The last, partial panel: computed full width into a tile, then trimmed.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private unsafe void Ragged(float* panel, float* input, int rows, int inputStride, float* output, int outputStride,
        int steps, int columns, bool accumulate)
    {
        int panelWidth = _panelWidth;
        Span<float> tile = stackalloc float[panelWidth];
        for (int row = 0; row < rows; ++row)
        {
            float* d = output + (long)row * outputStride;
            tile.Clear();
            if (accumulate) new ReadOnlySpan<float>(d, columns).CopyTo(tile);
            float* a0 = input + (long)row * inputStride;
            float* w = panel;
            for (int i = 0; i < steps; ++i, w += panelWidth)
            {
                float a = a0[i];
                for (int j = 0; j < panelWidth; ++j) tile[j] = MathF.FusedMultiplyAdd(a, w[j], tile[j]);
            }
            tile[..columns].CopyTo(new Span<float>(d, columns));
        }
    }
}

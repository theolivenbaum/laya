using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace StructuredDecisions.Numerics;

/// <summary>
/// A projection weight kept in <b>bf16</b>, panel-packed like <see cref="PackedMatrix"/> and widened to
/// fp32 inside the kernel.
///
/// <para>Mica's 4B parameters are 17 GB as fp32 — more than the machines this runs on have — and 8.4 GB
/// as the bf16 they are stored in. A bf16 value is the top half of an fp32, so widening is exact: the
/// products and fp32 accumulation are identical to converting the whole matrix at load time, which keeps
/// the "compute in fp32" rule intact while halving the bytes streamed per token row.</para>
///
/// <para>The widening costs no shuffles. Within a reduction step each pair of output vectors is stored
/// interleaved, so one 32-bit lane holds output <c>j</c> of the first vector in its low half and output
/// <c>j</c> of the second in its high half. One load then yields both: <c>lane &lt;&lt; 16</c> is the
/// first vector and <c>lane &amp; 0xFFFF0000</c> the second — two cheap integer ops per two weight
/// vectors, against the 12 FMAs each pair feeds.</para>
/// </summary>
public sealed class PackedMatrixBf16 : ILinear
{
    private readonly ushort[] _data;         // [panel][inFeatures][panelWidth], pairs interleaved
    private readonly int _vector;            // fp32 lanes per vector
    private readonly int _panelWidth;
    private readonly int _panels;
    private readonly int _panelStride;

    public int InFeatures { get; }
    public int OutFeatures { get; }
    public long Bytes => (long)_data.Length * sizeof(ushort);

    /// <summary>Repacks a PyTorch <c>[out, in]</c> weight given as raw bf16 bit patterns.</summary>
    public PackedMatrixBf16(ReadOnlySpan<ushort> rowMajor, int outFeatures, int inFeatures)
    {
        if (rowMajor.Length < (long)outFeatures * inFeatures)
        {
            throw new ArgumentException($"expected {(long)outFeatures * inFeatures} weights for [{outFeatures}, {inFeatures}].",
                nameof(rowMajor));
        }
        OutFeatures = outFeatures;
        InFeatures = inFeatures;
        _vector = SimdOps.UseVector512 ? 16 : Vector<float>.Count;
        _panelWidth = SimdOps.UseVector512 ? 4 * 16 : 2 * Vector<float>.Count;
        _panels = (outFeatures + _panelWidth - 1) / _panelWidth;
        _panelStride = inFeatures * _panelWidth;
        _data = new ushort[(long)_panels * _panelStride];

        for (int panel = 0; panel < _panels; ++panel)
        {
            int first = panel * _panelWidth;
            int columns = Math.Min(_panelWidth, outFeatures - first);
            long destination = (long)panel * _panelStride;
            for (int j = 0; j < columns; ++j)
            {
                int slot = Slot(j);
                var source = rowMajor.Slice((first + j) * inFeatures, inFeatures);
                for (int i = 0; i < inFeatures; ++i) _data[destination + (long)i * _panelWidth + slot] = source[i];
            }
        }
    }

    /// <summary>Where output column <paramref name="column"/> of a panel lives within one reduction step.</summary>
    private int Slot(int column)
    {
        int pair = column / (2 * _vector);
        int within = column % (2 * _vector);
        int half = within / _vector;
        int lane = within % _vector;
        return pair * 2 * _vector + 2 * lane + half;
    }

    public unsafe void Multiply(ReadOnlySpan<float> input, int rows, ReadOnlySpan<float> bias, Span<float> output,
        ParallelOptions? parallel = null)
        => Multiply(input, rows, InFeatures, bias, output, OutFeatures, parallel);

    public unsafe void Multiply(ReadOnlySpan<float> input, int rows, int inputStride,
        ReadOnlySpan<float> bias, Span<float> output, int outputStride, ParallelOptions? parallel = null)
    {
        if (input.Length < (long)(rows - 1) * inputStride + InFeatures) throw new ArgumentException("input is too small", nameof(input));
        if (output.Length < (long)(rows - 1) * outputStride + OutFeatures) throw new ArgumentException("output is too small", nameof(output));
        if (!bias.IsEmpty && bias.Length < OutFeatures) throw new ArgumentException("bias is too small", nameof(bias));
        if (rows == 0) return;

        fixed (ushort* weights = _data)
        fixed (float* inputPointer = input, biasPointer = bias, outputPointer = output)
        {
            float* bias0 = bias.IsEmpty ? null : biasPointer;
            int workers = DecisionRuntime.WorkersOf(parallel);
            if (workers <= 1 || (long)rows * OutFeatures * InFeatures <= 1_000_000)
            {
                for (int panel = 0; panel < _panels; ++panel)
                {
                    Panel(weights, inputPointer, rows, inputStride, bias0, outputPointer, outputStride, panel);
                }
                return;
            }

            nint weightAddress = (nint)weights, inputAddress = (nint)inputPointer;
            nint biasAddress = (nint)bias0, outputAddress = (nint)outputPointer;
            int chunk = Math.Max(1, _panels / (workers * 4));
            int chunks = (_panels + chunk - 1) / chunk;
            Parallel.For(0, chunks, DecisionRuntime.Resolve(parallel), index =>
            {
                int last = Math.Min(_panels, (index + 1) * chunk);
                for (int panel = index * chunk; panel < last; ++panel)
                {
                    Panel((ushort*)weightAddress, (float*)inputAddress, rows, inputStride, (float*)biasAddress,
                        (float*)outputAddress, outputStride, panel);
                }
            });
        }
    }

    private unsafe void Panel(ushort* weights, float* input, int rows, int inputStride, float* bias,
        float* output, int outputStride, int panel)
    {
        int columns = Math.Min(_panelWidth, OutFeatures - panel * _panelWidth);
        if (columns != _panelWidth) PanelRagged(weights, input, rows, inputStride, bias, output, outputStride, panel, columns);
        else if (SimdOps.UseVector512) Panel512(weights, input, rows, inputStride, bias, output, outputStride, panel);
        else PanelPortable(weights, input, rows, inputStride, bias, output, outputStride, panel);
    }

    /// <summary>64 outputs by six rows, as in <see cref="PackedMatrix"/>; two loads widen into four vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void Panel512(ushort* weights, float* input, int rows, int inputStride, float* bias,
        float* output, int outputStride, int panel)
    {
        const int width = 16;
        const int panelWidth = 4 * width;
        const int rowBlock = 6;
        int features = InFeatures;
        int n0 = panel * panelWidth;
        uint* panelBase = (uint*)(weights + (long)panel * _panelStride);
        var high = Vector512.Create(0xFFFF0000u);

        var seed0 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0 + 0 * width);
        var seed1 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0 + 1 * width);
        var seed2 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0 + 2 * width);
        var seed3 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0 + 3 * width);

        int row = 0;
        for (; row + rowBlock <= rows; row += rowBlock)
        {
            var c00 = seed0; var c01 = seed1; var c02 = seed2; var c03 = seed3;
            var c10 = seed0; var c11 = seed1; var c12 = seed2; var c13 = seed3;
            var c20 = seed0; var c21 = seed1; var c22 = seed2; var c23 = seed3;
            var c30 = seed0; var c31 = seed1; var c32 = seed2; var c33 = seed3;
            var c40 = seed0; var c41 = seed1; var c42 = seed2; var c43 = seed3;
            var c50 = seed0; var c51 = seed1; var c52 = seed2; var c53 = seed3;

            float* a0 = input + (long)(row + 0) * inputStride;
            float* a1 = input + (long)(row + 1) * inputStride;
            float* a2 = input + (long)(row + 2) * inputStride;
            float* a3 = input + (long)(row + 3) * inputStride;
            float* a4 = input + (long)(row + 4) * inputStride;
            float* a5 = input + (long)(row + 5) * inputStride;
            uint* w = panelBase;

            for (int i = 0; i < features; ++i, w += 2 * width)
            {
                var p01 = Vector512.Load(w);
                var p23 = Vector512.Load(w + width);
                var b0 = Vector512.ShiftLeft(p01, 16).AsSingle();
                var b1 = (p01 & high).AsSingle();
                var b2 = Vector512.ShiftLeft(p23, 16).AsSingle();
                var b3 = (p23 & high).AsSingle();
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

            float* d = output + (long)row * outputStride + n0;
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
            var t0 = seed0; var t1 = seed1; var t2 = seed2; var t3 = seed3;
            float* a0 = input + (long)row * inputStride;
            uint* w = panelBase;
            for (int i = 0; i < features; ++i, w += 2 * width)
            {
                var p01 = Vector512.Load(w);
                var p23 = Vector512.Load(w + width);
                var x = Vector512.Create(a0[i]);
                t0 = Vector512.FusedMultiplyAdd(x, Vector512.ShiftLeft(p01, 16).AsSingle(), t0);
                t1 = Vector512.FusedMultiplyAdd(x, (p01 & high).AsSingle(), t1);
                t2 = Vector512.FusedMultiplyAdd(x, Vector512.ShiftLeft(p23, 16).AsSingle(), t2);
                t3 = Vector512.FusedMultiplyAdd(x, (p23 & high).AsSingle(), t3);
            }
            float* d = output + (long)row * outputStride + n0;
            t0.Store(d + 0 * width); t1.Store(d + 1 * width); t2.Store(d + 2 * width); t3.Store(d + 3 * width);
        }
    }

    /// <summary>The same kernel over the portable vector width: one load widens into the panel's two vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void PanelPortable(ushort* weights, float* input, int rows, int inputStride, float* bias,
        float* output, int outputStride, int panel)
    {
        int width = Vector<float>.Count;
        const int rowBlock = 6;
        int features = InFeatures;
        int n0 = panel * _panelWidth;
        uint* panelBase = (uint*)(weights + (long)panel * _panelStride);
        var high = new Vector<uint>(0xFFFF0000u);

        var seed0 = bias is null ? Vector<float>.Zero : Vector.Load(bias + n0);
        var seed1 = bias is null ? Vector<float>.Zero : Vector.Load(bias + n0 + width);

        int row = 0;
        for (; row + rowBlock <= rows; row += rowBlock)
        {
            var c00 = seed0; var c01 = seed1;
            var c10 = seed0; var c11 = seed1;
            var c20 = seed0; var c21 = seed1;
            var c30 = seed0; var c31 = seed1;
            var c40 = seed0; var c41 = seed1;
            var c50 = seed0; var c51 = seed1;

            float* a0 = input + (long)(row + 0) * inputStride;
            float* a1 = input + (long)(row + 1) * inputStride;
            float* a2 = input + (long)(row + 2) * inputStride;
            float* a3 = input + (long)(row + 3) * inputStride;
            float* a4 = input + (long)(row + 4) * inputStride;
            float* a5 = input + (long)(row + 5) * inputStride;
            uint* w = panelBase;

            for (int i = 0; i < features; ++i, w += width)
            {
                var p = Vector.Load(w);
                var b0 = Vector.AsVectorSingle(Vector.ShiftLeft(p, 16));
                var b1 = Vector.AsVectorSingle(p & high);
                var v = new Vector<float>(a0[i]);
                c00 = Vector.FusedMultiplyAdd(v, b0, c00);
                c01 = Vector.FusedMultiplyAdd(v, b1, c01);
                v = new Vector<float>(a1[i]);
                c10 = Vector.FusedMultiplyAdd(v, b0, c10);
                c11 = Vector.FusedMultiplyAdd(v, b1, c11);
                v = new Vector<float>(a2[i]);
                c20 = Vector.FusedMultiplyAdd(v, b0, c20);
                c21 = Vector.FusedMultiplyAdd(v, b1, c21);
                v = new Vector<float>(a3[i]);
                c30 = Vector.FusedMultiplyAdd(v, b0, c30);
                c31 = Vector.FusedMultiplyAdd(v, b1, c31);
                v = new Vector<float>(a4[i]);
                c40 = Vector.FusedMultiplyAdd(v, b0, c40);
                c41 = Vector.FusedMultiplyAdd(v, b1, c41);
                v = new Vector<float>(a5[i]);
                c50 = Vector.FusedMultiplyAdd(v, b0, c50);
                c51 = Vector.FusedMultiplyAdd(v, b1, c51);
            }

            float* d = output + (long)row * outputStride + n0;
            c00.Store(d); c01.Store(d + width);
            d += outputStride;
            c10.Store(d); c11.Store(d + width);
            d += outputStride;
            c20.Store(d); c21.Store(d + width);
            d += outputStride;
            c30.Store(d); c31.Store(d + width);
            d += outputStride;
            c40.Store(d); c41.Store(d + width);
            d += outputStride;
            c50.Store(d); c51.Store(d + width);
        }

        for (; row < rows; ++row)
        {
            var c0 = seed0;
            var c1 = seed1;
            float* a0 = input + (long)row * inputStride;
            uint* w = panelBase;
            for (int i = 0; i < features; ++i, w += width)
            {
                var p = Vector.Load(w);
                var v = new Vector<float>(a0[i]);
                c0 = Vector.FusedMultiplyAdd(v, Vector.AsVectorSingle(Vector.ShiftLeft(p, 16)), c0);
                c1 = Vector.FusedMultiplyAdd(v, Vector.AsVectorSingle(p & high), c1);
            }
            float* d = output + (long)row * outputStride + n0;
            c0.Store(d);
            c1.Store(d + width);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private unsafe void PanelRagged(ushort* weights, float* input, int rows, int inputStride, float* bias,
        float* output, int outputStride, int panel, int columns)
    {
        int n0 = panel * _panelWidth;
        ushort* panelBase = weights + (long)panel * _panelStride;
        Span<int> slots = stackalloc int[columns];
        for (int j = 0; j < columns; ++j) slots[j] = Slot(j);

        for (int row = 0; row < rows; ++row)
        {
            float* a = input + (long)row * inputStride;
            float* destination = output + (long)row * outputStride + n0;
            for (int j = 0; j < columns; ++j)
            {
                // The same products in the same order as the vector kernels, one FMA at a time.
                float sum = bias is null ? 0f : bias[n0 + j];
                ushort* w = panelBase + slots[j];
                for (int i = 0; i < InFeatures; ++i, w += _panelWidth)
                {
                    sum = MathF.FusedMultiplyAdd(a[i], BitConverter.UInt32BitsToSingle((uint)*w << 16), sum);
                }
                destination[j] = sum;
            }
        }
    }
}

/// <summary>A linear projection <c>y = x · Wᵀ + b</c>, whatever its weights are stored as.</summary>
public interface ILinear
{
    int InFeatures { get; }
    int OutFeatures { get; }
    long Bytes { get; }

    void Multiply(ReadOnlySpan<float> input, int rows, ReadOnlySpan<float> bias, Span<float> output,
        ParallelOptions? parallel = null);

    void Multiply(ReadOnlySpan<float> input, int rows, int inputStride, ReadOnlySpan<float> bias, Span<float> output,
        int outputStride, ParallelOptions? parallel = null);
}

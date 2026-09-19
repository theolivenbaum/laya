using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Laya;

namespace Laya.Numerics;

/// <summary>
/// A projection weight, repacked once at load time into the layout the inner loop wants.
///
/// <para>PyTorch stores <c>nn.Linear</c> weights as <c>[out, in]</c>, which makes each output a dot
/// product over contiguous memory — correct, but it forces a horizontal reduction per output and
/// re-streams the whole weight matrix for every few rows of activations. This class repacks the
/// weights into <em>panels</em>: the outputs are cut into groups of two SIMD vectors, and within a
/// panel the weights are stored reduction-step major, so the kernel walks straight through memory.
/// For each reduction step it then broadcasts one activation scalar per token row and
/// multiply-accumulates the panel — eight FMAs against two contiguous vector loads and four
/// broadcasts, with unit-stride stores and no horizontal reduction anywhere.</para>
///
/// <para>Panel layout matters more than it looks: an earlier version stored the transpose plainly,
/// so the inner loop strode across the full output dimension (tens of kilobytes) on every step.
/// That defeats the hardware prefetcher and measured three times <em>slower</em> than the naive
/// dot-product kernel. The panels are what make the transposition pay.</para>
///
/// <para>On AVX-512 hardware the kernel uses 512-bit vectors explicitly. <c>Vector&lt;T&gt;</c>
/// stays 256-bit there by default — .NET does not widen it without
/// <c>PreferredVectorBitWidth</c> — so a portable-only kernel leaves half the machine idle, while
/// changing that switch would widen every other loop in the process as well. The panel width
/// follows whichever path is chosen, so the packing and the kernel always agree.</para>
/// </summary>
public sealed class PackedMatrix : IProjection
{
    private const int RowBlock = 4;

    private readonly float[] _data;          // [panel][inFeatures][panelWidth]
    private readonly int _panelWidth;
    private readonly int _panels;
    private readonly int _panelStride;

    public int InFeatures { get; }
    public int OutFeatures { get; }

    /// <summary>Repacks a PyTorch <c>[out, in]</c> weight.</summary>
    public PackedMatrix(ReadOnlySpan<float> rowMajor, int outFeatures, int inFeatures)
    {
        if (rowMajor.Length < (long)outFeatures * inFeatures)
        {
            throw new ArgumentException(
                $"expected {(long)outFeatures * inFeatures} weights for [{outFeatures}, {inFeatures}].",
                nameof(rowMajor));
        }

        OutFeatures = outFeatures;
        InFeatures = inFeatures;
        _panelWidth = SimdOps.UseVector512 ? 4 * 16 : 2 * Vector<float>.Count;
        _panels = (outFeatures + _panelWidth - 1) / _panelWidth;
        _panelStride = inFeatures * _panelWidth;
        // The last panel is zero-padded so the kernel can always load two whole vectors.
        _data = new float[(long)_panels * _panelStride];

        for (int panel = 0; panel < _panels; ++panel)
        {
            int first = panel * _panelWidth;
            int columns = Math.Min(_panelWidth, outFeatures - first);
            long destination = (long)panel * _panelStride;
            for (int i = 0; i < inFeatures; ++i)
            {
                for (int j = 0; j < columns; ++j)
                {
                    _data[destination + (long)i * _panelWidth + j] = rowMajor[(first + j) * inFeatures + i];
                }
            }
        }
    }

    /// <summary>Bytes this packed copy occupies, for load-time reporting.</summary>
    public long Bytes => (long)_data.Length * sizeof(float);

    /// <summary>
    /// <c>output = input · weightᵀ + bias</c> over <paramref name="rows"/> token rows of
    /// <see cref="InFeatures"/> values each. <paramref name="output"/> is row-major
    /// <c>[rows, OutFeatures]</c>.
    /// </summary>
    public unsafe void Multiply(ReadOnlySpan<float> input, int rows, ReadOnlySpan<float> bias, Span<float> output)
        => Multiply(input, rows, InFeatures, bias, output, OutFeatures);

    /// <summary>
    /// The same product over activations whose rows are strided — <paramref name="inputStride"/>
    /// and <paramref name="outputStride"/> are the distance between consecutive token rows, which
    /// may exceed the feature count.
    /// </summary>
    public unsafe void Multiply(ReadOnlySpan<float> input, int rows, int inputStride,
        ReadOnlySpan<float> bias, Span<float> output, int outputStride)
    {
        if (input.Length < (long)rows * inputStride) throw new ArgumentException("input is too small", nameof(input));
        if (output.Length < (long)rows * outputStride) throw new ArgumentException("output is too small", nameof(output));
        if (!bias.IsEmpty && bias.Length < OutFeatures) throw new ArgumentException("bias is too small", nameof(bias));

        fixed (float* weights = _data, inputPointer = input, biasPointer = bias, outputPointer = output)
        {
            float* bias0 = bias.IsEmpty ? null : biasPointer;

            // Panels are independent and each owns at least two whole vectors of the output, so
            // neighbouring workers share at most the cache line at a panel boundary.
            int workers = LayaRuntime.MaxDegreeOfParallelism;
            if (workers <= 1 || (long)rows * OutFeatures * InFeatures <= 1_000_000)
            {
                for (int panel = 0; panel < _panels; ++panel)
                {
                    Panel(weights, inputPointer, rows, inputStride, bias0, outputPointer, outputStride, panel);
                }
                return;
            }

            nint weightAddress = (nint)weights;
            nint inputAddress = (nint)inputPointer;
            nint biasAddress = (nint)bias0;
            nint outputAddress = (nint)outputPointer;

            int chunk = Math.Max(1, _panels / (workers * 4));
            int chunks = (_panels + chunk - 1) / chunk;
            Parallel.For(0, chunks, LayaRuntime.ParallelOptions, index =>
            {
                int first = index * chunk;
                int last = Math.Min(_panels, first + chunk);
                for (int panel = first; panel < last; ++panel)
                {
                    Panel((float*)weightAddress, (float*)inputAddress, rows, inputStride, (float*)biasAddress,
                        (float*)outputAddress, outputStride, panel);
                }
            });
        }
    }

    private unsafe void Panel(float* weights, float* input, int rows, int inputStride, float* bias,
        float* output, int outputStride, int panel)
    {
        int n0 = panel * _panelWidth;
        int columns = Math.Min(_panelWidth, OutFeatures - n0);

        // The ragged last panel is rare and small; keeping it out of the hot kernels means they
        // never carry a partial-store branch.
        if (columns != _panelWidth)
        {
            PanelRagged(weights, input, rows, inputStride, bias, output, outputStride, panel, columns);
            return;
        }

        if (SimdOps.UseVector512) Panel512(weights, input, rows, inputStride, bias, output, outputStride, panel);
        else PanelPortable(weights, input, rows, inputStride, bias, output, outputStride, panel);
    }

    /// <summary>
    /// The 512-bit kernel: 64 outputs and six token rows per pass.
    ///
    /// <para>The tile was swept, not reasoned about. Rows x vectors, GFLOP/s on the
    /// 404x1024x5248 projection: 6x2 = 84, 4x4 = 84, 5x4 = 92, <b>6x4 = 101</b>, 7x4 = 98,
    /// 3x4 = 67, 2x8 = 52, 3x8 = 67. Twenty-four accumulators, four weight vectors and one
    /// broadcast are 29 of the 32 zmm registers; every shape that needs more spills, and 512-bit
    /// register moves are not move-eliminated on this microarchitecture, so a spill costs a real
    /// uop on the same ports as the FMAs.</para>
    ///
    /// <para>One broadcast temp, reused. Giving each row its own made the live set 34 registers
    /// and the JIT emitted 28 <c>vmovaps</c> per reduction step — more uops than the FMAs they
    /// were feeding.</para>
    ///
    /// <para>What this does <em>not</em> fix, measured and rejected: padding the activation row
    /// stride to break L1 set aliasing (5%), blocking the rows so the activations fit L2 (worse —
    /// the weight panel then gets re-read per block), and wider panels to halve the activation
    /// traffic (much worse — register pressure). At 101 GFLOP/s this is 66% of the 152 GFLOP/s
    /// this core sustains on pure 512-bit FMA; the same instruction mix in isolation reaches
    /// ~130, and the rest is activation streaming from L3.</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void Panel512(float* weights, float* input, int rows, int inputStride, float* bias,
        float* output, int outputStride, int panel)
    {
        const int width = 16;
        const int panelWidth = 4 * width;
        const int rowBlock = 6;
        int features = InFeatures;
        int n0 = panel * panelWidth;
        float* panelBase = weights + (long)panel * _panelStride;

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
            float* w = panelBase;

            for (int i = 0; i < features; ++i, w += panelWidth)
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
            float* w = panelBase;
            for (int i = 0; i < features; ++i, w += panelWidth)
            {
                var x = Vector512.Create(a0[i]);
                t0 = Vector512.FusedMultiplyAdd(x, Vector512.Load(w + 0 * width), t0);
                t1 = Vector512.FusedMultiplyAdd(x, Vector512.Load(w + 1 * width), t1);
                t2 = Vector512.FusedMultiplyAdd(x, Vector512.Load(w + 2 * width), t2);
                t3 = Vector512.FusedMultiplyAdd(x, Vector512.Load(w + 3 * width), t3);
            }
            float* d = output + (long)row * outputStride + n0;
            t0.Store(d + 0 * width); t1.Store(d + 1 * width); t2.Store(d + 2 * width); t3.Store(d + 3 * width);
        }
    }

    /// <summary>The same kernel over the portable vector width, for machines without AVX-512.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void PanelPortable(float* weights, float* input, int rows, int inputStride, float* bias,
        float* output, int outputStride, int panel)
    {
        int width = Vector<float>.Count;
        const int rowBlock = 6;
        int panelWidth = _panelWidth;
        int features = InFeatures;
        int n0 = panel * panelWidth;
        float* panelBase = weights + (long)panel * _panelStride;

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
            float* w = panelBase;

            for (int i = 0; i < features; ++i, w += panelWidth)
            {
                var b0 = Vector.Load(w);
                var b1 = Vector.Load(w + width);
                var v0 = new Vector<float>(a0[i]);
                c00 = Vector.FusedMultiplyAdd(v0, b0, c00);
                c01 = Vector.FusedMultiplyAdd(v0, b1, c01);
                var v1 = new Vector<float>(a1[i]);
                c10 = Vector.FusedMultiplyAdd(v1, b0, c10);
                c11 = Vector.FusedMultiplyAdd(v1, b1, c11);
                var v2 = new Vector<float>(a2[i]);
                c20 = Vector.FusedMultiplyAdd(v2, b0, c20);
                c21 = Vector.FusedMultiplyAdd(v2, b1, c21);
                var v3 = new Vector<float>(a3[i]);
                c30 = Vector.FusedMultiplyAdd(v3, b0, c30);
                c31 = Vector.FusedMultiplyAdd(v3, b1, c31);
                var v4 = new Vector<float>(a4[i]);
                c40 = Vector.FusedMultiplyAdd(v4, b0, c40);
                c41 = Vector.FusedMultiplyAdd(v4, b1, c41);
                var v5 = new Vector<float>(a5[i]);
                c50 = Vector.FusedMultiplyAdd(v5, b0, c50);
                c51 = Vector.FusedMultiplyAdd(v5, b1, c51);
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
            float* w = panelBase;
            for (int i = 0; i < features; ++i, w += panelWidth)
            {
                var a = new Vector<float>(a0[i]);
                c0 = Vector.FusedMultiplyAdd(a, Vector.Load(w), c0);
                c1 = Vector.FusedMultiplyAdd(a, Vector.Load(w + width), c1);
            }
            float* d = output + (long)row * outputStride + n0;
            c0.Store(d);
            c1.Store(d + width);
        }
    }

    /// <summary>
    /// The last panel when the output dimension is not a whole number of panels. Computed the same
    /// way, into a full-width buffer, then trimmed — correctness over speed, since this is at most
    /// one panel out of dozens.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private unsafe void PanelRagged(float* weights, float* input, int rows, int inputStride, float* bias,
        float* output, int outputStride, int panel, int columns)
    {
        int panelWidth = _panelWidth;
        int features = InFeatures;
        int n0 = panel * panelWidth;
        float* panelBase = weights + (long)panel * _panelStride;
        Span<float> tile = stackalloc float[panelWidth];

        for (int row = 0; row < rows; ++row)
        {
            tile.Clear();
            float* a0 = input + (long)row * inputStride;
            float* w = panelBase;
            for (int i = 0; i < features; ++i, w += panelWidth)
            {
                float a = a0[i];
                for (int j = 0; j < panelWidth; ++j) tile[j] += a * w[j];
            }

            float* destination = output + (long)row * outputStride + n0;
            for (int j = 0; j < columns; ++j)
            {
                destination[j] = bias is null ? tile[j] : tile[j] + bias[n0 + j];
            }
        }
    }
}

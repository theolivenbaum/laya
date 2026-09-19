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
public sealed class PackedMatrix
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
        _panelWidth = 2 * SimdOps.PreferredLaneCount;
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
    {
        if (input.Length < (long)rows * InFeatures) throw new ArgumentException("input is too small", nameof(input));
        if (output.Length < (long)rows * OutFeatures) throw new ArgumentException("output is too small", nameof(output));
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
                    Panel(weights, inputPointer, rows, bias0, outputPointer, panel);
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
                    Panel((float*)weightAddress, (float*)inputAddress, rows, (float*)biasAddress,
                        (float*)outputAddress, panel);
                }
            });
        }
    }

    private unsafe void Panel(float* weights, float* input, int rows, float* bias, float* output, int panel)
    {
        int n0 = panel * _panelWidth;
        int columns = Math.Min(_panelWidth, OutFeatures - n0);

        // The ragged last panel is rare and small; keeping it out of the hot kernels means they
        // never carry a partial-store branch.
        if (columns != _panelWidth)
        {
            PanelRagged(weights, input, rows, bias, output, panel, columns);
            return;
        }

        if (SimdOps.UseVector512) Panel512(weights, input, rows, bias, output, panel);
        else PanelPortable(weights, input, rows, bias, output, panel);
    }

    /// <summary>
    /// The 512-bit kernel: 32 outputs and six token rows per pass.
    ///
    /// <para>The accumulators are seeded with the bias and stored straight to the destination.
    /// That is not a micro-optimization: an earlier version passed them by value to a small
    /// <c>Store</c> helper, the JIT declined to inline it, and the accumulators became address
    /// exposed — so every single FMA in the inner loop was followed by a 64-byte spill to the
    /// stack and a reload on the next iteration. It ran at a third of the speed. Anything that
    /// takes the address of an accumulator, directly or through a call, has to stay out of this
    /// loop.</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void Panel512(float* weights, float* input, int rows, float* bias, float* output, int panel)
    {
        const int width = 16;
        // The panel width is a compile-time constant on this path: _panelWidth is 2 * 16 whenever
        // the 512-bit kernel is selected, and letting the JIT know that turns the pointer bump at
        // the end of the inner loop from three instructions into a folded immediate.
        const int panelWidth = 2 * width;
        System.Diagnostics.Debug.Assert(panelWidth == _panelWidth, "panel width must match the kernel");
        // Six rows, measured: 4 rows gives 71 GFLOP/s on the 404x1024x5248 projection, 6 gives 87,
        // 8 gives 83 and 12 gives 80. Twelve accumulators plus the two weight vectors and one
        // broadcast fit the low sixteen zmm registers; wider blocks spill into zmm16-31 and lose
        // more than the extra reuse gains.
        const int rowBlock = 6;
        int outFeatures = OutFeatures;
        int features = InFeatures;
        int n0 = panel * panelWidth;
        float* panelBase = weights + (long)panel * _panelStride;

        var seed0 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0);
        var seed1 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0 + width);

        int row = 0;
        for (; row + rowBlock <= rows; row += rowBlock)
        {
            var c00 = seed0; var c01 = seed1;
            var c10 = seed0; var c11 = seed1;
            var c20 = seed0; var c21 = seed1;
            var c30 = seed0; var c31 = seed1;
            var c40 = seed0; var c41 = seed1;
            var c50 = seed0; var c51 = seed1;

            float* a0 = input + (long)(row + 0) * features;
            float* a1 = input + (long)(row + 1) * features;
            float* a2 = input + (long)(row + 2) * features;
            float* a3 = input + (long)(row + 3) * features;
            float* a4 = input + (long)(row + 4) * features;
            float* a5 = input + (long)(row + 5) * features;
            float* w = panelBase;

            for (int i = 0; i < features; ++i, w += panelWidth)
            {
                var b0 = Vector512.Load(w);
                var b1 = Vector512.Load(w + width);
                var v0 = Vector512.Create(a0[i]);
                c00 = Vector512.FusedMultiplyAdd(v0, b0, c00);
                c01 = Vector512.FusedMultiplyAdd(v0, b1, c01);
                var v1 = Vector512.Create(a1[i]);
                c10 = Vector512.FusedMultiplyAdd(v1, b0, c10);
                c11 = Vector512.FusedMultiplyAdd(v1, b1, c11);
                var v2 = Vector512.Create(a2[i]);
                c20 = Vector512.FusedMultiplyAdd(v2, b0, c20);
                c21 = Vector512.FusedMultiplyAdd(v2, b1, c21);
                var v3 = Vector512.Create(a3[i]);
                c30 = Vector512.FusedMultiplyAdd(v3, b0, c30);
                c31 = Vector512.FusedMultiplyAdd(v3, b1, c31);
                var v4 = Vector512.Create(a4[i]);
                c40 = Vector512.FusedMultiplyAdd(v4, b0, c40);
                c41 = Vector512.FusedMultiplyAdd(v4, b1, c41);
                var v5 = Vector512.Create(a5[i]);
                c50 = Vector512.FusedMultiplyAdd(v5, b0, c50);
                c51 = Vector512.FusedMultiplyAdd(v5, b1, c51);
            }

            float* d = output + (long)row * outFeatures + n0;
            c00.Store(d); c01.Store(d + width);
            d += outFeatures;
            c10.Store(d); c11.Store(d + width);
            d += outFeatures;
            c20.Store(d); c21.Store(d + width);
            d += outFeatures;
            c30.Store(d); c31.Store(d + width);
            d += outFeatures;
            c40.Store(d); c41.Store(d + width);
            d += outFeatures;
            c50.Store(d); c51.Store(d + width);
        }

        for (; row < rows; ++row)
        {
            var c0 = seed0;
            var c1 = seed1;
            float* a0 = input + (long)row * features;
            float* w = panelBase;
            for (int i = 0; i < features; ++i, w += panelWidth)
            {
                var a = Vector512.Create(a0[i]);
                c0 = Vector512.FusedMultiplyAdd(a, Vector512.Load(w), c0);
                c1 = Vector512.FusedMultiplyAdd(a, Vector512.Load(w + width), c1);
            }
            float* d = output + (long)row * outFeatures + n0;
            c0.Store(d);
            c1.Store(d + width);
        }
    }

    /// <summary>The same kernel over the portable vector width, for machines without AVX-512.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void PanelPortable(float* weights, float* input, int rows, float* bias, float* output, int panel)
    {
        int width = Vector<float>.Count;
        const int rowBlock = 6;
        int panelWidth = _panelWidth;
        int outFeatures = OutFeatures;
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

            float* a0 = input + (long)(row + 0) * features;
            float* a1 = input + (long)(row + 1) * features;
            float* a2 = input + (long)(row + 2) * features;
            float* a3 = input + (long)(row + 3) * features;
            float* a4 = input + (long)(row + 4) * features;
            float* a5 = input + (long)(row + 5) * features;
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

            float* d = output + (long)row * outFeatures + n0;
            c00.Store(d); c01.Store(d + width);
            d += outFeatures;
            c10.Store(d); c11.Store(d + width);
            d += outFeatures;
            c20.Store(d); c21.Store(d + width);
            d += outFeatures;
            c30.Store(d); c31.Store(d + width);
            d += outFeatures;
            c40.Store(d); c41.Store(d + width);
            d += outFeatures;
            c50.Store(d); c51.Store(d + width);
        }

        for (; row < rows; ++row)
        {
            var c0 = seed0;
            var c1 = seed1;
            float* a0 = input + (long)row * features;
            float* w = panelBase;
            for (int i = 0; i < features; ++i, w += panelWidth)
            {
                var a = new Vector<float>(a0[i]);
                c0 = Vector.FusedMultiplyAdd(a, Vector.Load(w), c0);
                c1 = Vector.FusedMultiplyAdd(a, Vector.Load(w + width), c1);
            }
            float* d = output + (long)row * outFeatures + n0;
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
    private unsafe void PanelRagged(float* weights, float* input, int rows, float* bias, float* output,
        int panel, int columns)
    {
        int panelWidth = _panelWidth;
        int features = InFeatures;
        int n0 = panel * panelWidth;
        float* panelBase = weights + (long)panel * _panelStride;
        Span<float> tile = stackalloc float[panelWidth];

        for (int row = 0; row < rows; ++row)
        {
            tile.Clear();
            float* a0 = input + (long)row * features;
            float* w = panelBase;
            for (int i = 0; i < features; ++i, w += panelWidth)
            {
                float a = a0[i];
                for (int j = 0; j < panelWidth; ++j) tile[j] += a * w[j];
            }

            float* destination = output + (long)row * OutFeatures + n0;
            for (int j = 0; j < columns; ++j)
            {
                destination[j] = bias is null ? tile[j] : tile[j] + bias[n0 + j];
            }
        }
    }
}

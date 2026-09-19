using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

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
        // The last panel is zero-padded so the kernel can always store two whole vectors.
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

        // The kernel runs on raw pointers: the inner loop is four broadcasts and two vector loads
        // per reduction step, and a bounds check on any of them costs more than the arithmetic.
        fixed (float* weights = _data, inputPointer = input, biasPointer = bias, outputPointer = output)
        {
            float* bias0 = bias.IsEmpty ? null : biasPointer;

            // Panels are independent and each owns at least two whole vectors of the output, so
            // neighbouring workers share at most the cache line at a panel boundary.
            int workers = Environment.ProcessorCount;
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
            Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = workers }, index =>
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void Panel(float* weights, float* input, int rows, float* bias, float* output, int panel)
    {
        if (SimdOps.UseVector512)
        {
            Panel512(weights, input, rows, bias, output, panel);
            return;
        }

        int width = Vector<float>.Count;
        int n0 = panel * _panelWidth;
        int columns = Math.Min(_panelWidth, OutFeatures - n0);
        float* panelBase = weights + (long)panel * _panelStride;
        int features = InFeatures;

        int row = 0;
        for (; row + RowBlock <= rows; row += RowBlock)
        {
            Vector<float> c00 = default, c01 = default, c10 = default, c11 = default;
            Vector<float> c20 = default, c21 = default, c30 = default, c31 = default;

            float* a0 = input + (long)row * features;
            float* a1 = a0 + features;
            float* a2 = a1 + features;
            float* a3 = a2 + features;
            float* w = panelBase;

            for (int i = 0; i < features; ++i, w += _panelWidth)
            {
                var b0 = Vector.Load(w);
                var b1 = Vector.Load(w + width);

                var a = new Vector<float>(a0[i]);
                c00 = Vector.FusedMultiplyAdd(a, b0, c00);
                c01 = Vector.FusedMultiplyAdd(a, b1, c01);
                a = new Vector<float>(a1[i]);
                c10 = Vector.FusedMultiplyAdd(a, b0, c10);
                c11 = Vector.FusedMultiplyAdd(a, b1, c11);
                a = new Vector<float>(a2[i]);
                c20 = Vector.FusedMultiplyAdd(a, b0, c20);
                c21 = Vector.FusedMultiplyAdd(a, b1, c21);
                a = new Vector<float>(a3[i]);
                c30 = Vector.FusedMultiplyAdd(a, b0, c30);
                c31 = Vector.FusedMultiplyAdd(a, b1, c31);
            }

            Store(output, bias, row, n0, columns, c00, c01);
            Store(output, bias, row + 1, n0, columns, c10, c11);
            Store(output, bias, row + 2, n0, columns, c20, c21);
            Store(output, bias, row + 3, n0, columns, c30, c31);
        }

        // Ragged tail: one row at a time, same kernel shape with a single accumulator pair.
        for (; row < rows; ++row)
        {
            Vector<float> c0 = default, c1 = default;
            float* a0 = input + (long)row * features;
            float* w = panelBase;
            for (int i = 0; i < features; ++i, w += _panelWidth)
            {
                var a = new Vector<float>(a0[i]);
                c0 = Vector.FusedMultiplyAdd(a, Vector.Load(w), c0);
                c1 = Vector.FusedMultiplyAdd(a, Vector.Load(w + width), c1);
            }
            Store(output, bias, row, n0, columns, c0, c1);
        }
    }

    /// <summary>
    /// The same kernel over explicit 512-bit vectors: 32 outputs and four token rows per pass.
    ///
    /// <para>Four rows, not eight: AVX-512 has the registers for sixteen accumulators, but holding
    /// eight row pointers alongside them measured about 10% slower here, so the weight panel is
    /// re-read more often and the register file stays comfortable.</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void Panel512(float* weights, float* input, int rows, float* bias, float* output, int panel)
    {
        const int width = 16;
        int n0 = panel * _panelWidth;
        int columns = Math.Min(_panelWidth, OutFeatures - n0);
        float* panelBase = weights + (long)panel * _panelStride;
        int features = InFeatures;

        int row = 0;
        for (; row + RowBlock <= rows; row += RowBlock)
        {
            Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default;
            Vector512<float> c20 = default, c21 = default, c30 = default, c31 = default;

            float* a0 = input + (long)row * features;
            float* a1 = a0 + features;
            float* a2 = a1 + features;
            float* a3 = a2 + features;
            float* w = panelBase;

            for (int i = 0; i < features; ++i, w += _panelWidth)
            {
                var b0 = Vector512.Load(w);
                var b1 = Vector512.Load(w + width);

                var a = Vector512.Create(a0[i]);
                c00 = Vector512.FusedMultiplyAdd(a, b0, c00);
                c01 = Vector512.FusedMultiplyAdd(a, b1, c01);
                a = Vector512.Create(a1[i]);
                c10 = Vector512.FusedMultiplyAdd(a, b0, c10);
                c11 = Vector512.FusedMultiplyAdd(a, b1, c11);
                a = Vector512.Create(a2[i]);
                c20 = Vector512.FusedMultiplyAdd(a, b0, c20);
                c21 = Vector512.FusedMultiplyAdd(a, b1, c21);
                a = Vector512.Create(a3[i]);
                c30 = Vector512.FusedMultiplyAdd(a, b0, c30);
                c31 = Vector512.FusedMultiplyAdd(a, b1, c31);
            }

            Store512(output, bias, row, n0, columns, c00, c01);
            Store512(output, bias, row + 1, n0, columns, c10, c11);
            Store512(output, bias, row + 2, n0, columns, c20, c21);
            Store512(output, bias, row + 3, n0, columns, c30, c31);
        }

        // Ragged tail: one row at a time, same kernel shape with a single accumulator pair.
        for (; row < rows; ++row)
        {
            Vector512<float> c0 = default, c1 = default;
            float* a0 = input + (long)row * features;
            float* w = panelBase;
            for (int i = 0; i < features; ++i, w += _panelWidth)
            {
                var broadcast = Vector512.Create(a0[i]);
                c0 = Vector512.FusedMultiplyAdd(broadcast, Vector512.Load(w), c0);
                c1 = Vector512.FusedMultiplyAdd(broadcast, Vector512.Load(w + width), c1);
            }
            Store512(output, bias, row, n0, columns, c0, c1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private unsafe void Store512(float* output, float* bias, int row, int n0, int columns,
        Vector512<float> low, Vector512<float> high)
    {
        const int width = 16;
        float* destination = output + (long)row * OutFeatures + n0;
        if (columns == 2 * width && bias is null)
        {
            low.Store(destination);
            high.Store(destination + width);
            return;
        }

        Span<float> tile = stackalloc float[2 * width];
        low.StoreUnsafe(ref tile[0]);
        high.StoreUnsafe(ref tile[width]);
        for (int i = 0; i < columns; ++i) destination[i] = bias is null ? tile[i] : tile[i] + bias[n0 + i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private unsafe void Store(float* output, float* bias, int row, int n0, int columns,
        Vector<float> low, Vector<float> high)
    {
        int width = Vector<float>.Count;
        float* destination = output + (long)row * OutFeatures + n0;

        if (columns == 2 * width && bias is null)
        {
            low.Store(destination);
            high.Store(destination + width);
            return;
        }

        Span<float> tile = stackalloc float[2 * Vector<float>.Count];
        low.StoreUnsafe(ref tile[0]);
        high.StoreUnsafe(ref tile[width]);
        if (bias is null)
        {
            for (int i = 0; i < columns; ++i) destination[i] = tile[i];
            return;
        }
        for (int i = 0; i < columns; ++i) destination[i] = tile[i] + bias[n0 + i];
    }
}

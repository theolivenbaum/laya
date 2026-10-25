using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace StructuredDecisions.Numerics;

/// <summary>
/// ggml's Q8_0: blocks of 32 values sharing one scale <c>d</c> (stored fp16), each value an int8 in −127…127.
/// The two quantizers are ggml's own, rounding included, because they differ: weights are converted by
/// <c>quantize_row_q8_0_ref</c> (<c>id = 1/d</c>, round half away from zero — what <c>convert_hf_to_gguf</c> and
/// <c>llama-quantize</c> write), activations by the x86 <c>quantize_row_q8_0</c> every CPU matmul runs on its
/// input (<c>id = 127/amax</c>, <c>_mm256_round_ps</c> to nearest even).
/// </summary>
public static class Q8_0
{
    public const int BlockSize = 32;

    /// <summary>One block of a weight row, as ggml's reference quantizer writes it; returns <c>d</c> as stored (fp16).</summary>
    public static float QuantizeWeightBlock(ReadOnlySpan<float> x, Span<sbyte> q)
    {
        float amax = 0f;
        for (int j = 0; j < BlockSize; ++j) amax = MathF.Max(amax, MathF.Abs(x[j]));
        float d = amax / 127f;
        float id = d != 0f ? 1f / d : 0f;
        for (int j = 0; j < BlockSize; ++j) q[j] = (sbyte)MathF.Round(x[j] * id, MidpointRounding.AwayFromZero);
        return (float)(Half)d;
    }

    /// <summary>
    /// <paramref name="rows"/> activation rows of <paramref name="features"/> values (a multiple of 32), as ggml's
    /// x86 quantizer does it. Stored offset by +128 as unsigned bytes, the operand form <c>vpdpbusd</c> wants;
    /// <paramref name="scales"/> gets each block's <c>d</c> as stored (fp16).
    /// </summary>
    public static void QuantizeActivations(ReadOnlySpan<float> input, int rows, int inputStride, int features,
        Span<byte> quantized, Span<float> scales)
    {
        int blocks = features / BlockSize;
        for (int r = 0; r < rows; ++r)
        {
            var source = input.Slice(r * inputStride, features);
            var target = quantized.Slice(r * features, features);
            for (int b = 0; b < blocks; ++b)
            {
                scales[r * blocks + b] = QuantizeActivationBlock(source.Slice(b * BlockSize, BlockSize), target.Slice(b * BlockSize, BlockSize));
            }
        }
    }

    private static float QuantizeActivationBlock(ReadOnlySpan<float> x, Span<byte> q)
    {
        var v0 = Vector256.Create(x);
        var v1 = Vector256.Create(x[8..]);
        var v2 = Vector256.Create(x[16..]);
        var v3 = Vector256.Create(x[24..]);
        var m = Vector256.Max(Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1)), Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3)));
        float amax = 0f;
        for (int j = 0; j < 8; ++j) amax = MathF.Max(amax, m[j]);
        float d = amax / 127f;
        float id = amax != 0f ? 127f / amax : 0f;
        var scale = Vector256.Create(id);
        var offset = Vector256.Create(128);
        var i0 = Vector256.ConvertToInt32(Vector256.Round(v0 * scale)) + offset;
        var i1 = Vector256.ConvertToInt32(Vector256.Round(v1 * scale)) + offset;
        var i2 = Vector256.ConvertToInt32(Vector256.Round(v2 * scale)) + offset;
        var i3 = Vector256.ConvertToInt32(Vector256.Round(v3 * scale)) + offset;
        var s01 = Vector256.Narrow(i0, i1);          // in order: Narrow is element-wise across the whole vector
        var s23 = Vector256.Narrow(i2, i3);
        Vector256.Narrow(s01.AsUInt16(), s23.AsUInt16()).CopyTo(q);
        return (float)(Half)d;
    }
}

/// <summary>How a model keeps its projection weights.</summary>
public enum WeightPrecision
{
    /// <summary>As the checkpoint stores them (bf16), computed in fp32: exact.</summary>
    Bf16,
    /// <summary>ggml Q8_0 weights and activations (<see cref="PackedMatrixQ8_0"/>): ~60% of the memory, faster, not exact.</summary>
    Q8_0,
}

/// <summary>Which integer dot-product kernel a <see cref="PackedMatrixQ8_0"/> runs; every kernel gives the same bits.</summary>
public enum Q8Kernel
{
    /// <summary>The fastest this process has: <see cref="Vnni512"/>, else <see cref="Vnni256"/>, else <see cref="Portable"/>.</summary>
    Best,
    Portable,
    /// <summary><c>vpdpbusd ymm</c> (AVX-VNNI), all .NET 10 exposes.</summary>
    Vnni256,
    /// <summary><c>vpdpbusd zmm</c> (<c>AvxVnni.V512</c>), .NET 11 only.</summary>
    Vnni512,
}

/// <summary>
/// A projection weight in ggml's <b>Q8_0</b>, multiplied the way ggml's CPU backend multiplies it: every input row is
/// quantized to Q8_0 too, each 32-value block's dot product is taken in exact integers, then scaled by
/// <c>d_w · d_x</c> and summed in fp32. 1.19 bytes per weight against bf16's 2 — and not exact: what it costs Mica is
/// measured in <c>tools/measure_mica_quant.py</c> (<c>q8_0-wa</c>).
///
/// <para>Layout, per 64-column panel: the int8 values as <c>[K/4][64 columns][4]</c> — one 4-byte group per column
/// per <c>vpdpbusd</c> step, so a zmm holds 16 columns and a ymm 8 — then per block the 64 columns' fp32 scales and
/// their negated value sums. <c>vpdpbusd</c> multiplies unsigned by signed bytes, so activations are stored +128 and
/// each block's integer accumulator starts at <c>−128 · Σw</c>, which makes the sum exactly <c>Σ x·w</c>.</para>
///
/// <para>The register tile is 3 rows × 64 columns: 12 integer and 12 float accumulators, both live through a block.</para>
/// </summary>
public sealed class PackedMatrixQ8_0 : ILinear
{
    private const int PanelWidth = 64;
    private const int RowTile = 3;
    private readonly sbyte[] _values;        // [panel][K/4][64][4]
    private readonly float[] _scales;        // [panel][block][64]
    private readonly short[] _negatedSums;   // [panel][block][64]: −Σ_block w
    private readonly int _panels, _blocks;

    public int InFeatures { get; }
    public int OutFeatures { get; }
    public long Bytes => _values.LongLength + _scales.LongLength * sizeof(float) + _negatedSums.LongLength * sizeof(short);

    /// <summary>The kernel <see cref="Multiply(ReadOnlySpan{float}, int, int, ReadOnlySpan{float}, Span{float}, int, ParallelOptions?)"/> runs; for tests and benchmarks.</summary>
    public Q8Kernel Kernel { get; set; } = Q8Kernel.Best;

    /// <summary>The kernel <see cref="Q8Kernel.Best"/> resolves to in this process.</summary>
    public static Q8Kernel BestKernel { get; } = ResolveBest();

    private static Q8Kernel ResolveBest()
    {
#if NET11_0_OR_GREATER
        if (SimdOps.UseVector512 && AvxVnni.V512.IsSupported) return Q8Kernel.Vnni512;
#endif
        return AvxVnni.IsSupported ? Q8Kernel.Vnni256 : Q8Kernel.Portable;
    }

    public static bool IsSupported(Q8Kernel kernel) => kernel switch
    {
        Q8Kernel.Best or Q8Kernel.Portable => true,
        Q8Kernel.Vnni256 => AvxVnni.IsSupported,
#if NET11_0_OR_GREATER
        Q8Kernel.Vnni512 => AvxVnni.V512.IsSupported,
#endif
        _ => false,
    };

    /// <summary>Quantizes a PyTorch <c>[out, in]</c> weight given as raw bf16 bit patterns.</summary>
    public PackedMatrixQ8_0(ReadOnlySpan<ushort> rowMajorBf16, int outFeatures, int inFeatures)
        : this(outFeatures, inFeatures)
    {
        if (rowMajorBf16.Length < (long)outFeatures * inFeatures) throw new ArgumentException("too few weights", nameof(rowMajorBf16));
        unsafe
        {
            fixed (ushort* bits = rowMajorBf16)
            {
                // Columns are quantized independently and land in disjoint bytes, so they can go in parallel.
                nint address = (nint)bits;
                Parallel.For(0, outFeatures, () => new float[inFeatures], (c, _, row) =>
                {
                    var source = new ReadOnlySpan<ushort>((ushort*)address + (long)c * inFeatures, inFeatures);
                    for (int i = 0; i < inFeatures; ++i) row[i] = BitConverter.UInt32BitsToSingle((uint)source[i] << 16);
                    Pack(c, row);
                    return row;
                }, _ => { });
            }
        }
    }

    /// <summary>Quantizes a PyTorch <c>[out, in]</c> fp32 weight.</summary>
    public PackedMatrixQ8_0(ReadOnlySpan<float> rowMajor, int outFeatures, int inFeatures)
        : this(outFeatures, inFeatures)
    {
        if (rowMajor.Length < (long)outFeatures * inFeatures) throw new ArgumentException("too few weights", nameof(rowMajor));
        for (int c = 0; c < outFeatures; ++c) Pack(c, rowMajor.Slice(c * inFeatures, inFeatures));
    }

    private PackedMatrixQ8_0(int outFeatures, int inFeatures)
    {
        if (inFeatures % Q8_0.BlockSize != 0) throw new ArgumentException($"Q8_0 needs a multiple of {Q8_0.BlockSize} inputs, got {inFeatures}.");
        OutFeatures = outFeatures;
        InFeatures = inFeatures;
        _panels = (outFeatures + PanelWidth - 1) / PanelWidth;
        _blocks = inFeatures / Q8_0.BlockSize;
        _values = new sbyte[(long)_panels * inFeatures * PanelWidth];
        _scales = new float[_panels * _blocks * PanelWidth];
        _negatedSums = new short[_panels * _blocks * PanelWidth];
    }

    private void Pack(int column, ReadOnlySpan<float> row)
    {
        int panel = column / PanelWidth, j = column % PanelWidth;
        Span<sbyte> q = stackalloc sbyte[Q8_0.BlockSize];
        long panelBase = (long)panel * InFeatures * PanelWidth;
        for (int b = 0; b < _blocks; ++b)
        {
            float d = Q8_0.QuantizeWeightBlock(row.Slice(b * Q8_0.BlockSize, Q8_0.BlockSize), q);
            int sum = 0;
            for (int t = 0; t < Q8_0.BlockSize; ++t)
            {
                int k = b * Q8_0.BlockSize + t;
                _values[panelBase + (long)(k / 4) * (4 * PanelWidth) + j * 4 + k % 4] = q[t];
                sum += q[t];
            }
            int at = (panel * _blocks + b) * PanelWidth + j;
            _scales[at] = d;
            _negatedSums[at] = (short)-sum;
        }
    }

    /// <summary>Column <paramref name="column"/>'s weights, dequantized (<c>q · d</c>, as ggml's <c>dequantize_row_q8_0</c>).</summary>
    public void Dequantize(int column, Span<float> destination)
    {
        int panel = column / PanelWidth, j = column % PanelWidth;
        long panelBase = (long)panel * InFeatures * PanelWidth;
        for (int k = 0; k < InFeatures; ++k)
        {
            float d = _scales[(panel * _blocks + k / Q8_0.BlockSize) * PanelWidth + j];
            destination[k] = _values[panelBase + (long)(k / 4) * (4 * PanelWidth) + j * 4 + k % 4] * d;
        }
    }

    public void Multiply(ReadOnlySpan<float> input, int rows, ReadOnlySpan<float> bias, Span<float> output,
        ParallelOptions? parallel = null)
        => Multiply(input, rows, InFeatures, bias, output, OutFeatures, parallel);

    public unsafe void Multiply(ReadOnlySpan<float> input, int rows, int inputStride,
        ReadOnlySpan<float> bias, Span<float> output, int outputStride, ParallelOptions? parallel = null)
    {
        if (input.Length < (long)(rows - 1) * inputStride + InFeatures) throw new ArgumentException("input is too small", nameof(input));
        if (output.Length < (long)(rows - 1) * outputStride + OutFeatures) throw new ArgumentException("output is too small", nameof(output));
        if (!bias.IsEmpty && bias.Length < OutFeatures) throw new ArgumentException("bias is too small", nameof(bias));
        if (rows == 0) return;
        var kernel = Kernel == Q8Kernel.Best ? BestKernel : Kernel;
        if (!IsSupported(kernel)) throw new PlatformNotSupportedException($"{kernel} is not supported in this process.");

        using var scratch = new ScratchBuffers();
        var quantized = ArrayPool<byte>.Shared.Rent(rows * InFeatures);
        try
        {
            var activationScales = scratch.Rent(rows * _blocks);
            Q8_0.QuantizeActivations(input, rows, inputStride, InFeatures, quantized, activationScales);
            Run(kernel, quantized, activationScales, rows, bias, output, outputStride, parallel);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(quantized);
        }
    }

    private unsafe void Run(Q8Kernel kernel, byte[] quantized, float[] activationScales, int rows, ReadOnlySpan<float> bias,
        Span<float> output, int outputStride, ParallelOptions? parallel)
    {

        fixed (sbyte* values = _values)
        fixed (float* scales = _scales)
        fixed (short* sums = _negatedSums)
        fixed (byte* x = quantized)
        fixed (float* dx = activationScales, biasPointer = bias, outputPointer = output)
        {
            var matrix = new Pointers(values, scales, sums, x, dx, bias.IsEmpty ? null : biasPointer, outputPointer, outputStride);
            int workers = DecisionRuntime.WorkersOf(parallel);
            if (workers <= 1 || (long)rows * OutFeatures * InFeatures <= 1_000_000)
            {
                for (int panel = 0; panel < _panels; ++panel) Panel(kernel, matrix, 0, rows, panel);
                return;
            }
            // As PackedMatrixBf16: few panels per worker are split by rows too, in whole tiles. Rows are independent.
            int rowGroups = 1;
            while (_panels * rowGroups < workers * 8 && rows / (rowGroups * 2) >= 8 * RowTile) rowGroups *= 2;
            int groupRows = ((rows + rowGroups - 1) / rowGroups + RowTile - 1) / RowTile * RowTile;
            int items = _panels * rowGroups;
            int chunk = Math.Max(1, items / (workers * 4));
            int chunks = (items + chunk - 1) / chunk;
            int rowCount = rows;
            Parallel.For(0, chunks, DecisionRuntime.Resolve(parallel), index =>
            {
                int last = Math.Min(items, (index + 1) * chunk);
                for (int item = index * chunk; item < last; ++item)
                {
                    int panel = item / rowGroups, group = item % rowGroups;
                    int r0 = group * groupRows, r1 = Math.Min(rowCount, r0 + groupRows);
                    if (r0 < r1) Panel(kernel, matrix, r0, r1, panel);
                }
            });
        }
    }

    private readonly unsafe struct Pointers(sbyte* values, float* scales, short* sums, byte* x, float* dx, float* bias,
        float* output, int outputStride)
    {
        public readonly nint Values = (nint)values, Scales = (nint)scales, Sums = (nint)sums, X = (nint)x, Dx = (nint)dx;
        public readonly nint Bias = (nint)bias, Output = (nint)output;
        public readonly int OutputStride = outputStride;
    }

    private unsafe void Panel(Q8Kernel kernel, in Pointers m, int r0, int r1, int panel)
    {
        int columns = Math.Min(PanelWidth, OutFeatures - panel * PanelWidth);
        if (columns != PanelWidth || kernel == Q8Kernel.Portable)
        {
            PanelPortable(m, r0, r1, panel, columns);
            return;
        }
        int row = r0;
#if NET11_0_OR_GREATER
        if (kernel == Q8Kernel.Vnni512)
        {
            for (; row + 3 <= r1; row += 3) Panel512<Three>(m, row, panel);
            if (r1 - row == 2) Panel512<Two>(m, row, panel);
            else if (r1 - row == 1) Panel512<One>(m, row, panel);
            return;
        }
#endif
        for (int half = 0; half < 2; ++half)
        {
            int r = row;
            for (; r + 3 <= r1; r += 3) Panel256<Three>(m, r, panel, half);
            if (r1 - r == 2) Panel256<Two>(m, r, panel, half);
            else if (r1 - r == 1) Panel256<One>(m, r, panel, half);
        }
    }

    private interface IRows { static abstract int Count { get; } }
    private struct One : IRows { public static int Count => 1; }
    private struct Two : IRows { public static int Count => 2; }
    private struct Three : IRows { public static int Count => 3; }

#if NET11_0_OR_GREATER
    /// <summary>Up to three rows by the panel's 64 columns: four zmm of 16 int32 lanes per row.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void Panel512<TR>(in Pointers m, int row, int panel) where TR : struct, IRows
    {
        int rows = TR.Count, features = InFeatures, blocks = _blocks;
        sbyte* w = (sbyte*)m.Values + (long)panel * features * PanelWidth;
        float* scale = (float*)m.Scales + panel * blocks * PanelWidth;
        short* sums = (short*)m.Sums + panel * blocks * PanelWidth;
        byte* a0 = (byte*)m.X + (long)row * features;
        byte* a1 = a0 + (rows > 1 ? features : 0);
        byte* a2 = a0 + (rows > 2 ? 2 * features : 0);
        float* d0 = (float*)m.Dx + (long)row * blocks;
        float* d1 = d0 + (rows > 1 ? blocks : 0);
        float* d2 = d0 + (rows > 2 ? 2 * blocks : 0);
        int n0 = panel * PanelWidth;
        float* bias = (float*)m.Bias;

        var seed0 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0);
        var seed1 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0 + 16);
        var seed2 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0 + 32);
        var seed3 = bias is null ? Vector512<float>.Zero : Vector512.Load(bias + n0 + 48);
        Vector512<float> f00 = seed0, f01 = seed1, f02 = seed2, f03 = seed3;
        Vector512<float> f10 = seed0, f11 = seed1, f12 = seed2, f13 = seed3;
        Vector512<float> f20 = seed0, f21 = seed1, f22 = seed2, f23 = seed3;

        for (int b = 0; b < blocks; ++b)
        {
            short* s = sums + b * PanelWidth;
            var z0 = Avx512F.ConvertToVector512Int32(Vector256.Load(s)) << 7;
            var z1 = Avx512F.ConvertToVector512Int32(Vector256.Load(s + 16)) << 7;
            var z2 = Avx512F.ConvertToVector512Int32(Vector256.Load(s + 32)) << 7;
            var z3 = Avx512F.ConvertToVector512Int32(Vector256.Load(s + 48)) << 7;
            Vector512<int> i00 = z0, i01 = z1, i02 = z2, i03 = z3;
            Vector512<int> i10 = z0, i11 = z1, i12 = z2, i13 = z3;
            Vector512<int> i20 = z0, i21 = z1, i22 = z2, i23 = z3;
            sbyte* wb = w + (long)b * Q8_0.BlockSize * PanelWidth;
            int k = b * Q8_0.BlockSize;
            for (int g = 0; g < Q8_0.BlockSize / 4; ++g, wb += 4 * PanelWidth, k += 4)
            {
                var w0 = Vector512.Load(wb);
                var w1 = Vector512.Load(wb + 64);
                var w2 = Vector512.Load(wb + 128);
                var w3 = Vector512.Load(wb + 192);
                var x = Vector512.Create(*(int*)(a0 + k)).AsByte();
                i00 = AvxVnni.V512.MultiplyWideningAndAdd(i00, x, w0);
                i01 = AvxVnni.V512.MultiplyWideningAndAdd(i01, x, w1);
                i02 = AvxVnni.V512.MultiplyWideningAndAdd(i02, x, w2);
                i03 = AvxVnni.V512.MultiplyWideningAndAdd(i03, x, w3);
                if (rows > 1)
                {
                    x = Vector512.Create(*(int*)(a1 + k)).AsByte();
                    i10 = AvxVnni.V512.MultiplyWideningAndAdd(i10, x, w0);
                    i11 = AvxVnni.V512.MultiplyWideningAndAdd(i11, x, w1);
                    i12 = AvxVnni.V512.MultiplyWideningAndAdd(i12, x, w2);
                    i13 = AvxVnni.V512.MultiplyWideningAndAdd(i13, x, w3);
                }
                if (rows > 2)
                {
                    x = Vector512.Create(*(int*)(a2 + k)).AsByte();
                    i20 = AvxVnni.V512.MultiplyWideningAndAdd(i20, x, w0);
                    i21 = AvxVnni.V512.MultiplyWideningAndAdd(i21, x, w1);
                    i22 = AvxVnni.V512.MultiplyWideningAndAdd(i22, x, w2);
                    i23 = AvxVnni.V512.MultiplyWideningAndAdd(i23, x, w3);
                }
            }
            float* sc = scale + b * PanelWidth;
            var dw0 = Vector512.Load(sc);
            var dw1 = Vector512.Load(sc + 16);
            var dw2 = Vector512.Load(sc + 32);
            var dw3 = Vector512.Load(sc + 48);
            var dx = Vector512.Create(d0[b]);
            f00 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i00), dw0 * dx, f00);
            f01 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i01), dw1 * dx, f01);
            f02 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i02), dw2 * dx, f02);
            f03 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i03), dw3 * dx, f03);
            if (rows > 1)
            {
                dx = Vector512.Create(d1[b]);
                f10 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i10), dw0 * dx, f10);
                f11 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i11), dw1 * dx, f11);
                f12 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i12), dw2 * dx, f12);
                f13 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i13), dw3 * dx, f13);
            }
            if (rows > 2)
            {
                dx = Vector512.Create(d2[b]);
                f20 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i20), dw0 * dx, f20);
                f21 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i21), dw1 * dx, f21);
                f22 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i22), dw2 * dx, f22);
                f23 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(i23), dw3 * dx, f23);
            }
        }

        float* o = (float*)m.Output + (long)row * m.OutputStride + n0;
        f00.Store(o); f01.Store(o + 16); f02.Store(o + 32); f03.Store(o + 48);
        if (rows > 1) { o += m.OutputStride; f10.Store(o); f11.Store(o + 16); f12.Store(o + 32); f13.Store(o + 48); }
        if (rows > 2) { o += m.OutputStride; f20.Store(o); f21.Store(o + 16); f22.Store(o + 32); f23.Store(o + 48); }
    }
#endif

    /// <summary>Up to three rows by half a panel (32 columns): four ymm of 8 int32 lanes per row.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void Panel256<TR>(in Pointers m, int row, int panel, int half) where TR : struct, IRows
    {
        int rows = TR.Count, features = InFeatures, blocks = _blocks;
        int c0 = half * 32;
        sbyte* w = (sbyte*)m.Values + (long)panel * features * PanelWidth + c0 * 4;
        float* scale = (float*)m.Scales + panel * blocks * PanelWidth + c0;
        short* sums = (short*)m.Sums + panel * blocks * PanelWidth + c0;
        byte* a0 = (byte*)m.X + (long)row * features;
        byte* a1 = a0 + (rows > 1 ? features : 0);
        byte* a2 = a0 + (rows > 2 ? 2 * features : 0);
        float* d0 = (float*)m.Dx + (long)row * blocks;
        float* d1 = d0 + (rows > 1 ? blocks : 0);
        float* d2 = d0 + (rows > 2 ? 2 * blocks : 0);
        int n0 = panel * PanelWidth + c0;
        float* bias = (float*)m.Bias;

        var seed0 = bias is null ? Vector256<float>.Zero : Vector256.Load(bias + n0);
        var seed1 = bias is null ? Vector256<float>.Zero : Vector256.Load(bias + n0 + 8);
        var seed2 = bias is null ? Vector256<float>.Zero : Vector256.Load(bias + n0 + 16);
        var seed3 = bias is null ? Vector256<float>.Zero : Vector256.Load(bias + n0 + 24);
        Vector256<float> f00 = seed0, f01 = seed1, f02 = seed2, f03 = seed3;
        Vector256<float> f10 = seed0, f11 = seed1, f12 = seed2, f13 = seed3;
        Vector256<float> f20 = seed0, f21 = seed1, f22 = seed2, f23 = seed3;

        for (int b = 0; b < blocks; ++b)
        {
            short* s = sums + b * PanelWidth;
            var z0 = Avx2.ConvertToVector256Int32(Vector128.Load(s)) << 7;
            var z1 = Avx2.ConvertToVector256Int32(Vector128.Load(s + 8)) << 7;
            var z2 = Avx2.ConvertToVector256Int32(Vector128.Load(s + 16)) << 7;
            var z3 = Avx2.ConvertToVector256Int32(Vector128.Load(s + 24)) << 7;
            Vector256<int> i00 = z0, i01 = z1, i02 = z2, i03 = z3;
            Vector256<int> i10 = z0, i11 = z1, i12 = z2, i13 = z3;
            Vector256<int> i20 = z0, i21 = z1, i22 = z2, i23 = z3;
            sbyte* wb = w + (long)b * Q8_0.BlockSize * PanelWidth;
            int k = b * Q8_0.BlockSize;
            for (int g = 0; g < Q8_0.BlockSize / 4; ++g, wb += 4 * PanelWidth, k += 4)
            {
                var w0 = Vector256.Load(wb);
                var w1 = Vector256.Load(wb + 32);
                var w2 = Vector256.Load(wb + 64);
                var w3 = Vector256.Load(wb + 96);
                var x = Vector256.Create(*(int*)(a0 + k)).AsByte();
                i00 = AvxVnni.MultiplyWideningAndAdd(i00, x, w0);
                i01 = AvxVnni.MultiplyWideningAndAdd(i01, x, w1);
                i02 = AvxVnni.MultiplyWideningAndAdd(i02, x, w2);
                i03 = AvxVnni.MultiplyWideningAndAdd(i03, x, w3);
                if (rows > 1)
                {
                    x = Vector256.Create(*(int*)(a1 + k)).AsByte();
                    i10 = AvxVnni.MultiplyWideningAndAdd(i10, x, w0);
                    i11 = AvxVnni.MultiplyWideningAndAdd(i11, x, w1);
                    i12 = AvxVnni.MultiplyWideningAndAdd(i12, x, w2);
                    i13 = AvxVnni.MultiplyWideningAndAdd(i13, x, w3);
                }
                if (rows > 2)
                {
                    x = Vector256.Create(*(int*)(a2 + k)).AsByte();
                    i20 = AvxVnni.MultiplyWideningAndAdd(i20, x, w0);
                    i21 = AvxVnni.MultiplyWideningAndAdd(i21, x, w1);
                    i22 = AvxVnni.MultiplyWideningAndAdd(i22, x, w2);
                    i23 = AvxVnni.MultiplyWideningAndAdd(i23, x, w3);
                }
            }
            float* sc = scale + b * PanelWidth;
            var dw0 = Vector256.Load(sc);
            var dw1 = Vector256.Load(sc + 8);
            var dw2 = Vector256.Load(sc + 16);
            var dw3 = Vector256.Load(sc + 24);
            var dx = Vector256.Create(d0[b]);
            f00 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i00), dw0 * dx, f00);
            f01 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i01), dw1 * dx, f01);
            f02 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i02), dw2 * dx, f02);
            f03 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i03), dw3 * dx, f03);
            if (rows > 1)
            {
                dx = Vector256.Create(d1[b]);
                f10 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i10), dw0 * dx, f10);
                f11 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i11), dw1 * dx, f11);
                f12 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i12), dw2 * dx, f12);
                f13 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i13), dw3 * dx, f13);
            }
            if (rows > 2)
            {
                dx = Vector256.Create(d2[b]);
                f20 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i20), dw0 * dx, f20);
                f21 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i21), dw1 * dx, f21);
                f22 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i22), dw2 * dx, f22);
                f23 = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(i23), dw3 * dx, f23);
            }
        }

        float* o = (float*)m.Output + (long)row * m.OutputStride + n0;
        f00.Store(o); f01.Store(o + 8); f02.Store(o + 16); f03.Store(o + 24);
        if (rows > 1) { o += m.OutputStride; f10.Store(o); f11.Store(o + 8); f12.Store(o + 16); f13.Store(o + 24); }
        if (rows > 2) { o += m.OutputStride; f20.Store(o); f21.Store(o + 8); f22.Store(o + 16); f23.Store(o + 24); }
    }

    /// <summary>The same integer sums and the same fp32 operations in the same order, one output at a time.</summary>
    private unsafe void PanelPortable(in Pointers m, int r0, int r1, int panel, int columns)
    {
        int features = InFeatures, blocks = _blocks;
        sbyte* w = (sbyte*)m.Values + (long)panel * features * PanelWidth;
        float* scale = (float*)m.Scales + panel * blocks * PanelWidth;
        short* sums = (short*)m.Sums + panel * blocks * PanelWidth;
        float* bias = (float*)m.Bias;
        int n0 = panel * PanelWidth;
        for (int row = r0; row < r1; ++row)
        {
            byte* a = (byte*)m.X + (long)row * features;
            float* dx = (float*)m.Dx + (long)row * blocks;
            float* o = (float*)m.Output + (long)row * m.OutputStride + n0;
            for (int j = 0; j < columns; ++j)
            {
                float acc = bias is null ? 0f : bias[n0 + j];
                for (int b = 0; b < blocks; ++b)
                {
                    int sum = sums[b * PanelWidth + j] << 7;
                    for (int t = 0; t < Q8_0.BlockSize; ++t)
                    {
                        int k = b * Q8_0.BlockSize + t;
                        sum += a[k] * w[(long)(k / 4) * (4 * PanelWidth) + j * 4 + k % 4];
                    }
                    acc = MathF.FusedMultiplyAdd(sum, scale[b * PanelWidth + j] * dx[b], acc);
                }
                o[j] = acc;
            }
        }
    }
}

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Laya.Numerics;

/// <summary>
/// A projection weight kept in bfloat16 and multiplied in fp32: <c>output = input · weightᵀ</c>.
///
/// <para>LFM2 checkpoints are stored in bf16, and bf16 is exactly the top half of an fp32, so the
/// weights PyTorch computes with in fp32 are recovered bit for bit by a 16-bit shift. Keeping them
/// in bf16 halves the resident size (d1-3B's text stack is 5 GB instead of 10) and the bytes every
/// product streams from memory, without moving a result: the arithmetic is fp32 FMA on the widened
/// values, as in PyTorch.</para>
///
/// <para>The kernel is the BLIS layout with the roles chosen for this orientation. The register
/// tile is 32 token rows x 12 output columns: two vectors of 16 rows times twelve broadcast
/// weights, 24 accumulators. The activations are packed once per call into groups of 16 rows,
/// reduction-step major (<c>[group][k][16]</c>), so a tile's rows are two contiguous streams; the
/// weights are stored in panels of 12 columns (<c>[panel][k][12]</c>), and a block of
/// <see cref="BlockDepth"/> steps of one panel — 12 KiB once widened — stays in L1 while every row
/// group streams past it from L2. The tile accumulates into a column-major scratch (each
/// accumulator is 16 rows of one column), which is transposed into the row-major output once per
/// block of <see cref="BlockColumns"/> columns.</para>
///
/// <para>The previous layout — 6 rows x 64 columns, vectors across the outputs — had to read four
/// weight vectors per step from L2 because a 64-column panel never fits in L1; at 21 bytes a cycle
/// that capped it near 90 GFLOP/s. Here the per-step L1 traffic is two activation vectors and
/// twelve 4-byte broadcasts.</para>
///
/// <para>Each output is still one fp32 FMA chain over the reduction in order, k = 0, 1, 2, …, so
/// blocking changes no result.</para>
/// </summary>
public sealed class BFloat16Matrix
{
    /// <summary>Output columns per weight panel: the broadcast side of the register tile.</summary>
    public const int PanelWidth = 12;

    private readonly ushort[] _data;          // [panel][inFeatures][PanelWidth]
    private readonly int _panels;

    public int InFeatures { get; }
    public int OutFeatures { get; }

    /// <summary>
    /// Reduction steps per block. Each block ends with the accumulators stored and the next starts by
    /// reloading them, so longer is better until the 32-row activation tile (32 KiB at 256) and the
    /// widened weight block stop fitting L1 and L2. Swept 64-512 on d1's shapes; 256 was best.
    /// </summary>
    public static int BlockDepth { get; set; } = ReadSetting("LAYA_GEMM_KC", 256);

    /// <summary>Output columns per block (a multiple of 12): the transposed accumulator is this many columns tall.</summary>
    public static int BlockColumns { get; set; } = ReadSetting("LAYA_GEMM_NC", 192) / PanelWidth * PanelWidth;

    private static int ReadSetting(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int value) && value >= 16 ? value : fallback;

    /// <summary>Rows per packed activation group: one vector.</summary>
    private static int Lanes => SimdOps.UseVector512 ? 16 : Vector<float>.Count;

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
        _panels = (outFeatures + PanelWidth - 1) / PanelWidth;
        _data = new ushort[(long)_panels * inFeatures * PanelWidth];

        for (int panel = 0; panel < _panels; ++panel)
        {
            int first = panel * PanelWidth;
            int columns = Math.Min(PanelWidth, outFeatures - first);
            long destination = (long)panel * inFeatures * PanelWidth;
            for (int j = 0; j < columns; ++j)
            {
                var source = rowMajor.Slice((int)((long)(first + j) * inFeatures), inFeatures);
                for (int i = 0; i < inFeatures; ++i)
                {
                    _data[destination + (long)i * PanelWidth + j] = FlushSubnormal(source[i]);
                }
            }
        }
    }

    /// <summary>
    /// A subnormal bf16 (exponent bits all zero, mantissa not) becomes a signed zero.
    ///
    /// <para>d1's short-conv input projections are about a third subnormal. On x86 every FMA that
    /// reads a subnormal operand takes a microcode assist, and neither .NET nor PyTorch sets
    /// flush-to-zero / denormals-are-zero, so those projections ran 20 times slower than the same
    /// shape on normal weights — it was most of PyTorch's time on this model too. A subnormal weight
    /// is below 1.2e-38; times an activation of order one it is invisible in an fp32 sum of order
    /// one, so the outputs do not move (the parity tests hold unchanged).</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FlushSubnormal(ushort bits) => (bits & 0x7F80) == 0 ? (ushort)(bits & 0x8000) : bits;

    public long Bytes => (long)_data.Length * sizeof(ushort);

    /// <summary>Row <paramref name="column"/> of the original <c>[out, in]</c> matrix, widened.</summary>
    public void ReadRow(int column, Span<float> destination)
    {
        long start = (long)(column / PanelWidth) * InFeatures * PanelWidth + column % PanelWidth;
        var bits = MemoryMarshal.Cast<float, uint>(destination);
        for (int i = 0; i < InFeatures; ++i) bits[i] = (uint)_data[start + (long)i * PanelWidth] << 16;
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

        int lanes = Lanes;
        int groups = (rows + lanes - 1) / lanes;
        int paddedRows = groups * lanes;
        int k = InFeatures;
        float[] packed = ArrayPool<float>.Shared.Rent(groups * k * lanes);
        try
        {
            fixed (ushort* weights = _data)
            fixed (float* inputPointer = input, outputPointer = output, a = packed)
            {
                PackRows(inputPointer, rows, inputStride, a, groups, lanes);

                int blockColumns = Math.Max(PanelWidth, BlockColumns);
                int blocks = (OutFeatures + blockColumns - 1) / blockColumns;
                int workers = LayaRuntime.WorkersOf(parallel);
                if (workers <= 1 || blocks == 1 || (long)rows * OutFeatures * InFeatures <= 4_000_000)
                {
                    for (int block = 0; block < blocks; ++block)
                    {
                        Block(weights, a, groups, rows, paddedRows, outputPointer, outputStride, block * blockColumns, blockColumns);
                    }
                    return;
                }

                nint w = (nint)weights, packedAddress = (nint)a, o = (nint)outputPointer;
                Parallel.For(0, blocks, LayaRuntime.Resolve(parallel), block =>
                    Block((ushort*)w, (float*)packedAddress, groups, rows, paddedRows, (float*)o, outputStride,
                        block * blockColumns, blockColumns));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packed);
        }
    }

    /// <summary>
    /// Activations <c>[rows][k]</c> to <c>[group][k][lanes]</c>, the rows past the end zero, and
    /// subnormals flushed to zero as the weights are: SwiGLU products are a few percent subnormal in
    /// some layers, and one in a vector is enough to send that FMA to a microcode assist.
    /// </summary>
    private unsafe void PackRows(float* input, int rows, int inputStride, float* packed, int groups, int lanes)
    {
        int k = InFeatures;
        uint** sources = stackalloc uint*[lanes];
        for (int g = 0; g < groups; ++g)
        {
            float* destination = packed + (long)g * k * lanes;
            int count = Math.Min(lanes, rows - g * lanes);
            for (int r = 0; r < count; ++r) sources[r] = (uint*)(input + (long)(g * lanes + r) * inputStride);
            int i = 0;
            if (count == 16 && lanes == 16 && Avx512F.IsSupported)
            {
                for (; i + 16 <= k; i += 16) Transpose16x16(sources, i, destination + (long)i * 16, 16);
            }
            else if (count < lanes)
            {
                new Span<float>(destination, k * lanes).Clear();
            }

            // Step outer, row inner: the stores walk the destination sequentially, the reads are
            // `count` sequential streams. Row outer made every store a new cache line.
            uint* d = (uint*)destination + (long)i * lanes;
            for (; i < k; ++i, d += lanes)
            {
                for (int r = 0; r < count; ++r) d[r] = Flush(sources[r][i]);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Flush(uint value) => (value & 0x7F800000u) == 0 ? value & 0x80000000u : value;

    /// <summary>
    /// Sixteen rows' steps <c>[i, i + 16)</c> into sixteen step-major vectors, in registers: pairwise
    /// unpacks, 64-bit shuffles, then two rounds of 128-bit lane shuffles. Subnormals are flushed on
    /// the way out.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Transpose16x16(uint** rows, int i, float* destination, int destinationStride)
    {
        var r0 = Vector512.Load((float*)rows[0] + i); var r1 = Vector512.Load((float*)rows[1] + i);
        var r2 = Vector512.Load((float*)rows[2] + i); var r3 = Vector512.Load((float*)rows[3] + i);
        var r4 = Vector512.Load((float*)rows[4] + i); var r5 = Vector512.Load((float*)rows[5] + i);
        var r6 = Vector512.Load((float*)rows[6] + i); var r7 = Vector512.Load((float*)rows[7] + i);
        var r8 = Vector512.Load((float*)rows[8] + i); var r9 = Vector512.Load((float*)rows[9] + i);
        var r10 = Vector512.Load((float*)rows[10] + i); var r11 = Vector512.Load((float*)rows[11] + i);
        var r12 = Vector512.Load((float*)rows[12] + i); var r13 = Vector512.Load((float*)rows[13] + i);
        var r14 = Vector512.Load((float*)rows[14] + i); var r15 = Vector512.Load((float*)rows[15] + i);

        var t0 = Avx512F.UnpackLow(r0, r1); var t1 = Avx512F.UnpackHigh(r0, r1);
        var t2 = Avx512F.UnpackLow(r2, r3); var t3 = Avx512F.UnpackHigh(r2, r3);
        var t4 = Avx512F.UnpackLow(r4, r5); var t5 = Avx512F.UnpackHigh(r4, r5);
        var t6 = Avx512F.UnpackLow(r6, r7); var t7 = Avx512F.UnpackHigh(r6, r7);
        var t8 = Avx512F.UnpackLow(r8, r9); var t9 = Avx512F.UnpackHigh(r8, r9);
        var t10 = Avx512F.UnpackLow(r10, r11); var t11 = Avx512F.UnpackHigh(r10, r11);
        var t12 = Avx512F.UnpackLow(r12, r13); var t13 = Avx512F.UnpackHigh(r12, r13);
        var t14 = Avx512F.UnpackLow(r14, r15); var t15 = Avx512F.UnpackHigh(r14, r15);

        r0 = Avx512F.Shuffle(t0, t2, 0x44); r1 = Avx512F.Shuffle(t0, t2, 0xEE);
        r2 = Avx512F.Shuffle(t1, t3, 0x44); r3 = Avx512F.Shuffle(t1, t3, 0xEE);
        r4 = Avx512F.Shuffle(t4, t6, 0x44); r5 = Avx512F.Shuffle(t4, t6, 0xEE);
        r6 = Avx512F.Shuffle(t5, t7, 0x44); r7 = Avx512F.Shuffle(t5, t7, 0xEE);
        r8 = Avx512F.Shuffle(t8, t10, 0x44); r9 = Avx512F.Shuffle(t8, t10, 0xEE);
        r10 = Avx512F.Shuffle(t9, t11, 0x44); r11 = Avx512F.Shuffle(t9, t11, 0xEE);
        r12 = Avx512F.Shuffle(t12, t14, 0x44); r13 = Avx512F.Shuffle(t12, t14, 0xEE);
        r14 = Avx512F.Shuffle(t13, t15, 0x44); r15 = Avx512F.Shuffle(t13, t15, 0xEE);

        t0 = Avx512F.Shuffle4x128(r0, r4, 0x88); t1 = Avx512F.Shuffle4x128(r1, r5, 0x88);
        t2 = Avx512F.Shuffle4x128(r2, r6, 0x88); t3 = Avx512F.Shuffle4x128(r3, r7, 0x88);
        t4 = Avx512F.Shuffle4x128(r0, r4, 0xDD); t5 = Avx512F.Shuffle4x128(r1, r5, 0xDD);
        t6 = Avx512F.Shuffle4x128(r2, r6, 0xDD); t7 = Avx512F.Shuffle4x128(r3, r7, 0xDD);
        t8 = Avx512F.Shuffle4x128(r8, r12, 0x88); t9 = Avx512F.Shuffle4x128(r9, r13, 0x88);
        t10 = Avx512F.Shuffle4x128(r10, r14, 0x88); t11 = Avx512F.Shuffle4x128(r11, r15, 0x88);
        t12 = Avx512F.Shuffle4x128(r8, r12, 0xDD); t13 = Avx512F.Shuffle4x128(r9, r13, 0xDD);
        t14 = Avx512F.Shuffle4x128(r10, r14, 0xDD); t15 = Avx512F.Shuffle4x128(r11, r15, 0xDD);

        Store(Avx512F.Shuffle4x128(t0, t8, 0x88), destination + 0 * destinationStride);
        Store(Avx512F.Shuffle4x128(t1, t9, 0x88), destination + 1 * destinationStride);
        Store(Avx512F.Shuffle4x128(t2, t10, 0x88), destination + 2 * destinationStride);
        Store(Avx512F.Shuffle4x128(t3, t11, 0x88), destination + 3 * destinationStride);
        Store(Avx512F.Shuffle4x128(t4, t12, 0x88), destination + 4 * destinationStride);
        Store(Avx512F.Shuffle4x128(t5, t13, 0x88), destination + 5 * destinationStride);
        Store(Avx512F.Shuffle4x128(t6, t14, 0x88), destination + 6 * destinationStride);
        Store(Avx512F.Shuffle4x128(t7, t15, 0x88), destination + 7 * destinationStride);
        Store(Avx512F.Shuffle4x128(t0, t8, 0xDD), destination + 8 * destinationStride);
        Store(Avx512F.Shuffle4x128(t1, t9, 0xDD), destination + 9 * destinationStride);
        Store(Avx512F.Shuffle4x128(t2, t10, 0xDD), destination + 10 * destinationStride);
        Store(Avx512F.Shuffle4x128(t3, t11, 0xDD), destination + 11 * destinationStride);
        Store(Avx512F.Shuffle4x128(t4, t12, 0xDD), destination + 12 * destinationStride);
        Store(Avx512F.Shuffle4x128(t5, t13, 0xDD), destination + 13 * destinationStride);
        Store(Avx512F.Shuffle4x128(t6, t14, 0xDD), destination + 14 * destinationStride);
        Store(Avx512F.Shuffle4x128(t7, t15, 0xDD), destination + 15 * destinationStride);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Store(Vector512<float> value, float* destination)
        {
            var bits = value.AsUInt32();
            var subnormal = Vector512.Equals(bits & Vector512.Create(0x7F800000u), Vector512<uint>.Zero);
            Vector512.ConditionalSelect(subnormal, bits & Vector512.Create(0x80000000u), bits).AsSingle().Store(destination);
        }
    }

    [ThreadStatic] private static float[]? t_widened;
    [ThreadStatic] private static float[]? t_columns;

    private static unsafe float* Pinned(ref float[]? slot, int length)
    {
        var array = slot;
        if (array is null || array.Length < length)
        {
            array = GC.AllocateUninitializedArray<float>(length, pinned: true);
            slot = array;
        }
        return (float*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array));
    }

    /// <summary>
    /// Output columns <c>[first, first + width)</c> for every row: all reduction blocks into a
    /// column-major accumulator, then one transpose into the output.
    /// </summary>
    private unsafe void Block(ushort* weights, float* packed, int groups, int rows, int paddedRows, float* output,
        int outputStride, int first, int width)
    {
        int lanes = Lanes;
        int k = InFeatures;
        int columns = Math.Min(width, OutFeatures - first);
        int panels = (columns + PanelWidth - 1) / PanelWidth;
        int depth = BlockDepth;
        float* widened = Pinned(ref t_widened, panels * depth * PanelWidth);
        float* accumulated = Pinned(ref t_columns, panels * PanelWidth * paddedRows);   // [panel][group][12][lanes]
        int firstPanel = first / PanelWidth;

        for (int k0 = 0; k0 < k; k0 += depth)
        {
            int steps = Math.Min(depth, k - k0);
            bool accumulate = k0 > 0;

            // Widen this reduction block of every panel once: [panel][steps][12], at most
            // BlockColumns x BlockDepth floats, L2-resident.
            for (int p = 0; p < panels; ++p)
            {
                Widen(weights + ((long)(firstPanel + p) * k + k0) * PanelWidth, widened + (long)p * steps * PanelWidth,
                    steps * PanelWidth);
            }

            // Row tile outer, panel inner: the 32-row activation tile (two vectors per step) stays in
            // L1 while the 12-column weight micro-panels stream from L2 — 48 bytes a step instead of 128.
            int g = 0;
            if (SimdOps.UseVector512)
            {
                for (; g + 2 <= groups; g += 2)
                {
                    float* a0 = packed + ((long)g * k + k0) * 16;
                    float* a1 = packed + ((long)(g + 1) * k + k0) * 16;
                    for (int p = 0; p < panels; ++p)
                    {
                        Tile32x12(a0, a1, widened + (long)p * steps * PanelWidth, steps,
                            accumulated + ((long)p * groups + g) * PanelWidth * 16, accumulate);
                    }
                }
                for (; g < groups; ++g)
                {
                    float* a0 = packed + ((long)g * k + k0) * 16;
                    for (int p = 0; p < panels; ++p)
                    {
                        Tile16x12(a0, widened + (long)p * steps * PanelWidth, steps,
                            accumulated + ((long)p * groups + g) * PanelWidth * 16, accumulate);
                    }
                }
            }
            else
            {
                for (; g < groups; ++g)
                {
                    float* a0 = packed + ((long)g * k + k0) * lanes;
                    for (int p = 0; p < panels; ++p)
                    {
                        TilePortable(a0, widened + (long)p * steps * PanelWidth, steps,
                            accumulated + ((long)p * groups + g) * PanelWidth * lanes, accumulate);
                    }
                }
            }
        }

        // [panel][group][column][lane] -> output[group · lanes + lane][first + panel · 12 + column]:
        // sixteen columns at a time through the register transpose when a whole group of 16 rows is
        // there, the ragged edges element by element.
        uint** columnVectors = stackalloc uint*[16];
        for (int g = 0; g < groups; ++g)
        {
            int count = Math.Min(lanes, rows - g * lanes);
            float* rowsOut = output + (long)g * lanes * outputStride + first;
            int j0 = 0;
            if (count == 16 && lanes == 16 && Avx512F.IsSupported)
            {
                for (; j0 + 16 <= columns; j0 += 16)
                {
                    for (int j = 0; j < 16; ++j)
                    {
                        int column = j0 + j;
                        columnVectors[j] = (uint*)(accumulated + (((long)(column / PanelWidth) * groups + g) * PanelWidth + column % PanelWidth) * lanes);
                    }
                    Transpose16x16(columnVectors, 0, rowsOut + j0, outputStride);
                }
            }
            for (int column = j0; column < columns; ++column)
            {
                float* source = accumulated + (((long)(column / PanelWidth) * groups + g) * PanelWidth + column % PanelWidth) * lanes;
                for (int r = 0; r < count; ++r) rowsOut[(long)r * outputStride + column] = source[r];
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
                var (low, high) = Vector512.Widen(Vector512.Load(source + i));
                Vector512.ShiftLeft(low, 16).AsSingle().Store(destination + i);
                Vector512.ShiftLeft(high, 16).AsSingle().Store(destination + i + 16);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (; i + 16 <= count; i += 16)
            {
                var (low, high) = Vector256.Widen(Vector256.Load(source + i));
                Vector256.ShiftLeft(low, 16).AsSingle().Store(destination + i);
                Vector256.ShiftLeft(high, 16).AsSingle().Store(destination + i + 8);
            }
        }
        uint* bits = (uint*)destination;
        for (; i < count; ++i) bits[i] = (uint)source[i] << 16;
    }

    /// <summary>
    /// The register tile: 32 rows (two packed groups) x 12 columns. Per reduction step, two vector
    /// loads, twelve broadcasts from the L1-resident weight block, twenty-four FMAs. One broadcast
    /// temp, reused, so the live set is 24 accumulators + 2 activations + 1 broadcast.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile32x12(float* a0, float* a1, float* b, int steps, float* c, bool accumulate)
    {
        Vector512<float> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51,
            c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1;
        if (accumulate)
        {
            float* e = c;
            c00 = Vector512.Load(e); c01 = Vector512.Load(e + 192); e += 16;
            c10 = Vector512.Load(e); c11 = Vector512.Load(e + 192); e += 16;
            c20 = Vector512.Load(e); c21 = Vector512.Load(e + 192); e += 16;
            c30 = Vector512.Load(e); c31 = Vector512.Load(e + 192); e += 16;
            c40 = Vector512.Load(e); c41 = Vector512.Load(e + 192); e += 16;
            c50 = Vector512.Load(e); c51 = Vector512.Load(e + 192); e += 16;
            c60 = Vector512.Load(e); c61 = Vector512.Load(e + 192); e += 16;
            c70 = Vector512.Load(e); c71 = Vector512.Load(e + 192); e += 16;
            c80 = Vector512.Load(e); c81 = Vector512.Load(e + 192); e += 16;
            c90 = Vector512.Load(e); c91 = Vector512.Load(e + 192); e += 16;
            ca0 = Vector512.Load(e); ca1 = Vector512.Load(e + 192); e += 16;
            cb0 = Vector512.Load(e); cb1 = Vector512.Load(e + 192);
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
        c00.Store(s); c01.Store(s + 192); s += 16;
        c10.Store(s); c11.Store(s + 192); s += 16;
        c20.Store(s); c21.Store(s + 192); s += 16;
        c30.Store(s); c31.Store(s + 192); s += 16;
        c40.Store(s); c41.Store(s + 192); s += 16;
        c50.Store(s); c51.Store(s + 192); s += 16;
        c60.Store(s); c61.Store(s + 192); s += 16;
        c70.Store(s); c71.Store(s + 192); s += 16;
        c80.Store(s); c81.Store(s + 192); s += 16;
        c90.Store(s); c91.Store(s + 192); s += 16;
        ca0.Store(s); ca1.Store(s + 192); s += 16;
        cb0.Store(s); cb1.Store(s + 192);
    }

    /// <summary>The last odd group of 16 rows: one vector x 12 columns.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile16x12(float* a, float* b, int steps, float* c, bool accumulate)
    {
        Vector512<float> c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb;
        if (accumulate)
        {
            c0 = Vector512.Load(c); c1 = Vector512.Load(c + 16); c2 = Vector512.Load(c + 2 * 16);
            c3 = Vector512.Load(c + 3 * 16); c4 = Vector512.Load(c + 4 * 16); c5 = Vector512.Load(c + 5 * 16);
            c6 = Vector512.Load(c + 6 * 16); c7 = Vector512.Load(c + 7 * 16); c8 = Vector512.Load(c + 8 * 16);
            c9 = Vector512.Load(c + 9 * 16); ca = Vector512.Load(c + 10 * 16); cb = Vector512.Load(c + 11 * 16);
        }
        else
        {
            c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = ca = cb = Vector512<float>.Zero;
        }
        for (int i = 0; i < steps; ++i, a += 16, b += 12)
        {
            var x = Vector512.Load(a);
            c0 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[0]), c0);
            c1 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[1]), c1);
            c2 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[2]), c2);
            c3 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[3]), c3);
            c4 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[4]), c4);
            c5 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[5]), c5);
            c6 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[6]), c6);
            c7 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[7]), c7);
            c8 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[8]), c8);
            c9 = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[9]), c9);
            ca = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[10]), ca);
            cb = Vector512.FusedMultiplyAdd(x, Vector512.Create(b[11]), cb);
        }
        c0.Store(c); c1.Store(c + 16); c2.Store(c + 2 * 16); c3.Store(c + 3 * 16);
        c4.Store(c + 4 * 16); c5.Store(c + 5 * 16); c6.Store(c + 6 * 16); c7.Store(c + 7 * 16);
        c8.Store(c + 8 * 16); c9.Store(c + 9 * 16); ca.Store(c + 10 * 16); cb.Store(c + 11 * 16);
    }

    /// <summary>The portable tile: one <c>Vector&lt;float&gt;</c> of rows x 12 columns (14 of 16 ymm registers on AVX2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void TilePortable(float* a, float* b, int steps, float* c, bool accumulate)
    {
        int lanes = Vector<float>.Count;
        Vector<float> c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb;
        if (accumulate)
        {
            c0 = Vector.Load(c); c1 = Vector.Load(c + lanes); c2 = Vector.Load(c + 2 * lanes);
            c3 = Vector.Load(c + 3 * lanes); c4 = Vector.Load(c + 4 * lanes); c5 = Vector.Load(c + 5 * lanes);
            c6 = Vector.Load(c + 6 * lanes); c7 = Vector.Load(c + 7 * lanes); c8 = Vector.Load(c + 8 * lanes);
            c9 = Vector.Load(c + 9 * lanes); ca = Vector.Load(c + 10 * lanes); cb = Vector.Load(c + 11 * lanes);
        }
        else
        {
            c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = ca = cb = Vector<float>.Zero;
        }
        for (int i = 0; i < steps; ++i, a += lanes, b += 12)
        {
            var x = Vector.Load(a);
            c0 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[0]), c0);
            c1 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[1]), c1);
            c2 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[2]), c2);
            c3 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[3]), c3);
            c4 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[4]), c4);
            c5 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[5]), c5);
            c6 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[6]), c6);
            c7 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[7]), c7);
            c8 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[8]), c8);
            c9 = Vector.FusedMultiplyAdd(x, new Vector<float>(b[9]), c9);
            ca = Vector.FusedMultiplyAdd(x, new Vector<float>(b[10]), ca);
            cb = Vector.FusedMultiplyAdd(x, new Vector<float>(b[11]), cb);
        }
        c0.Store(c); c1.Store(c + lanes); c2.Store(c + 2 * lanes); c3.Store(c + 3 * lanes);
        c4.Store(c + 4 * lanes); c5.Store(c + 5 * lanes); c6.Store(c + 6 * lanes); c7.Store(c + 7 * lanes);
        c8.Store(c + 8 * lanes); c9.Store(c + 9 * lanes); ca.Store(c + 10 * lanes); cb.Store(c + 11 * lanes);
    }
}

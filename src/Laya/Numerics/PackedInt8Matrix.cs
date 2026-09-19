using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Laya;

namespace Laya.Numerics;

/// <summary>
/// The int8 counterpart of <see cref="PackedMatrix"/>: 8-bit per-output-channel symmetric weights
/// against per-row dynamically quantized activations (W8A8).
///
/// <para><b>The layout is the point.</b> The obvious int8 GEMM makes each output a dot product over
/// contiguous <c>[out, in]</c> weights and ends every tile in a horizontal reduction. That is the
/// shape <see cref="PackedMatrix"/> exists to avoid — a plain transpose measured three times slower
/// than the naive kernel here. So this class keeps the panel structure exactly and interleaves four
/// reduction steps into each 32-bit lane instead: <c>[panel][k/4][column][k%4]</c>. One dword
/// broadcast of four consecutive activations then feeds a <c>vpdpbusd</c>-shaped dot against a
/// panel vector, each lane still owns one output column, and the loop still ends in a store rather
/// than a shuffle chain. The register tile is unchanged at six token rows by four output vectors;
/// only the accumulators change from float to int32, and each step now consumes four reduction
/// steps instead of one.</para>
///
/// <para><b>What it buys, honestly.</b> A quarter of the weight bytes. The arithmetic is only
/// faster where the hardware has a real int8-VNNI instruction: <c>vpdpbusd</c> does 64 products in
/// one uop against the FMA's 16, but the <c>vpmaddubsw</c> fallback needs three uops for those 64
/// and the widening fallback needs eight, which is at best a small win and at worst a loss against
/// fp32. See <see cref="Vnni"/> for which sequence a given host gets.</para>
/// </summary>
public sealed class PackedInt8Matrix : IProjection
{
    private const int RowBlock = 6;

    private readonly sbyte[] _q;        // [panel][kBlocks][panelWidth][4]
    private readonly float[] _channelScale;   // [panels * panelWidth] weight scale per output column
    private readonly float[] _channelOffset;  // [panels * panelWidth] scale * 128 * sum_k q[o, k]
    private readonly int _lanes;
    private readonly int _panelWidth;
    private readonly int _panels;
    private readonly long _panelStride;
    private readonly int _kBlocks;
    private readonly int _kPadded;
    private readonly int _outlierChannels;

    public int InFeatures { get; }
    public int OutFeatures { get; }

    /// <summary>Repacks and quantizes a PyTorch <c>[out, in]</c> weight.</summary>
    /// <param name="outlierChannels">
    /// How many input channels to hold out of the int8 path and accumulate in float; see
    /// <see cref="Vnni.OutlierChannels"/> for why they exist. Defaults to the process setting.
    /// </param>
    public PackedInt8Matrix(ReadOnlySpan<float> rowMajor, int outFeatures, int inFeatures,
        int? outlierChannels = null)
    {
        if (rowMajor.Length < (long)outFeatures * inFeatures)
        {
            throw new ArgumentException(
                $"expected {(long)outFeatures * inFeatures} weights for [{outFeatures}, {inFeatures}].",
                nameof(rowMajor));
        }

        OutFeatures = outFeatures;
        InFeatures = inFeatures;
        _lanes = SimdOps.UseVector512 ? 16 : 8;
        _panelWidth = 4 * _lanes;
        _panels = (outFeatures + _panelWidth - 1) / _panelWidth;
        _kBlocks = (inFeatures + 3) / 4;
        _kPadded = _kBlocks * 4;
        _panelStride = (long)_kBlocks * _panelWidth * 4;
        _outlierChannels = outlierChannels ?? Vnni.OutlierChannels;

        _q = new sbyte[_panels * _panelStride];
        _channelScale = new float[_panels * _panelWidth];
        _channelOffset = new float[_panels * _panelWidth];

        int qmax = Vnni.WeightQMax;

        for (int o = 0; o < outFeatures; ++o)
        {
            var row = rowMajor.Slice(o * inFeatures, inFeatures);
            float amax = 0f;
            for (int i = 0; i < inFeatures; ++i) amax = MathF.Max(amax, MathF.Abs(row[i]));

            float scale = amax > 0 ? amax / qmax : 1f;
            float inverse = 1f / scale;

            int panel = o / _panelWidth;
            int column = o - panel * _panelWidth;
            long panelBase = panel * _panelStride;

            int sum = 0;
            for (int i = 0; i < inFeatures; ++i)
            {
                int v = Math.Clamp((int)MathF.Round(row[i] * inverse), -qmax, qmax);
                sum += v;
                _q[panelBase + (long)(i >> 2) * _panelWidth * 4 + column * 4 + (i & 3)] = (sbyte)v;
            }

            _channelScale[panel * _panelWidth + column] = scale;
            _channelOffset[panel * _panelWidth + column] = scale * Vnni.ZeroPoint * sum;
        }
        // Columns past OutFeatures in the last panel keep zero weights, a unit scale and a zero
        // offset, so the kernel can always compute a whole panel and simply not store the tail.
        for (int c = outFeatures; c < _panels * _panelWidth; ++c) _channelScale[c] = 1f;
    }

    public long Bytes => _q.Length + (long)(_channelScale.Length + _channelOffset.Length) * sizeof(float);

    /// <summary>Which instruction sequence this matrix's kernel will use.</summary>
    public static string Kernel => Vnni.Description;

    public void Multiply(ReadOnlySpan<float> input, int rows, ReadOnlySpan<float> bias, Span<float> output)
        => Multiply(input, rows, InFeatures, bias, output, OutFeatures);

    public unsafe void Multiply(ReadOnlySpan<float> input, int rows, int inputStride,
        ReadOnlySpan<float> bias, Span<float> output, int outputStride)
    {
        if (input.Length < (long)rows * inputStride) throw new ArgumentException("input is too small", nameof(input));
        if (output.Length < (long)rows * outputStride) throw new ArgumentException("output is too small", nameof(output));
        if (!bias.IsEmpty && bias.Length < OutFeatures) throw new ArgumentException("bias is too small", nameof(bias));

        byte[] activations = ArrayPool<byte>.Shared.Rent(rows * _kPadded);
        float[] activationScale = ArrayPool<float>.Shared.Rent(rows);
        int outlierCount = Math.Min(_outlierChannels, InFeatures - 1);
        int[] outliers = outlierCount > 0 ? ArrayPool<int>.Shared.Rent(outlierCount) : [];
        try
        {
            if (outlierCount > 0)
            {
                outlierCount = SelectOutlierChannels(input, rows, inputStride, outliers, outlierCount);
            }
            QuantizeActivations(input, rows, inputStride, activations, activationScale,
                outliers.AsSpan(0, outlierCount));


            fixed (sbyte* weights = _q)
            fixed (byte* activationPointer = activations)
            fixed (float* activationScalePointer = activationScale)
            fixed (float* scalePointer = _channelScale, offsetPointer = _channelOffset)
            fixed (float* biasPointer = bias, outputPointer = output)
            {
                float* bias0 = bias.IsEmpty ? null : biasPointer;
                var context = new Context
                {
                    Weights = (nint)weights,
                    Activations = (nint)activationPointer,
                    ActivationScale = (nint)activationScalePointer,
                    ChannelScale = (nint)scalePointer,
                    ChannelOffset = (nint)offsetPointer,
                    Bias = (nint)bias0,
                    Output = (nint)outputPointer,
                    OutputStride = outputStride,
                    Rows = rows,
                };

                int workers = LayaRuntime.MaxDegreeOfParallelism;
                if (workers <= 1 || (long)rows * OutFeatures * InFeatures <= 1_000_000)
                {
                    for (int panel = 0; panel < _panels; ++panel) Panel(in context, panel);
                }
                else
                {
                    int chunk = Math.Max(1, _panels / (workers * 4));
                    int chunks = (_panels + chunk - 1) / chunk;
                    Parallel.For(0, chunks, LayaRuntime.ParallelOptions, index =>
                    {
                        int first = index * chunk;
                        int last = Math.Min(_panels, first + chunk);
                        for (int panel = first; panel < last; ++panel) Panel(in context, panel);
                    });
                }

                // The panels store rather than accumulate, so the float correction for the held-out
                // channels has to follow them. Parallel.For has already joined by here.
                if (outlierCount > 0)
                {
                    AddOutlierContribution(input, rows, inputStride, outliers.AsSpan(0, outlierCount),
                        output, outputStride);
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(activations);
            ArrayPool<float>.Shared.Return(activationScale);
            if (outliers.Length > 0) ArrayPool<int>.Shared.Return(outliers);
        }
    }

    /// <summary>
    /// The pointers for one <see cref="Multiply"/>, held as <c>nint</c> so the struct can be copied
    /// into the parallel closure — a struct with pointer fields cannot. The kernels cast once on
    /// entry, which is free, and it keeps the panel kernels down to one argument rather than ten.
    /// </summary>
    private readonly struct Context
    {
        public required nint Weights { get; init; }
        public required nint Activations { get; init; }
        public required nint ActivationScale { get; init; }
        public required nint ChannelScale { get; init; }
        public required nint ChannelOffset { get; init; }
        public required nint Bias { get; init; }
        public required nint Output { get; init; }
        public required int OutputStride { get; init; }
        public required int Rows { get; init; }
    }

    /// <summary>
    /// Per-row symmetric quantization of the activations into the uint8 range the int8 dot expects:
    /// <c>ua = round(x * 127 / amax) + 128</c>. Dynamic rather than calibrated, because the range of
    /// these activations varies far more across tokens than it does across runs.
    /// </summary>
    private unsafe void QuantizeActivations(ReadOnlySpan<float> input, int rows, int inputStride,
        byte[] activations, float[] activationScale, ReadOnlySpan<int> outliers)
    {
        fixed (float* inputPointer = input)
        fixed (byte* destinationPointer = activations)
        fixed (float* scalePointer = activationScale)
        {
        // A mask rather than the index list: the row loop asks "is this channel held" once per
        // element, and a linear scan of the list there would cost features x heldCount per row.
        bool[] heldMask = ArrayPool<bool>.Shared.Rent(InFeatures);
        Array.Clear(heldMask, 0, InFeatures);
        foreach (int channel in outliers) heldMask[channel] = true;

        fixed (bool* maskPointer = heldMask)
        {
            nint source = (nint)inputPointer, destination = (nint)destinationPointer, scales = (nint)scalePointer;
            nint held = (nint)maskPointer;
            int features = InFeatures, padded = _kPadded, heldCount = outliers.Length;

            int workers = LayaRuntime.MaxDegreeOfParallelism;
            if (workers <= 1 || (long)rows * features < 200_000)
            {
                for (int row = 0; row < rows; ++row)
                {
                    QuantizeRow(source, destination, scales, row, inputStride, features, padded, held, heldCount);
                }
            }
            else
            {
                Parallel.For(0, rows, LayaRuntime.ParallelOptions,
                    row => QuantizeRow(source, destination, scales, row, inputStride, features, padded, held, heldCount));
            }
        }
        ArrayPool<bool>.Shared.Return(heldMask);
        }
    }

    private static unsafe void QuantizeRow(nint source, nint destination, nint scales, int row,
        int inputStride, int features, int padded, nint held, int heldCount)
    {
        int qmax = Vnni.ActivationQMax;
        var values = new ReadOnlySpan<float>((float*)source + (long)row * inputStride, features);
        byte* target = (byte*)destination + (long)row * padded;

        bool* heldMask = (bool*)held;

        // The scale is set by the largest value that is actually going through the int8 path, so the
        // held-out channels must not be allowed to set it.
        var (amax, meanSquare) = SimdOps.Spread(values);
        if (heldCount > 0)
        {
            amax = 0f;
            for (int i = 0; i < features; ++i)
            {
                if (!heldMask[i]) amax = MathF.Max(amax, MathF.Abs(values[i]));
            }
        }

        // Clipping the range instead of holding channels out is measured and rejected; see
        // Vnni.ClipSigmas. The switch stays so the result can be reproduced.
        float clip = amax;
        float sigmas = Vnni.ClipSigmas;
        if (sigmas > 0 && meanSquare > 0) clip = MathF.Min(amax, sigmas * MathF.Sqrt(meanSquare));

        float scale = clip > 0 ? clip / qmax : 1f;
        float inverse = 1f / scale;
        ((float*)scales)[row] = scale;

        for (int i = 0; i < features; ++i)
        {
            target[i] = (byte)(Math.Clamp((int)MathF.Round(values[i] * inverse), -qmax, qmax) + Vnni.ZeroPoint);
        }
        for (int i = features; i < padded; ++i) target[i] = Vnni.ZeroPoint;

        // Pinning a held-out channel to the zero point makes its int8 contribution exactly the
        // 128 * q[o, k] that the epilogue's offset correction already subtracts, so it cancels and
        // the float pass can add the real thing back.
        if (heldCount > 0)
        {
            for (int i = 0; i < features; ++i)
            {
                if (heldMask[i]) target[i] = Vnni.ZeroPoint;
            }
        }
    }

    /// <summary>
    /// Picks the input channels to keep in float: the ones whose largest magnitude over this batch
    /// of rows is biggest. They are selected per call rather than calibrated once, because it costs
    /// one pass over the activations — the same order as the quantization that follows — and a
    /// dynamic choice cannot go stale on an input distribution nobody calibrated for.
    /// </summary>
    private int SelectOutlierChannels(ReadOnlySpan<float> input, int rows, int inputStride,
        int[] outliers, int wanted)
    {
        int features = InFeatures;
        float[] channelMax = ArrayPool<float>.Shared.Rent(features);
        try
        {
            var magnitude = channelMax.AsSpan(0, features);
            magnitude.Clear();
            for (int row = 0; row < rows; ++row)
            {
                SimdOps.MaxMagnitudeInto(input.Slice(row * inputStride, features), magnitude);
            }

            // A partial selection: `wanted` is sixteen and `features` is thousands, so repeatedly
            // taking the largest beats sorting the whole thing.
            int taken = 0;
            for (; taken < wanted; ++taken)
            {
                int best = -1;
                float bestValue = float.NegativeInfinity;
                for (int i = 0; i < features; ++i)
                {
                    if (magnitude[i] > bestValue) { bestValue = magnitude[i]; best = i; }
                }
                if (best < 0 || bestValue <= 0) break;
                outliers[taken] = best;
                magnitude[best] = float.NegativeInfinity;
            }
            return taken;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(channelMax);
        }
    }

    /// <summary>
    /// Adds the held-out channels' contribution in float: <c>y[row, o] += x[row, k] * W[o, k]</c>
    /// over the held-out <c>k</c>.
    ///
    /// <para><c>W</c> here is the dequantized int8 weight, not the original float one. That is
    /// deliberate and it is what makes this cheap: the weights were never the problem — full-range
    /// weights measured <em>worse</em> end to end than the reduced range, because the error is
    /// dominated by the activations — so there is no second copy of the weights to keep. The columns
    /// are gathered out of the panel layout once per call into a small dense <c>[k][out]</c> tile,
    /// which then multiplies as a plain rank-k update.</para>
    /// </summary>
    private unsafe void AddOutlierContribution(ReadOnlySpan<float> input, int rows, int inputStride,
        ReadOnlySpan<int> outliers, Span<float> output, int outputStride)
    {
        int count = outliers.Length;
        float[] columns = ArrayPool<float>.Shared.Rent(count * OutFeatures);
        try
        {
            for (int h = 0; h < count; ++h)
            {
                int channel = outliers[h];
                int kb = channel >> 2, kk = channel & 3;
                var destination = columns.AsSpan(h * OutFeatures, OutFeatures);
                for (int o = 0; o < OutFeatures; ++o)
                {
                    int panel = o / _panelWidth;
                    int column = o - panel * _panelWidth;
                    sbyte q = _q[panel * _panelStride + (long)kb * _panelWidth * 4 + column * 4 + kk];
                    destination[o] = q * _channelScale[panel * _panelWidth + column];
                }
            }

            for (int row = 0; row < rows; ++row)
            {
                var values = input.Slice(row * inputStride, InFeatures);
                var target = output.Slice(row * outputStride, OutFeatures);
                for (int h = 0; h < count; ++h)
                {
                    float x = values[outliers[h]];
                    if (x == 0f) continue;
                    SimdOps.AddScaled(target, columns.AsSpan(h * OutFeatures, OutFeatures), x);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(columns);
        }
    }

    private unsafe void Panel(in Context context, int panel)
    {
        int n0 = panel * _panelWidth;
        int columns = Math.Min(_panelWidth, OutFeatures - n0);

        if (columns != _panelWidth)
        {
            PanelRagged(in context, panel, columns);
            return;
        }

        if (_lanes == 16) Panel512(in context, panel);
        else Panel256(in context, panel);
    }

    /// <summary>
    /// The 512-bit kernel: 64 output columns and six token rows per pass, four reduction steps per
    /// iteration. Twenty-four int32 accumulators, four weight vectors and one broadcast — the same
    /// 29-register budget the float kernel was swept to, so that tile is kept rather than re-derived.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void Panel512(in Context context, int panel)
    {
        const int lanes = 16;
        int n0 = panel * _panelWidth;
        int kBlocks = _kBlocks, kPadded = _kPadded, rows = context.Rows, outputStride = context.OutputStride;
        sbyte* panelBase = (sbyte*)context.Weights + panel * _panelStride;
        byte* activations = (byte*)context.Activations;
        float* activationScale = (float*)context.ActivationScale;
        float* channelScale = (float*)context.ChannelScale;
        float* channelOffset = (float*)context.ChannelOffset;
        float* biasPointer = (float*)context.Bias;
        float* output = (float*)context.Output;

        var s0 = Vector512.Load(channelScale + n0 + 0 * lanes);
        var s1 = Vector512.Load(channelScale + n0 + 1 * lanes);
        var s2 = Vector512.Load(channelScale + n0 + 2 * lanes);
        var s3 = Vector512.Load(channelScale + n0 + 3 * lanes);
        var f0 = Vector512.Load(channelOffset + n0 + 0 * lanes);
        var f1 = Vector512.Load(channelOffset + n0 + 1 * lanes);
        var f2 = Vector512.Load(channelOffset + n0 + 2 * lanes);
        var f3 = Vector512.Load(channelOffset + n0 + 3 * lanes);
        var b0 = biasPointer is null ? Vector512<float>.Zero : Vector512.Load(biasPointer + n0 + 0 * lanes);
        var b1 = biasPointer is null ? Vector512<float>.Zero : Vector512.Load(biasPointer + n0 + 1 * lanes);
        var b2 = biasPointer is null ? Vector512<float>.Zero : Vector512.Load(biasPointer + n0 + 2 * lanes);
        var b3 = biasPointer is null ? Vector512<float>.Zero : Vector512.Load(biasPointer + n0 + 3 * lanes);

        int row = 0;
        for (; row + RowBlock <= rows; row += RowBlock)
        {
            var c00 = Vector512<int>.Zero; var c01 = Vector512<int>.Zero; var c02 = Vector512<int>.Zero; var c03 = Vector512<int>.Zero;
            var c10 = Vector512<int>.Zero; var c11 = Vector512<int>.Zero; var c12 = Vector512<int>.Zero; var c13 = Vector512<int>.Zero;
            var c20 = Vector512<int>.Zero; var c21 = Vector512<int>.Zero; var c22 = Vector512<int>.Zero; var c23 = Vector512<int>.Zero;
            var c30 = Vector512<int>.Zero; var c31 = Vector512<int>.Zero; var c32 = Vector512<int>.Zero; var c33 = Vector512<int>.Zero;
            var c40 = Vector512<int>.Zero; var c41 = Vector512<int>.Zero; var c42 = Vector512<int>.Zero; var c43 = Vector512<int>.Zero;
            var c50 = Vector512<int>.Zero; var c51 = Vector512<int>.Zero; var c52 = Vector512<int>.Zero; var c53 = Vector512<int>.Zero;

            int* a0 = (int*)(activations + (long)(row + 0) * kPadded);
            int* a1 = (int*)(activations + (long)(row + 1) * kPadded);
            int* a2 = (int*)(activations + (long)(row + 2) * kPadded);
            int* a3 = (int*)(activations + (long)(row + 3) * kPadded);
            int* a4 = (int*)(activations + (long)(row + 4) * kPadded);
            int* a5 = (int*)(activations + (long)(row + 5) * kPadded);
            sbyte* w = panelBase;

            for (int kb = 0; kb < kBlocks; ++kb, w += _panelWidth * 4)
            {
                var w0 = Vector512.Load(w + 0 * 64);
                var w1 = Vector512.Load(w + 1 * 64);
                var w2 = Vector512.Load(w + 2 * 64);
                var w3 = Vector512.Load(w + 3 * 64);

                var x = Vector512.Create(a0[kb]).AsByte();
                c00 = Vnni.DotAccumulate512(c00, x, w0);
                c01 = Vnni.DotAccumulate512(c01, x, w1);
                c02 = Vnni.DotAccumulate512(c02, x, w2);
                c03 = Vnni.DotAccumulate512(c03, x, w3);
                x = Vector512.Create(a1[kb]).AsByte();
                c10 = Vnni.DotAccumulate512(c10, x, w0);
                c11 = Vnni.DotAccumulate512(c11, x, w1);
                c12 = Vnni.DotAccumulate512(c12, x, w2);
                c13 = Vnni.DotAccumulate512(c13, x, w3);
                x = Vector512.Create(a2[kb]).AsByte();
                c20 = Vnni.DotAccumulate512(c20, x, w0);
                c21 = Vnni.DotAccumulate512(c21, x, w1);
                c22 = Vnni.DotAccumulate512(c22, x, w2);
                c23 = Vnni.DotAccumulate512(c23, x, w3);
                x = Vector512.Create(a3[kb]).AsByte();
                c30 = Vnni.DotAccumulate512(c30, x, w0);
                c31 = Vnni.DotAccumulate512(c31, x, w1);
                c32 = Vnni.DotAccumulate512(c32, x, w2);
                c33 = Vnni.DotAccumulate512(c33, x, w3);
                x = Vector512.Create(a4[kb]).AsByte();
                c40 = Vnni.DotAccumulate512(c40, x, w0);
                c41 = Vnni.DotAccumulate512(c41, x, w1);
                c42 = Vnni.DotAccumulate512(c42, x, w2);
                c43 = Vnni.DotAccumulate512(c43, x, w3);
                x = Vector512.Create(a5[kb]).AsByte();
                c50 = Vnni.DotAccumulate512(c50, x, w0);
                c51 = Vnni.DotAccumulate512(c51, x, w1);
                c52 = Vnni.DotAccumulate512(c52, x, w2);
                c53 = Vnni.DotAccumulate512(c53, x, w3);
            }

            float* d = output + (long)row * outputStride + n0;
            Store512(d, activationScale[row + 0], c00, c01, c02, c03, s0, s1, s2, s3, f0, f1, f2, f3, b0, b1, b2, b3);
            d += outputStride;
            Store512(d, activationScale[row + 1], c10, c11, c12, c13, s0, s1, s2, s3, f0, f1, f2, f3, b0, b1, b2, b3);
            d += outputStride;
            Store512(d, activationScale[row + 2], c20, c21, c22, c23, s0, s1, s2, s3, f0, f1, f2, f3, b0, b1, b2, b3);
            d += outputStride;
            Store512(d, activationScale[row + 3], c30, c31, c32, c33, s0, s1, s2, s3, f0, f1, f2, f3, b0, b1, b2, b3);
            d += outputStride;
            Store512(d, activationScale[row + 4], c40, c41, c42, c43, s0, s1, s2, s3, f0, f1, f2, f3, b0, b1, b2, b3);
            d += outputStride;
            Store512(d, activationScale[row + 5], c50, c51, c52, c53, s0, s1, s2, s3, f0, f1, f2, f3, b0, b1, b2, b3);
        }

        for (; row < rows; ++row)
        {
            var c0 = Vector512<int>.Zero; var c1 = Vector512<int>.Zero; var c2 = Vector512<int>.Zero; var c3 = Vector512<int>.Zero;
            int* a0 = (int*)(activations + (long)row * kPadded);
            sbyte* w = panelBase;
            for (int kb = 0; kb < kBlocks; ++kb, w += _panelWidth * 4)
            {
                var x = Vector512.Create(a0[kb]).AsByte();
                c0 = Vnni.DotAccumulate512(c0, x, Vector512.Load(w + 0 * 64));
                c1 = Vnni.DotAccumulate512(c1, x, Vector512.Load(w + 1 * 64));
                c2 = Vnni.DotAccumulate512(c2, x, Vector512.Load(w + 2 * 64));
                c3 = Vnni.DotAccumulate512(c3, x, Vector512.Load(w + 3 * 64));
            }
            Store512(output + (long)row * outputStride + n0, activationScale[row],
                c0, c1, c2, c3, s0, s1, s2, s3, f0, f1, f2, f3, b0, b1, b2, b3);
        }
    }

    /// <summary>
    /// The epilogue: <c>y = aScale * (channelScale * acc - channelOffset) + bias</c>, where the
    /// offset undoes the +128 the activations were shifted by. This runs once per row per panel, not
    /// in the reduction loop, so passing the vectors by value here costs nothing.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store512(float* d, float activationScale,
        Vector512<int> c0, Vector512<int> c1, Vector512<int> c2, Vector512<int> c3,
        Vector512<float> s0, Vector512<float> s1, Vector512<float> s2, Vector512<float> s3,
        Vector512<float> f0, Vector512<float> f1, Vector512<float> f2, Vector512<float> f3,
        Vector512<float> b0, Vector512<float> b1, Vector512<float> b2, Vector512<float> b3)
    {
        var a = Vector512.Create(activationScale);
        (Vector512.FusedMultiplyAdd(a, Vector512.ConvertToSingle(c0) * s0 - f0, b0)).Store(d + 0 * 16);
        (Vector512.FusedMultiplyAdd(a, Vector512.ConvertToSingle(c1) * s1 - f1, b1)).Store(d + 1 * 16);
        (Vector512.FusedMultiplyAdd(a, Vector512.ConvertToSingle(c2) * s2 - f2, b2)).Store(d + 2 * 16);
        (Vector512.FusedMultiplyAdd(a, Vector512.ConvertToSingle(c3) * s3 - f3, b3)).Store(d + 3 * 16);
    }

    /// <summary>The 256-bit kernel, same tile, eight columns per accumulator.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void Panel256(in Context context, int panel)
    {
        const int lanes = 8;
        int n0 = panel * _panelWidth;
        int kBlocks = _kBlocks, kPadded = _kPadded, rows = context.Rows, outputStride = context.OutputStride;
        sbyte* panelBase = (sbyte*)context.Weights + panel * _panelStride;
        byte* activations = (byte*)context.Activations;
        float* activationScale = (float*)context.ActivationScale;
        float* channelScale = (float*)context.ChannelScale;
        float* channelOffset = (float*)context.ChannelOffset;
        float* biasPointer = (float*)context.Bias;
        float* output = (float*)context.Output;

        var s0 = Vector256.Load(channelScale + n0 + 0 * lanes);
        var s1 = Vector256.Load(channelScale + n0 + 1 * lanes);
        var s2 = Vector256.Load(channelScale + n0 + 2 * lanes);
        var s3 = Vector256.Load(channelScale + n0 + 3 * lanes);
        var f0 = Vector256.Load(channelOffset + n0 + 0 * lanes);
        var f1 = Vector256.Load(channelOffset + n0 + 1 * lanes);
        var f2 = Vector256.Load(channelOffset + n0 + 2 * lanes);
        var f3 = Vector256.Load(channelOffset + n0 + 3 * lanes);
        var b0 = biasPointer is null ? Vector256<float>.Zero : Vector256.Load(biasPointer + n0 + 0 * lanes);
        var b1 = biasPointer is null ? Vector256<float>.Zero : Vector256.Load(biasPointer + n0 + 1 * lanes);
        var b2 = biasPointer is null ? Vector256<float>.Zero : Vector256.Load(biasPointer + n0 + 2 * lanes);
        var b3 = biasPointer is null ? Vector256<float>.Zero : Vector256.Load(biasPointer + n0 + 3 * lanes);

        for (int row = 0; row < rows; ++row)
        {
            var c0 = Vector256<int>.Zero; var c1 = Vector256<int>.Zero; var c2 = Vector256<int>.Zero; var c3 = Vector256<int>.Zero;
            int* a0 = (int*)(activations + (long)row * kPadded);
            sbyte* w = panelBase;
            for (int kb = 0; kb < kBlocks; ++kb, w += _panelWidth * 4)
            {
                var x = Vector256.Create(a0[kb]).AsByte();
                c0 = Vnni.DotAccumulate256(c0, x, Vector256.Load(w + 0 * 32));
                c1 = Vnni.DotAccumulate256(c1, x, Vector256.Load(w + 1 * 32));
                c2 = Vnni.DotAccumulate256(c2, x, Vector256.Load(w + 2 * 32));
                c3 = Vnni.DotAccumulate256(c3, x, Vector256.Load(w + 3 * 32));
            }

            float* d = output + (long)row * outputStride + n0;
            var a = Vector256.Create(activationScale[row]);
            (Vector256.FusedMultiplyAdd(a, Vector256.ConvertToSingle(c0) * s0 - f0, b0)).Store(d + 0 * lanes);
            (Vector256.FusedMultiplyAdd(a, Vector256.ConvertToSingle(c1) * s1 - f1, b1)).Store(d + 1 * lanes);
            (Vector256.FusedMultiplyAdd(a, Vector256.ConvertToSingle(c2) * s2 - f2, b2)).Store(d + 2 * lanes);
            (Vector256.FusedMultiplyAdd(a, Vector256.ConvertToSingle(c3) * s3 - f3, b3)).Store(d + 3 * lanes);
        }
    }

    /// <summary>
    /// The last panel when the output dimension is not a whole number of panels: computed scalar into
    /// a full-width tile and trimmed. Correctness over speed — none of this model's projections have
    /// a ragged panel, since every output dimension is a multiple of 64.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private unsafe void PanelRagged(in Context context, int panel, int columns)
    {
        int n0 = panel * _panelWidth;
        sbyte* panelBase = (sbyte*)context.Weights + panel * _panelStride;
        byte* activations = (byte*)context.Activations;
        float* activationScale = (float*)context.ActivationScale;
        float* channelScale = (float*)context.ChannelScale;
        float* channelOffset = (float*)context.ChannelOffset;
        float* biasPointer = (float*)context.Bias;
        float* output = (float*)context.Output;
        Span<int> tile = stackalloc int[_panelWidth];

        for (int row = 0; row < context.Rows; ++row)
        {
            tile.Clear();
            byte* a0 = activations + (long)row * _kPadded;
            for (int kb = 0; kb < _kBlocks; ++kb)
            {
                sbyte* w = panelBase + (long)kb * _panelWidth * 4;
                for (int j = 0; j < _panelWidth; ++j)
                {
                    int sum = 0;
                    for (int kk = 0; kk < 4; ++kk) sum += a0[kb * 4 + kk] * w[j * 4 + kk];
                    tile[j] += sum;
                }
            }

            float scale = activationScale[row];
            float* destination = output + (long)row * context.OutputStride + n0;
            for (int j = 0; j < columns; ++j)
            {
                float value = scale * (tile[j] * channelScale[n0 + j] - channelOffset[n0 + j]);
                destination[j] = biasPointer is null ? value : value + biasPointer[n0 + j];
            }
        }
    }
}

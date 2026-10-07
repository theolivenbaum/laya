using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Laya.Diagnostics;
using Laya.Io;
using Laya.Numerics;

namespace Laya.Models;

/// <summary>
/// How the rows of one LFM2 pass relate: a <em>trunk</em> of shared tokens followed by any number
/// of <em>branches</em> that each continue it, packed back to back with no padding —
/// <c>[trunk | branch 0 | branch 1 | …]</c>. A plain prompt is a trunk with no branches.
///
/// <para>This is <c>hybrid.Tree</c> from the d1 reference. A branch sees the whole trunk and its own
/// earlier tokens and nothing of its siblings, so N questions over one state read the state once:
/// every token-wise operation (norms, projections, the MLP) runs over the packed rows in one
/// product, and only the two token-mixing operators look at the tree. Mathematically each branch is
/// exactly <c>trunk + branch</c> run on its own.</para>
/// </summary>
public sealed class Lfm2Tree
{
    public int TrunkLength { get; }
    public IReadOnlyList<int> BranchLengths { get; }
    public int Total { get; }

    /// <summary>The previous token of each row in its own chain, or -1 at the start of the prompt.</summary>
    internal int[] Parent { get; }

    /// <summary>The RoPE position of each row: a branch continues where the trunk stops.</summary>
    internal int[] Position { get; }

    /// <summary>The first row of the branch a row belongs to, or -1 for a trunk row.</summary>
    internal int[] BranchStart { get; }

    /// <summary>The last row of each branch — where the answer is read — or of the trunk when there are none.</summary>
    public int[] Leaves { get; }

    private Lfm2Tree(int trunk, int[] branches)
    {
        TrunkLength = trunk;
        BranchLengths = branches;
        Total = trunk + branches.Sum();
        Parent = new int[Total];
        Position = new int[Total];
        BranchStart = new int[Total];
        for (int t = 0; t < trunk; ++t)
        {
            Parent[t] = t - 1;
            Position[t] = t;
            BranchStart[t] = -1;
        }

        Leaves = branches.Length == 0 ? [trunk - 1] : new int[branches.Length];
        int row = trunk;
        for (int b = 0; b < branches.Length; ++b)
        {
            if (branches[b] <= 0) throw new ArgumentException("every branch needs at least one token.", nameof(branches));
            for (int o = 0; o < branches[b]; ++o, ++row)
            {
                Parent[row] = o == 0 ? trunk - 1 : row - 1;
                Position[row] = trunk + o;
                BranchStart[row] = row - o;
            }
            Leaves[b] = row - 1;
        }
    }

    /// <summary>One causal chain of <paramref name="length"/> tokens.</summary>
    public static Lfm2Tree Chain(int length)
    {
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));
        return new Lfm2Tree(length, []);
    }

    /// <summary>A trunk of <paramref name="trunk"/> tokens (may be 0) continued by each of the branches.</summary>
    public static Lfm2Tree Branched(int trunk, IReadOnlyList<int> branches)
    {
        if (trunk < 0) throw new ArgumentOutOfRangeException(nameof(trunk));
        if (branches.Count == 0) return Chain(trunk);
        return new Lfm2Tree(trunk, [.. branches]);
    }

    public bool IsChain => BranchLengths.Count == 0;
}

/// <summary>Weights of one LFM2 decoder layer.</summary>
internal sealed class Lfm2LayerWeights
{
    public required Lfm2LayerKind Kind;
    public required float[] OperatorNorm;
    public required float[] FfnNorm;

    // conv: in_proj [3d, d] emitting B | C | x, the depthwise taps [d][taps], out_proj [d, d]
    public BFloat16Matrix? ConvIn;
    public float[]? ConvTaps;
    public float[]? ConvTapBias;
    public float[]? ConvInBias;
    public float[]? ConvOutBias;
    public BFloat16Matrix? ConvOut;

    // attention: q | k | v fused [d + 2 kv, d], per-head RMSNorm weights, out_proj [d, d]
    public BFloat16Matrix? Qkv;
    public float[]? QNorm;
    public float[]? KNorm;
    public BFloat16Matrix? AttnOut;

    // MLP: w1 | w3 fused [2 ff, d] emitting gate | up, and w2 [d, ff]
    public required BFloat16Matrix GateUp;
    public required BFloat16Matrix Down;
}

/// <summary>
/// The LFM2 language model — embeddings, the hybrid conv/attention stack, the final norm and the
/// (tied) LM head — forward only, in fp32 over bf16 weights.
///
/// <para>The structure is d1's <c>hybrid.py</c> line for line: each layer is
/// <c>h = x + op(rms(x)); h = h + mlp(rms(h))</c>, where <c>op</c> is either the gated short
/// convolution <c>out(C · conv(B · x))</c> with three causal taps, or GQA attention with per-head
/// q/k RMSNorm and half-split RoPE over the whole head. The final norm is called
/// <c>embedding_norm</c> in the checkpoint, despite where it is applied.</para>
/// </summary>
public sealed class Lfm2LanguageModel
{
    private readonly Lfm2Config _config;
    private readonly BFloat16Matrix? _untiedHead;
    private readonly ushort[] _embeddings;         // [vocab, hidden], bf16 bits
    private readonly float[] _finalNorm;
    private readonly Lfm2LayerWeights[] _layers;
    private readonly float[] _inverseFrequency;

    public Lfm2Config Config => _config;

    /// <summary>Bytes the weights occupy in managed memory.</summary>
    public long WeightBytes { get; }

    public Lfm2LanguageModel(Lfm2Config config, SafetensorsFile weights, string prefix = "model.language_model.",
        string headName = "lm_head.weight")
    {
        _config = config;
        int d = config.HiddenSize;
        int kv = config.KeyValueDim;
        int ff = config.IntermediateSize;

        _embeddings = weights.ReadBFloat16Bits(prefix + "embed_tokens.weight");
        if (_embeddings.Length != (long)config.VocabSize * d)
        {
            throw new InvalidDataException($"embed_tokens has {_embeddings.Length} values, expected {config.VocabSize} x {d}.");
        }
        if (!config.TieWordEmbeddings || weights.Contains(headName))
        {
            _untiedHead = weights.Contains(headName)
                ? new BFloat16Matrix(weights.ReadBFloat16Bits(headName), config.VocabSize, d)
                : throw new InvalidDataException($"the config unties the LM head but there is no '{headName}'.");
        }
        _finalNorm = weights.ReadFloat32(prefix + "embedding_norm.weight");

        _layers = new Lfm2LayerWeights[config.NumLayers];
        long bytes = (long)_embeddings.Length * 2 + (_untiedHead?.Bytes ?? 0);
        for (int i = 0; i < config.NumLayers; ++i)
        {
            string p = $"{prefix}layers.{i}.";
            var kind = config.LayerKinds[i];
            var layer = new Lfm2LayerWeights
            {
                Kind = kind,
                OperatorNorm = weights.ReadFloat32(p + "operator_norm.weight"),
                FfnNorm = weights.ReadFloat32(p + "ffn_norm.weight"),
                GateUp = new BFloat16Matrix(
                    Concat(weights.ReadBFloat16Bits(p + "feed_forward.w1.weight"), weights.ReadBFloat16Bits(p + "feed_forward.w3.weight")),
                    2 * ff, d),
                Down = new BFloat16Matrix(weights.ReadBFloat16Bits(p + "feed_forward.w2.weight"), d, ff),
            };
            if (kind == Lfm2LayerKind.Conv)
            {
                layer.ConvIn = new BFloat16Matrix(weights.ReadBFloat16Bits(p + "conv.in_proj.weight"), 3 * d, d);
                layer.ConvOut = new BFloat16Matrix(weights.ReadBFloat16Bits(p + "conv.out_proj.weight"), d, d);
                layer.ConvTaps = weights.ReadFloat32(p + "conv.conv.weight");   // [d, 1, taps]
                if (layer.ConvTaps.Length != d * config.ConvTaps)
                {
                    throw new InvalidDataException($"{p}conv.conv.weight: expected [{d}, 1, {config.ConvTaps}].");
                }
                if (config.ConvBias)
                {
                    layer.ConvTapBias = weights.ReadFloat32(p + "conv.conv.bias");
                    layer.ConvInBias = weights.ReadFloat32(p + "conv.in_proj.bias");
                    layer.ConvOutBias = weights.ReadFloat32(p + "conv.out_proj.bias");
                }
                bytes += layer.ConvIn.Bytes + layer.ConvOut.Bytes;
            }
            else
            {
                layer.Qkv = new BFloat16Matrix(
                    Concat(weights.ReadBFloat16Bits(p + "self_attn.q_proj.weight"),
                        weights.ReadBFloat16Bits(p + "self_attn.k_proj.weight"),
                        weights.ReadBFloat16Bits(p + "self_attn.v_proj.weight")),
                    d + 2 * kv, d);
                layer.QNorm = weights.ReadFloat32(p + "self_attn.q_layernorm.weight");
                layer.KNorm = weights.ReadFloat32(p + "self_attn.k_layernorm.weight");
                layer.AttnOut = new BFloat16Matrix(weights.ReadBFloat16Bits(p + "self_attn.out_proj.weight"), d, d);
                bytes += layer.Qkv.Bytes + layer.AttnOut.Bytes;
            }
            bytes += layer.GateUp.Bytes + layer.Down.Bytes;
            _layers[i] = layer;
        }
        WeightBytes = bytes;

        // RoPE frequencies exactly as hybrid.rope computes them: fp32 pow, fp32 reciprocal.
        int headDim = config.HeadDim;
        _inverseFrequency = new float[headDim / 2];
        for (int i = 0; i < headDim / 2; ++i)
        {
            float exponent = 2 * i / (float)headDim;
            _inverseFrequency[i] = 1f / MathF.Pow((float)config.RopeTheta, exponent);
        }
    }

    private static ushort[] Concat(params ushort[][] parts)
    {
        var result = new ushort[parts.Sum(p => (long)p.Length)];
        long offset = 0;
        foreach (var part in parts)
        {
            part.CopyTo(result, offset);
            offset += part.Length;
        }
        return result;
    }

    /// <summary>
    /// Runs the stack over <paramref name="tokenIds"/> laid out as <paramref name="tree"/> and returns
    /// the final-normed hidden states, row-major <c>[tree.Total, hidden]</c>.
    /// </summary>
    public float[] Forward(ReadOnlySpan<int> tokenIds, Lfm2Tree tree, IStateRecorder? recorder = null,
        ParallelOptions? parallel = null)
    {
        if (tokenIds.Length != tree.Total) throw new ArgumentException("the tree does not cover the tokens.", nameof(tree));
        int tokens = tree.Total;
        int d = _config.HiddenSize;

        var hidden = new float[tokens * d];
        using (ForwardTiming.Measure("lfm2.embeddings"))
        {
            for (int t = 0; t < tokens; ++t)
            {
                int id = tokenIds[t];
                if ((uint)id >= (uint)_config.VocabSize)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"token id {id} is outside the vocabulary.");
                }
                Widen(_embeddings.AsSpan(id * d, d), hidden.AsSpan(t * d, d));
            }
        }
        recorder?.Record("embeddings", hidden, [tokens, d]);
        return Run(hidden, tree, recorder, parallel);
    }

    /// <summary>
    /// The stack over embeddings already in hand — the text embeddings with image features scattered
    /// into their slots, for a multimodal prompt. <paramref name="hidden"/> is overwritten.
    /// </summary>
    public float[] Run(float[] hidden, Lfm2Tree tree, IStateRecorder? recorder = null, ParallelOptions? parallel = null)
    {
        int tokens = tree.Total;
        int d = _config.HiddenSize;
        int ff = _config.IntermediateSize;
        int qkvWidth = d + 2 * _config.KeyValueDim;

        using var scratch = new ScratchBuffers();
        var normed = scratch.Rent(tokens * d);
        var projected = scratch.Rent(tokens * d);
        var mixed = scratch.Rent(tokens * d);
        var wide = scratch.Rent(tokens * Math.Max(2 * ff, Math.Max(3 * d, qkvWidth)));
        var activated = scratch.Rent(tokens * ff);

        for (int index = 0; index < _layers.Length; ++index)
        {
            var layer = _layers[index];
            bool detailed = recorder is not null && IsProbed(index);

            using (ForwardTiming.Measure("lfm2.norm")) RmsNormRows(hidden, layer.OperatorNorm, normed, tokens, d);
            if (detailed) recorder!.Record($"layers.{index}.operator_norm", normed.AsSpan(0, tokens * d), [tokens, d]);

            if (layer.Kind == Lfm2LayerKind.Conv)
            {
                using (ForwardTiming.Measure("lfm2.conv_in")) layer.ConvIn!.Multiply(normed.AsSpan(0, tokens * d), tokens, wide, parallel);
                if (layer.ConvInBias is not null) AddBias(wide, layer.ConvInBias, tokens);
                if (detailed) recorder!.Record($"layers.{index}.conv.in_proj", wide.AsSpan(0, tokens * 3 * d), [tokens, 3 * d]);
                using (ForwardTiming.Measure("lfm2.conv")) ShortConv(wide, tree, layer, mixed);
                using (ForwardTiming.Measure("lfm2.conv_out")) layer.ConvOut!.Multiply(mixed.AsSpan(0, tokens * d), tokens, projected, parallel);
                if (layer.ConvOutBias is not null) AddBias(projected, layer.ConvOutBias, tokens);
            }
            else
            {
                using (ForwardTiming.Measure("lfm2.qkv")) layer.Qkv!.Multiply(normed.AsSpan(0, tokens * d), tokens, wide, parallel);
                using (ForwardTiming.Measure("lfm2.qk_norm_rope")) NormalizeAndRotate(wide, tree, layer, detailed ? recorder : null, index);
                using (ForwardTiming.Measure("lfm2.attention")) Attention(wide, tree, mixed, parallel);
                using (ForwardTiming.Measure("lfm2.attn_out")) layer.AttnOut!.Multiply(mixed.AsSpan(0, tokens * d), tokens, projected, parallel);
            }
            if (detailed) recorder!.Record($"layers.{index}.operator", projected.AsSpan(0, tokens * d), [tokens, d]);
            using (ForwardTiming.Measure("lfm2.residual")) SimdOps.Add(hidden.AsSpan(0, tokens * d), projected.AsSpan(0, tokens * d));

            using (ForwardTiming.Measure("lfm2.norm")) RmsNormRows(hidden, layer.FfnNorm, normed, tokens, d);
            if (detailed) recorder!.Record($"layers.{index}.ffn_norm", normed.AsSpan(0, tokens * d), [tokens, d]);
            using (ForwardTiming.Measure("lfm2.mlp_in")) layer.GateUp.Multiply(normed.AsSpan(0, tokens * d), tokens, wide, parallel);
            using (ForwardTiming.Measure("lfm2.swiglu")) SwiGlu(wide, tokens, ff, activated);
            using (ForwardTiming.Measure("lfm2.mlp_out")) layer.Down.Multiply(activated.AsSpan(0, tokens * ff), tokens, projected, parallel);
            if (detailed) recorder!.Record($"layers.{index}.feed_forward", projected.AsSpan(0, tokens * d), [tokens, d]);
            using (ForwardTiming.Measure("lfm2.residual")) SimdOps.Add(hidden.AsSpan(0, tokens * d), projected.AsSpan(0, tokens * d));
            recorder?.Record($"layers.{index}.output", hidden.AsSpan(0, tokens * d), [tokens, d]);
        }

        using (ForwardTiming.Measure("lfm2.norm")) RmsNormRows(hidden, _finalNorm, hidden, tokens, d);
        recorder?.Record("final_norm", hidden.AsSpan(0, tokens * d), [tokens, d]);
        return hidden;
    }

    /// <summary>The layers whose internals the parity dump records: the first conv and the first attention layer.</summary>
    private bool IsProbed(int index)
        => index == Array.FindIndex(_layers, l => l.Kind == Lfm2LayerKind.Conv)
        || index == Array.FindIndex(_layers, l => l.Kind == Lfm2LayerKind.Attention);

    /// <summary>The LM-head logit of one token for one final hidden state.</summary>
    public float Logit(ReadOnlySpan<float> hiddenRow, int tokenId)
    {
        int d = _config.HiddenSize;
        Span<float> row = d <= 8192 ? stackalloc float[d] : new float[d];
        HeadRow(tokenId, row);
        return SimdOps.Dot(hiddenRow, row);
    }

    /// <summary>The LM-head logits of every vocabulary entry for one final hidden state.</summary>
    public void Logits(ReadOnlySpan<float> hiddenRow, Span<float> destination, ParallelOptions? parallel = null)
    {
        int d = _config.HiddenSize;
        if (_untiedHead is not null)
        {
            _untiedHead.Multiply(hiddenRow, 1, destination, parallel);
            return;
        }
        float[] row = ArrayPool<float>.Shared.Rent(d);
        try
        {
            for (int v = 0; v < _config.VocabSize; ++v)
            {
                Widen(_embeddings.AsSpan(v * d, d), row.AsSpan(0, d));
                destination[v] = SimdOps.Dot(hiddenRow, row.AsSpan(0, d));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(row);
        }
    }

    private void HeadRow(int tokenId, Span<float> destination)
    {
        int d = _config.HiddenSize;
        if ((uint)tokenId >= (uint)_config.VocabSize) throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (_untiedHead is not null) _untiedHead.ReadRow(tokenId, destination);
        else Widen(_embeddings.AsSpan(tokenId * d, d), destination);
    }

    /// <summary>The embedding row of a token, widened to fp32.</summary>
    public void Embedding(int tokenId, Span<float> destination)
    {
        int d = _config.HiddenSize;
        if ((uint)tokenId >= (uint)_config.VocabSize) throw new ArgumentOutOfRangeException(nameof(tokenId));
        Widen(_embeddings.AsSpan(tokenId * d, d), destination);
    }

    private static void Widen(ReadOnlySpan<ushort> source, Span<float> destination)
    {
        var bits = MemoryMarshal.Cast<float, uint>(destination);
        for (int i = 0; i < source.Length; ++i) bits[i] = (uint)source[i] << 16;
    }

    private void RmsNormRows(float[] source, float[] weight, float[] destination, int tokens, int width)
    {
        for (int t = 0; t < tokens; ++t)
        {
            RmsNorm(source.AsSpan(t * width, width), weight, _config.NormEps, destination.AsSpan(t * width, width));
        }
    }

    /// <summary><c>x · rsqrt(mean(x²) + eps) · w</c>, as hybrid.RMSNorm computes it in fp32.</summary>
    internal static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, float epsilon, Span<float> output)
    {
        int n = input.Length;
        int width = Vector<float>.Count;
        var accumulator = Vector<float>.Zero;
        int i = 0;
        for (; i <= n - width; i += width)
        {
            var v = Vector.LoadUnsafe(in input[i]);
            accumulator = Vector.FusedMultiplyAdd(v, v, accumulator);
        }
        float sum = Vector.Sum(accumulator);
        for (; i < n; ++i) sum += input[i] * input[i];

        float scale = 1f / MathF.Sqrt(sum / n + epsilon);
        var scaleVector = new Vector<float>(scale);
        i = 0;
        for (; i <= n - width; i += width)
        {
            (Vector.LoadUnsafe(in input[i]) * scaleVector * Vector.LoadUnsafe(in weight[i])).StoreUnsafe(ref output[i]);
        }
        for (; i < n; ++i) output[i] = input[i] * scale * weight[i];
    }

    private void AddBias(float[] rows, float[] bias, int tokens)
    {
        for (int t = 0; t < tokens; ++t) SimdOps.Add(rows.AsSpan(t * bias.Length, bias.Length), bias);
    }

    /// <summary>
    /// The gated short convolution's middle: <c>y = C · conv(B · x)</c>, where <c>in_proj</c> emitted
    /// <c>[B | C | x]</c> per token and the depthwise convolution is causal over the token's own
    /// chain: tap <c>k</c> reads the input <c>taps - 1 - k</c> steps back, following
    /// <see cref="Lfm2Tree.Parent"/>, so a branch's first tokens read the trunk's last ones.
    /// </summary>
    private void ShortConv(float[] bcx, Lfm2Tree tree, Lfm2LayerWeights layer, float[] destination)
    {
        int d = _config.HiddenSize;
        int taps = _config.ConvTaps;
        int tokens = tree.Total;
        float[] gatedRows = ArrayPool<float>.Shared.Rent(tokens * d);   // B · x per token
        float[] tapsTransposed = ArrayPool<float>.Shared.Rent(taps * d); // [taps][d]
        try
        {
            for (int c = 0; c < d; ++c)
            {
                for (int k = 0; k < taps; ++k) tapsTransposed[k * d + c] = layer.ConvTaps![c * taps + k];
            }
            for (int t = 0; t < tokens; ++t)
            {
                var row = bcx.AsSpan(t * 3 * d, 3 * d);
                var gated = gatedRows.AsSpan(t * d, d);
                row[..d].CopyTo(gated);
                SimdOps.Multiply(gated, row.Slice(2 * d, d));
            }

            Span<int> sources = stackalloc int[taps];
            int width = Vector<float>.Count;
            for (int t = 0; t < tokens; ++t)
            {
                // sources[k] is the row tap k reads: the newest tap is the token itself.
                int current = t;
                for (int k = taps - 1; k >= 0; --k)
                {
                    sources[k] = current;
                    current = current < 0 ? -1 : tree.Parent[current];
                }

                var output = destination.AsSpan(t * d, d);
                if (layer.ConvTapBias is not null) layer.ConvTapBias.CopyTo(output);
                else output.Clear();

                // Accumulate oldest tap first, the order F.conv1d sums its window in.
                for (int k = 0; k < taps; ++k)
                {
                    if (sources[k] < 0) continue;
                    var input = gatedRows.AsSpan(sources[k] * d, d);
                    var weight = tapsTransposed.AsSpan(k * d, d);
                    int c = 0;
                    for (; c <= d - width; c += width)
                    {
                        var acc = Vector.LoadUnsafe(ref output[c]);
                        acc = Vector.FusedMultiplyAdd(Vector.LoadUnsafe(in weight[c]), Vector.LoadUnsafe(in input[c]), acc);
                        acc.StoreUnsafe(ref output[c]);
                    }
                    for (; c < d; ++c) output[c] = MathF.FusedMultiplyAdd(weight[c], input[c], output[c]);
                }

                SimdOps.Multiply(output, bcx.AsSpan(t * 3 * d + d, d));   // · C
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gatedRows);
            ArrayPool<float>.Shared.Return(tapsTransposed);
        }
    }

    /// <summary>Per-head RMSNorm of q and k, then RoPE at each row's tree position, in place.</summary>
    private void NormalizeAndRotate(float[] qkv, Lfm2Tree tree, Lfm2LayerWeights layer, IStateRecorder? recorder, int index)
    {
        int d = _config.HiddenSize;
        int headDim = _config.HeadDim;
        int heads = _config.NumAttentionHeads;
        int kvHeads = _config.NumKeyValueHeads;
        int stride = d + 2 * _config.KeyValueDim;
        int tokens = tree.Total;
        int half = headDim / 2;
        Span<float> cos = stackalloc float[half];
        Span<float> sin = stackalloc float[half];

        for (int t = 0; t < tokens; ++t)
        {
            var row = qkv.AsSpan(t * stride, stride);
            for (int h = 0; h < heads; ++h)
            {
                var head = row.Slice(h * headDim, headDim);
                RmsNorm(head, layer.QNorm, _config.NormEps, head);
            }
            for (int h = 0; h < kvHeads; ++h)
            {
                var head = row.Slice(d + h * headDim, headDim);
                RmsNorm(head, layer.KNorm, _config.NormEps, head);
            }
        }

        if (recorder is not null)
        {
            var q = new float[tokens * d];
            var k = new float[tokens * _config.KeyValueDim];
            var v = new float[tokens * _config.KeyValueDim];
            for (int t = 0; t < tokens; ++t)
            {
                qkv.AsSpan(t * stride, d).CopyTo(q.AsSpan(t * d, d));
                qkv.AsSpan(t * stride + d, _config.KeyValueDim).CopyTo(k.AsSpan(t * _config.KeyValueDim));
                qkv.AsSpan(t * stride + d + _config.KeyValueDim, _config.KeyValueDim).CopyTo(v.AsSpan(t * _config.KeyValueDim));
            }
            recorder.Record($"layers.{index}.attn.q_norm", q, [tokens, heads, headDim]);
            recorder.Record($"layers.{index}.attn.k_norm", k, [tokens, kvHeads, headDim]);
            recorder.Record($"layers.{index}.attn.v", v, [tokens, _config.KeyValueDim]);
        }

        for (int t = 0; t < tokens; ++t)
        {
            // angle = float(position) * inv_freq, both fp32, as torch computes it.
            float position = tree.Position[t];
            for (int i = 0; i < half; ++i)
            {
                float angle = position * _inverseFrequency[i];
                cos[i] = MathF.Cos(angle);
                sin[i] = MathF.Sin(angle);
            }
            var row = qkv.AsSpan(t * stride, stride);
            for (int h = 0; h < heads; ++h) Rotate(row.Slice(h * headDim, headDim), cos, sin);
            for (int h = 0; h < kvHeads; ++h) Rotate(row.Slice(d + h * headDim, headDim), cos, sin);
        }
    }

    /// <summary>Half-split RoPE: <c>x·cos + [-b, a]·sin</c> over the halves <c>a | b</c> of the head.</summary>
    private static void Rotate(Span<float> head, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        int half = cos.Length;
        for (int i = 0; i < half; ++i)
        {
            float a = head[i];
            float b = head[i + half];
            head[i] = a * cos[i] - b * sin[i];
            head[i + half] = b * cos[i] + a * sin[i];
        }
    }

    /// <summary>
    /// Causal GQA attention over the tree: a trunk row sees the trunk up to itself; a branch row
    /// sees the whole trunk and its own branch up to itself. Query head <c>h</c> reads key/value head
    /// <c>h / (heads / kvHeads)</c>, as <c>repeat_interleave</c> arranges.
    ///
    /// <para>The keys of one kv head are gathered once, transposed, and shared by its query group;
    /// adjacent keys sit in adjacent lanes, so a score vector finishes with a store rather than a
    /// horizontal reduction.</para>
    /// </summary>
    private unsafe void Attention(float[] qkv, Lfm2Tree tree, float[] destination, ParallelOptions? parallel)
    {
        int d = _config.HiddenSize;
        int headDim = _config.HeadDim;
        int heads = _config.NumAttentionHeads;
        int kvHeads = _config.NumKeyValueHeads;
        int group = heads / kvHeads;
        int kvDim = _config.KeyValueDim;
        int stride = d + 2 * kvDim;
        int tokens = tree.Total;
        int trunk = tree.TrunkLength;
        float scale = 1f / MathF.Sqrt(headDim);
        int keyStride = AttentionKernels.PaddedKeyStride(tokens) + AttentionKernels.PaddedKeyStride(1);
        int[] branchStart = tree.BranchStart;

        void Unit(int kvHead)
        {
            int keyFloats = headDim * keyStride;
            int valueFloats = tokens * headDim;
            float[] rented = ArrayPool<float>.Shared.Rent(keyFloats + valueFloats + keyStride + headDim);
            try
            {
                fixed (float* source = qkv, output = destination, scratch = rented)
                {
                    float* keysTransposed = scratch;
                    float* values = keysTransposed + keyFloats;
                    float* scores = values + valueFloats;
                    float* partial = scores + keyStride;
                    new Span<float>(keysTransposed, keyFloats).Clear();

                    for (int t = 0; t < tokens; ++t)
                    {
                        float* row = source + (long)t * stride;
                        float* key = row + d + kvHead * headDim;
                        for (int j = 0; j < headDim; ++j) keysTransposed[j * keyStride + t] = key[j];
                        Buffer.MemoryCopy(row + d + kvDim + kvHead * headDim, values + t * headDim, headDim * 4, headDim * 4);
                    }

                    for (int h = kvHead * group; h < (kvHead + 1) * group; ++h)
                    {
                        for (int t = 0; t < tokens; ++t)
                        {
                            float* query = source + (long)t * stride + h * headDim;
                            float* result = output + (long)t * d + h * headDim;
                            int start = branchStart[t];
                            // The trunk part: [0, t] for a trunk row, the whole trunk for a branch row.
                            int first = start < 0 ? t + 1 : trunk;
                            int own = start < 0 ? 0 : t - start + 1;
                            int count = first + own;

                            if (first > 0)
                            {
                                AttentionKernels.Scores(query, keysTransposed, keyStride, headDim, first, scale, scores);
                            }
                            if (own > 0)
                            {
                                AttentionKernels.Scores(query, keysTransposed + start, keyStride, headDim, own, scale, scores + first);
                            }
                            SimdOps.Softmax(new Span<float>(scores, count));

                            if (first > 0)
                            {
                                AttentionKernels.WeightedSum(values, scores, first, headDim, result);
                            }
                            if (own > 0)
                            {
                                if (first > 0)
                                {
                                    AttentionKernels.WeightedSum(values + (long)start * headDim, scores + first, own, headDim, partial);
                                    for (int j = 0; j < headDim; ++j) result[j] += partial[j];
                                }
                                else
                                {
                                    AttentionKernels.WeightedSum(values + (long)start * headDim, scores, own, headDim, result);
                                }
                            }
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        if (LayaRuntime.WorkersOf(parallel) <= 1)
        {
            for (int g = 0; g < kvHeads; ++g) Unit(g);
        }
        else
        {
            Parallel.For(0, kvHeads, LayaRuntime.Resolve(parallel), Unit);
        }
    }

    /// <summary><c>silu(gate) · up</c> over rows of <c>[gate | up]</c>.</summary>
    private static void SwiGlu(float[] gateUp, int tokens, int ff, float[] destination)
    {
        int width = Vector<float>.Count;
        for (int t = 0; t < tokens; ++t)
        {
            var gate = gateUp.AsSpan(t * 2 * ff, ff);
            var up = gateUp.AsSpan(t * 2 * ff + ff, ff);
            var output = destination.AsSpan(t * ff, ff);
            int i = 0;
            for (; i <= ff - width; i += width)
            {
                var g = Vector.LoadUnsafe(in gate[i]);
                var silu = g / (Vector<float>.One + SimdOps.Exp(-g));
                (silu * Vector.LoadUnsafe(in up[i])).StoreUnsafe(ref output[i]);
            }
            for (; i < ff; ++i)
            {
                float g = gate[i];
                output[i] = g / (1f + MathF.Exp(-g)) * up[i];
            }
        }
    }
}

using System.Buffers;
using Laya.Diagnostics;
using Laya.Io;
using Laya.Numerics;

namespace Laya.Models;

/// <summary>Weights of one ModernBERT layer, with the projections already repacked.</summary>
internal sealed class ModernBertLayerWeights
{
    public float[]? AttnNormWeight;           // null on layer 0, where HF uses nn.Identity
    public float[]? AttnNormBias;
    public required PackedMatrix Wqkv;        // [3 * hidden, hidden]
    public float[] WqkvBias = [];
    public required PackedMatrix AttnWo;      // [hidden, hidden]
    public float[] AttnWoBias = [];
    public required float[] MlpNormWeight;
    public float[] MlpNormBias = [];
    public required PackedMatrix MlpWi;       // [2 * intermediate, hidden]
    public float[] MlpWiBias = [];
    public required PackedMatrix MlpWo;       // [hidden, intermediate]
    public float[] MlpWoBias = [];
    public required AttentionKind Kind;
}

/// <summary>
/// One sequence inside a batch: where its tokens start and how many there are.
/// </summary>
public readonly record struct Segment(int Start, int Length)
{
    public int End => Start + Length;
}

/// <summary>
/// The ModernBERT encoder, forward pass only.
///
/// <para>Sequences are concatenated rather than padded into a rectangle. Every projection then sees
/// one tall matrix, so the 421M parameters are streamed from memory once for the whole batch
/// instead of once per question — which is the difference between being weight-bandwidth bound and
/// being compute bound. Attention is the only operation that mixes tokens, and it runs per
/// segment, which is exactly what PyTorch's padding mask achieves: a token never attends outside
/// its own sequence.</para>
/// </summary>
public sealed class ModernBertEncoder
{
    private readonly ModernBertConfig _config;
    private readonly float[] _tokenEmbeddings;   // [vocab, hidden]
    private readonly float[] _embeddingNormWeight;
    private readonly float[] _embeddingNormBias;
    private readonly float[] _finalNormWeight;
    private readonly float[] _finalNormBias;
    private readonly ModernBertLayerWeights[] _layers;
    private readonly RopeCache _globalRope;
    private readonly RopeCache _localRope;

    public ModernBertConfig Config => _config;

    /// <summary>Bytes the embeddings and repacked projections occupy in managed memory.</summary>
    public long WeightBytes { get; private set; }

    public ModernBertEncoder(ModernBertConfig config, SafetensorsFile weights, string prefix = "encoder.")
    {
        _config = config;
        _tokenEmbeddings = weights.ReadFloat32(prefix + "embeddings.tok_embeddings.weight");
        _embeddingNormWeight = weights.ReadFloat32(prefix + "embeddings.norm.weight");
        _embeddingNormBias = Optional(weights, prefix + "embeddings.norm.bias");
        _finalNormWeight = weights.ReadFloat32(prefix + "final_norm.weight");
        _finalNormBias = Optional(weights, prefix + "final_norm.bias");

        _layers = new ModernBertLayerWeights[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; ++i)
        {
            string p = $"{prefix}layers.{i}.";
            _layers[i] = new ModernBertLayerWeights
            {
                // Layer 0's attn_norm is nn.Identity in HF, so the checkpoint has no tensor for it.
                AttnNormWeight = weights.Contains(p + "attn_norm.weight") ? weights.ReadFloat32(p + "attn_norm.weight") : null,
                AttnNormBias = weights.Contains(p + "attn_norm.bias") ? weights.ReadFloat32(p + "attn_norm.bias") : null,
                Wqkv = Pack(weights, p + "attn.Wqkv.weight", 3 * config.HiddenSize, config.HiddenSize),
                WqkvBias = Optional(weights, p + "attn.Wqkv.bias"),
                AttnWo = Pack(weights, p + "attn.Wo.weight", config.HiddenSize, config.HiddenSize),
                AttnWoBias = Optional(weights, p + "attn.Wo.bias"),
                MlpNormWeight = weights.ReadFloat32(p + "mlp_norm.weight"),
                MlpNormBias = Optional(weights, p + "mlp_norm.bias"),
                MlpWi = Pack(weights, p + "mlp.Wi.weight", 2 * config.IntermediateSize, config.HiddenSize),
                MlpWiBias = Optional(weights, p + "mlp.Wi.bias"),
                MlpWo = Pack(weights, p + "mlp.Wo.weight", config.HiddenSize, config.IntermediateSize),
                MlpWoBias = Optional(weights, p + "mlp.Wo.bias"),
                Kind = config.LayerKinds.Length > i
                    ? config.LayerKinds[i]
                    : (i % config.GlobalAttentionEveryNLayers == 0 ? AttentionKind.Global : AttentionKind.Sliding),
            };
        }

        _globalRope = new RopeCache(config.HeadDim, config.GlobalRopeTheta);
        _localRope = new RopeCache(config.HeadDim, config.LocalRopeTheta);

        WeightBytes = (long)sizeof(float) * (_tokenEmbeddings.Length + _embeddingNormWeight.Length
                + _embeddingNormBias.Length + _finalNormWeight.Length + _finalNormBias.Length)
            + _layers.Sum(l => l.Wqkv.Bytes + l.AttnWo.Bytes + l.MlpWi.Bytes + l.MlpWo.Bytes);
    }

    private static float[] Optional(SafetensorsFile weights, string name)
        => weights.Contains(name) ? weights.ReadFloat32(name) : [];

    /// <summary>Reads a <c>[out, in]</c> weight and repacks it for the broadcast GEMM kernel.</summary>
    internal static PackedMatrix Pack(SafetensorsFile weights, string name, int outFeatures, int inFeatures)
        => new(weights.ReadFloat32(name), outFeatures, inFeatures);

    /// <summary>Runs the encoder over one sequence.</summary>
    public float[] Forward(ReadOnlySpan<int> tokenIds, IStateRecorder? recorder = null, ParallelOptions? parallel = null)
        => Forward(tokenIds, [new Segment(0, tokenIds.Length)], recorder is null ? null : new BatchRecorder(recorder, [""]), parallel);

    /// <summary>
    /// Runs the encoder over a batch of concatenated sequences and returns the final hidden states,
    /// row-major <c>[totalTokens, hidden]</c>.
    /// </summary>
    /// <param name="parallel">
    /// The threads every kernel of this pass may use; null means <see cref="LayaRuntime.ParallelOptions"/>.
    /// Pin it below the core count to run several passes side by side without oversubscribing.
    /// </param>
    public float[] Forward(ReadOnlySpan<int> tokenIds, IReadOnlyList<Segment> segments, BatchRecorder? recorder,
        ParallelOptions? parallel = null)
    {
        int tokens = tokenIds.Length;
        int hidden = _config.HiddenSize;
        int intermediate = _config.IntermediateSize;

        var hiddenStates = new float[tokens * hidden];
        using (ForwardTiming.Measure("embeddings"))
        {
        for (int t = 0; t < tokens; ++t)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)_config.VocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"token id {id} is outside the vocabulary.");
            }
            _tokenEmbeddings.AsSpan(id * hidden, hidden).CopyTo(hiddenStates.AsSpan(t * hidden, hidden));
        }
            NormalizeInto(hiddenStates, hiddenStates, tokens, hidden, _embeddingNormWeight, _embeddingNormBias);
        }
        recorder?.Record("encoder.embeddings", hiddenStates, segments, hidden);

        var units = AttentionUnits(segments);
        using var scratch = new ScratchBuffers();
        var normed = scratch.Rent(tokens * hidden);
        var qkv = scratch.Rent(tokens * 3 * hidden);
        var attention = scratch.Rent(tokens * hidden);
        var projected = scratch.Rent(tokens * hidden);
        var mlpHidden = scratch.Rent(tokens * 2 * intermediate);
        var activated = scratch.Rent(tokens * intermediate);

        for (int layerIndex = 0; layerIndex < _layers.Length; ++layerIndex)
        {
            var layer = _layers[layerIndex];

            using (ForwardTiming.Measure("encoder.norm"))
            {
                if (layer.AttnNormWeight is null)
                {
                    hiddenStates.AsSpan(0, tokens * hidden).CopyTo(normed.AsSpan(0, tokens * hidden));
                }
                else
                {
                    NormalizeInto(hiddenStates, normed, tokens, hidden, layer.AttnNormWeight, layer.AttnNormBias ?? []);
                }
            }

            using (ForwardTiming.Measure("encoder.qkv")) layer.Wqkv.Multiply(normed.AsSpan(0, tokens * hidden), tokens, layer.WqkvBias, qkv, parallel);
            using (ForwardTiming.Measure("encoder.attention")) Attention(qkv, segments, layer.Kind, attention, units, parallel);
            using (ForwardTiming.Measure("encoder.attn_out")) layer.AttnWo.Multiply(attention.AsSpan(0, tokens * hidden), tokens, layer.AttnWoBias, projected, parallel);
            using (ForwardTiming.Measure("encoder.residual")) SimdOps.Add(hiddenStates.AsSpan(0, tokens * hidden), projected.AsSpan(0, tokens * hidden));
            recorder?.Record($"encoder.layers.{layerIndex}.attn_residual", hiddenStates, segments, hidden);

            using (ForwardTiming.Measure("encoder.norm")) NormalizeInto(hiddenStates, normed, tokens, hidden, layer.MlpNormWeight, layer.MlpNormBias);
            using (ForwardTiming.Measure("encoder.mlp_in")) layer.MlpWi.Multiply(normed.AsSpan(0, tokens * hidden), tokens, layer.MlpWiBias, mlpHidden, parallel);
            using (ForwardTiming.Measure("encoder.geglu")) GeGlu(mlpHidden, tokens, intermediate, activated);
            using (ForwardTiming.Measure("encoder.mlp_out")) layer.MlpWo.Multiply(activated.AsSpan(0, tokens * intermediate), tokens, layer.MlpWoBias, projected, parallel);
            using (ForwardTiming.Measure("encoder.residual")) SimdOps.Add(hiddenStates.AsSpan(0, tokens * hidden), projected.AsSpan(0, tokens * hidden));
            recorder?.Record($"encoder.layers.{layerIndex}.output", hiddenStates, segments, hidden);
        }

        NormalizeInto(hiddenStates, hiddenStates, tokens, hidden, _finalNormWeight, _finalNormBias);
        recorder?.Record("encoder.last_hidden_state", hiddenStates, segments, hidden);
        return hiddenStates;
    }

    private void NormalizeInto(float[] source, float[] destination, int tokens, int hidden, float[] weight, float[] bias)
    {
        for (int t = 0; t < tokens; ++t)
        {
            SimdOps.LayerNorm(source.AsSpan(t * hidden, hidden), weight, bias, _config.NormEps,
                destination.AsSpan(t * hidden, hidden));
        }
    }

    /// <summary>
    /// GeGLU: <c>Wi</c> emits <c>[input | gate]</c> and the activation is applied to the first half
    /// only. Getting the halves the wrong way round still produces plausible-looking numbers, so
    /// this order is pinned by the parity dumps.
    /// </summary>
    private static void GeGlu(float[] wide, int tokens, int intermediate, float[] destination)
    {
        for (int t = 0; t < tokens; ++t)
        {
            var row = wide.AsSpan(t * 2 * intermediate, 2 * intermediate);
            var output = destination.AsSpan(t * intermediate, intermediate);
            row[..intermediate].CopyTo(output);
            SimdOps.Gelu(output);
            SimdOps.Multiply(output, row.Slice(intermediate, intermediate));
        }
    }

    /// <summary>
    /// Multi-head attention over each segment independently.
    ///
    /// <para>Scratch space comes from the array pool rather than <c>new</c>: this runs 28 times per
    /// call for every (segment, head) pair, and allocating the rotated query and key per unit was
    /// costing well over a hundred megabytes of garbage per forward pass.</para>
    ///
    /// <para>The query, key and value for a head are gathered into contiguous scratch first — they
    /// are otherwise strided by the packed QKV layout — and the keys are written out
    /// <em>transposed</em>, so <see cref="AttentionKernels"/> can keep adjacent keys in SIMD lanes
    /// and never reduce a vector to a scalar. A forward pass has about 14 million query-key pairs;
    /// a horizontal sum on each of them is the difference between attention being a fifth of the
    /// time and a twentieth.</para>
    /// </summary>
    private unsafe void Attention(float[] qkv, IReadOnlyList<Segment> segments, AttentionKind kind,
        float[] destination, (Segment Segment, int Head)[] units, ParallelOptions? parallel)
    {
        int hidden = _config.HiddenSize;
        int headDim = _config.HeadDim;
        int stride = 3 * hidden;
        float scale = 1f / MathF.Sqrt(headDim);
        int window = kind == AttentionKind.Sliding ? _config.SlidingHalfWindow : int.MaxValue;
        var rope = kind == AttentionKind.Global ? _globalRope : _localRope;

        Parallel.For(0, units.Length, LayaRuntime.Resolve(parallel), unit =>
        {
            var (segment, head) = units[unit];
            int length = segment.Length;
            int keyStride = AttentionKernels.PaddedKeyStride(length);
            int qBase = head * headDim;
            int kBase = hidden + head * headDim;
            int vBase = 2 * hidden + head * headDim;

            // query [length][headDim] | keysT [headDim][keyStride] | values [length][headDim] | scores
            int queryFloats = length * headDim;
            int keyFloats = headDim * keyStride;
            float[] rented = ArrayPool<float>.Shared.Rent(2 * queryFloats + keyFloats + keyStride);
            try
            {
                fixed (float* source = qkv, output = destination, scratch = rented)
                {
                    float* query = scratch;
                    float* keysTransposed = query + queryFloats;
                    float* values = keysTransposed + keyFloats;
                    float* scores = values + queryFloats;

                    // Padding lanes must be zero: they produce scores the softmax never reads, but
                    // NaNs there would still poison the multiply.
                    new Span<float>(keysTransposed, keyFloats).Clear();

                    float* rotated = stackalloc float[headDim];
                    for (int t = 0; t < length; ++t)
                    {
                        float* row = source + (long)(segment.Start + t) * stride;
                        float* q = query + t * headDim;
                        Buffer.MemoryCopy(row + qBase, q, headDim * 4, headDim * 4);
                        Buffer.MemoryCopy(row + vBase, values + t * headDim, headDim * 4, headDim * 4);

                        // Positions restart at 0 for every sequence, as they do in a padded batch.
                        rope.Apply(new Span<float>(q, headDim), t);

                        Buffer.MemoryCopy(row + kBase, rotated, headDim * 4, headDim * 4);
                        rope.Apply(new Span<float>(rotated, headDim), t);
                        for (int d = 0; d < headDim; ++d) keysTransposed[d * keyStride + t] = rotated[d];
                    }

                    for (int t = 0; t < length; ++t)
                    {
                        int first = window == int.MaxValue ? 0 : Math.Max(0, t - window);
                        int last = window == int.MaxValue ? length - 1 : Math.Min(length - 1, t + window);
                        int count = last - first + 1;

                        AttentionKernels.Scores(query + t * headDim, keysTransposed + first, keyStride,
                            headDim, count, scale, scores);
                        SimdOps.Softmax(new Span<float>(scores, count));

                        AttentionKernels.WeightedSum(values + (long)first * headDim, scores, count, headDim,
                            output + (long)(segment.Start + t) * hidden + head * headDim);
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        });
    }

    /// <summary>The (segment, head) work items, built once per call and reused by every layer.</summary>
    private (Segment Segment, int Head)[] AttentionUnits(IReadOnlyList<Segment> segments)
    {
        var units = new (Segment, int)[segments.Count * _config.NumAttentionHeads];
        int index = 0;
        foreach (var segment in segments)
        {
            for (int head = 0; head < _config.NumAttentionHeads; ++head) units[index++] = (segment, head);
        }
        return units;
    }
}

/// <summary>
/// Records a batched activation as one tensor per sequence, so a dump of a five-question batch is
/// indistinguishable from five single-question dumps and can be compared to PyTorch directly.
/// </summary>
public sealed class BatchRecorder(IStateRecorder recorder, IReadOnlyList<string> labels)
{
    public void Record(string name, float[] values, IReadOnlyList<Segment> segments, int channels)
    {
        for (int i = 0; i < segments.Count; ++i)
        {
            var segment = segments[i];
            recorder.Record(labels[i] + name, values.AsSpan(segment.Start * channels, segment.Length * channels),
                [segment.Length, channels]);
        }
    }

    public void RecordOne(int index, string name, ReadOnlySpan<float> values, ReadOnlySpan<int> shape)
        => recorder.Record(labels[index] + name, values, shape);
}

/// <summary>
/// Rotary position embeddings, precomputed per attention kind because ModernBERT gives the global
/// and sliding layers different bases.
/// </summary>
public sealed class RopeCache
{
    private readonly int _headDim;
    private readonly double _theta;
    private float[] _cos = [];
    private float[] _sin = [];
    private int _positions;
    private readonly Lock _gate = new();

    public RopeCache(int headDim, double theta)
    {
        _headDim = headDim;
        _theta = theta;
        Grow(512);
    }

    private void Grow(int positions)
    {
        lock (_gate)
        {
            if (positions <= _positions) return;
            int half = _headDim / 2;
            var cos = new float[positions * half];
            var sin = new float[positions * half];
            for (int i = 0; i < half; ++i)
            {
                double inverseFrequency = 1.0 / Math.Pow(_theta, 2.0 * i / _headDim);
                for (int p = 0; p < positions; ++p)
                {
                    double angle = p * inverseFrequency;
                    cos[p * half + i] = (float)Math.Cos(angle);
                    sin[p * half + i] = (float)Math.Sin(angle);
                }
            }
            _cos = cos;
            _sin = sin;
            _positions = positions;
        }
    }

    /// <summary>
    /// Applies the rotation in place. This is the "rotate half" convention:
    /// <c>x' = x·cos + rotate_half(x)·sin</c> with <c>rotate_half([a, b]) = [-b, a]</c> over the two
    /// halves of the head dimension — not the interleaved pairs some other models use.
    /// </summary>
    public void Apply(Span<float> vector, int position)
    {
        if (position >= _positions) Grow(Math.Max(position + 1, _positions * 2));

        int half = _headDim / 2;
        var cos = _cos.AsSpan(position * half, half);
        var sin = _sin.AsSpan(position * half, half);
        for (int i = 0; i < half; ++i)
        {
            float a = vector[i];
            float b = vector[i + half];
            vector[i] = a * cos[i] - b * sin[i];
            vector[i + half] = b * cos[i] + a * sin[i];
        }
    }
}

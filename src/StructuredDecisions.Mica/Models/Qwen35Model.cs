using StructuredDecisions.Diagnostics;
using StructuredDecisions.Io;
using StructuredDecisions.Numerics;

namespace StructuredDecisions.Mica.Models;

/// <summary>
/// What a Qwen3.5 decoder remembers of the tokens it has read: keys and values for each full-attention
/// layer, and the convolution tail plus the delta-rule state for each DeltaNet layer.
///
/// <para>Reading a prompt in two calls through one state gives bit-for-bit the activations of reading it
/// in one: every GEMM row, every attention row and the recurrence are computed the same way either way.
/// That is what lets Mica read the system prompt and the state once and then branch per question with
/// <see cref="Clone"/>.</para>
/// </summary>
public sealed class Qwen35State
{
    internal readonly float[]?[] Keys;          // per layer: [capacity, kvHeads * headDim], full attention only
    internal readonly float[]?[] Values;
    internal readonly float[]?[] ConvTail;      // per layer: [kernel - 1, convDim], the last pre-convolution rows
    internal readonly float[]?[] Recurrent;     // per layer: [vHeads, dk, dv]

    /// <summary>How many tokens have been read into this state.</summary>
    public int Length { get; internal set; }

    internal Qwen35State(Qwen35Config config)
    {
        int layers = config.NumHiddenLayers;
        Keys = new float[layers][];
        Values = new float[layers][];
        ConvTail = new float[layers][];
        Recurrent = new float[layers][];
        for (int i = 0; i < layers; ++i)
        {
            if (config.LayerKinds[i] == Qwen35LayerKind.FullAttention)
            {
                Keys[i] = [];
                Values[i] = [];
            }
            else
            {
                ConvTail[i] = new float[(config.LinearConvKernel - 1) * config.LinearConvDim];
                Recurrent[i] = new float[config.LinearNumValueHeads * config.LinearKeyHeadDim * config.LinearValueHeadDim];
            }
        }
    }

    private Qwen35State(Qwen35State other)
    {
        int layers = other.Keys.Length;
        Keys = new float[layers][];
        Values = new float[layers][];
        ConvTail = new float[layers][];
        Recurrent = new float[layers][];
        for (int i = 0; i < layers; ++i)
        {
            Keys[i] = other.Keys[i]?.ToArray();
            Values[i] = other.Values[i]?.ToArray();
            ConvTail[i] = other.ConvTail[i]?.ToArray();
            Recurrent[i] = other.Recurrent[i]?.ToArray();
        }
        Length = other.Length;
    }

    /// <summary>An independent copy: what is read into one never reaches the other.</summary>
    public Qwen35State Clone() => new(this);

    /// <summary>
    /// A point to come back to: the recurrent and convolution state as they are now. Keys and values are
    /// not copied — rewinding only moves <see cref="Length"/> back, and whatever is read next overwrites
    /// the cache from there. Reading one long shared prefix and then several suffixes therefore costs one
    /// copy of the (small, fixed-size) DeltaNet state per suffix instead of a copy of the whole KV cache.
    /// </summary>
    public Mark Save() => new(Length, [.. ConvTail.Select(c => c?.ToArray())], [.. Recurrent.Select(r => r?.ToArray())]);

    /// <summary>Returns to a <see cref="Save"/>d point of this same state.</summary>
    public void Rewind(Mark mark)
    {
        if (mark.Length > Length) throw new ArgumentException("a mark can only rewind, not advance", nameof(mark));
        for (int i = 0; i < ConvTail.Length; ++i)
        {
            mark.ConvTail[i]?.CopyTo(ConvTail[i]!, 0);
            mark.Recurrent[i]?.CopyTo(Recurrent[i]!, 0);
        }
        Length = mark.Length;
    }

    /// <summary>A saved point of a <see cref="Qwen35State"/>.</summary>
    public sealed record Mark(int Length, float[]?[] ConvTail, float[]?[] Recurrent);

    /// <summary>Bytes held, to judge what keeping a prefix costs.</summary>
    public long Bytes => sizeof(float) * (Keys.Sum(k => (long)(k?.Length ?? 0)) + Values.Sum(v => (long)(v?.Length ?? 0))
        + ConvTail.Sum(c => (long)(c?.Length ?? 0)) + Recurrent.Sum(r => (long)(r?.Length ?? 0)));
}

/// <summary>
/// The Qwen3.5 text decoder (<c>Qwen3_5TextModel</c> in transformers), forward only, the backbone of Mica.
///
/// <para>Thirty-two pre-norm layers; three in four mix tokens with a Gated DeltaNet and every fourth with
/// gated softmax attention. Every layer ends in a SwiGLU MLP. Mica only ever needs the next-token logits
/// of the last prompt position, and only for its label tokens, so the 248k-row LM head is never
/// multiplied out: <see cref="Logits"/> takes the dot products it needs from the tied embedding.</para>
///
/// <para>Weights stay in the checkpoint's bf16 (<see cref="PackedMatrixBf16"/>); compute is fp32
/// throughout. The DeltaNet runs as its token-by-token recurrence. transformers evaluates the same
/// recurrence chunk-parallel for a prefill — algebraically identical, and within fp32 rounding of it.</para>
/// </summary>
public sealed class Qwen35Model
{
    /// <summary>Rows processed per pass through a layer: bounds the wide intermediates for long prompts.</summary>
    public int RowsPerChunk { get; set; } = 1024;

    private readonly Qwen35Config _config;
    private readonly TokenTable _embeddings;
    private readonly TokenTable _head;
    private readonly float[] _finalNorm;          // 1 + weight
    private readonly Layer[] _layers;
    private readonly float[] _inverseFrequency;

    public Qwen35Config Config => _config;
    public long WeightBytes { get; }

    private sealed class Layer
    {
        public required Qwen35LayerKind Kind;
        public required float[] InputNorm;        // 1 + weight
        public required float[] PostAttentionNorm;
        public required ILinear Mlp;              // [gate; up], 2 * intermediate outputs
        public required ILinear Down;

        // Full attention.
        public ILinear? Qkv;                      // [q|gate per head; k; v]
        public ILinear? Output;
        public float[]? QueryNorm, KeyNorm;

        // Gated DeltaNet.
        public ILinear? InProjection;             // [qkv (conv dim); z (value dim); b (v heads); a (v heads)]
        public float[]? ConvWeight;               // [convDim, kernel]
        public float[]? NegativeA;                // -exp(A_log)
        public float[]? DtBias;
        public float[]? GatedNorm;
        public ILinear? OutProjection;
    }

    public Qwen35Model(Qwen35Config config, SafetensorsCheckpoint weights, string? prefix = null)
    {
        if (!config.AttentionOutputGate) throw new NotSupportedException("Qwen3.5 without attn_output_gate is not implemented.");
        _config = config;
        prefix ??= weights.Contains("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.";

        _embeddings = TokenTable.Load(weights, prefix + "embed_tokens.weight", config.VocabSize, config.HiddenSize);
        _head = config.TieWordEmbeddings || !weights.Contains("lm_head.weight")
            ? _embeddings
            : TokenTable.Load(weights, "lm_head.weight", config.VocabSize, config.HiddenSize);
        _finalNorm = OnePlus(weights.ReadFloat32(prefix + "norm.weight"));

        int hidden = config.HiddenSize;
        _layers = new Layer[config.NumHiddenLayers];
        for (int i = 0; i < _layers.Length; ++i)
        {
            string p = $"{prefix}layers.{i}.";
            var kind = config.LayerKinds[i];
            var layer = new Layer
            {
                Kind = kind,
                InputNorm = OnePlus(weights.ReadFloat32(p + "input_layernorm.weight")),
                PostAttentionNorm = OnePlus(weights.ReadFloat32(p + "post_attention_layernorm.weight")),
                Mlp = Stack(weights, [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"], hidden),
                Down = Stack(weights, [p + "mlp.down_proj.weight"], config.IntermediateSize),
            };
            if (kind == Qwen35LayerKind.FullAttention)
            {
                layer.Qkv = Stack(weights, [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"], hidden);
                layer.Output = Stack(weights, [p + "self_attn.o_proj.weight"], config.NumAttentionHeads * config.HeadDim);
                layer.QueryNorm = OnePlus(weights.ReadFloat32(p + "self_attn.q_norm.weight"));
                layer.KeyNorm = OnePlus(weights.ReadFloat32(p + "self_attn.k_norm.weight"));
            }
            else
            {
                string a = p + "linear_attn.";
                layer.InProjection = Stack(weights,
                    [a + "in_proj_qkv.weight", a + "in_proj_z.weight", a + "in_proj_b.weight", a + "in_proj_a.weight"], hidden);
                layer.ConvWeight = weights.ReadFloat32(a + "conv1d.weight");       // [convDim, 1, kernel]
                layer.NegativeA = [.. weights.ReadFloat32(a + "A_log").Select(v => -MathF.Exp(v))];
                layer.DtBias = weights.ReadFloat32(a + "dt_bias");
                layer.GatedNorm = weights.ReadFloat32(a + "norm.weight");
                layer.OutProjection = Stack(weights, [a + "out_proj.weight"], config.LinearValueDim);
            }
            _layers[i] = layer;
        }

        // inv_freq = 1 / theta^(2i / rotary_dim), in fp32 as transformers builds it.
        int half = config.RotaryDim / 2;
        _inverseFrequency = new float[half];
        for (int i = 0; i < half; ++i)
        {
            _inverseFrequency[i] = 1f / MathF.Pow((float)config.RopeTheta, 2f * i / config.RotaryDim);
        }

        WeightBytes = _embeddings.Bytes + (_head == _embeddings ? 0 : _head.Bytes) + _layers.Sum(l =>
            l.Mlp.Bytes + l.Down.Bytes + (l.Qkv?.Bytes ?? 0) + (l.Output?.Bytes ?? 0)
            + (l.InProjection?.Bytes ?? 0) + (l.OutProjection?.Bytes ?? 0));
    }

    private static float[] OnePlus(float[] weight)
    {
        for (int i = 0; i < weight.Length; ++i) weight[i] = 1f + weight[i];
        return weight;
    }

    /// <summary>
    /// Packs the named <c>[out, in]</c> weights stacked along the output dimension into one projection, so
    /// a layer's sibling projections run as one GEMM. Each output column is computed independently, so
    /// stacking changes no value. bf16 weights stay bf16; anything else is widened to fp32.
    /// </summary>
    private static ILinear Stack(SafetensorsCheckpoint weights, string[] names, int inFeatures)
    {
        var entries = names.Select(weights.Entry).ToArray();
        foreach (var entry in entries)
        {
            if (entry.Shape.Length != 2 || entry.Shape[1] != inFeatures)
            {
                throw new InvalidDataException($"{entry.Name}: expected [*, {inFeatures}], found [{string.Join(", ", entry.Shape)}].");
            }
        }
        int outFeatures = entries.Sum(e => e.Shape[0]);
        if (entries.All(e => e.DType == "BF16"))
        {
            if (names.Length == 1) return new PackedMatrixBf16(weights.ReadBFloat16Bits(names[0]), outFeatures, inFeatures);
            var stacked = new ushort[(long)outFeatures * inFeatures];
            long offset = 0;
            foreach (string name in names)
            {
                var part = weights.ReadBFloat16Bits(name);
                part.CopyTo(stacked.AsSpan((int)offset));
                offset += part.Length;
            }
            return new PackedMatrixBf16(stacked, outFeatures, inFeatures);
        }

        var values = new float[(long)outFeatures * inFeatures];
        long at = 0;
        foreach (string name in names)
        {
            var part = weights.ReadFloat32(name);
            part.CopyTo(values.AsSpan((int)at));
            at += part.Length;
        }
        return new PackedMatrix(values, outFeatures, inFeatures);
    }

    /// <summary>A state that has read nothing yet.</summary>
    public Qwen35State NewState() => new(_config);

    /// <summary>
    /// Reads <paramref name="tokens"/> after whatever <paramref name="state"/> already holds and returns
    /// the final-normed hidden state of the last one (<c>[hidden]</c>), which is what the LM head reads.
    /// </summary>
    public float[] Forward(ReadOnlySpan<int> tokens, Qwen35State state, IStateRecorder? recorder = null,
        ParallelOptions? parallel = null)
    {
        if (tokens.Length == 0) throw new ArgumentException("nothing to read", nameof(tokens));
        int rows = tokens.Length;
        int hidden = _config.HiddenSize;
        int start = state.Length;

        var x = new float[(long)rows * hidden];
        using (ForwardTiming.Measure("embeddings"))
        {
            for (int t = 0; t < rows; ++t) _embeddings.Row(tokens[t], x.AsSpan(t * hidden, hidden));
        }
        recorder?.Record("embeddings", x, [rows, hidden]);

        var mixed = recorder is null ? null : new float[x.Length];
        for (int i = 0; i < _layers.Length; ++i)
        {
            var layer = _layers[i];
            for (int chunk = 0; chunk < rows; chunk += RowsPerChunk)
            {
                int count = Math.Min(RowsPerChunk, rows - chunk);
                var slice = x.AsSpan(chunk * hidden, count * hidden);
                if (layer.Kind == Qwen35LayerKind.FullAttention)
                {
                    using var stage = ForwardTiming.Measure("full_attention");
                    FullAttention(layer, i, slice, count, start + chunk, state, mixed is null ? default : mixed.AsSpan(chunk * hidden, count * hidden), parallel);
                }
                else
                {
                    using var stage = ForwardTiming.Measure("linear_attention");
                    LinearAttention(layer, i, slice, count, state, mixed is null ? default : mixed.AsSpan(chunk * hidden, count * hidden), parallel);
                }
                using (ForwardTiming.Measure("mlp")) Mlp(layer, slice, count, parallel);
            }
            if (recorder is not null)
            {
                recorder.Record($"layers.{i}.mixer", mixed, [rows, hidden]);
                recorder.Record($"layers.{i}", x, [rows, hidden]);
            }
        }
        state.Length = start + rows;

        var last = new float[hidden];
        DecoderKernels.RmsNorm(x.AsSpan((rows - 1) * hidden, hidden), _finalNorm, _config.RmsNormEps, last);
        if (recorder is not null)
        {
            var normed = new float[x.Length];
            for (int t = 0; t < rows; ++t)
            {
                DecoderKernels.RmsNorm(x.AsSpan(t * hidden, hidden), _finalNorm, _config.RmsNormEps, normed.AsSpan(t * hidden, hidden));
            }
            recorder.Record("norm", normed, [rows, hidden]);
        }
        return last;
    }

    /// <summary>The LM head's logits for just <paramref name="ids"/>: <c>hidden · E[id]</c>.</summary>
    public void Logits(ReadOnlySpan<float> hidden, ReadOnlySpan<int> ids, Span<float> logits)
    {
        var buffer = new float[_config.HiddenSize];
        for (int i = 0; i < ids.Length; ++i)
        {
            _head.Row(ids[i], buffer);
            logits[i] = SimdOps.Dot(hidden, buffer);
        }
    }

    private void Mlp(Layer layer, Span<float> x, int rows, ParallelOptions? parallel)
    {
        int hidden = _config.HiddenSize, inter = _config.IntermediateSize;
        using var scratch = new ScratchBuffers();
        var normed = scratch.Rent(rows * hidden);
        var gateUp = scratch.Rent(rows * 2 * inter);
        var act = scratch.Rent(rows * inter);
        var down = scratch.Rent(rows * hidden);

        for (int t = 0; t < rows; ++t)
        {
            DecoderKernels.RmsNorm(x.Slice(t * hidden, hidden), layer.PostAttentionNorm, _config.RmsNormEps, normed.AsSpan(t * hidden, hidden));
        }
        layer.Mlp.Multiply(normed.AsSpan(0, rows * hidden), rows, [], gateUp.AsSpan(0, rows * 2 * inter), parallel);
        ForRows(rows, parallel, t => DecoderKernels.SiluMultiply(
            gateUp.AsSpan(t * 2 * inter, inter), gateUp.AsSpan(t * 2 * inter + inter, inter), act.AsSpan(t * inter, inter)));
        layer.Down.Multiply(act.AsSpan(0, rows * inter), rows, [], down.AsSpan(0, rows * hidden), parallel);
        SimdOps.Add(x, down.AsSpan(0, rows * hidden));
    }

    /// <summary>
    /// Gated softmax attention over everything read so far: per-head RMSNorm on q and k, RoPE on the first
    /// <see cref="Qwen35Config.RotaryDim"/> dimensions of each head, causal GQA, then <c>· sigmoid(gate)</c>.
    /// </summary>
    private void FullAttention(Layer layer, int index, Span<float> x, int rows, int position, Qwen35State state,
        Span<float> record, ParallelOptions? parallel)
    {
        var c = _config;
        int hidden = c.HiddenSize, headDim = c.HeadDim, heads = c.NumAttentionHeads, kvHeads = c.NumKeyValueHeads;
        int qWidth = 2 * heads * headDim, kvWidth = kvHeads * headDim, width = qWidth + 2 * kvWidth;

        using var scratch = new ScratchBuffers();
        var normed = scratch.Rent(rows * hidden);
        var qkv = scratch.Rent(rows * width);
        var attention = scratch.Rent(rows * heads * headDim);
        var projected = scratch.Rent(rows * hidden);

        for (int t = 0; t < rows; ++t)
        {
            DecoderKernels.RmsNorm(x.Slice(t * hidden, hidden), layer.InputNorm, c.RmsNormEps, normed.AsSpan(t * hidden, hidden));
        }
        layer.Qkv!.Multiply(normed.AsSpan(0, rows * hidden), rows, [], qkv.AsSpan(0, rows * width), parallel);

        // Append this chunk's keys and values to the cache, normed and rotated.
        int total = position + rows;
        var keys = Grow(ref state.Keys[index]!, total * kvWidth, position * kvWidth);
        var values = Grow(ref state.Values[index]!, total * kvWidth, position * kvWidth);
        ForRows(rows, parallel, t =>
        {
            var row = qkv.AsSpan(t * width, width);
            int pos = position + t;
            Span<float> cos = stackalloc float[c.RotaryDim / 2];
            Span<float> sin = stackalloc float[c.RotaryDim / 2];
            Rotation(pos, cos, sin);
            for (int h = 0; h < heads; ++h)
            {
                var q = row.Slice(h * 2 * headDim, headDim);
                DecoderKernels.RmsNorm(q, layer.QueryNorm, c.RmsNormEps, q);
                Rotate(q, cos, sin);
            }
            for (int h = 0; h < kvHeads; ++h)
            {
                var k = row.Slice(qWidth + h * headDim, headDim);
                DecoderKernels.RmsNorm(k, layer.KeyNorm, c.RmsNormEps, k);
                Rotate(k, cos, sin);
            }
            row.Slice(qWidth, kvWidth).CopyTo(keys.AsSpan(pos * kvWidth, kvWidth));
            row.Slice(qWidth + kvWidth, kvWidth).CopyTo(values.AsSpan(pos * kvWidth, kvWidth));
        });

        float scale = 1f / MathF.Sqrt(headDim);
        int groups = heads / kvHeads;
        int work = rows * heads;
        Parallel.For(0, work, DecisionRuntime.Resolve(parallel), () => new float[total], (item, _, scores) =>
        {
            int t = item / heads, h = item % heads, kvh = h / groups;
            int pos = position + t;
            var q = qkv.AsSpan(t * width + h * 2 * headDim, headDim);
            float max = float.NegativeInfinity;
            for (int j = 0; j <= pos; ++j)
            {
                float s = SimdOps.Dot(q, keys.AsSpan(j * kvWidth + kvh * headDim, headDim)) * scale;
                scores[j] = s;
                if (s > max) max = s;
            }
            float sum = 0f;
            for (int j = 0; j <= pos; ++j)
            {
                float e = MathF.Exp(scores[j] - max);
                scores[j] = e;
                sum += e;
            }
            var output = attention.AsSpan((t * heads + h) * headDim, headDim);
            output.Clear();
            float inv = 1f / sum;
            for (int j = 0; j <= pos; ++j)
            {
                SimdOps.AddScaled(output, values.AsSpan(j * kvWidth + kvh * headDim, headDim), scores[j] * inv);
            }
            DecoderKernels.MultiplySigmoid(output, qkv.AsSpan(t * width + h * 2 * headDim + headDim, headDim));
            return scores;
        }, _ => { });

        layer.Output!.Multiply(attention.AsSpan(0, rows * heads * headDim), rows, [], projected.AsSpan(0, rows * hidden), parallel);
        if (!record.IsEmpty) projected.AsSpan(0, rows * hidden).CopyTo(record);
        SimdOps.Add(x, projected.AsSpan(0, rows * hidden));
    }

    /// <summary>cos/sin of <c>position · inv_freq</c>, the product taken in fp32 as transformers does.</summary>
    private void Rotation(int position, Span<float> cos, Span<float> sin)
    {
        for (int i = 0; i < cos.Length; ++i)
        {
            float angle = position * _inverseFrequency[i];
            cos[i] = MathF.Cos(angle);
            sin[i] = MathF.Sin(angle);
        }
    }

    /// <summary><c>rotate_half</c> RoPE on the first <c>2 · cos.Length</c> values; the rest pass through.</summary>
    private static void Rotate(Span<float> head, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        int half = cos.Length;
        for (int i = 0; i < half; ++i)
        {
            float a = head[i], b = head[i + half];
            head[i] = a * cos[i] - b * sin[i];
            head[i + half] = b * cos[i] + a * sin[i];
        }
    }

    /// <summary>
    /// Gated DeltaNet: causal depthwise conv + SiLU over [q; k; v], L2-normalised q and k, then per value
    /// head the gated delta rule
    /// <code>
    /// S ← S · exp(g);  S ← S + k ⊗ β (v − Sᵀk);  o = Sᵀq
    /// </code>
    /// with <c>g = −exp(A_log) · softplus(a + dt_bias)</c> and <c>β = sigmoid(b)</c>; the output goes
    /// through <c>RMSNormGated(o, z)</c> and <c>out_proj</c>.
    /// </summary>
    private void LinearAttention(Layer layer, int index, Span<float> x, int rows, Qwen35State state,
        Span<float> record, ParallelOptions? parallel)
    {
        var c = _config;
        int hidden = c.HiddenSize, convDim = c.LinearConvDim, kernel = c.LinearConvKernel;
        int keyDim = c.LinearKeyDim, valueDim = c.LinearValueDim;
        int vHeads = c.LinearNumValueHeads, kHeads = c.LinearNumKeyHeads, dk = c.LinearKeyHeadDim, dv = c.LinearValueHeadDim;
        int width = convDim + valueDim + 2 * vHeads;
        int tail = kernel - 1;

        using var scratch = new ScratchBuffers();
        var normed = scratch.Rent(rows * hidden);
        var projected = scratch.Rent(rows * width);
        var history = scratch.Rent((tail + rows) * convDim);     // conv input: the state's tail, then this chunk
        var conv = scratch.Rent(rows * convDim);
        var decay = scratch.Rent(rows * vHeads);
        var beta = scratch.Rent(rows * vHeads);
        var core = scratch.Rent(rows * valueDim);
        var output = scratch.Rent(rows * hidden);

        for (int t = 0; t < rows; ++t)
        {
            DecoderKernels.RmsNorm(x.Slice(t * hidden, hidden), layer.InputNorm, c.RmsNormEps, normed.AsSpan(t * hidden, hidden));
        }
        layer.InProjection!.Multiply(normed.AsSpan(0, rows * hidden), rows, [], projected.AsSpan(0, rows * width), parallel);

        var convTail = state.ConvTail[index]!;
        convTail.AsSpan().CopyTo(history);
        for (int t = 0; t < rows; ++t)
        {
            projected.AsSpan(t * width, convDim).CopyTo(history.AsSpan((tail + t) * convDim, convDim));
        }
        history.AsSpan(rows * convDim, tail * convDim).CopyTo(convTail);

        var weight = layer.ConvWeight!;
        ForRows(rows, parallel, t =>
        {
            var destination = conv.AsSpan(t * convDim, convDim);
            for (int ch = 0; ch < convDim; ++ch)
            {
                float sum = 0f;
                for (int j = 0; j < kernel; ++j) sum += weight[ch * kernel + j] * history[(t + j) * convDim + ch];
                destination[ch] = sum;
            }
            DecoderKernels.Silu(destination);
            float queryScale = 1f / MathF.Sqrt(dk);
            for (int h = 0; h < kHeads; ++h)
            {
                DecoderKernels.L2Normalize(destination.Slice(h * dk, dk), 1e-6f, queryScale);
                DecoderKernels.L2Normalize(destination.Slice(keyDim + h * dk, dk), 1e-6f);
            }
            var row = projected.AsSpan(t * width, width);
            for (int h = 0; h < vHeads; ++h)
            {
                float b = row[convDim + valueDim + h];
                float a = row[convDim + valueDim + vHeads + h];
                beta[t * vHeads + h] = DecoderKernels.Sigmoid(b);
                decay[t * vHeads + h] = MathF.Exp(layer.NegativeA![h] * DecoderKernels.Softplus(a + layer.DtBias![h]));
            }
        });

        var recurrent = state.Recurrent[index]!;
        int ratio = vHeads / kHeads;
        Parallel.For(0, vHeads, DecisionRuntime.Resolve(parallel), h =>
        {
            var s = recurrent.AsSpan(h * dk * dv, dk * dv);
            int kh = h / ratio;
            Span<float> memory = stackalloc float[dv];
            Span<float> delta = stackalloc float[dv];
            for (int t = 0; t < rows; ++t)
            {
                var row = conv.AsSpan(t * convDim, convDim);
                var q = row.Slice(kh * dk, dk);
                var k = row.Slice(keyDim + kh * dk, dk);
                var v = row.Slice(2 * keyDim + h * dv, dv);
                float g = decay[t * vHeads + h];
                float b = beta[t * vHeads + h];

                memory.Clear();
                for (int i = 0; i < dk; ++i)
                {
                    var si = s.Slice(i * dv, dv);
                    SimdOps.Scale(si, g);
                    SimdOps.AddScaled(memory, si, k[i]);
                }
                for (int j = 0; j < dv; ++j) delta[j] = (v[j] - memory[j]) * b;

                var o = core.AsSpan(t * valueDim + h * dv, dv);
                o.Clear();
                for (int i = 0; i < dk; ++i)
                {
                    var si = s.Slice(i * dv, dv);
                    SimdOps.AddScaled(si, delta, k[i]);
                    SimdOps.AddScaled(o, si, q[i]);
                }
            }
        });

        ForRows(rows, parallel, t =>
        {
            var z = projected.AsSpan(t * width + convDim, valueDim);
            for (int h = 0; h < vHeads; ++h)
            {
                var o = core.AsSpan(t * valueDim + h * dv, dv);
                DecoderKernels.RmsNormGated(o, z.Slice(h * dv, dv), layer.GatedNorm, c.RmsNormEps, o);
            }
        });
        layer.OutProjection!.Multiply(core.AsSpan(0, rows * valueDim), rows, [], output.AsSpan(0, rows * hidden), parallel);
        if (!record.IsEmpty) output.AsSpan(0, rows * hidden).CopyTo(record);
        SimdOps.Add(x, output.AsSpan(0, rows * hidden));
    }

    private static float[] Grow(ref float[] buffer, int needed, int keep)
    {
        if (buffer.Length >= needed) return buffer;
        var grown = new float[Math.Max(needed, buffer.Length * 2)];
        buffer.AsSpan(0, keep).CopyTo(grown);
        buffer = grown;
        return grown;
    }

    private static void ForRows(int rows, ParallelOptions? parallel, Action<int> body)
    {
        if (rows < 4 || DecisionRuntime.WorkersOf(parallel) <= 1)
        {
            for (int t = 0; t < rows; ++t) body(t);
            return;
        }
        Parallel.For(0, rows, DecisionRuntime.Resolve(parallel), body);
    }

    /// <summary>The embedding table (and, tied, the LM head), kept in bf16 when the checkpoint is.</summary>
    private sealed class TokenTable
    {
        private readonly ushort[]? _bf16;
        private readonly float[]? _fp32;
        private readonly int _rows, _width;

        private TokenTable(ushort[]? bf16, float[]? fp32, int rows, int width)
        {
            _bf16 = bf16;
            _fp32 = fp32;
            _rows = rows;
            _width = width;
        }

        public long Bytes => _bf16 is not null ? _bf16.LongLength * 2 : _fp32!.LongLength * 4;

        public static TokenTable Load(SafetensorsCheckpoint weights, string name, int rows, int width)
        {
            var entry = weights.Entry(name);
            if (entry.Shape.Length != 2 || entry.Shape[1] != width)
            {
                throw new InvalidDataException($"{name}: expected [*, {width}], found [{string.Join(", ", entry.Shape)}].");
            }
            rows = entry.Shape[0];
            return entry.DType == "BF16"
                ? new TokenTable(weights.ReadBFloat16Bits(name), null, rows, width)
                : new TokenTable(null, weights.ReadFloat32(name), rows, width);
        }

        public void Row(int id, Span<float> destination)
        {
            if ((uint)id >= (uint)_rows) throw new ArgumentOutOfRangeException(nameof(id), $"token id {id} is outside the vocabulary.");
            if (_fp32 is not null)
            {
                _fp32.AsSpan(id * _width, _width).CopyTo(destination);
                return;
            }
            var source = _bf16.AsSpan((int)((long)id * _width), _width);
            for (int i = 0; i < _width; ++i) destination[i] = BitConverter.UInt32BitsToSingle((uint)source[i] << 16);
        }
    }
}

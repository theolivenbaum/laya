using System.Buffers;
using Laya.Diagnostics;
using Laya.Io;
using Laya.Numerics;

namespace Laya.Models;

/// <summary>Everything one forward pass produces for a single question.</summary>
public sealed record DecisionOutput(float[] OptionLogits, float[] ActionProbabilities);

/// <summary>
/// The typed decision head that sits on top of the encoder.
///
/// <para>It is a faithful port of <c>DecisionModel</c> in <c>.reference/laya/common.py</c>:
/// a question-type embedding is added to every position, two pre-norm transformer layers refine
/// the sequence, a small scorer reads each option's <c>[MASK]</c> marker, and an action head
/// predicts whether to escalate from the pooled <c>[CLS]</c> state plus four calibration
/// features.</para>
/// </summary>
public sealed class DecisionModel
{
    private readonly ModernBertEncoder _encoder;
    private readonly int _hidden;

    private readonly float[] _typeEmbedding;         // [3, hidden]
    private readonly HeadLayer[] _headLayers;

    private readonly float[] _scorerNormWeight;
    private readonly float[] _scorerNormBias;
    private readonly float[] _scorerLinear1;
    private readonly float[] _scorerLinear1Bias;
    private readonly float[] _scorerLinear2;
    private readonly float[] _scorerLinear2Bias;

    private readonly float[] _actLinear1;
    private readonly float[] _actLinear1Bias;
    private readonly float[] _actLinear2;
    private readonly float[] _actLinear2Bias;
    private readonly int _actionCount;

    /// <summary>Bytes the repacked weights occupy in managed memory.</summary>
    public long WeightBytes { get; }

    /// <summary>The <c>temperature</c> buffer stored in the checkpoint (one per question type).</summary>
    public float[] CheckpointTemperature { get; }

    public ModernBertEncoder Encoder => _encoder;

    public DecisionModel(ModernBertConfig encoderConfig, LayaConfig config, SafetensorsFile weights)
    {
        _encoder = new ModernBertEncoder(encoderConfig, weights);
        _hidden = encoderConfig.HiddenSize;

        _typeEmbedding = weights.ReadFloat32("type_emb.weight");

        _headLayers = new HeadLayer[config.HeadLayers];
        for (int i = 0; i < config.HeadLayers; ++i)
        {
            string p = $"head.layers.{i}.";
            _headLayers[i] = new HeadLayer
            {
                Norm1Weight = weights.ReadFloat32(p + "norm1.weight"),
                Norm1Bias = weights.ReadFloat32(p + "norm1.bias"),
                Norm2Weight = weights.ReadFloat32(p + "norm2.weight"),
                Norm2Bias = weights.ReadFloat32(p + "norm2.bias"),
                InProjWeight = ModernBertEncoder.Pack(weights, p + "self_attn.in_proj_weight", 3 * _hidden, _hidden),
                InProjBias = weights.ReadFloat32(p + "self_attn.in_proj_bias"),
                OutProjWeight = ModernBertEncoder.Pack(weights, p + "self_attn.out_proj.weight", _hidden, _hidden),
                OutProjBias = weights.ReadFloat32(p + "self_attn.out_proj.bias"),
                Linear1Weight = ModernBertEncoder.Pack(weights, p + "linear1.weight", 4 * _hidden, _hidden),
                Linear1Bias = weights.ReadFloat32(p + "linear1.bias"),
                Linear2Weight = ModernBertEncoder.Pack(weights, p + "linear2.weight", _hidden, 4 * _hidden),
                Linear2Bias = weights.ReadFloat32(p + "linear2.bias"),
            };
        }

        _scorerNormWeight = weights.ReadFloat32("scorer.0.weight");
        _scorerNormBias = weights.ReadFloat32("scorer.0.bias");
        _scorerLinear1 = weights.ReadFloat32("scorer.1.weight");
        _scorerLinear1Bias = weights.ReadFloat32("scorer.1.bias");
        _scorerLinear2 = weights.ReadFloat32("scorer.3.weight");
        _scorerLinear2Bias = weights.ReadFloat32("scorer.3.bias");

        _actLinear1 = weights.ReadFloat32("act_head.0.weight");
        _actLinear1Bias = weights.ReadFloat32("act_head.0.bias");
        _actLinear2 = weights.ReadFloat32("act_head.2.weight");
        _actLinear2Bias = weights.ReadFloat32("act_head.2.bias");
        _actionCount = _actLinear2Bias.Length;

        CheckpointTemperature = weights.Contains("temperature") ? weights.ReadFloat32("temperature") : [1f, 1f, 1f];

        WeightBytes = _encoder.WeightBytes
            + (long)sizeof(float) * (_typeEmbedding.Length + _scorerNormWeight.Length + _scorerNormBias.Length
                + _scorerLinear1.Length + _scorerLinear1Bias.Length + _scorerLinear2.Length + _scorerLinear2Bias.Length
                + _actLinear1.Length + _actLinear1Bias.Length + _actLinear2.Length + _actLinear2Bias.Length)
            + _headLayers.Sum(l => l.InProjWeight.Bytes + l.OutProjWeight.Bytes
                + l.Linear1Weight.Bytes + l.Linear2Weight.Bytes);
    }

    /// <summary>Number of heads the decision head uses: <c>max(1, hidden / 64)</c>, as in Python.</summary>
    public int HeadAttentionHeads => Math.Max(1, _hidden / 64);

    public int ActionCount => _actionCount;

    /// <summary>One question in a batch: its tokens, its option markers and its type.</summary>
    public sealed record BatchItem(int[] TokenIds, int[] MarkerPositions, int Type);

    /// <summary>Runs one question on its own.</summary>
    public DecisionOutput Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> markerPositions, int questionType,
        IStateRecorder? recorder = null)
    {
        var item = new BatchItem([.. tokenIds], [.. markerPositions], questionType);
        return Forward([item], recorder is null ? null : new BatchRecorder(recorder, [""]))[0];
    }

    /// <summary>
    /// Runs a batch of questions in one pass.
    ///
    /// <para>The sequences are concatenated, so every projection in the encoder and the head sees
    /// one tall matrix and the weights are streamed once for the whole batch. Attention, the type
    /// embedding, the marker gather and the calibration features are all per question, which is
    /// what makes this identical to running them one at a time.</para>
    /// </summary>
    public DecisionOutput[] Forward(IReadOnlyList<BatchItem> items, BatchRecorder? recorder = null)
    {
        if (items.Count == 0) return [];

        var segments = new Segment[items.Count];
        int total = 0;
        for (int i = 0; i < items.Count; ++i)
        {
            segments[i] = new Segment(total, items[i].TokenIds.Length);
            total += items[i].TokenIds.Length;
        }

        var tokens = new int[total];
        for (int i = 0; i < items.Count; ++i) items[i].TokenIds.CopyTo(tokens, segments[i].Start);

        float[] states = _encoder.Forward(tokens, segments, recorder);

        // h = h + type_emb(qtype)[:, None, :], with each question's own type over its own rows.
        for (int i = 0; i < items.Count; ++i)
        {
            var typeVector = _typeEmbedding.AsSpan(items[i].Type * _hidden, _hidden);
            for (int t = segments[i].Start; t < segments[i].End; ++t)
            {
                SimdOps.Add(states.AsSpan(t * _hidden, _hidden), typeVector);
            }
        }
        recorder?.Record("head.input", states, segments, _hidden);

        for (int layer = 0; layer < _headLayers.Length; ++layer)
        {
            HeadForward(_headLayers[layer], states, total, segments);
            recorder?.Record($"head.layers.{layer}.output", states, segments, _hidden);
        }

        var outputs = new DecisionOutput[items.Count];
        for (int i = 0; i < items.Count; ++i)
        {
            outputs[i] = ReadOut(items[i], segments[i], states, recorder, i);
        }
        return outputs;
    }

    private DecisionOutput ReadOut(BatchItem item, Segment segment, float[] states, BatchRecorder? recorder, int index)
    {
        int options = item.MarkerPositions.Length;
        var logits = new float[options];
        var scorerBuffer = new float[_hidden];
        var scorerHidden = new float[_hidden];

        for (int o = 0; o < options; ++o)
        {
            int position = segment.Start + Math.Max(0, item.MarkerPositions[o]);
            SimdOps.LayerNorm(states.AsSpan(position * _hidden, _hidden), _scorerNormWeight, _scorerNormBias,
                1e-5f, scorerBuffer);
            SimdOps.Linear(scorerBuffer, _scorerLinear1, _scorerLinear1Bias, scorerHidden);
            SimdOps.Gelu(scorerHidden);
            logits[o] = SimdOps.Dot(scorerHidden, _scorerLinear2) + _scorerLinear2Bias[0];
        }
        recorder?.RecordOne(index, "logits", logits, [options]);

        // Calibration features come from the untempered softmax of the option logits.
        var probabilities = logits.AsSpan().ToArray();
        SimdOps.Softmax(probabilities);
        float k = Math.Max(2, options);
        float entropy = 0f;
        foreach (float p in probabilities) entropy -= p * MathF.Log(Math.Clamp(p, 1e-9f, float.MaxValue));
        entropy /= MathF.Log(k);

        float top1 = 0f;
        float top2 = 0f;
        foreach (float p in probabilities)
        {
            if (p > top1)
            {
                top2 = top1;
                top1 = p;
            }
            else if (p > top2)
            {
                top2 = p;
            }
        }

        var actionInput = new float[_hidden + 4];
        states.AsSpan(segment.Start * _hidden, _hidden).CopyTo(actionInput);   // pooled = h[:, 0]
        actionInput[_hidden] = top1;
        actionInput[_hidden + 1] = top1 - top2;
        actionInput[_hidden + 2] = entropy;
        actionInput[_hidden + 3] = k / 255f;

        var actionHidden = new float[_actLinear1Bias.Length];
        SimdOps.Linear(actionInput, _actLinear1, _actLinear1Bias, actionHidden);
        SimdOps.Gelu(actionHidden);
        var actionLogits = new float[_actionCount];
        SimdOps.Linear(actionHidden, _actLinear2, _actLinear2Bias, actionLogits);
        recorder?.RecordOne(index, "act_logits", actionLogits, [_actionCount]);

        var actionProbabilities = actionLogits.AsSpan().ToArray();
        SimdOps.Softmax(actionProbabilities);
        return new DecisionOutput(logits, actionProbabilities);
    }

    /// <summary>
    /// One <c>nn.TransformerEncoderLayer(norm_first=True)</c>:
    /// <c>x = x + attn(norm1(x)); x = x + ff(norm2(x))</c>. PyTorch's default activation is ReLU,
    /// which is what the checkpoint was trained with even though the rest of the model uses GELU.
    /// </summary>
    private void HeadForward(HeadLayer layer, float[] states, int tokens, IReadOnlyList<Segment> segments)
    {
        int hidden = _hidden;
        int heads = HeadAttentionHeads;
        int headDim = hidden / heads;
        float scale = 1f / MathF.Sqrt(headDim);

        using var scratch = new ScratchBuffers();
        var normed = scratch.Rent(tokens * hidden);
        using (ForwardTiming.Measure("head.norm"))
        {
            for (int t = 0; t < tokens; ++t)
            {
                SimdOps.LayerNorm(states.AsSpan(t * hidden, hidden), layer.Norm1Weight, layer.Norm1Bias, 1e-5f,
                    normed.AsSpan(t * hidden, hidden));
            }
        }

        var qkv = scratch.Rent(tokens * 3 * hidden);
        using (ForwardTiming.Measure("head.qkv"))
        {
            layer.InProjWeight.Multiply(normed.AsSpan(0, tokens * hidden), tokens, layer.InProjBias,
                qkv.AsSpan(0, tokens * 3 * hidden));
        }

        var context = scratch.Rent(tokens * hidden);
        int stride = 3 * hidden;
        var units = new (Segment Segment, int Head)[segments.Count * heads];
        int unitIndex = 0;
        foreach (var segment in segments)
        {
            for (int head = 0; head < heads; ++head) units[unitIndex++] = (segment, head);
        }

        using var attentionStage = ForwardTiming.Measure("head.attention");
        HeadAttention(qkv, context, segments, units, hidden, headDim, scale);
        attentionStage.Dispose();

        var projected = scratch.Rent(tokens * hidden);
        using (ForwardTiming.Measure("head.attn_out"))
        {
            layer.OutProjWeight.Multiply(context.AsSpan(0, tokens * hidden), tokens, layer.OutProjBias,
                projected.AsSpan(0, tokens * hidden));
        }
        SimdOps.Add(states.AsSpan(0, tokens * hidden), projected.AsSpan(0, tokens * hidden));

        using (ForwardTiming.Measure("head.norm"))
        {
            for (int t = 0; t < tokens; ++t)
            {
                SimdOps.LayerNorm(states.AsSpan(t * hidden, hidden), layer.Norm2Weight, layer.Norm2Bias, 1e-5f,
                    normed.AsSpan(t * hidden, hidden));
            }
        }

        int feedForward = layer.Linear1Bias.Length;
        var wide = scratch.Rent(tokens * feedForward);
        using (ForwardTiming.Measure("head.ff_in"))
        {
            layer.Linear1Weight.Multiply(normed.AsSpan(0, tokens * hidden), tokens, layer.Linear1Bias,
                wide.AsSpan(0, tokens * feedForward));
        }
        using (ForwardTiming.Measure("head.relu")) SimdOps.Relu(wide.AsSpan(0, tokens * feedForward));
        using (ForwardTiming.Measure("head.ff_out"))
        {
            layer.Linear2Weight.Multiply(wide.AsSpan(0, tokens * feedForward), tokens, layer.Linear2Bias,
                projected.AsSpan(0, tokens * hidden));
        }
        SimdOps.Add(states.AsSpan(0, tokens * hidden), projected.AsSpan(0, tokens * hidden));
    }

    /// <summary>
    /// Full attention inside each question's own sequence, on pointers for the same reason the
    /// encoder's is: the head dimension is 64 floats, so per-pair span construction dominates.
    /// </summary>
    private static unsafe void HeadAttention(float[] qkv, float[] context, IReadOnlyList<Segment> segments,
        (Segment Segment, int Head)[] units, int hidden, int headDim, float scale)
    {
        int stride = 3 * hidden;
        Parallel.For(0, units.Length, LayaRuntime.ParallelOptions, unit =>
        {
            var (segment, head) = units[unit];
            int length = segment.Length;
            float[] rented = ArrayPool<float>.Shared.Rent(length);
            try
            {
                fixed (float* source = qkv, output = context, scores = rented)
                {
                    float* keys = source + (long)segment.Start * stride + hidden + head * headDim;
                    float* values = source + (long)segment.Start * stride + 2 * hidden + head * headDim;

                    for (int t = 0; t < length; ++t)
                    {
                        float* query = source + (long)(segment.Start + t) * stride + head * headDim;
                        float* k = keys;
                        for (int j = 0; j < length; ++j, k += stride)
                        {
                            scores[j] = SimdOps.Dot(query, k, headDim) * scale;
                        }
                        SimdOps.Softmax(new Span<float>(scores, length));

                        float* destination = output + (long)(segment.Start + t) * hidden + head * headDim;
                        new Span<float>(destination, headDim).Clear();
                        float* v = values;
                        for (int j = 0; j < length; ++j, v += stride)
                        {
                            float weight = scores[j];
                            if (weight != 0f) SimdOps.AddScaled(destination, v, headDim, weight);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        });
    }

    private sealed class HeadLayer
    {
        public required float[] Norm1Weight;
        public required float[] Norm1Bias;
        public required float[] Norm2Weight;
        public required float[] Norm2Bias;
        public required PackedMatrix InProjWeight;
        public required float[] InProjBias;
        public required PackedMatrix OutProjWeight;
        public required float[] OutProjBias;
        public required PackedMatrix Linear1Weight;
        public required float[] Linear1Bias;
        public required PackedMatrix Linear2Weight;
        public required float[] Linear2Bias;
    }
}

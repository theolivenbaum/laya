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

        var normed = new float[tokens * hidden];
        for (int t = 0; t < tokens; ++t)
        {
            SimdOps.LayerNorm(states.AsSpan(t * hidden, hidden), layer.Norm1Weight, layer.Norm1Bias, 1e-5f,
                normed.AsSpan(t * hidden, hidden));
        }

        var qkv = new float[tokens * 3 * hidden];
        layer.InProjWeight.Multiply(normed, tokens, layer.InProjBias, qkv);

        var context = new float[tokens * hidden];
        int stride = 3 * hidden;
        var units = new List<(Segment Segment, int Head)>(segments.Count * heads);
        foreach (var segment in segments)
        {
            for (int head = 0; head < heads; ++head) units.Add((segment, head));
        }

        Parallel.For(0, units.Count, unit =>
        {
            var (segment, head) = units[unit];
            int length = segment.Length;
            var scores = new float[length];
            for (int t = 0; t < length; ++t)
            {
                int row = segment.Start + t;
                var query = qkv.AsSpan(row * stride + head * headDim, headDim);
                for (int j = 0; j < length; ++j)
                {
                    scores[j] = SimdOps.Dot(query,
                        qkv.AsSpan((segment.Start + j) * stride + hidden + head * headDim, headDim)) * scale;
                }
                SimdOps.Softmax(scores);

                var output = context.AsSpan(row * hidden + head * headDim, headDim);
                output.Clear();
                for (int j = 0; j < length; ++j)
                {
                    float weight = scores[j];
                    if (weight == 0f) continue;
                    var value = qkv.AsSpan((segment.Start + j) * stride + 2 * hidden + head * headDim, headDim);
                    for (int d = 0; d < headDim; ++d) output[d] += weight * value[d];
                }
            }
        });

        var projected = new float[tokens * hidden];
        layer.OutProjWeight.Multiply(context, tokens, layer.OutProjBias, projected);
        SimdOps.Add(states.AsSpan(0, tokens * hidden), projected);

        for (int t = 0; t < tokens; ++t)
        {
            SimdOps.LayerNorm(states.AsSpan(t * hidden, hidden), layer.Norm2Weight, layer.Norm2Bias, 1e-5f,
                normed.AsSpan(t * hidden, hidden));
        }

        int feedForward = layer.Linear1Bias.Length;
        var wide = new float[tokens * feedForward];
        layer.Linear1Weight.Multiply(normed, tokens, layer.Linear1Bias, wide);
        SimdOps.Relu(wide);
        layer.Linear2Weight.Multiply(wide, tokens, layer.Linear2Bias, projected);
        SimdOps.Add(states.AsSpan(0, tokens * hidden), projected);
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

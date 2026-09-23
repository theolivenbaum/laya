using System.Text.Json;
using System.Text.Json.Nodes;
using Laya.Io;
using Laya.Models;
using Laya.Numerics;

namespace Laya.Training;

/// <summary>One training sequence: tokens, option markers, question type, and the target distribution.</summary>
public sealed record TrainingItem(int[] TokenIds, int[] MarkerPositions, int QuestionType, float[] Target)
{
    /// <summary>Index of the most probable target option (the notebook's <c>label</c>).</summary>
    public int Label
    {
        get
        {
            int best = 0;
            for (int i = 1; i < Target.Length; ++i)
            {
                if (Target[i] > Target[best]) best = i;
            }
            return best;
        }
    }

    public int Options => MarkerPositions.Length;
}

/// <summary>What one forward pass kept for its backward pass.</summary>
public sealed class ForwardTape
{
    internal ForwardTape() { }

    internal IReadOnlyList<TrainingItem> Items { get; init; } = null!;
    internal Segment[] Segments { get; init; } = null!;
    internal int[] Tokens { get; init; } = null!;
    internal int[] MarkerRows { get; init; } = null!;
    internal bool Training { get; init; }
    internal ulong Seed { get; init; }

    /// <summary>Input to every encoder layer from the first trainable one, then to the final norm.</summary>
    internal float[]?[] EncoderInputs { get; init; } = null!;
    internal float[]? FinalNormInput { get; set; }
    internal float[]? EmbeddingInput { get; set; }
    internal float[][] HeadInputs { get; init; } = null!;

    internal float[] ScorerInput { get; init; } = null!;       // [markers, hidden] (gathered head output)
    internal float[] ScorerNormed { get; init; } = null!;
    internal float[] ScorerPreActivation { get; init; } = null!;
    internal float[] ScorerActivation { get; init; } = null!;

    public int TokenCount => Tokens.Length;
}

/// <summary>
/// The decision model with gradients: <c>DecisionModel</c> from <c>.reference/laya/common.py</c> —
/// ModernBERT, the type embedding, two pre-norm transformer layers and the option scorer — with a
/// backward pass for every piece, in fp32 on the CPU.
///
/// <para>Every layer is checkpointed: the forward pass keeps only each layer's input, and the
/// backward pass recomputes the layer before differentiating it. That costs one extra forward pass
/// and caps activation memory at one layer's worth, which is what makes a 421M-parameter encoder
/// trainable in ordinary RAM. Sequences are concatenated, as in inference, so no compute is spent on
/// padding.</para>
///
/// <para>The action head is not differentiated. The reference training script adds
/// <c>0.0 * act.sum()</c> to its loss, so its gradients are exactly zero; its parameters still pass
/// through the optimizer, which is where the decoupled weight decay touches them, exactly as there.</para>
/// </summary>
public sealed class TrainableDecisionModel
{
    private readonly List<Parameter> _parameters = [];
    private readonly Dictionary<string, Parameter> _byName = new(StringComparer.Ordinal);
    private readonly EncoderLayer[] _encoderLayers;
    private readonly HeadLayer[] _headLayers;
    private readonly Rotary _globalRotary;
    private readonly Rotary _localRotary;

    private readonly Parameter _tokenEmbeddings;
    private readonly Parameter _embeddingNorm;
    private readonly Parameter? _embeddingNormBias;
    private readonly Parameter _finalNorm;
    private readonly Parameter? _finalNormBias;
    private readonly Parameter _typeEmbedding;
    private readonly Parameter _scorerNorm, _scorerNormBias, _scorer1, _scorer1Bias, _scorer2, _scorer2Bias;

    public ModernBertConfig EncoderConfig { get; }
    public int Hidden { get; }
    public int HeadHeads { get; }

    /// <summary>Dropout inside the head's transformer layers during training (PyTorch's default, 0.1).</summary>
    public float HeadDropout { get; set; } = 0.1f;

    /// <summary>Every tensor of the checkpoint, in file order.</summary>
    public IReadOnlyList<Parameter> Parameters => _parameters;

    public Parameter this[string name] => _byName[name];

    /// <summary>The lowest encoder layer that receives gradients; <c>0</c> when the whole encoder trains.</summary>
    public int FirstTrainableEncoderLayer { get; private set; }

    public bool EmbeddingsTrainable => _tokenEmbeddings.Trainable;

    private TrainableDecisionModel(ModernBertConfig encoderConfig, int headLayers, SafetensorsFile weights)
    {
        EncoderConfig = encoderConfig;
        Hidden = encoderConfig.HiddenSize;
        HeadHeads = Math.Max(1, Hidden / 64);

        foreach (var entry in weights.Entries.Values.OrderBy(e => e.Start))
        {
            var parameter = new Parameter(entry.Name, entry.Shape, weights.ReadFloat32(entry.Name), entry.DType);
            _parameters.Add(parameter);
            _byName[entry.Name] = parameter;
        }
        // The temperature buffer is a buffer, not a parameter.
        if (_byName.TryGetValue("temperature", out var temperature)) temperature.Trainable = false;

        const string e = "encoder.";
        _tokenEmbeddings = Required(e + "embeddings.tok_embeddings.weight");
        _embeddingNorm = Required(e + "embeddings.norm.weight");
        _embeddingNormBias = Optional(e + "embeddings.norm.bias");
        _finalNorm = Required(e + "final_norm.weight");
        _finalNormBias = Optional(e + "final_norm.bias");

        _encoderLayers = new EncoderLayer[encoderConfig.NumHiddenLayers];
        for (int i = 0; i < _encoderLayers.Length; ++i)
        {
            string p = $"{e}layers.{i}.";
            _encoderLayers[i] = new EncoderLayer
            {
                AttnNorm = Optional(p + "attn_norm.weight"),
                AttnNormBias = Optional(p + "attn_norm.bias"),
                Wqkv = Required(p + "attn.Wqkv.weight"),
                WqkvBias = Optional(p + "attn.Wqkv.bias"),
                Wo = Required(p + "attn.Wo.weight"),
                WoBias = Optional(p + "attn.Wo.bias"),
                MlpNorm = Required(p + "mlp_norm.weight"),
                MlpNormBias = Optional(p + "mlp_norm.bias"),
                Wi = Required(p + "mlp.Wi.weight"),
                WiBias = Optional(p + "mlp.Wi.bias"),
                MlpWo = Required(p + "mlp.Wo.weight"),
                MlpWoBias = Optional(p + "mlp.Wo.bias"),
                Kind = encoderConfig.LayerKinds.Length > i
                    ? encoderConfig.LayerKinds[i]
                    : (i % encoderConfig.GlobalAttentionEveryNLayers == 0 ? AttentionKind.Global : AttentionKind.Sliding),
            };
        }

        _headLayers = new HeadLayer[headLayers];
        for (int i = 0; i < headLayers; ++i)
        {
            string p = $"head.layers.{i}.";
            _headLayers[i] = new HeadLayer
            {
                Norm1 = Required(p + "norm1.weight"),
                Norm1Bias = Required(p + "norm1.bias"),
                InProj = Required(p + "self_attn.in_proj_weight"),
                InProjBias = Required(p + "self_attn.in_proj_bias"),
                OutProj = Required(p + "self_attn.out_proj.weight"),
                OutProjBias = Required(p + "self_attn.out_proj.bias"),
                Norm2 = Required(p + "norm2.weight"),
                Norm2Bias = Required(p + "norm2.bias"),
                Linear1 = Required(p + "linear1.weight"),
                Linear1Bias = Required(p + "linear1.bias"),
                Linear2 = Required(p + "linear2.weight"),
                Linear2Bias = Required(p + "linear2.bias"),
            };
        }

        _typeEmbedding = Required("type_emb.weight");
        _scorerNorm = Required("scorer.0.weight");
        _scorerNormBias = Required("scorer.0.bias");
        _scorer1 = Required("scorer.1.weight");
        _scorer1Bias = Required("scorer.1.bias");
        _scorer2 = Required("scorer.3.weight");
        _scorer2Bias = Required("scorer.3.bias");

        _globalRotary = new Rotary(encoderConfig.HeadDim, encoderConfig.GlobalRopeTheta);
        _localRotary = new Rotary(encoderConfig.HeadDim, encoderConfig.LocalRopeTheta);
    }

    private Parameter Required(string name) => _byName.TryGetValue(name, out var p)
        ? p
        : throw new InvalidDataException($"the checkpoint has no '{name}' tensor; is it a laya decision model?");

    private Parameter? Optional(string name) => _byName.GetValueOrDefault(name);

    /// <summary>Loads every tensor of a checkpoint directory into trainable fp32 parameters.</summary>
    public static TrainableDecisionModel FromDirectory(string directory)
    {
        var encoderConfig = ModernBertConfig.Load(Path.Combine(directory, "encoder", "config.json"));
        var config = LayaConfig.Load(Path.Combine(directory, "rl_agent_config.json"));
        using var weights = new SafetensorsFile(Path.Combine(directory, "model.safetensors"));
        return new TrainableDecisionModel(encoderConfig, config.HeadLayers, weights);
    }

    /// <summary>
    /// Chooses what trains. <paramref name="encoderLayers"/> null trains everything, embeddings
    /// included (the reference notebook); <c>n</c> trains only the top <c>n</c> encoder layers and the
    /// final norm; <c>0</c> freezes the whole encoder. The head, type embedding, scorer and action
    /// head always train. Frozen layers run without a tape and keep no gradient, so the cheaper
    /// settings cost proportionally less time and memory.
    /// </summary>
    public void SetTrainableEncoderLayers(int? encoderLayers)
    {
        int count = _encoderLayers.Length;
        int trainable = Math.Clamp(encoderLayers ?? count, 0, count);
        FirstTrainableEncoderLayer = count - trainable;
        bool all = encoderLayers is null || trainable == count && encoderLayers >= count;
        foreach (var parameter in _parameters)
        {
            if (parameter.Name == "temperature") continue;
            if (!parameter.Name.StartsWith("encoder.", StringComparison.Ordinal))
            {
                parameter.Trainable = true;
                continue;
            }
            bool train;
            if (parameter.Name.StartsWith("encoder.embeddings.", StringComparison.Ordinal)) train = all;
            else if (parameter.Name.StartsWith("encoder.final_norm.", StringComparison.Ordinal)) train = trainable > 0;
            else
            {
                int layer = int.Parse(parameter.Name.Split('.')[2], System.Globalization.CultureInfo.InvariantCulture);
                train = layer >= FirstTrainableEncoderLayer;
            }
            parameter.Trainable = train;
            if (!train) parameter.ReleaseGradient();
        }
    }

    public long TrainableParameterCount => _parameters.Where(p => p.Trainable).Sum(p => (long)p.Length);

    public long ParameterCount => _parameters.Where(p => p.Name != "temperature").Sum(p => (long)p.Length);

    public void ZeroGradients()
    {
        foreach (var parameter in _parameters) parameter.ZeroGradient();
    }

    // ------------------------------------------------------------------------------------ forward

    /// <summary>
    /// Runs the batch and returns the option logits of every item. <paramref name="training"/> turns
    /// the head's dropout on; <paramref name="keepTape"/> keeps what <see cref="Backward"/> needs.
    /// </summary>
    public (float[][] Logits, ForwardTape? Tape) Forward(IReadOnlyList<TrainingItem> items, bool training,
        bool keepTape, ulong seed = 0, ParallelOptions? parallel = null)
    {
        var options = LayaRuntime.Resolve(parallel);
        int hidden = Hidden;

        var segments = new Segment[items.Count];
        int total = 0;
        int markers = 0;
        for (int i = 0; i < items.Count; ++i)
        {
            segments[i] = new Segment(total, items[i].TokenIds.Length);
            total += items[i].TokenIds.Length;
            markers += items[i].MarkerPositions.Length;
        }
        var tokens = new int[total];
        for (int i = 0; i < items.Count; ++i) items[i].TokenIds.CopyTo(tokens, segments[i].Start);
        int longest = segments.Length == 0 ? 0 : segments.Max(s => s.Length);
        _globalRotary.Ensure(longest);
        _localRotary.Ensure(longest);

        var markerRows = new int[markers];
        int m = 0;
        for (int i = 0; i < items.Count; ++i)
        {
            foreach (int position in items[i].MarkerPositions) markerRows[m++] = segments[i].Start + Math.Max(0, position);
        }

        // Embeddings.
        var embedded = new float[total * hidden];
        for (int t = 0; t < total; ++t)
        {
            int id = tokens[t];
            if ((uint)id >= (uint)EncoderConfig.VocabSize) throw new ArgumentOutOfRangeException(nameof(items), $"token id {id} is outside the vocabulary.");
            _tokenEmbeddings.Data.AsSpan(id * hidden, hidden).CopyTo(embedded.AsSpan(t * hidden, hidden));
        }
        var states = new float[total * hidden];
        Ops.LayerNorm(embedded, total, hidden, _embeddingNorm, _embeddingNormBias, EncoderConfig.NormEps, states, options);

        var encoderInputs = new float[]?[_encoderLayers.Length];
        for (int l = 0; l < _encoderLayers.Length; ++l)
        {
            if (keepTape && l >= FirstTrainableEncoderLayer) encoderInputs[l] = (float[])states.Clone();
            states = EncoderLayerForward(_encoderLayers[l], states, segments, options, tape: null);
        }

        float[]? finalNormInput = keepTape && FirstTrainableEncoderLayer < _encoderLayers.Length ? (float[])states.Clone() : null;
        var normed = new float[total * hidden];
        Ops.LayerNorm(states, total, hidden, _finalNorm, _finalNormBias, EncoderConfig.NormEps, normed, options);
        states = normed;

        // h = h + type_emb(qtype), each question's type over its own rows.
        for (int i = 0; i < items.Count; ++i)
        {
            var typeVector = _typeEmbedding.Data.AsSpan(items[i].QuestionType * hidden, hidden);
            for (int t = segments[i].Start; t < segments[i].End; ++t) SimdOps.Add(states.AsSpan(t * hidden, hidden), typeVector);
        }

        var headInputs = new float[_headLayers.Length][];
        for (int l = 0; l < _headLayers.Length; ++l)
        {
            if (keepTape) headInputs[l] = (float[])states.Clone();
            states = HeadLayerForward(_headLayers[l], l, states, segments, training, seed, options, tape: null);
        }

        // Scorer: LayerNorm -> Linear -> GELU -> Linear at every option's [MASK] marker.
        var scorerInput = new float[markers * hidden];
        for (int r = 0; r < markers; ++r) states.AsSpan(markerRows[r] * hidden, hidden).CopyTo(scorerInput.AsSpan(r * hidden, hidden));
        var scorerNormed = new float[markers * hidden];
        Ops.LayerNorm(scorerInput, markers, hidden, _scorerNorm, _scorerNormBias, 1e-5f, scorerNormed, options);
        var preActivation = new float[markers * hidden];
        Ops.Linear(_scorer1, _scorer1Bias, scorerNormed, markers, preActivation, options);
        var activation = (float[])preActivation.Clone();
        SimdOps.Gelu(activation);
        var scores = new float[markers];
        Ops.Linear(_scorer2, _scorer2Bias, activation, markers, scores, options);

        var logits = new float[items.Count][];
        m = 0;
        for (int i = 0; i < items.Count; ++i)
        {
            logits[i] = scores.AsSpan(m, items[i].Options).ToArray();
            m += items[i].Options;
        }

        ForwardTape? tape = keepTape
            ? new ForwardTape
            {
                Items = items,
                Segments = segments,
                Tokens = tokens,
                MarkerRows = markerRows,
                Training = training,
                Seed = seed,
                EncoderInputs = encoderInputs,
                FinalNormInput = finalNormInput,
                EmbeddingInput = EmbeddingsTrainable ? embedded : null,
                HeadInputs = headInputs,
                ScorerInput = scorerInput,
                ScorerNormed = scorerNormed,
                ScorerPreActivation = preActivation,
                ScorerActivation = activation,
            }
            : null;
        return (logits, tape);
    }

    /// <summary>What an encoder layer's backward pass needs, captured by a recomputation.</summary>
    private sealed class EncoderLayerTape
    {
        public required float[] Normed, Qkv, Context, Residual, MlpNormed, Wide, Activated;
    }

    private float[] EncoderLayerForward(EncoderLayer layer, float[] x, Segment[] segments, ParallelOptions options,
        EncoderLayerTape? tape)
    {
        int tokens = x.Length / Hidden, hidden = Hidden, intermediate = EncoderConfig.IntermediateSize;
        float epsilon = EncoderConfig.NormEps;

        float[] normed;
        if (layer.AttnNorm is null)
        {
            normed = x;   // layer 0: nn.Identity
        }
        else
        {
            normed = new float[tokens * hidden];
            Ops.LayerNorm(x, tokens, hidden, layer.AttnNorm, layer.AttnNormBias, epsilon, normed, options);
        }

        var qkv = new float[tokens * 3 * hidden];
        Ops.Linear(layer.Wqkv, layer.WqkvBias, normed, tokens, qkv, options);
        var context = new float[tokens * hidden];
        Ops.AttentionForward(qkv, segments, Shape(layer), context, options);
        var residual = new float[tokens * hidden];
        Ops.Linear(layer.Wo, layer.WoBias, context, tokens, residual, options);
        SimdOps.Add(residual, x);                                   // x1 = x + attn

        var mlpNormed = new float[tokens * hidden];
        Ops.LayerNorm(residual, tokens, hidden, layer.MlpNorm, layer.MlpNormBias, epsilon, mlpNormed, options);
        var wide = new float[tokens * 2 * intermediate];
        Ops.Linear(layer.Wi, layer.WiBias, mlpNormed, tokens, wide, options);
        var activated = new float[tokens * intermediate];
        Parallel.For(0, tokens, options, t =>
        {
            var row = wide.AsSpan(t * 2 * intermediate, 2 * intermediate);
            var output = activated.AsSpan(t * intermediate, intermediate);
            row[..intermediate].CopyTo(output);
            SimdOps.Gelu(output);
            SimdOps.Multiply(output, row.Slice(intermediate, intermediate));
        });
        var output = new float[tokens * hidden];
        Ops.Linear(layer.MlpWo, layer.MlpWoBias, activated, tokens, output, options);
        SimdOps.Add(output, residual);                              // x2 = x1 + mlp

        if (tape is not null)
        {
            tape.Normed = normed;
            tape.Qkv = qkv;
            tape.Context = context;
            tape.Residual = residual;
            tape.MlpNormed = mlpNormed;
            tape.Wide = wide;
            tape.Activated = activated;
        }
        return output;
    }

    private Ops.AttentionShape Shape(EncoderLayer layer) => new(
        Hidden, EncoderConfig.NumAttentionHeads, 1f / MathF.Sqrt(EncoderConfig.HeadDim),
        layer.Kind == AttentionKind.Sliding ? EncoderConfig.SlidingHalfWindow : int.MaxValue,
        layer.Kind == AttentionKind.Global ? _globalRotary : _localRotary, 0f, 0);

    private sealed class HeadLayerTape
    {
        public required float[] Normed, Qkv, Context, Residual, FfNormed, Wide, Dropped;
    }

    private ulong Salt(ulong seed, int layer, int site) => Ops.Mix(seed, (ulong)(layer * 16 + site + 1));

    private float[] HeadLayerForward(HeadLayer layer, int index, float[] x, Segment[] segments, bool training,
        ulong seed, ParallelOptions options, HeadLayerTape? tape)
    {
        int tokens = x.Length / Hidden, hidden = Hidden, feedForward = layer.Linear1.Shape[0];
        float dropout = training ? HeadDropout : 0f;

        var normed = new float[tokens * hidden];
        Ops.LayerNorm(x, tokens, hidden, layer.Norm1, layer.Norm1Bias, 1e-5f, normed, options);
        var qkv = new float[tokens * 3 * hidden];
        Ops.Linear(layer.InProj, layer.InProjBias, normed, tokens, qkv, options);
        var context = new float[tokens * hidden];
        Ops.AttentionForward(qkv, segments, HeadShape(index, dropout, seed), context, options);
        var residual = new float[tokens * hidden];
        Ops.Linear(layer.OutProj, layer.OutProjBias, context, tokens, residual, options);
        Ops.Dropout(residual, tokens * hidden, dropout, Salt(seed, index, 1), options);     // dropout1
        SimdOps.Add(residual, x);

        var ffNormed = new float[tokens * hidden];
        Ops.LayerNorm(residual, tokens, hidden, layer.Norm2, layer.Norm2Bias, 1e-5f, ffNormed, options);
        var wide = new float[tokens * feedForward];
        Ops.Linear(layer.Linear1, layer.Linear1Bias, ffNormed, tokens, wide, options);
        var dropped = (float[])wide.Clone();
        SimdOps.Relu(dropped);
        Ops.Dropout(dropped, tokens * feedForward, dropout, Salt(seed, index, 2), options); // dropout (ff)
        var output = new float[tokens * hidden];
        Ops.Linear(layer.Linear2, layer.Linear2Bias, dropped, tokens, output, options);
        Ops.Dropout(output, tokens * hidden, dropout, Salt(seed, index, 3), options);        // dropout2
        SimdOps.Add(output, residual);

        if (tape is not null)
        {
            tape.Normed = normed;
            tape.Qkv = qkv;
            tape.Context = context;
            tape.Residual = residual;
            tape.FfNormed = ffNormed;
            tape.Wide = wide;
            tape.Dropped = dropped;
        }
        return output;
    }

    private Ops.AttentionShape HeadShape(int index, float dropout, ulong seed)
        => new(Hidden, HeadHeads, 1f / MathF.Sqrt(Hidden / HeadHeads), int.MaxValue, null, dropout, Salt(seed, index, 0));

    // ----------------------------------------------------------------------------------- backward

    /// <summary>
    /// Backpropagates <c>∂loss/∂logits</c> (one array per item, one entry per option) through the
    /// batch and accumulates every trainable parameter's gradient.
    /// </summary>
    public void Backward(ForwardTape tape, IReadOnlyList<float[]> logitGradients, ParallelOptions? parallel = null)
    {
        var options = LayaRuntime.Resolve(parallel);
        int hidden = Hidden, tokens = tape.TokenCount, markers = tape.MarkerRows.Length;

        // Scorer.
        var dScores = new float[markers];
        int m = 0;
        for (int i = 0; i < tape.Items.Count; ++i)
        {
            logitGradients[i].AsSpan(0, tape.Items[i].Options).CopyTo(dScores.AsSpan(m));
            m += tape.Items[i].Options;
        }
        var dActivation = new float[markers * hidden];
        Ops.LinearBackward(_scorer2, _scorer2Bias, tape.ScorerActivation, dScores, markers, dActivation, options);
        Ops.GeluBackward(tape.ScorerPreActivation, dActivation);
        var dNormed = new float[markers * hidden];
        Ops.LinearBackward(_scorer1, _scorer1Bias, tape.ScorerNormed, dActivation, markers, dNormed, options);
        var dMarkers = new float[markers * hidden];
        Ops.LayerNormBackward(tape.ScorerInput, markers, hidden, _scorerNorm, _scorerNormBias, 1e-5f, dNormed, dMarkers, options);

        var dStates = new float[tokens * hidden];
        for (int r = 0; r < markers; ++r)
        {
            SimdOps.Add(dStates.AsSpan(tape.MarkerRows[r] * hidden, hidden), dMarkers.AsSpan(r * hidden, hidden));
        }

        // Head layers, last first, each recomputed from its checkpointed input.
        for (int l = _headLayers.Length - 1; l >= 0; --l)
        {
            dStates = HeadLayerBackward(_headLayers[l], l, tape.HeadInputs[l], tape, dStates, options);
        }

        // Type embedding: every row of a question received its type's vector.
        if (_typeEmbedding.Trainable)
        {
            var gradient = _typeEmbedding.GradientBuffer();
            for (int i = 0; i < tape.Items.Count; ++i)
            {
                var row = gradient.AsSpan(tape.Items[i].QuestionType * hidden, hidden);
                for (int t = tape.Segments[i].Start; t < tape.Segments[i].End; ++t) SimdOps.Add(row, dStates.AsSpan(t * hidden, hidden));
            }
        }

        if (tape.FinalNormInput is null) return;     // the encoder is frozen

        var dEncoder = new float[tokens * hidden];
        Ops.LayerNormBackward(tape.FinalNormInput, tokens, hidden, _finalNorm, _finalNormBias, EncoderConfig.NormEps,
            dStates, dEncoder, options);
        dStates = dEncoder;

        for (int l = _encoderLayers.Length - 1; l >= FirstTrainableEncoderLayer; --l)
        {
            // Below the lowest trainable layer nothing needs a gradient, unless the embeddings train.
            bool needInputGradient = l > FirstTrainableEncoderLayer || EmbeddingsTrainable;
            dStates = EncoderLayerBackward(_encoderLayers[l], tape.EncoderInputs[l]!, tape.Segments, dStates,
                needInputGradient, options);
        }

        if (EmbeddingsTrainable && FirstTrainableEncoderLayer == 0 && tape.EmbeddingInput is not null)
        {
            var dEmbedded = new float[tokens * hidden];
            Ops.LayerNormBackward(tape.EmbeddingInput, tokens, hidden, _embeddingNorm, _embeddingNormBias,
                EncoderConfig.NormEps, dStates, dEmbedded, options);
            var gradient = _tokenEmbeddings.GradientBuffer();
            for (int t = 0; t < tokens; ++t)
            {
                SimdOps.Add(gradient.AsSpan(tape.Tokens[t] * hidden, hidden), dEmbedded.AsSpan(t * hidden, hidden));
            }
        }
    }

    private float[] EncoderLayerBackward(EncoderLayer layer, float[] x, Segment[] segments, float[] dOutput,
        bool needInputGradient, ParallelOptions options)
    {
        int tokens = x.Length / Hidden, hidden = Hidden, intermediate = EncoderConfig.IntermediateSize;
        float epsilon = EncoderConfig.NormEps;
        var tape = new EncoderLayerTape { Normed = [], Qkv = [], Context = [], Residual = [], MlpNormed = [], Wide = [], Activated = [] };
        EncoderLayerForward(layer, x, segments, options, tape);

        // x2 = x1 + Wo2 · (gelu(input) ⊙ gate)
        var dResidual = (float[])dOutput.Clone();
        var dActivated = new float[tokens * intermediate];
        Ops.LinearBackward(layer.MlpWo, layer.MlpWoBias, tape.Activated, dOutput, tokens, dActivated, options);
        var dWide = new float[tokens * 2 * intermediate];
        Parallel.For(0, tokens, options, t =>
        {
            var row = tape.Wide.AsSpan(t * 2 * intermediate, 2 * intermediate);
            var input = row[..intermediate];
            var gate = row.Slice(intermediate, intermediate);
            var upstream = dActivated.AsSpan(t * intermediate, intermediate);
            var dInput = dWide.AsSpan(t * 2 * intermediate, intermediate);
            var dGate = dWide.AsSpan(t * 2 * intermediate + intermediate, intermediate);

            input.CopyTo(dGate);
            SimdOps.Gelu(dGate);
            SimdOps.Multiply(dGate, upstream);                   // dgate = dy · gelu(input)
            upstream.CopyTo(dInput);
            SimdOps.Multiply(dInput, gate);                      // dy · gate
            Ops.GeluBackward(input, dInput);                     // · gelu'(input)
        });
        tape.Activated = [];
        var dMlpNormed = new float[tokens * hidden];
        Ops.LinearBackward(layer.Wi, layer.WiBias, tape.MlpNormed, dWide, tokens, dMlpNormed, options);
        Ops.LayerNormBackward(tape.Residual, tokens, hidden, layer.MlpNorm, layer.MlpNormBias, epsilon, dMlpNormed, dResidual, options);

        // x1 = x + Wo · attention(Wqkv · norm(x))
        var dContext = new float[tokens * hidden];
        Ops.LinearBackward(layer.Wo, layer.WoBias, tape.Context, dResidual, tokens, dContext, options);
        var dQkv = new float[tokens * 3 * hidden];
        Ops.AttentionBackward(tape.Qkv, segments, Shape(layer), dContext, dQkv, options);

        if (!needInputGradient)
        {
            Ops.LinearBackward(layer.Wqkv, layer.WqkvBias, tape.Normed, dQkv, tokens, null, options);
            return dResidual;
        }

        var dNormed = new float[tokens * hidden];
        Ops.LinearBackward(layer.Wqkv, layer.WqkvBias, tape.Normed, dQkv, tokens, dNormed, options);
        var dx = dResidual;
        if (layer.AttnNorm is null) SimdOps.Add(dx, dNormed);
        else Ops.LayerNormBackward(x, tokens, hidden, layer.AttnNorm, layer.AttnNormBias, epsilon, dNormed, dx, options);
        return dx;
    }

    private float[] HeadLayerBackward(HeadLayer layer, int index, float[] x, ForwardTape forward, float[] dOutput,
        ParallelOptions options)
    {
        int tokens = x.Length / Hidden, hidden = Hidden, feedForward = layer.Linear1.Shape[0];
        float dropout = forward.Training ? HeadDropout : 0f;
        ulong seed = forward.Seed;
        var tape = new HeadLayerTape { Normed = [], Qkv = [], Context = [], Residual = [], FfNormed = [], Wide = [], Dropped = [] };
        HeadLayerForward(layer, index, x, forward.Segments, forward.Training, seed, options, tape);

        // x2 = x1 + drop2(linear2(drop(relu(linear1(norm2(x1))))))
        var dResidual = (float[])dOutput.Clone();
        var dy = (float[])dOutput.Clone();
        Ops.Dropout(dy, tokens * hidden, dropout, Salt(seed, index, 3), options);
        var dDropped = new float[tokens * feedForward];
        Ops.LinearBackward(layer.Linear2, layer.Linear2Bias, tape.Dropped, dy, tokens, dDropped, options);
        Ops.Dropout(dDropped, tokens * feedForward, dropout, Salt(seed, index, 2), options);
        for (int i = 0; i < dDropped.Length; ++i)
        {
            if (tape.Wide[i] <= 0f) dDropped[i] = 0f;            // relu
        }
        var dFfNormed = new float[tokens * hidden];
        Ops.LinearBackward(layer.Linear1, layer.Linear1Bias, tape.FfNormed, dDropped, tokens, dFfNormed, options);
        Ops.LayerNormBackward(tape.Residual, tokens, hidden, layer.Norm2, layer.Norm2Bias, 1e-5f, dFfNormed, dResidual, options);

        // x1 = x + drop1(out_proj(attention(in_proj(norm1(x)))))
        var dAttention = (float[])dResidual.Clone();
        Ops.Dropout(dAttention, tokens * hidden, dropout, Salt(seed, index, 1), options);
        var dContext = new float[tokens * hidden];
        Ops.LinearBackward(layer.OutProj, layer.OutProjBias, tape.Context, dAttention, tokens, dContext, options);
        var dQkv = new float[tokens * 3 * hidden];
        Ops.AttentionBackward(tape.Qkv, forward.Segments, HeadShape(index, dropout, seed), dContext, dQkv, options);
        var dNormed = new float[tokens * hidden];
        Ops.LinearBackward(layer.InProj, layer.InProjBias, tape.Normed, dQkv, tokens, dNormed, options);
        Ops.LayerNormBackward(x, tokens, hidden, layer.Norm1, layer.Norm1Bias, 1e-5f, dNormed, dResidual, options);
        return dResidual;
    }

    // --------------------------------------------------------------------------------------- save

    /// <summary>
    /// Writes a complete checkpoint directory that <c>Agent.FromDirectory</c> loads: the weights in
    /// the dtypes they were read in, <paramref name="config"/> as <c>rl_agent_config.json</c>, and the
    /// encoder config and tokenizer copied from <paramref name="sourceDirectory"/>.
    /// </summary>
    public void Save(string directory, string sourceDirectory, JsonObject config)
    {
        Directory.CreateDirectory(directory);
        SafetensorsWriter.Write(Path.Combine(directory, "model.safetensors"),
            [.. _parameters.Select(p => new SafetensorsTensor(p.Name, p.Shape, p.Data, p.StoredDType))]);

        CopyDirectory(Path.Combine(sourceDirectory, "encoder"), Path.Combine(directory, "encoder"));
        string tokenizer = Path.Combine(sourceDirectory, "tokenizer");
        if (Directory.Exists(tokenizer)) CopyDirectory(tokenizer, Path.Combine(directory, "tokenizer"));

        File.WriteAllText(Path.Combine(directory, "rl_agent_config.json"),
            config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void CopyDirectory(string source, string destination)
    {
        if (Path.GetFullPath(source) == Path.GetFullPath(destination)) return;
        Directory.CreateDirectory(destination);
        foreach (string file in Directory.GetFiles(source))
        {
            File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), overwrite: true);
        }
    }

    private sealed class EncoderLayer
    {
        public Parameter? AttnNorm, AttnNormBias, WqkvBias, WoBias, MlpNormBias, WiBias, MlpWoBias;
        public required Parameter Wqkv, Wo, MlpNorm, Wi, MlpWo;
        public required AttentionKind Kind;
    }

    private sealed class HeadLayer
    {
        public required Parameter Norm1, Norm1Bias, InProj, InProjBias, OutProj, OutProjBias,
            Norm2, Norm2Bias, Linear1, Linear1Bias, Linear2, Linear2Bias;
    }
}

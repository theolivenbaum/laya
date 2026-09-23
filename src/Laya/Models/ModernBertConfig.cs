using System.Text.Json;

namespace Laya.Models;

/// <summary>Which attention a ModernBERT layer uses.</summary>
public enum AttentionKind
{
    Global,
    Sliding,
}

/// <summary>
/// The parts of <c>encoder/config.json</c> the port needs.
///
/// <para>ModernBERT interleaves a full-attention layer with two sliding-window layers, and each
/// kind has its own RoPE base. The English checkpoint uses 160000 for global and 10000 for local;
/// the multilingual one uses 160000 for both and disables positions entirely
/// (<c>position_embedding_type: "sans_pos"</c>).</para>
/// </summary>
public sealed class ModernBertConfig
{
    public required int HiddenSize { get; init; }
    public required int IntermediateSize { get; init; }
    public required int NumHiddenLayers { get; init; }
    public required int NumAttentionHeads { get; init; }
    public required int VocabSize { get; init; }
    public required float NormEps { get; init; }
    public required bool NormBias { get; init; }
    public required bool AttentionBias { get; init; }
    public required bool MlpBias { get; init; }
    public required int LocalAttention { get; init; }
    public required int GlobalAttentionEveryNLayers { get; init; }
    public required double GlobalRopeTheta { get; init; }
    public required double LocalRopeTheta { get; init; }
    public required int MaxPositionEmbeddings { get; init; }
    public required string PositionEmbeddingType { get; init; }
    public AttentionKind[] LayerKinds { get; init; } = [];

    public int HeadDim => HiddenSize / NumAttentionHeads;

    /// <summary>
    /// Always true. The multilingual checkpoint's config says
    /// <c>position_embedding_type: "sans_pos"</c>, but transformers' ModernBERT implementation has
    /// no such switch — it reads neither that key nor any other positional option and always
    /// applies RoPE. Honouring the key here made layer 0 diverge immediately, so the field is kept
    /// for reference and ignored, exactly as the reference implementation ignores it.
    /// </summary>
    public bool UsesRope => true;

    /// <summary>Half-window of a sliding layer: a token sees <c>[i - w, i + w]</c>.</summary>
    public int SlidingHalfWindow => LocalAttention / 2;

    public static ModernBertConfig Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        return FromJson(document.RootElement);
    }

    public static ModernBertConfig FromJson(JsonElement root)
    {
        int layers = root.GetProperty("num_hidden_layers").GetInt32();
        int everyN = root.TryGetProperty("global_attn_every_n_layers", out var every) ? every.GetInt32() : 3;

        var kinds = new AttentionKind[layers];
        if (root.TryGetProperty("layer_types", out var layerTypes) && layerTypes.ValueKind == JsonValueKind.Array)
        {
            int index = 0;
            foreach (var entry in layerTypes.EnumerateArray())
            {
                if (index >= layers) break;
                kinds[index++] = entry.GetString() == "sliding_attention" ? AttentionKind.Sliding : AttentionKind.Global;
            }
        }
        else
        {
            for (int i = 0; i < layers; ++i) kinds[i] = i % everyN == 0 ? AttentionKind.Global : AttentionKind.Sliding;
        }

        double globalTheta = 160000d;
        double localTheta = 10000d;
        if (root.TryGetProperty("rope_parameters", out var rope) && rope.ValueKind == JsonValueKind.Object)
        {
            if (rope.TryGetProperty("full_attention", out var full) && full.TryGetProperty("rope_theta", out var ft))
            {
                globalTheta = ft.GetDouble();
            }
            if (rope.TryGetProperty("sliding_attention", out var sliding) && sliding.TryGetProperty("rope_theta", out var st))
            {
                localTheta = st.GetDouble();
            }
        }
        if (root.TryGetProperty("global_rope_theta", out var legacyGlobal)) globalTheta = legacyGlobal.GetDouble();
        if (root.TryGetProperty("local_rope_theta", out var legacyLocal)) localTheta = legacyLocal.GetDouble();

        return new ModernBertConfig
        {
            HiddenSize = root.GetProperty("hidden_size").GetInt32(),
            IntermediateSize = root.GetProperty("intermediate_size").GetInt32(),
            NumHiddenLayers = layers,
            NumAttentionHeads = root.GetProperty("num_attention_heads").GetInt32(),
            VocabSize = root.GetProperty("vocab_size").GetInt32(),
            NormEps = root.TryGetProperty("norm_eps", out var eps) ? (float)eps.GetDouble() : 1e-5f,
            NormBias = root.TryGetProperty("norm_bias", out var nb) && nb.GetBoolean(),
            AttentionBias = root.TryGetProperty("attention_bias", out var ab) && ab.GetBoolean(),
            MlpBias = root.TryGetProperty("mlp_bias", out var mb) && mb.GetBoolean(),
            LocalAttention = root.TryGetProperty("local_attention", out var la) ? la.GetInt32() : 128,
            GlobalAttentionEveryNLayers = everyN,
            GlobalRopeTheta = globalTheta,
            LocalRopeTheta = localTheta,
            MaxPositionEmbeddings = root.TryGetProperty("max_position_embeddings", out var mpe) ? mpe.GetInt32() : 8192,
            PositionEmbeddingType = root.TryGetProperty("position_embedding_type", out var pet)
                ? pet.GetString() ?? "absolute"
                : "absolute",
            LayerKinds = kinds,
        };
    }
}

/// <summary>
/// <c>rl_agent_config.json</c>: the decision-head hyperparameters and the calibration temperatures
/// fitted at the end of training.
/// </summary>
public sealed class LayaConfig
{
    public string Encoder { get; init; } = "answerdotai/ModernBERT-large";
    public int HeadLayers { get; init; } = 2;
    public int MaxLength { get; init; } = 512;
    public int HeadMaxLength { get; init; } = 192;
    public string ModelName { get; init; } = "rl-agent";
    public string AmpDType { get; init; } = "fp16";
    public IReadOnlyList<float> Temperature { get; init; } = [1f, 1f, 1f];
    public IReadOnlyDictionary<string, float> TemperatureByOptions { get; init; } =
        new Dictionary<string, float>(StringComparer.Ordinal);
    public IReadOnlyDictionary<string, float> ActionCosts { get; init; } =
        new Dictionary<string, float>(StringComparer.Ordinal);

    /// <summary>Number of action-head outputs: one per cost plus the implicit "do nothing".</summary>
    public int ActionCount => ActionCosts.Count + 1;

    // A temperature that is not a number is kept as NaN, which Calibration.ClampTemperature turns
    // into the neutral 1.0, as the Python's clamp_temperature does.
    private static float NumberOrNaN(JsonElement value)
        => value.ValueKind == JsonValueKind.Number ? (float)value.GetDouble() : float.NaN;

    public static LayaConfig Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        var temperature = new List<float> { 1f, 1f, 1f };
        if (root.TryGetProperty("temperature", out var temp) && temp.ValueKind == JsonValueKind.Array)
        {
            temperature.Clear();
            foreach (var value in temp.EnumerateArray()) temperature.Add(NumberOrNaN(value));
        }

        var byOptions = new Dictionary<string, float>(StringComparer.Ordinal);
        if (root.TryGetProperty("temperature_by_options", out var byOpt) && byOpt.ValueKind == JsonValueKind.Object)
        {
            foreach (var entry in byOpt.EnumerateObject()) byOptions[entry.Name] = NumberOrNaN(entry.Value);
        }

        var costs = new Dictionary<string, float>(StringComparer.Ordinal);
        if (root.TryGetProperty("act_costs", out var actCosts) && actCosts.ValueKind == JsonValueKind.Object)
        {
            foreach (var entry in actCosts.EnumerateObject()) costs[entry.Name] = (float)entry.Value.GetDouble();
        }

        return new LayaConfig
        {
            Encoder = root.TryGetProperty("encoder", out var enc) ? enc.GetString() ?? "" : "",
            HeadLayers = root.TryGetProperty("head_layers", out var hl) ? hl.GetInt32() : 2,
            MaxLength = root.TryGetProperty("max_len", out var ml) ? ml.GetInt32() : 512,
            HeadMaxLength = root.TryGetProperty("head_max_len", out var hml) ? hml.GetInt32() : 192,
            ModelName = root.TryGetProperty("model_name", out var mn) ? mn.GetString() ?? "rl-agent" : "rl-agent",
            AmpDType = root.TryGetProperty("amp_dtype", out var amp) ? amp.GetString() ?? "fp16" : "fp16",
            Temperature = temperature,
            TemperatureByOptions = byOptions,
            ActionCosts = costs,
        };
    }
}

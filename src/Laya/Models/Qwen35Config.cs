using System.Text.Json;

namespace Laya.Models;

/// <summary>Which token mixer a Qwen3.5 decoder layer uses.</summary>
public enum Qwen35LayerKind
{
    /// <summary>Gated DeltaNet: a causal depthwise convolution, then the gated delta rule recurrence.</summary>
    LinearAttention,

    /// <summary>Causal softmax attention with an output gate, partial RoPE and grouped KV heads.</summary>
    FullAttention,
}

/// <summary>
/// The text half of a Qwen3.5 <c>config.json</c> — the decoder Mica is a fine-tune of. The vision tower
/// and the multi-token-prediction head in the same checkpoint are not used: Mica reads the next-token
/// logits of the prompt's last position and nothing else.
/// </summary>
public sealed class Qwen35Config
{
    public required int HiddenSize { get; init; }
    public required int IntermediateSize { get; init; }
    public required int NumHiddenLayers { get; init; }
    public required int VocabSize { get; init; }
    public required float RmsNormEps { get; init; }
    public required Qwen35LayerKind[] LayerKinds { get; init; }

    // Full attention.
    public required int NumAttentionHeads { get; init; }
    public required int NumKeyValueHeads { get; init; }
    public required int HeadDim { get; init; }
    public required bool AttentionOutputGate { get; init; }
    public required double RopeTheta { get; init; }
    public required double PartialRotaryFactor { get; init; }

    // Gated DeltaNet.
    public required int LinearNumKeyHeads { get; init; }
    public required int LinearNumValueHeads { get; init; }
    public required int LinearKeyHeadDim { get; init; }
    public required int LinearValueHeadDim { get; init; }
    public required int LinearConvKernel { get; init; }

    public required bool TieWordEmbeddings { get; init; }

    /// <summary>Dimensions of each head RoPE rotates (the rest pass through): 64 of 256 for Mica.</summary>
    public int RotaryDim => (int)(HeadDim * PartialRotaryFactor);

    public int LinearKeyDim => LinearNumKeyHeads * LinearKeyHeadDim;
    public int LinearValueDim => LinearNumValueHeads * LinearValueHeadDim;

    /// <summary>Channels of the depthwise convolution: query, key and value concatenated.</summary>
    public int LinearConvDim => 2 * LinearKeyDim + LinearValueDim;

    public static Qwen35Config Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        return FromJson(document.RootElement);
    }

    /// <summary>Reads either a full multimodal config (its <c>text_config</c>) or a bare text config.</summary>
    public static Qwen35Config FromJson(JsonElement root)
    {
        var text = root.TryGetProperty("text_config", out var nested) ? nested : root;
        int layers = text.GetProperty("num_hidden_layers").GetInt32();

        var kinds = new Qwen35LayerKind[layers];
        if (text.TryGetProperty("layer_types", out var types) && types.ValueKind == JsonValueKind.Array)
        {
            int i = 0;
            foreach (var type in types.EnumerateArray())
            {
                if (i >= layers) break;
                kinds[i++] = type.GetString() switch
                {
                    "linear_attention" => Qwen35LayerKind.LinearAttention,
                    "full_attention" => Qwen35LayerKind.FullAttention,
                    var other => throw new NotSupportedException($"Qwen3.5 layer type '{other}' is not implemented."),
                };
            }
        }
        else
        {
            int interval = text.TryGetProperty("full_attention_interval", out var fi) ? fi.GetInt32() : 4;
            for (int i = 0; i < layers; ++i)
            {
                kinds[i] = (i + 1) % interval == 0 ? Qwen35LayerKind.FullAttention : Qwen35LayerKind.LinearAttention;
            }
        }

        double theta = 10_000_000d, partial = 0.25;
        if (text.TryGetProperty("rope_parameters", out var rope) && rope.ValueKind == JsonValueKind.Object)
        {
            if (rope.TryGetProperty("rope_type", out var rt) && rt.GetString() is { } ropeType && ropeType != "default")
            {
                throw new NotSupportedException($"Qwen3.5 rope_type '{ropeType}' is not implemented.");
            }
            if (rope.TryGetProperty("rope_theta", out var t)) theta = t.GetDouble();
            if (rope.TryGetProperty("partial_rotary_factor", out var p)) partial = p.GetDouble();
        }

        int hidden = text.GetProperty("hidden_size").GetInt32();
        int heads = text.GetProperty("num_attention_heads").GetInt32();
        bool tie = root.TryGetProperty("tie_word_embeddings", out var tw) ? tw.GetBoolean()
            : !text.TryGetProperty("tie_word_embeddings", out var ttw) || ttw.GetBoolean();
        if (text.TryGetProperty("attention_bias", out var ab) && ab.GetBoolean())
        {
            throw new NotSupportedException("Qwen3.5 with attention_bias is not implemented.");
        }
        string act = text.TryGetProperty("hidden_act", out var ha) ? ha.GetString() ?? "silu" : "silu";
        if (act != "silu") throw new NotSupportedException($"Qwen3.5 hidden_act '{act}' is not implemented.");

        return new Qwen35Config
        {
            HiddenSize = hidden,
            IntermediateSize = text.GetProperty("intermediate_size").GetInt32(),
            NumHiddenLayers = layers,
            VocabSize = text.GetProperty("vocab_size").GetInt32(),
            RmsNormEps = text.TryGetProperty("rms_norm_eps", out var eps) ? eps.GetSingle() : 1e-6f,
            LayerKinds = kinds,
            NumAttentionHeads = heads,
            NumKeyValueHeads = text.GetProperty("num_key_value_heads").GetInt32(),
            HeadDim = text.TryGetProperty("head_dim", out var hd) ? hd.GetInt32() : hidden / heads,
            AttentionOutputGate = !text.TryGetProperty("attn_output_gate", out var g) || g.GetBoolean(),
            RopeTheta = theta,
            PartialRotaryFactor = partial,
            LinearNumKeyHeads = text.GetProperty("linear_num_key_heads").GetInt32(),
            LinearNumValueHeads = text.GetProperty("linear_num_value_heads").GetInt32(),
            LinearKeyHeadDim = text.GetProperty("linear_key_head_dim").GetInt32(),
            LinearValueHeadDim = text.GetProperty("linear_value_head_dim").GetInt32(),
            LinearConvKernel = text.TryGetProperty("linear_conv_kernel_dim", out var ck) ? ck.GetInt32() : 4,
            TieWordEmbeddings = tie,
        };
    }
}

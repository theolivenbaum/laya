using System.Text.Json;

namespace Laya.Models;

/// <summary>What one LFM2 decoder layer mixes tokens with.</summary>
public enum Lfm2LayerKind
{
    /// <summary>A gated short convolution: <c>out(C · conv(B · x))</c>, three taps, causal.</summary>
    Conv,
    /// <summary>Causal grouped-query attention with per-head q/k RMSNorm and RoPE.</summary>
    Attention,
}

/// <summary>
/// The parts of an LFM2 <c>text_config</c> the port needs.
///
/// <para>LFM2 is a hybrid: most layers are short gated convolutions, every few is full causal
/// attention, and every layer has a SwiGLU MLP. d1-3B (LFM2.5-VL-3B) has 30 layers, 10 of them
/// attention, hidden 2048, a 10752-wide MLP, 32 query heads over 8 key/value heads, and its LM head
/// tied to the token embeddings.</para>
/// </summary>
public sealed class Lfm2Config
{
    public required int HiddenSize { get; init; }
    public required int IntermediateSize { get; init; }
    public required int NumAttentionHeads { get; init; }
    public required int NumKeyValueHeads { get; init; }
    public required int VocabSize { get; init; }
    public required float NormEps { get; init; }
    public required double RopeTheta { get; init; }
    public required int ConvTaps { get; init; }
    public required bool ConvBias { get; init; }
    public required Lfm2LayerKind[] LayerKinds { get; init; }
    public required bool TieWordEmbeddings { get; init; }
    public int MaxPositionEmbeddings { get; init; } = 32768;
    public int BosTokenId { get; init; } = -1;
    public int EosTokenId { get; init; } = -1;
    public int PadTokenId { get; init; } = -1;

    public int NumLayers => LayerKinds.Length;
    public int HeadDim => HiddenSize / NumAttentionHeads;
    public int KeyValueDim => NumKeyValueHeads * HeadDim;

    /// <summary>Loads the <c>text_config</c> of an LFM2-VL <c>config.json</c>, or a plain LFM2 one.</summary>
    public static Lfm2Config Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;
        return FromJson(root.TryGetProperty("text_config", out var text) ? text : root);
    }

    public static Lfm2Config FromJson(JsonElement root)
    {
        string modelType = root.TryGetProperty("model_type", out var mt) ? mt.GetString() ?? "" : "";
        if (modelType != "lfm2")
        {
            throw new InvalidDataException($"expected an LFM2 text config (model_type 'lfm2'), got '{modelType}'.");
        }

        int layers = root.GetProperty("num_hidden_layers").GetInt32();
        var kinds = new Lfm2LayerKind[layers];
        if (root.TryGetProperty("layer_types", out var types) && types.ValueKind == JsonValueKind.Array)
        {
            int index = 0;
            foreach (var entry in types.EnumerateArray())
            {
                if (index >= layers) break;
                kinds[index++] = entry.GetString() switch
                {
                    "conv" => Lfm2LayerKind.Conv,
                    "full_attention" => Lfm2LayerKind.Attention,
                    string other => throw new NotSupportedException($"LFM2 layer type '{other}' is not implemented."),
                    null => throw new InvalidDataException("null LFM2 layer type."),
                };
            }
            if (index != layers) throw new InvalidDataException($"layer_types has {index} entries for {layers} layers.");
        }
        else if (root.TryGetProperty("full_attn_idxs", out var full) && full.ValueKind == JsonValueKind.Array)
        {
            var attention = full.EnumerateArray().Select(e => e.GetInt32()).ToHashSet();
            for (int i = 0; i < layers; ++i) kinds[i] = attention.Contains(i) ? Lfm2LayerKind.Attention : Lfm2LayerKind.Conv;
        }
        else
        {
            throw new InvalidDataException("LFM2 config has neither layer_types nor full_attn_idxs.");
        }

        // HF's sizing rule, kept exactly: with block_auto_adjust_ff_dim the configured size is a
        // SwiGLU-equivalent budget, shrunk by 2/3 and rounded up to block_multiple_of.
        int intermediate = root.TryGetProperty("intermediate_size", out var isz)
            ? isz.GetInt32()
            : root.GetProperty("block_ff_dim").GetInt32();
        if (root.TryGetProperty("block_auto_adjust_ff_dim", out var adjust) && adjust.GetBoolean())
        {
            intermediate = (int)(2 * intermediate / 3.0);
            if (root.TryGetProperty("block_ffn_dim_multiplier", out var multiplier) && multiplier.ValueKind == JsonValueKind.Number)
            {
                intermediate = (int)(multiplier.GetDouble() * intermediate);
                int multipleOf = root.TryGetProperty("block_multiple_of", out var mo) ? mo.GetInt32() : 256;
                intermediate = multipleOf * ((intermediate + multipleOf - 1) / multipleOf);
            }
        }

        double theta = 1_000_000d;
        if (root.TryGetProperty("rope_parameters", out var rope) && rope.ValueKind == JsonValueKind.Object
            && rope.TryGetProperty("rope_theta", out var rt))
        {
            theta = rt.GetDouble();
        }
        else if (root.TryGetProperty("rope_theta", out var legacy))
        {
            theta = legacy.GetDouble();
        }

        return new Lfm2Config
        {
            HiddenSize = root.GetProperty("hidden_size").GetInt32(),
            IntermediateSize = intermediate,
            NumAttentionHeads = root.GetProperty("num_attention_heads").GetInt32(),
            NumKeyValueHeads = root.GetProperty("num_key_value_heads").GetInt32(),
            VocabSize = root.GetProperty("vocab_size").GetInt32(),
            NormEps = root.TryGetProperty("norm_eps", out var eps) ? (float)eps.GetDouble() : 1e-5f,
            RopeTheta = theta,
            ConvTaps = root.TryGetProperty("conv_L_cache", out var taps) ? taps.GetInt32() : 3,
            ConvBias = root.TryGetProperty("conv_bias", out var cb) && cb.GetBoolean(),
            LayerKinds = kinds,
            TieWordEmbeddings = !root.TryGetProperty("tie_word_embeddings", out var tie) || tie.GetBoolean(),
            MaxPositionEmbeddings = root.TryGetProperty("max_position_embeddings", out var mpe) ? mpe.GetInt32() : 32768,
            BosTokenId = root.TryGetProperty("bos_token_id", out var bos) && bos.ValueKind == JsonValueKind.Number ? bos.GetInt32() : -1,
            EosTokenId = root.TryGetProperty("eos_token_id", out var eos) && eos.ValueKind == JsonValueKind.Number ? eos.GetInt32() : -1,
            PadTokenId = root.TryGetProperty("pad_token_id", out var pad) && pad.ValueKind == JsonValueKind.Number ? pad.GetInt32() : -1,
        };
    }
}

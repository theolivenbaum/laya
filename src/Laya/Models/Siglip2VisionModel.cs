using System.Buffers;
using System.Numerics;
using System.Text.Json;
using Laya.Diagnostics;
using Laya.Io;
using Laya.Numerics;

namespace Laya.Models;

/// <summary>The parts of a SigLIP2 <c>vision_config</c> the port needs.</summary>
public sealed class Siglip2VisionConfig
{
    public required int HiddenSize { get; init; }
    public required int IntermediateSize { get; init; }
    public required int NumAttentionHeads { get; init; }
    public required int NumHiddenLayers { get; init; }
    public required int PatchSize { get; init; }
    public required int NumChannels { get; init; }
    public required int NumPatches { get; init; }
    public required float LayerNormEps { get; init; }
    public required string HiddenAct { get; init; }

    public int HeadDim => HiddenSize / NumAttentionHeads;
    public int PatchFeatures => NumChannels * PatchSize * PatchSize;
    public int PositionGrid => (int)Math.Round(Math.Sqrt(NumPatches));

    public static Siglip2VisionConfig FromJson(JsonElement root) => new()
    {
        HiddenSize = root.GetProperty("hidden_size").GetInt32(),
        IntermediateSize = root.GetProperty("intermediate_size").GetInt32(),
        NumAttentionHeads = root.GetProperty("num_attention_heads").GetInt32(),
        NumHiddenLayers = root.GetProperty("num_hidden_layers").GetInt32(),
        PatchSize = root.TryGetProperty("patch_size", out var ps) ? ps.GetInt32() : 16,
        NumChannels = root.TryGetProperty("num_channels", out var nc) ? nc.GetInt32() : 3,
        NumPatches = root.TryGetProperty("num_patches", out var np) ? np.GetInt32() : 256,
        LayerNormEps = root.TryGetProperty("layer_norm_eps", out var eps) ? (float)eps.GetDouble() : 1e-6f,
        HiddenAct = root.TryGetProperty("hidden_act", out var act) ? act.GetString() ?? "gelu_pytorch_tanh" : "gelu_pytorch_tanh",
    };
}

internal sealed class Siglip2LayerWeights
{
    public required float[] Norm1Weight, Norm1Bias, Norm2Weight, Norm2Bias;
    public required BFloat16Matrix Qkv, Out, Fc1, Fc2;
    public required float[] QkvBias, OutBias, Fc1Bias, Fc2Bias;
}

/// <summary>
/// The SigLIP2 NaFlex vision tower (no pooling head), forward only: patch embedding, the 16 x 16
/// learned position grid resized to the image's patch grid, pre-norm encoder layers with
/// bidirectional attention and a tanh-GELU MLP, and the post layer norm.
///
/// <para>A tile is processed on its own with exactly its own patches, which is what the reference's
/// padding mask computes: padded patches are masked out of every key set and their outputs are
/// discarded.</para>
/// </summary>
public sealed class Siglip2VisionModel
{
    private readonly Siglip2VisionConfig _config;
    private readonly BFloat16Matrix _patchEmbedding;
    private readonly float[] _patchBias;
    private readonly float[] _positions;         // [grid, grid, hidden]
    private readonly Siglip2LayerWeights[] _layers;
    private readonly float[] _postNormWeight, _postNormBias;

    public Siglip2VisionConfig Config => _config;
    public long WeightBytes { get; }

    public Siglip2VisionModel(Siglip2VisionConfig config, SafetensorsFile weights,
        string prefix = "model.vision_tower.vision_model.")
    {
        _config = config;
        int d = config.HiddenSize;
        _patchEmbedding = Pack(weights, prefix + "embeddings.patch_embedding.weight", d, config.PatchFeatures);
        _patchBias = weights.ReadFloat32(prefix + "embeddings.patch_embedding.bias");
        _positions = weights.ReadFloat32(prefix + "embeddings.position_embedding.weight");
        _postNormWeight = weights.ReadFloat32(prefix + "post_layernorm.weight");
        _postNormBias = weights.ReadFloat32(prefix + "post_layernorm.bias");

        _layers = new Siglip2LayerWeights[config.NumHiddenLayers];
        long bytes = _patchEmbedding.Bytes + _positions.Length * 4L;
        for (int i = 0; i < _layers.Length; ++i)
        {
            string p = $"{prefix}encoder.layers.{i}.";
            string a = p + "self_attn.";
            var layer = new Siglip2LayerWeights
            {
                Norm1Weight = weights.ReadFloat32(p + "layer_norm1.weight"),
                Norm1Bias = weights.ReadFloat32(p + "layer_norm1.bias"),
                Norm2Weight = weights.ReadFloat32(p + "layer_norm2.weight"),
                Norm2Bias = weights.ReadFloat32(p + "layer_norm2.bias"),
                Qkv = new BFloat16Matrix([.. Bits(weights, a + "q_proj.weight"), .. Bits(weights, a + "k_proj.weight"),
                    .. Bits(weights, a + "v_proj.weight")], 3 * d, d),
                QkvBias = [.. weights.ReadFloat32(a + "q_proj.bias"), .. weights.ReadFloat32(a + "k_proj.bias"),
                    .. weights.ReadFloat32(a + "v_proj.bias")],
                Out = Pack(weights, a + "out_proj.weight", d, d),
                OutBias = weights.ReadFloat32(a + "out_proj.bias"),
                Fc1 = Pack(weights, p + "mlp.fc1.weight", config.IntermediateSize, d),
                Fc1Bias = weights.ReadFloat32(p + "mlp.fc1.bias"),
                Fc2 = Pack(weights, p + "mlp.fc2.weight", d, config.IntermediateSize),
                Fc2Bias = weights.ReadFloat32(p + "mlp.fc2.bias"),
            };
            bytes += layer.Qkv.Bytes + layer.Out.Bytes + layer.Fc1.Bytes + layer.Fc2.Bytes;
            _layers[i] = layer;
        }
        WeightBytes = bytes;
    }

    private static ushort[] Bits(SafetensorsFile weights, string name) => weights.ReadBFloat16Bits(name);

    private static BFloat16Matrix Pack(SafetensorsFile weights, string name, int outFeatures, int inFeatures)
        => new(weights.ReadBFloat16Bits(name), outFeatures, inFeatures);

    /// <summary>
    /// Encodes one tile: <paramref name="patches"/> is <c>[height · width, 3 · patch²]</c>, row-major
    /// over the patch grid. Returns the post-normed hidden states <c>[height · width, hidden]</c>.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> patches, int height, int width, IStateRecorder? recorder = null,
        string name = "vision", ParallelOptions? parallel = null)
    {
        int tokens = height * width;
        int d = _config.HiddenSize;
        int ff = _config.IntermediateSize;
        if (patches.Length != tokens * _config.PatchFeatures) throw new ArgumentException("patch count does not match the grid.", nameof(patches));

        var hidden = new float[tokens * d];
        using (ForwardTiming.Measure("vision.embed"))
        {
            _patchEmbedding.Multiply(patches, tokens, hidden, parallel);
            var positions = Resampling.ResizePositionGrid(_positions, _config.PositionGrid, d, height, width);
            for (int t = 0; t < tokens; ++t)
            {
                var row = hidden.AsSpan(t * d, d);
                SimdOps.Add(row, _patchBias);
                SimdOps.Add(row, positions.AsSpan(t * d, d));
            }
        }
        recorder?.Record($"{name}.embeddings", hidden, [tokens, d]);

        using var scratch = new ScratchBuffers();
        var normed = scratch.Rent(tokens * d);
        var qkv = scratch.Rent(tokens * 3 * d);
        var mixed = scratch.Rent(tokens * d);
        var projected = scratch.Rent(tokens * d);
        var wide = scratch.Rent(tokens * ff);

        for (int index = 0; index < _layers.Length; ++index)
        {
            var layer = _layers[index];
            using (ForwardTiming.Measure("vision.norm")) LayerNormRows(hidden, layer.Norm1Weight, layer.Norm1Bias, normed, tokens);
            using (ForwardTiming.Measure("vision.qkv")) layer.Qkv.Multiply(normed.AsSpan(0, tokens * d), tokens, qkv, parallel);
            AddBias(qkv, layer.QkvBias, tokens);
            using (ForwardTiming.Measure("vision.attention")) Attention(qkv, tokens, mixed, parallel);
            using (ForwardTiming.Measure("vision.attn_out")) layer.Out.Multiply(mixed.AsSpan(0, tokens * d), tokens, projected, parallel);
            AddBias(projected, layer.OutBias, tokens);
            SimdOps.Add(hidden.AsSpan(0, tokens * d), projected.AsSpan(0, tokens * d));

            using (ForwardTiming.Measure("vision.norm")) LayerNormRows(hidden, layer.Norm2Weight, layer.Norm2Bias, normed, tokens);
            using (ForwardTiming.Measure("vision.fc1")) layer.Fc1.Multiply(normed.AsSpan(0, tokens * d), tokens, wide, parallel);
            using (ForwardTiming.Measure("vision.gelu"))
            {
                AddBias(wide, layer.Fc1Bias, tokens);
                GeluTanh(wide.AsSpan(0, tokens * ff));
            }
            using (ForwardTiming.Measure("vision.fc2")) layer.Fc2.Multiply(wide.AsSpan(0, tokens * ff), tokens, projected, parallel);
            AddBias(projected, layer.Fc2Bias, tokens);
            SimdOps.Add(hidden.AsSpan(0, tokens * d), projected.AsSpan(0, tokens * d));
            recorder?.Record($"{name}.layers.{index}.output", hidden.AsSpan(0, tokens * d), [tokens, d]);
        }

        LayerNormRows(hidden, _postNormWeight, _postNormBias, hidden, tokens);
        recorder?.Record($"{name}.post_layernorm", hidden, [tokens, d]);
        return hidden;
    }

    private void LayerNormRows(float[] source, float[] weight, float[] bias, float[] destination, int tokens)
    {
        int d = _config.HiddenSize;
        for (int t = 0; t < tokens; ++t)
        {
            SimdOps.LayerNorm(source.AsSpan(t * d, d), weight, bias, _config.LayerNormEps, destination.AsSpan(t * d, d));
        }
    }

    private static void AddBias(float[] rows, float[] bias, int tokens)
    {
        for (int t = 0; t < tokens; ++t) SimdOps.Add(rows.AsSpan(t * bias.Length, bias.Length), bias);
    }

    /// <summary>
    /// <c>gelu_pytorch_tanh</c>: <c>0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))</c>, with
    /// <c>tanh(u) = 1 - 2 / (e^{2u} + 1)</c> on the vectorized exponential.
    /// </summary>
    internal static void GeluTanh(Span<float> values)
    {
        const float k = 0.7978845608028654f;   // sqrt(2 / pi)
        const float c = 0.044715f;
        int width = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var two = new Vector<float>(2f);
        int i = 0;
        for (; i <= values.Length - width; i += width)
        {
            var x = Vector.LoadUnsafe(ref values[i]);
            var u = new Vector<float>(k) * (x + new Vector<float>(c) * x * x * x);
            var tanh = one - two / (SimdOps.Exp(two * u) + one);
            (half * x * (one + tanh)).StoreUnsafe(ref values[i]);
        }
        for (; i < values.Length; ++i)
        {
            float x = values[i];
            values[i] = 0.5f * x * (1f + MathF.Tanh(k * (x + c * x * x * x)));
        }
    }

    /// <summary>Full bidirectional multi-head attention over one tile's patches.</summary>
    private unsafe void Attention(float[] qkv, int tokens, float[] destination, ParallelOptions? parallel)
    {
        int d = _config.HiddenSize;
        int headDim = _config.HeadDim;
        int heads = _config.NumAttentionHeads;
        int stride = 3 * d;
        float scale = 1f / MathF.Sqrt(headDim);
        int keyStride = AttentionKernels.PaddedKeyStride(tokens);

        void Head(int h)
        {
            int keyFloats = headDim * keyStride;
            float[] rented = ArrayPool<float>.Shared.Rent(keyFloats + tokens * headDim + keyStride);
            try
            {
                fixed (float* source = qkv, output = destination, scratch = rented)
                {
                    float* keysTransposed = scratch;
                    float* values = scratch + keyFloats;
                    float* scores = values + tokens * headDim;
                    new Span<float>(keysTransposed, keyFloats).Clear();
                    for (int t = 0; t < tokens; ++t)
                    {
                        float* row = source + (long)t * stride;
                        for (int j = 0; j < headDim; ++j) keysTransposed[j * keyStride + t] = row[d + h * headDim + j];
                        Buffer.MemoryCopy(row + 2 * d + h * headDim, values + t * headDim, headDim * 4, headDim * 4);
                    }
                    for (int t = 0; t < tokens; ++t)
                    {
                        AttentionKernels.Scores(source + (long)t * stride + h * headDim, keysTransposed, keyStride, headDim,
                            tokens, scale, scores);
                        SimdOps.Softmax(new Span<float>(scores, tokens));
                        AttentionKernels.WeightedSum(values, scores, tokens, headDim, output + (long)t * d + h * headDim);
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
            for (int h = 0; h < heads; ++h) Head(h);
        }
        else
        {
            Parallel.For(0, heads, LayaRuntime.Resolve(parallel), Head);
        }
    }
}

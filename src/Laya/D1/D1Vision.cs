using System.Text.Json;
using Laya.Diagnostics;
using Laya.Io;
using Laya.Models;
using Laya.Numerics;

namespace Laya.D1;

/// <summary>
/// d1's image path from processed tiles to language-model embeddings: the SigLIP2 tower, the 2 x 2
/// pixel unshuffle, and the two-layer GELU projector (<c>Lfm2VlMultiModalProjector</c>).
/// </summary>
public sealed class D1Vision
{
    private readonly Siglip2VisionModel _tower;
    private readonly BFloat16Matrix _linear1;
    private readonly float[] _bias1;
    private readonly BFloat16Matrix _linear2;
    private readonly float[] _bias2;
    private readonly float[]? _normWeight, _normBias;
    private readonly int _factor;

    public Siglip2VisionModel Tower => _tower;
    public int OutputSize { get; }
    public long WeightBytes => _tower.WeightBytes + _linear1.Bytes + _linear2.Bytes;

    public D1Vision(JsonElement config, SafetensorsFile weights)
    {
        var visionConfig = Siglip2VisionConfig.FromJson(config.GetProperty("vision_config"));
        _tower = new Siglip2VisionModel(visionConfig, weights);
        _factor = config.TryGetProperty("downsample_factor", out var f) ? f.GetInt32() : 2;
        string act = config.TryGetProperty("projector_hidden_act", out var a) ? a.GetString() ?? "gelu" : "gelu";
        if (act != "gelu") throw new NotSupportedException($"projector activation '{act}' is not implemented.");

        int inFeatures = visionConfig.HiddenSize * _factor * _factor;
        int hidden = config.TryGetProperty("projector_hidden_size", out var ph) ? ph.GetInt32() : 2560;
        OutputSize = config.GetProperty("text_config").GetProperty("hidden_size").GetInt32();
        const string p = "model.multi_modal_projector.";
        if (config.TryGetProperty("projector_use_layernorm", out var ln) && ln.GetBoolean())
        {
            _normWeight = weights.ReadFloat32(p + "layer_norm.weight");
            _normBias = weights.ReadFloat32(p + "layer_norm.bias");
        }
        _linear1 = new BFloat16Matrix(weights.ReadBFloat16Bits(p + "linear_1.weight"), hidden, inFeatures);
        _linear2 = new BFloat16Matrix(weights.ReadBFloat16Bits(p + "linear_2.weight"), OutputSize, hidden);
        bool bias = !config.TryGetProperty("projector_bias", out var pb) || pb.GetBoolean();
        _bias1 = bias ? weights.ReadFloat32(p + "linear_1.bias") : new float[hidden];
        _bias2 = bias ? weights.ReadFloat32(p + "linear_2.bias") : new float[OutputSize];
    }

    /// <summary>
    /// The language-model embeddings of one picture: every tile through the tower and the projector,
    /// concatenated in tile order — the order of its <c>&lt;image&gt;</c> tokens.
    /// </summary>
    public float[] Embed(ProcessedImage image, IStateRecorder? recorder = null, string name = "image",
        ParallelOptions? parallel = null)
    {
        var parts = new List<float[]>();
        for (int t = 0; t < image.Tiles.Count; ++t)
        {
            var tile = image.Tiles[t];
            string label = $"{name}.tile{t}";
            recorder?.Record($"{label}.patches", tile.Patches, [tile.Rows * tile.Columns, tile.Patches.Length / (tile.Rows * tile.Columns)]);
            var features = _tower.Forward(tile.Patches, tile.Rows, tile.Columns, recorder, label, parallel);
            parts.Add(Project(features, tile.Rows, tile.Columns, recorder, label, parallel));
        }
        return [.. parts.SelectMany(p => p)];
    }

    /// <summary>
    /// <c>pixel_unshuffle</c> then the projector. Output token <c>(i, j)</c> holds the four patches
    /// <c>(2i + dy, 2j + dx)</c> as channels <c>[(dy · 2 + dx) · C + c]</c>.
    /// </summary>
    private float[] Project(float[] features, int rows, int columns, IStateRecorder? recorder, string label,
        ParallelOptions? parallel)
    {
        int c = _tower.Config.HiddenSize;
        int f = _factor;
        if (rows % f != 0 || columns % f != 0)
        {
            throw new InvalidOperationException($"a {rows} x {columns} patch grid does not unshuffle by {f}.");
        }
        int outRows = rows / f, outColumns = columns / f;
        int tokens = outRows * outColumns;
        int width = c * f * f;
        var unshuffled = new float[tokens * width];
        for (int i = 0; i < outRows; ++i)
        {
            for (int j = 0; j < outColumns; ++j)
            {
                for (int dy = 0; dy < f; ++dy)
                {
                    for (int dx = 0; dx < f; ++dx)
                    {
                        features.AsSpan(((i * f + dy) * columns + j * f + dx) * c, c)
                            .CopyTo(unshuffled.AsSpan((i * outColumns + j) * width + (dy * f + dx) * c, c));
                    }
                }
            }
        }
        if (_normWeight is not null)
        {
            for (int t = 0; t < tokens; ++t)
            {
                var row = unshuffled.AsSpan(t * width, width);
                SimdOps.LayerNorm(row, _normWeight, _normBias!, 1e-5f, row);
            }
        }

        var hidden = new float[tokens * _bias1.Length];
        using (ForwardTiming.Measure("vision.projector"))
        {
            _linear1.Multiply(unshuffled, tokens, hidden, parallel);
            for (int t = 0; t < tokens; ++t) SimdOps.Add(hidden.AsSpan(t * _bias1.Length, _bias1.Length), _bias1);
            SimdOps.Gelu(hidden);
            var output = new float[tokens * OutputSize];
            _linear2.Multiply(hidden, tokens, output, parallel);
            for (int t = 0; t < tokens; ++t) SimdOps.Add(output.AsSpan(t * OutputSize, OutputSize), _bias2);
            recorder?.Record($"{label}.projected", output, [tokens, OutputSize]);
            return output;
        }
    }
}

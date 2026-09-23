using System.Text.Json;
using Laya.Io;
using Laya.Training;
using Xunit;
using Xunit.Abstractions;

namespace Laya.Tests;

/// <summary>
/// The backward pass against PyTorch autograd. <c>tools/dump_training_reference.py</c> builds the
/// reference <c>DecisionModel</c> on a tiny random ModernBERT (4 layers, a sliding window shorter
/// than the sequences, both head layers), backpropagates a fixed linear function of the logits,
/// and dumps every parameter's gradient; this compares all of them.
/// </summary>
public class TrainingGradientTests(ITestOutputHelper output)
{
    private static string Fixture => Path.Combine(TestModels.FixtureRoot, "training-tiny");

    private sealed record Item(int[] Ids, int[] Markers, int QType, float[] Coeff, float[] Logits);

    private static List<Item> Batch()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixture, "batch.json")));
        return [.. document.RootElement.GetProperty("items").EnumerateArray().Select(e => new Item(
            [.. e.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32())],
            [.. e.GetProperty("markers").EnumerateArray().Select(x => x.GetInt32())],
            e.GetProperty("qtype").GetInt32(),
            [.. e.GetProperty("coeff").EnumerateArray().Select(x => x.GetSingle())],
            [.. e.GetProperty("logits").EnumerateArray().Select(x => x.GetSingle())]))];
    }

    private static (TrainableDecisionModel Model, List<Item> Batch, float[][] Logits) RunBackward()
    {
        var model = TrainableDecisionModel.FromDirectory(Fixture);
        model.SetTrainableEncoderLayers(null);
        var batch = Batch();
        var items = batch.Select(b => new TrainingItem(b.Ids, b.Markers, b.QType, new float[b.Markers.Length])).ToList();
        var (logits, tape) = model.Forward(items, training: false, keepTape: true);
        model.Backward(tape!, [.. batch.Select(b => b.Coeff)]);
        return (model, batch, logits);
    }

    [Fact]
    public void LogitsMatchPyTorch()
    {
        var (_, batch, logits) = RunBackward();
        for (int i = 0; i < batch.Count; ++i)
        {
            for (int k = 0; k < batch[i].Logits.Length; ++k)
            {
                Assert.True(Math.Abs(batch[i].Logits[k] - logits[i][k]) < 1e-4,
                    $"item {i} option {k}: {logits[i][k]} vs torch {batch[i].Logits[k]}");
            }
        }
    }

    [Fact]
    public void EveryGradientMatchesPyTorch()
    {
        var (model, _, _) = RunBackward();
        using var reference = new SafetensorsFile(Path.Combine(Fixture, "grads.safetensors"));

        int compared = 0;
        foreach (var (name, _) in reference.Entries)
        {
            float[] expected = reference.ReadFloat32(name);
            float[] actual = model[name].Gradient ?? new float[expected.Length];
            double maxError = 0d, scale = 0d;
            for (int i = 0; i < expected.Length; ++i)
            {
                maxError = Math.Max(maxError, Math.Abs(expected[i] - actual[i]));
                scale = Math.Max(scale, Math.Abs(expected[i]));
            }
            double relative = maxError / Math.Max(scale, 1e-12);
            output.WriteLine($"{name,-50} max|g|={scale:E2} max err={maxError:E2} rel={relative:E2}");
            Assert.True(relative < 1e-4, $"{name}: relative error {relative:E2} (max |g| {scale:E2})");
            compared++;
        }
        Assert.Equal(reference.Entries.Count, compared);

        // Nothing outside the loss received a gradient.
        Assert.All(model.Parameters.Where(p => p.Name.StartsWith("act_head.", StringComparison.Ordinal)),
            p => Assert.True(p.Gradient is null || p.Gradient.All(g => g == 0f)));
    }

    [Fact]
    public void FreezingTheEncoderLeavesItsGradientsEmpty()
    {
        var model = TrainableDecisionModel.FromDirectory(Fixture);
        model.SetTrainableEncoderLayers(1);
        var batch = Batch();
        var items = batch.Select(b => new TrainingItem(b.Ids, b.Markers, b.QType, new float[b.Markers.Length])).ToList();
        var (_, tape) = model.Forward(items, training: false, keepTape: true);
        model.Backward(tape!, [.. batch.Select(b => b.Coeff)]);

        Assert.Null(model["encoder.layers.0.attn.Wqkv.weight"].Gradient);
        Assert.Null(model["encoder.embeddings.tok_embeddings.weight"].Gradient);
        Assert.NotNull(model["encoder.layers.3.attn.Wqkv.weight"].Gradient);

        // The top layer's gradient does not depend on what is frozen below it.
        var (full, _, _) = RunBackward();
        float[] expected = full["encoder.layers.3.mlp.Wi.weight"].Gradient!;
        float[] actual = model["encoder.layers.3.mlp.Wi.weight"].Gradient!;
        for (int i = 0; i < expected.Length; ++i) Assert.Equal(expected[i], actual[i], 1e-6f);
    }

    [Fact]
    public void DropoutGradientsMatchFiniteDifferences()
    {
        // PyTorch draws its dropout masks from its own generator, so the training-mode path is checked
        // against itself instead: the same seed regenerates the same masks, which makes the forward
        // pass a deterministic function whose derivative can be taken numerically.
        var model = TrainableDecisionModel.FromDirectory(Fixture);
        model.SetTrainableEncoderLayers(null);
        model.HeadDropout = 0.3f;
        var batch = Batch();
        var items = batch.Select(b => new TrainingItem(b.Ids, b.Markers, b.QType, new float[b.Markers.Length])).ToList();
        const ulong seed = 12345;

        double Loss()
        {
            var (logits, _) = model.Forward(items, training: true, keepTape: false, seed: seed);
            double total = 0d;
            for (int i = 0; i < batch.Count; ++i)
            {
                for (int k = 0; k < logits[i].Length; ++k) total += (double)logits[i][k] * batch[i].Coeff[k];
            }
            return total;
        }

        var (_, tape) = model.Forward(items, training: true, keepTape: true, seed: seed);
        model.Backward(tape!, [.. batch.Select(b => b.Coeff)]);

        foreach (string name in (string[])["head.layers.0.linear1.weight", "head.layers.1.self_attn.in_proj_weight",
                     "head.layers.0.self_attn.out_proj.bias", "encoder.layers.3.mlp.Wi.weight", "scorer.1.weight"])
        {
            var parameter = model[name];
            float[] gradient = parameter.Gradient!;
            // The coordinates with the largest analytic gradient, so the difference is well above fp32 noise.
            var coordinates = Enumerable.Range(0, gradient.Length).OrderByDescending(i => Math.Abs(gradient[i])).Take(3);
            foreach (int index in coordinates)
            {
                float original = parameter.Data[index];
                const float h = 2e-3f;
                parameter.Data[index] = original + h;
                parameter.Invalidate();
                double plus = Loss();
                parameter.Data[index] = original - h;
                parameter.Invalidate();
                double minus = Loss();
                parameter.Data[index] = original;
                parameter.Invalidate();

                double numeric = (plus - minus) / (2 * h);
                output.WriteLine($"{name}[{index}] analytic={gradient[index]:E3} numeric={numeric:E3}");
                Assert.True(Math.Abs(numeric - gradient[index]) <= 0.05 * Math.Abs(gradient[index]) + 1e-5,
                    $"{name}[{index}]: analytic {gradient[index]:E3} vs numeric {numeric:E3}");
            }
        }
    }

    private static string EnglishDump => Path.Combine(TestModels.RepositoryRoot, "artifacts", "dumps", "training-english");

    /// <summary>
    /// The same comparison on the real 28-layer English checkpoint, for a subset of tensors. Run
    /// <c>tools/dump_training_reference.py --checkpoint artifacts/models/english --data … --out artifacts/dumps/training-english</c>
    /// first; the test skips without it.
    /// </summary>
    [ModelFact]
    public void EnglishCheckpointGradientsMatchPyTorch()
    {
        if (!File.Exists(Path.Combine(EnglishDump, "grads-subset.safetensors")))
        {
            output.WriteLine("no dump in " + EnglishDump + "; nothing to compare");
            return;
        }

        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(EnglishDump, "batch.json")));
        var batch = document.RootElement.GetProperty("items").EnumerateArray().Select(e => new Item(
            [.. e.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32())],
            [.. e.GetProperty("markers").EnumerateArray().Select(x => x.GetInt32())],
            e.GetProperty("qtype").GetInt32(),
            [.. e.GetProperty("coeff").EnumerateArray().Select(x => x.GetSingle())],
            [.. e.GetProperty("logits").EnumerateArray().Select(x => x.GetSingle())])).ToList();

        var model = TrainableDecisionModel.FromDirectory(TestModels.CheckpointDirectory("english"));
        model.SetTrainableEncoderLayers(null);
        var items = batch.Select(b => new TrainingItem(b.Ids, b.Markers, b.QType, new float[b.Markers.Length])).ToList();
        var (logits, tape) = model.Forward(items, training: false, keepTape: true);
        model.Backward(tape!, [.. batch.Select(b => b.Coeff)]);

        for (int i = 0; i < batch.Count; ++i)
        {
            for (int k = 0; k < batch[i].Logits.Length; ++k) Assert.Equal(batch[i].Logits[k], logits[i][k], 3);
        }

        using var reference = new SafetensorsFile(Path.Combine(EnglishDump, "grads-subset.safetensors"));
        foreach (var (name, _) in reference.Entries)
        {
            float[] expected = reference.ReadFloat32(name);
            float[] actual = model[name].Gradient!;
            double error = 0d, scale = 0d, dot = 0d, na = 0d, nb = 0d;
            for (int j = 0; j < expected.Length; ++j)
            {
                error = Math.Max(error, Math.Abs(expected[j] - actual[j]));
                scale = Math.Max(scale, Math.Abs(expected[j]));
                dot += (double)expected[j] * actual[j];
                na += (double)expected[j] * expected[j];
                nb += (double)actual[j] * actual[j];
            }
            double cosine = dot / Math.Sqrt(na * nb);
            output.WriteLine($"{name,-45} max|g|={scale:E2} rel={error / scale:E2} cos={cosine:F7}");
            Assert.True(error / scale < 1e-3 && cosine > 0.99999, $"{name}: rel {error / scale:E2}, cosine {cosine:F7}");
        }
    }
}

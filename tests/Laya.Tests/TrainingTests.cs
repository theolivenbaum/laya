using System.Text.Json;
using System.Text.Json.Nodes;
using Laya.Io;
using Laya.Models;
using Laya.Runtime;
using Laya.Training;
using Xunit;

namespace Laya.Tests;

/// <summary>The pieces of the training loop, each against the reference or a closed form.</summary>
public class TrainingTests
{
    private static string Fixture => Path.Combine(TestModels.FixtureRoot, "training-tiny");

    [Fact]
    public void ObjectiveMatchesTheNotebookLossAndGradient()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixture, "objective.json")));
        var root = document.RootElement;
        static float[] Floats(JsonElement e) => [.. e.EnumerateArray().Select(x => x.GetSingle())];

        var logits = root.GetProperty("logits").EnumerateArray().Select(Floats).ToArray();
        var targets = root.GetProperty("target").EnumerateArray().Select(Floats).ToArray();
        var types = root.GetProperty("qtypes").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var items = Enumerable.Range(0, logits.Length)
            .Select(i => new TrainingItem([0], new int[logits[i].Length], types[i], targets[i])).ToList();
        double[][][] draws = [.. root.GetProperty("draws").EnumerateArray().Select(g =>
            g.EnumerateArray().Select(i => i.EnumerateArray().Select(x => x.GetDouble()).ToArray()).ToArray())];

        var result = new RlcdObjective().Compute(items, logits, root.GetProperty("sigma").GetDouble(), draws);

        Assert.Equal(root.GetProperty("loss_ce").GetDouble(), result.CrossEntropy, 5);
        Assert.Equal(root.GetProperty("loss_rl").GetDouble(), result.PolicyLoss, 4);
        Assert.Equal(root.GetProperty("reward_mean").GetDouble(), result.MeanReward, 4);
        var expected = root.GetProperty("grad").EnumerateArray().Select(Floats).ToArray();
        for (int i = 0; i < expected.Length; ++i)
        {
            for (int k = 0; k < expected[i].Length; ++k) Assert.Equal(expected[i][k], result.LogitGradients[i][k], 4);
        }
    }

    [Fact]
    public void AdamWAndTheCosineScheduleMatchPyTorch()
    {
        // torch.optim.AdamW(lr=0.1, weight_decay=0.01) + CosineAnnealingLR(T_max=3, eta_min=1e-6).
        var parameter = new Parameter("w", [4], [0.5f, -1f, 2f, 0f], "F32");
        var optimizer = new AdamW([new ParameterGroup([parameter], 0.1)], weightDecay: 0.01);
        var schedule = new CosineSchedule(optimizer, 3, 1e-6);
        float[][] gradients = [[0.1f, -0.2f, 0.3f, 0f], [0.5f, 0.5f, -0.5f, 0f], [-0.3f, 0.1f, 0.2f, 1f]];
        float[][] expected =
        [
            [0.39950001f, -0.89900005f, 1.898f, 0f],
            [0.3346217f, -0.93149203f, 1.91859365f, 0f],
            [0.32829964f, -0.94257891f, 1.9182955f, -0.01597082f],
        ];
        double[] rates = [0.07500025, 0.025000750000000013, 1e-6];

        for (int step = 0; step < 3; ++step)
        {
            gradients[step].CopyTo(parameter.GradientBuffer(), 0);
            optimizer.Step();
            schedule.Step();
            for (int i = 0; i < 4; ++i) Assert.Equal(expected[step][i], parameter.Data[i], 5);
            Assert.Equal(rates[step], optimizer.Groups[0].LearningRate, 9);
        }
    }

    [Fact]
    public void ClippingScalesToTheMaximumNorm()
    {
        var a = new Parameter("a", [2], [0f, 0f], "F32");
        var b = new Parameter("b", [1], [0f], "F32");
        a.GradientBuffer()[0] = 3f;
        a.GradientBuffer()[1] = 0f;
        b.GradientBuffer()[0] = 4f;
        Assert.Equal(5d, Gradients.ClipByGlobalNorm([a, b], 1.0), 6);
        Assert.Equal(0.6f, a.Gradient![0], 5);
        Assert.Equal(0.8f, b.Gradient![0], 5);
    }

    [Fact]
    public void TemperatureFittingRecoversAKnownTemperature()
    {
        // Targets drawn from softmax(z / 2) are best explained by T = 2.
        var random = new Random(5);
        var samples = new List<CalibrationSample>();
        for (int i = 0; i < 400; ++i)
        {
            var logits = Enumerable.Range(0, 4).Select(_ => (float)(random.NextDouble() * 8 - 4)).ToArray();
            var target = logits.Select(z => z / 2f).ToArray();
            Numerics.SimdOps.Softmax(target);
            samples.Add(new CalibrationSample(0, logits, target));
        }
        Assert.Equal(2f, TemperatureFitting.FitOne(samples), 3);
        Assert.Equal(1f, TemperatureFitting.FitOne(samples.Take(5).ToList()));

        var fit = TemperatureFitting.Fit(samples, minimumBucketSamples: 30);
        Assert.Equal(2f, fit.ByType[0], 3);
        Assert.Equal(TemperatureFitting.MissingTypeTemperature, fit.ByType[2]);
        Assert.Equal(2f, fit.ByBucket["choice:3-5"], 3);
    }

    [Fact]
    public void TargetsFollowTheNotebook()
    {
        var gold = new GoldAnswer("b", new Dictionary<string, double> { ["a"] = 1, ["b"] = 3 }, null, null);
        Assert.Equal([0.25f, 0.75f, 0f], DecisionDataset.Target(Question.Choice("x", "a", "b", "c"), gold));
        var none = new GoldAnswer(null, new Dictionary<string, double>(), null, null);
        Assert.Equal([0.5f, 0.5f], DecisionDataset.Target(Question.Noul("x"), none));
        Assert.Equal([1 / 3f, 1 / 3f, 1 / 3f], DecisionDataset.Target(Question.Score("x", "l", "m", "h"), none));
    }

    [Fact]
    public void ParsesHubRowsWithStringEncodedFields()
    {
        string row = JsonSerializer.Serialize(new
        {
            id = "c1",
            workflow = "customer_service",
            state = "{\"body\": \"charged twice\"}",
            questions = "{\"urgent\": {\"type\": \"noul\", \"instructions\": \"Is it urgent?\"}}",
            gold = "{\"urgent\": {\"label\": \"true\", \"noul\": 0.8, \"probabilities\": {\"false\": 0.2, \"true\": 0.8}}}",
        });
        using var document = JsonDocument.Parse(row);
        var parsed = DecisionDataset.ParseCase(document.RootElement);
        Assert.Equal("customer_service", parsed.Workflow);
        Assert.Equal("{\"body\": \"charged twice\"}", SequenceBuilder.SerializeState(parsed.State));
        Assert.Equal(QuestionType.Noul, parsed.Questions["urgent"].Type);
        Assert.Equal(0.8, parsed.Gold["urgent"].Noul);
    }

    [Fact]
    public void ASavedCheckpointIsWhatInferenceRuns()
    {
        // Train a couple of steps, save, then load the result with the *inference* model: its logits
        // must equal the trainer's own evaluation pass on the saved (fp16-rounded) weights.
        var model = TrainableDecisionModel.FromDirectory(Fixture);
        model.SetTrainableEncoderLayers(null);
        var random = new Random(9);
        var items = Enumerable.Range(0, 4).Select(i =>
        {
            int length = 10 + 3 * i;
            int[] ids = [.. Enumerable.Range(0, length).Select(_ => random.Next(3, 97))];
            int k = 2 + i % 3;
            int[] markers = [.. Enumerable.Range(0, k).Select(o => 1 + 2 * o)];
            var target = Enumerable.Range(0, k).Select(_ => (float)random.NextDouble()).ToArray();
            float sum = target.Sum();
            return new TrainingItem(ids, markers, i % 3, [.. target.Select(t => t / sum)]);
        }).ToList();

        var optimizer = new AdamW([new ParameterGroup([.. model.Parameters.Where(p => p.Trainable)], 1e-3)]);
        var objective = new RlcdObjective();
        for (int step = 0; step < 2; ++step)
        {
            var (logits, tape) = model.Forward(items, training: true, keepTape: true, seed: (ulong)step);
            model.Backward(tape!, objective.Compute(items, logits, 0.4, random).LogitGradients);
            optimizer.Step();
            model.ZeroGradients();
        }

        string directory = Path.Combine(Path.GetTempPath(), "laya-train-" + Guid.NewGuid().ToString("N"));
        try
        {
            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(Fixture, "rl_agent_config.json")))!.AsObject();
            model.Save(directory, Fixture, config);

            var reloaded = TrainableDecisionModel.FromDirectory(directory);
            var (expected, _) = reloaded.Forward(items, training: false, keepTape: false);

            using var weights = new SafetensorsFile(Path.Combine(directory, "model.safetensors"));
            var inference = new DecisionModel(ModernBertConfig.Load(Path.Combine(directory, "encoder", "config.json")),
                LayaConfig.Load(Path.Combine(directory, "rl_agent_config.json")), weights);
            var outputs = inference.Forward([.. items.Select(i => new DecisionModel.BatchItem(i.TokenIds, i.MarkerPositions, i.QuestionType))]);
            for (int i = 0; i < items.Count; ++i)
            {
                for (int k = 0; k < items[i].Options; ++k) Assert.Equal(expected[i][k], outputs[i].OptionLogits[k], 4);
            }

            // Training moved the weights, and the file kept every tensor in its original dtype.
            using var original = new SafetensorsFile(Path.Combine(Fixture, "model.safetensors"));
            Assert.Equal(original.Entries.Keys.Order(), weights.Entries.Keys.Order());
            Assert.All(weights.Entries.Values, e => Assert.Equal(original.Entries[e.Name].DType, e.DType));
            Assert.NotEqual(original.ReadFloat32("scorer.1.weight"), weights.ReadFloat32("scorer.1.weight"));
        }
        finally
        {
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void SafetensorsRoundTrips()
    {
        string path = Path.Combine(Path.GetTempPath(), "laya-st-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            SafetensorsWriter.Write(path,
            [
                new SafetensorsTensor("a", [2, 3], [1f, -2f, 3.5f, 0f, 65504f, 1e-3f], "F16"),
                new SafetensorsTensor("b", [2], [0.1f, -7.25f], "F32"),
                new SafetensorsTensor("c", [1], [3.140625f], "BF16"),
            ]);
            using var file = new SafetensorsFile(path);
            Assert.Equal([2, 3], file.Entry("a").Shape);
            Assert.Equal([1f, -2f, 3.5f, 0f, 65504f, (float)(Half)1e-3f], file.ReadFloat32("a"));
            Assert.Equal([0.1f, -7.25f], file.ReadFloat32("b"));
            Assert.Equal([3.140625f], file.ReadFloat32("c"));
        }
        finally
        {
            File.Delete(path);
        }
    }
}

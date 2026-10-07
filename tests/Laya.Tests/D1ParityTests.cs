using System.Globalization;
using System.Text.Json;
using Laya.D1;
using Laya.Diagnostics;
using Laya.Io;
using Laya.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace Laya.Tests;

/// <summary>
/// Parity of the d1 port against LiquidAI's own PyTorch code (<c>modeling_d1.py</c> and friends),
/// as dumped by <c>tools/dump_d1_reference.py</c>.
///
/// <para>Four cases pin both of the reference's code paths: a lone question (the whole prompt as
/// one causal chain) and several questions over one state (the tree: the state is the trunk, each
/// question a branch). Each compares the rendered prompt's token ids exactly, then the embeddings,
/// every decoder layer, the operator and MLP outputs inside the first conv and the first attention
/// layer, the final norm, the log-softmax over the whole vocabulary at every answer slot, the option
/// probabilities, and the answers.</para>
///
/// <para>With the full dumps in <c>artifacts/dumps/d1</c> every value of every tensor is compared;
/// otherwise the committed fixture's sampled values (the start of the first and of the last row)
/// are.</para>
/// </summary>
public class D1ParityTests(ITestOutputHelper output, D1ParityTests.Loaded loaded) : IClassFixture<D1ParityTests.Loaded>
{
    /// <summary>
    /// Bound on max |Δ| relative to the tensor's own magnitude. The residual stream reaches ~25 by
    /// layer 4; 30 layers of fp32 GEMMs whose summation order differs from MKL's leave a few 1e-5 of
    /// that.
    /// </summary>
    private const double RelativeTolerance = 1e-4;

    /// <summary>
    /// Bounds for the picture cases. The SigLIP2 tower carries values up to ~1600 by its last layer,
    /// and 27 layers of fp32 round-off leave ~1e-4 of that (5e-4 allows for it); the language model
    /// then carries the projected features' ~2e-5 differences through 30 layers on the image rows,
    /// to ~1e-3 of the residual stream. <see cref="LanguageModelOverReferenceImageFeaturesMatchesPyTorch"/>
    /// shows that is amplified noise and not the language model: fed PyTorch's own features, the
    /// image rows match to the text cases' 1e-4. The log-probabilities and answers keep their bounds.
    /// </summary>
    private const double VisionTowerTolerance = 5e-4;
    private const double VisionLanguageModelTolerance = 2e-3;

    /// <summary>Bound on the log-probabilities at the answer slot, which feed the answer directly.</summary>
    private const double LogProbabilityTolerance = 2e-3;

    /// <summary>The checkpoint, loaded once for the whole class: repacking 2.6B weights takes a while.</summary>
    public sealed class Loaded : IDisposable
    {
        public D1Agent? Agent { get; } = TestModels.D1Available ? D1Agent.FromDirectory(TestModels.D1Directory) : null;

        public void Dispose() => Agent?.Dispose();
    }

    [D1ModelFact]
    public void SingleQuestionChainMatchesPyTorch() => Compare("single");

    [D1ModelFact]
    public void QuestionTreeMatchesPyTorch() => Compare("tree");

    [D1ModelFact]
    public void JsonStateMatchesPyTorch() => Compare("json");

    [D1ModelFact]
    public void MultilingualTreeMatchesPyTorch() => Compare("ruling");

    /// <summary>A picture that fits, resized whole; the picture is the whole state; one question.</summary>
    [D1ModelFact]
    public void SmallImageMatchesPyTorch() => Compare("small", vision: true);

    /// <summary>A large picture: two 512-pixel tiles and a thumbnail, plus text, two questions (the tree).</summary>
    [D1ModelFact]
    public void TiledImageMatchesPyTorch() => Compare("tiled", vision: true);

    /// <summary>Over a megapixel: Pillow's bicubic shrinks it first, then it is tiled.</summary>
    [D1ModelFact]
    public void CappedImageMatchesPyTorch() => Compare("capped", vision: true);

    /// <summary>
    /// The language model alone over PyTorch's own projected image features: with the tower's fp32
    /// noise taken out, the image rows must match as tightly as text does. This separates "the LM is
    /// right and the tower's noise grows through 30 layers" from a bug in the multimodal LM path.
    /// </summary>
    [D1ModelFact]
    public void LanguageModelOverReferenceImageFeaturesMatchesPyTorch()
    {
        string dump = Path.Combine(TestModels.RepositoryRoot, "artifacts", "dumps", "d1-vision", "tiled.safetensors");
        if (!File.Exists(dump)) return;   // needs the full dump; the fixture samples too little of the features
        using var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(TestModels.FixtureRoot, "torch-d1-vision.json")));
        var meta = fixture.RootElement.GetProperty("cases").GetProperty("tiled");
        using var reference = new SafetensorsFile(dump);
        var features = new List<float>();
        for (int t = 0; reference.Contains($"image0.tile{t}.projected"); ++t) features.AddRange(reference.ReadFloat32($"image0.tile{t}.projected"));

        int[] ids = ExpectedIds(meta);
        int trunk = meta.GetProperty("trunk").GetArrayLength();
        var branches = meta.GetProperty("ids").EnumerateArray().Select(r => r.GetArrayLength()).ToArray();
        var recorder = new StateRecorder();
        Agent.Forward(ids, Models.Lfm2Tree.Branched(trunk, branches), [.. features], recorder);
        var actual = recorder.States.ToDictionary(s => s.Name, s => s, StringComparer.Ordinal);

        var failures = new List<string>();
        foreach (var (name, entry) in reference.Entries)
        {
            if (!name.StartsWith("layers.", StringComparison.Ordinal) && name != "final_norm") continue;
            Check(name, entry.Shape, reference.ReadFloat32(name), actual, failures, sampled: false);
        }
        Assert.Empty(failures);
    }

    private D1Agent Agent => loaded.Agent!;

    private void Compare(string name, bool vision = false)
    {
        string fixturePath = Path.Combine(TestModels.FixtureRoot, vision ? "torch-d1-vision.json" : "torch-d1.json");
        Assert.True(File.Exists(fixturePath), $"missing parity fixture {fixturePath}");
        using var fixture = JsonDocument.Parse(File.ReadAllText(fixturePath));
        var meta = fixture.RootElement.GetProperty("cases").GetProperty(name);

        object? state = meta.GetProperty("state") switch
        {
            { ValueKind: JsonValueKind.String } text => text.GetString(),
            { ValueKind: JsonValueKind.Null } => null,
            var other => other.Clone(),
        };
        var questions = QuestionJson.Parse(meta.GetProperty("questions"));
        var questionList = questions.Select(q => q.Value).ToArray();

        // The prompt first: every format string of prompt.py, and the tokenizer under it.
        RgbImage[] images = [];
        if (vision)
        {
            images = [RgbImage.Load(Path.Combine(TestModels.FixtureRoot, "d1-images", meta.GetProperty("image").GetString()!))];
            Assert.Equal(meta.GetProperty("prefix").GetString(), Agent.Prompt.Prefix(state, D1ImageProcessor.ImageToken));
            var suffixes = meta.GetProperty("suffixes").EnumerateArray().Select(p => p.GetString()!).ToArray();
            for (int i = 0; i < questionList.Length; ++i) Assert.Equal(suffixes[i], Agent.Prompt.Suffix(questionList[i]));
        }
        else
        {
            var prompts = meta.GetProperty("prompts").EnumerateArray().Select(p => p.GetString()!).ToArray();
            for (int i = 0; i < questionList.Length; ++i) Assert.Equal(prompts[i], Agent.Prompt.Render(state, questionList[i]));
        }
        int[] expectedIds = ExpectedIds(meta);

        var recorder = new StateRecorder();
        var result = vision ? Agent.SystemOne(state, questions, images, recorder) : Agent.SystemOne(state, questions, recorder);
        var actual = recorder.States.ToDictionary(s => s.Name, s => s, StringComparer.Ordinal);

        Assert.Equal(expectedIds, actual["input_ids"].Values.Select(v => (int)v).ToArray());
        Assert.Equal(meta.GetProperty("usage").GetProperty("input_tokens").GetInt32(), result.Usage.InputTokens);

        var failures = new List<string>();
        int compared = 0;
        string fullDump = Path.Combine(TestModels.RepositoryRoot, "artifacts", "dumps", vision ? "d1-vision" : "d1", name + ".safetensors");
        if (File.Exists(fullDump))
        {
            using var reference = new SafetensorsFile(fullDump);
            foreach (var (tensorName, entry) in reference.Entries)
            {
                float[] expected = reference.ReadFloat32(tensorName);
                compared++;
                Check(tensorName, entry.Shape, expected, actual, failures, sampled: false, vision);
            }
        }
        else
        {
            foreach (var tensor in fixture.RootElement.GetProperty("tensors").EnumerateArray())
            {
                string full = tensor.GetProperty("name").GetString()!;
                if (!full.StartsWith(name + ".", StringComparison.Ordinal)) continue;
                int[] shape = [.. tensor.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32())];
                float[] expected = [.. tensor.GetProperty("values").EnumerateArray().Select(v => v.GetSingle())];
                compared++;
                Check(full[(name.Length + 1)..], shape, expected, actual, failures, sampled: true, vision);
            }
        }

        output.WriteLine($"{name}: compared {compared} tensors ({(File.Exists(fullDump) ? "full" : "sampled")})");
        Assert.Empty(failures);
        CompareAnswers(meta.GetProperty("answers"), result);
    }

    private static int[] ExpectedIds(JsonElement meta)
    {
        var ids = new List<int>();
        if (meta.GetProperty("trunk") is { ValueKind: JsonValueKind.Array } trunk)
        {
            ids.AddRange(trunk.EnumerateArray().Select(i => i.GetInt32()));
        }
        foreach (var row in meta.GetProperty("ids").EnumerateArray()) ids.AddRange(row.EnumerateArray().Select(i => i.GetInt32()));
        return [.. ids];
    }

    private void Check(string name, int[] shape, float[] expected, Dictionary<string, RecordedState> actual,
        List<string> failures, bool sampled, bool vision = false)
    {
        if (!actual.TryGetValue(name, out var got))
        {
            failures.Add($"{name}: not recorded by the .NET implementation");
            return;
        }
        Assert.True(shape.SequenceEqual(got.Shape), $"{name}: shape [{string.Join(",", got.Shape)}] vs [{string.Join(",", shape)}]");

        float[] values = got.Values;
        if (sampled && shape.Length == 2 && shape[0] > 2)
        {
            // The fixture samples the first half of row 0 and the first half of the last row.
            int half = expected.Length / 2;
            values = [.. got.Values.AsSpan(0, half), .. got.Values.AsSpan((shape[0] - 1) * shape[1], half)];
        }

        double maxAbsolute = 0, scale = 1;
        for (int i = 0; i < expected.Length; ++i)
        {
            maxAbsolute = Math.Max(maxAbsolute, Math.Abs(expected[i] - values[i]));
            scale = Math.Max(scale, Math.Abs(expected[i]));
        }

        double relative = !vision ? RelativeTolerance
            : name.StartsWith("image", StringComparison.Ordinal) ? VisionTowerTolerance
            : VisionLanguageModelTolerance;
        double bound = name.StartsWith("logz", StringComparison.Ordinal) || name.StartsWith("probabilities", StringComparison.Ordinal)
            ? LogProbabilityTolerance
            : relative * scale;
        string line = string.Format(CultureInfo.InvariantCulture, "{0,-28} max|Δ| = {1:E2}  (scale {2:F2}, bound {3:E1})",
            name, maxAbsolute, scale, bound);
        output.WriteLine(line);
        if (maxAbsolute > bound) failures.Add(line);
    }

    private void CompareAnswers(JsonElement expected, DecisionResult actual)
    {
        foreach (var property in expected.EnumerateObject())
        {
            Assert.True(actual.TryGet(property.Name, out var answer), $"missing answer '{property.Name}'");
            var reference = property.Value;
            Assert.Equal(reference.GetProperty("type").GetString(), answer.Type);
            if (reference.TryGetProperty("choice", out var choice)) Assert.Equal(choice.GetString(), answer.Choice);
            if (reference.TryGetProperty("score", out var score)) Assert.Equal(score.GetDouble(), answer.Score!.Value, 3);
            if (reference.TryGetProperty("noul", out var noul)) Assert.Equal(noul.GetDouble(), answer.Noul!.Value, 3);
            if (reference.TryGetProperty("confidence", out var confidence)) Assert.Equal(confidence.GetDouble(), answer.Confidence, 3);
            if (reference.TryGetProperty("probabilities", out var probabilities))
            {
                foreach (var option in probabilities.EnumerateObject())
                {
                    Assert.Equal(option.Value.GetDouble(), answer.ProbabilityOf(option.Name), 3);
                }
            }
            if (reference.TryGetProperty("legend", out var legend))
            {
                foreach (var level in legend.EnumerateObject())
                {
                    Assert.Equal(level.Value.GetString(), answer.Legend!.First(l => l.Key == level.Name).Value);
                }
            }
            output.WriteLine($"  {property.Name}: {answer.Type} matched");
        }
    }
}

/// <summary>A fact that skips itself when the d1 checkpoint is not on disk.</summary>
internal sealed class D1ModelFactAttribute : FactAttribute
{
    public D1ModelFactAttribute()
    {
        if (!TestModels.D1Available)
        {
            Skip = $"d1 is not in {TestModels.D1Directory}; run: dotnet run --project src/Laya.Cli -- download --model d1 " +
                   "--cache artifacts/models-cache (or set LAYA_D1_MODEL)";
        }
    }
}

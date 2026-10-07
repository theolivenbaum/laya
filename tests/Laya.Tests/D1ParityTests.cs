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

    private D1Agent Agent => loaded.Agent!;

    private void Compare(string name)
    {
        string fixturePath = Path.Combine(TestModels.FixtureRoot, "torch-d1.json");
        Assert.True(File.Exists(fixturePath), $"missing parity fixture {fixturePath}");
        using var fixture = JsonDocument.Parse(File.ReadAllText(fixturePath));
        var meta = fixture.RootElement.GetProperty("cases").GetProperty(name);

        object? state = meta.GetProperty("state") is { ValueKind: JsonValueKind.String } text
            ? text.GetString()
            : meta.GetProperty("state").Clone();
        var questions = QuestionJson.Parse(meta.GetProperty("questions"));

        // The prompt first: every format string of prompt.py, and the tokenizer under it.
        var prompts = meta.GetProperty("prompts").EnumerateArray().Select(p => p.GetString()!).ToArray();
        var questionList = questions.Select(q => q.Value).ToArray();
        for (int i = 0; i < questionList.Length; ++i)
        {
            Assert.Equal(prompts[i], Agent.Prompt.Render(state, questionList[i]));
        }
        int[] expectedIds = ExpectedIds(meta);

        var recorder = new StateRecorder();
        var result = Agent.SystemOne(state, questions, recorder);
        var actual = recorder.States.ToDictionary(s => s.Name, s => s, StringComparer.Ordinal);

        Assert.Equal(expectedIds, actual["input_ids"].Values.Select(v => (int)v).ToArray());
        Assert.Equal(meta.GetProperty("usage").GetProperty("input_tokens").GetInt32(), result.Usage.InputTokens);

        var failures = new List<string>();
        int compared = 0;
        string fullDump = Path.Combine(TestModels.RepositoryRoot, "artifacts", "dumps", "d1", name + ".safetensors");
        if (File.Exists(fullDump))
        {
            using var reference = new SafetensorsFile(fullDump);
            foreach (var (tensorName, entry) in reference.Entries)
            {
                float[] expected = reference.ReadFloat32(tensorName);
                compared++;
                Check(tensorName, entry.Shape, expected, actual, failures, sampled: false);
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
                Check(full[(name.Length + 1)..], shape, expected, actual, failures, sampled: true);
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
        List<string> failures, bool sampled)
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

        double bound = name.StartsWith("logz", StringComparison.Ordinal) || name.StartsWith("probabilities", StringComparison.Ordinal)
            ? LogProbabilityTolerance
            : RelativeTolerance * scale;
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

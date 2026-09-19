using System.Globalization;
using System.Text.Json;
using Laya.Diagnostics;
using Laya.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace Laya.Tests;

/// <summary>
/// The parity suite: run a checkpoint through the .NET implementation and compare every layer
/// against tensors dumped from the original PyTorch model by <c>tools/dump_reference.py</c>.
///
/// <para>The fixtures carry the state and questions they were produced from, so a test cannot
/// drift away from its golden file, and they carry the reference answers as well — matching
/// hidden states is necessary, but agreeing on the answer is the point.</para>
/// </summary>
public class ParityTests(ITestOutputHelper output)
{
    /// <summary>
    /// Absolute tolerance. ModernBERT-large carries activations in the tens of thousands through
    /// its residual stream, so the bound scales with the tensor's own magnitude; the constant is
    /// fp32 round-off over a 28-layer network, not a fudge factor.
    /// </summary>
    private const double AbsoluteTolerance = 2e-2;

    private const double RelativeTolerance = 5e-3;

    [ModelFact("english")]
    public void EnglishCheckpointMatchesPyTorch() => Compare("english", "torch-english-triage.json");

    [ModelFact("multilingual")]
    public void MultilingualCheckpointMatchesPyTorch()
        => Compare("multilingual", "torch-multilingual-triage.json");

    [ModelFact("typed-decisions")]
    public void TypedDecisionsCheckpointMatchesPyTorch()
        => Compare("typed-decisions", "torch-typed-decisions.json");

    private void Compare(string checkpoint, string fixtureName)
    {
        string path = Path.Combine(TestModels.FixtureRoot, fixtureName);
        Assert.True(File.Exists(path), $"missing parity fixture {path}");

        using var fixture = JsonDocument.Parse(File.ReadAllText(path));
        var root = fixture.RootElement;
        var input = root.GetProperty("input");
        object? state = ReadState(input.GetProperty("state"));
        var questions = ReadQuestions(input.GetProperty("questions"));

        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory(checkpoint));
        var recorder = new StateRecorder();
        var result = agent.SystemOne(state, questions, recorder);

        var actual = recorder.States.ToDictionary(s => s.Name, s => s, StringComparer.Ordinal);
        int compared = 0;
        var failures = new List<string>();

        foreach (var tensor in root.GetProperty("tensors").EnumerateArray())
        {
            string name = tensor.GetProperty("name").GetString()!;
            if (!actual.TryGetValue(name, out var got))
            {
                failures.Add($"{name}: not recorded by the .NET implementation");
                continue;
            }

            int[] shape = [.. tensor.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32())];
            Assert.Equal(shape, got.Shape);

            double maxAbsolute = 0;
            double maxRelative = 0;
            int index = 0;
            foreach (var value in tensor.GetProperty("values").EnumerateArray())
            {
                if (index >= got.Values.Length) break;
                double expected = value.GetDouble();
                double difference = Math.Abs(expected - got.Values[index]);
                maxAbsolute = Math.Max(maxAbsolute, difference);
                maxRelative = Math.Max(maxRelative, difference / Math.Max(Math.Abs(expected), 1e-6));
                index++;
            }

            double scale = Math.Max(1d, Math.Abs(tensor.GetProperty("max_abs").GetDouble()));
            bool withinAbsolute = maxAbsolute <= AbsoluteTolerance * scale;
            if (!withinAbsolute && maxRelative > RelativeTolerance)
            {
                failures.Add(string.Format(CultureInfo.InvariantCulture,
                    "{0}: max |Δ| = {1:E3} (scale {2:E3}), max relative Δ = {3:E3}",
                    name, maxAbsolute, scale, maxRelative));
            }
            compared++;
        }

        output.WriteLine($"{checkpoint}: compared {compared} tensors against {fixtureName}");
        Assert.Empty(failures);

        CompareAnswers(root.GetProperty("answers"), result);
    }

    private void CompareAnswers(JsonElement expected, DecisionResult actual)
    {
        foreach (var property in expected.EnumerateObject())
        {
            Assert.True(actual.TryGet(property.Name, out var answer), $"missing answer '{property.Name}'");
            var reference = property.Value;

            Assert.Equal(reference.GetProperty("type").GetString(), answer.Type);

            if (reference.TryGetProperty("choice", out var choice))
            {
                Assert.Equal(choice.GetString(), answer.Choice);
            }
            if (reference.TryGetProperty("score", out var score))
            {
                Assert.Equal(score.GetDouble(), answer.Score!.Value, 3);
            }
            if (reference.TryGetProperty("noul", out var noul))
            {
                Assert.Equal(noul.GetDouble(), answer.Noul!.Value, 3);
            }
            Assert.Equal(reference.GetProperty("confidence").GetDouble(), answer.Confidence, 3);

            if (reference.TryGetProperty("probabilities", out var probabilities))
            {
                foreach (var option in probabilities.EnumerateObject())
                {
                    Assert.Equal(option.Value.GetDouble(), answer.ProbabilityOf(option.Name), 3);
                }
            }

            output.WriteLine($"  {property.Name}: matched");
        }
    }

    private static object? ReadState(JsonElement state)
        => state.ValueKind == JsonValueKind.String ? state.GetString() : state.Clone();

    /// <summary>
    /// Rebuilds the question set from the fixture. This is the same shape the Python API takes,
    /// and the order of the JSON object is the label order, so it is preserved.
    /// </summary>
    private static QuestionSet ReadQuestions(JsonElement questions)
    {
        var set = new QuestionSet();
        foreach (var property in questions.EnumerateObject())
        {
            var definition = property.Value;
            string type = definition.GetProperty("type").GetString()!;
            object instructions = definition.GetProperty("instructions").GetString()!;
            bool hasCriteria = definition.TryGetProperty("criteria", out var criteria)
                && criteria.ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined);

            set.Add(property.Name, QuestionTypes.Parse(type) switch
            {
                QuestionType.Choice when criteria.ValueKind == JsonValueKind.Array =>
                    Question.Choice(instructions, [.. criteria.EnumerateArray().Select(c => c.GetString()!)]),
                QuestionType.Choice => Question.Choice(instructions, criteria.EnumerateObject()
                    .Select(p => new KeyValuePair<string, object?>(p.Name,
                        p.Value.ValueKind == JsonValueKind.String ? p.Value.GetString() : null))),
                QuestionType.Score => Question.Score(instructions,
                    [.. criteria.EnumerateArray().Select(object? (c) => c.GetString())]),
                _ when !hasCriteria => Question.Noul(instructions),
                _ => Question.Noul(instructions,
                    criteria.TryGetProperty("true", out var t) ? t.GetString() : null,
                    criteria.TryGetProperty("false", out var f) ? f.GetString() : null),
            });
        }
        return set;
    }
}

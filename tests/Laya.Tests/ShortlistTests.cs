using Laya.Runtime;
using Xunit;

namespace Laya.Tests;

/// <summary>The opt-in embedding shortlist. Ported from <c>.reference/tests/test_shortlist.py</c>.</summary>
public class ShortlistTests
{
    /// <summary>Looks each text up in a table, and records every call.</summary>
    private sealed class TableEmbed(Dictionary<string, float[]> vectors)
    {
        public List<IReadOnlyList<string>> Calls { get; } = [];

        public float[][] Embed(IReadOnlyList<string> texts)
        {
            Calls.Add([.. texts]);
            return [.. texts.Select(t => vectors[t])];
        }
    }

    private static Question Criteria(string instructions = "") => Question.Choice(instructions,
        ("alpha", null), ("beta", ""), ("gamma", "mid"), ("delta", "same"));

    private static TableEmbed Table() => new(new Dictionary<string, float[]>
    {
        ["pay me"] = [1f, 0f],
        ["alpha"] = [1f, 0f],
        ["beta"] = [0f, 1f],
        ["gamma: mid"] = [0.6f, 0.8f],
        ["delta: same"] = [1f, 0f],
    });

    [Fact]
    public void KeepsTheTopKWithTiesInInputOrder()
    {
        var table = Table();
        Assert.Equal(["alpha", "delta"], Shortlist.ShortlistChoice("pay me", Criteria(), table.Embed, k: 2));
        Assert.Single(table.Calls);
        Assert.Equal("pay me", table.Calls[0][0]);
        Assert.Equal(Criteria().RenderOptions(), table.Calls[0].Skip(1));
        Assert.Equal(["alpha"], Shortlist.ShortlistChoice("pay me", Criteria(), table.Embed, k: 1));
        Assert.Equal(["alpha", "delta", "gamma"], Shortlist.ShortlistChoice("pay me", Criteria(), table.Embed, k: 3));
    }

    [Fact]
    public void ZeroAndNonFiniteVectorsScoreZero()
    {
        var zero = new TableEmbed(new Dictionary<string, float[]>
        {
            ["pay me"] = [0f, 0f], ["alpha"] = [1f, 0f], ["beta"] = [0f, 1f],
            ["gamma: mid"] = [0.6f, 0.8f], ["delta: same"] = [3f, 4f],
        });
        Assert.Equal(["alpha", "beta"], Shortlist.ShortlistChoice("pay me", Criteria(), zero.Embed, k: 2));

        var nan = new TableEmbed(new Dictionary<string, float[]>
        {
            ["pay me"] = [1f, 0f], ["alpha"] = [float.NaN, float.NaN], ["beta"] = [1f, 0f],
        });
        Assert.Equal(["beta"], Shortlist.ShortlistChoice("pay me", Question.Choice("", "alpha", "beta"), nan.Embed, k: 1));
    }

    [Fact]
    public void InstructionsAndStructuredStateFormTheQuery()
    {
        var table = new TableEmbed(new Dictionary<string, float[]>
        {
            ["Classify\npay me"] = [0f, 1f], ["alpha"] = [1f, 0f], ["beta"] = [0f, 1f], ["gamma"] = [0f, 0.2f],
        });
        Assert.Equal(["beta", "gamma"],
            Shortlist.ShortlistChoice("pay me", Question.Choice("Classify", "alpha", "beta", "gamma"), table.Embed, k: 2));

        var json = new TableEmbed(new Dictionary<string, float[]>
        {
            ["Classify\n{\"text\": \"hi\"}"] = [1f, 0f], ["alpha"] = [1f, 0f], ["beta"] = [0f, 1f],
        });
        var state = new Dictionary<string, object?> { ["text"] = "hi" };
        Assert.Equal(["alpha"], Shortlist.ShortlistChoice(state, Question.Choice("Classify", "alpha", "beta"), json.Embed, k: 1));
    }

    [Fact]
    public void PassesThroughWithoutEmbeddingWhenKCoversEveryLabel()
    {
        EmbedFunction boom = _ => throw new InvalidOperationException("must not be called");
        Assert.Equal(["alpha", "beta", "gamma", "delta"], Shortlist.ShortlistChoice("pay me", Criteria(), boom, k: 4));
        Assert.Equal(["alpha", "beta", "gamma", "delta"], Shortlist.ShortlistChoice("pay me", Criteria(), boom, k: 20));
    }

    [Fact]
    public void RejectsBadInput()
    {
        var table = Table();
        Assert.Throws<ArgumentOutOfRangeException>(() => Shortlist.ShortlistChoice("pay me", Criteria(), table.Embed, k: 0));
        EmbedFunction wrongShape = texts => [[1f]];
        Assert.Throws<InvalidOperationException>(() => Shortlist.ShortlistChoice("pay me", Criteria(), wrongShape, k: 1));
    }

    /// <summary>Records the questions it was asked and answers each choice with its first option.</summary>
    private sealed class RecordingEngine : IDecisionEngine
    {
        public List<QuestionSet> Calls { get; } = [];

        public DecisionResult SystemOne(object? state, QuestionSet questions)
        {
            Calls.Add(questions);
            return new DecisionResult
            {
                Model = "stub",
                Answers = [.. questions.Select(q => new KeyValuePair<string, Answer>(q.Key, new Answer
                {
                    Type = QuestionTypes.Name(q.Value.Type),
                    Choice = q.Value.OptionKeys.FirstOrDefault(),
                    Confidence = 1,
                    Action = new ActionEstimate(0),
                }))],
                Usage = new Usage(0, 0),
            };
        }

        public void Dispose() { }
    }

    [Fact]
    public void PredictShortlistReducesChoicesAndLeavesTheRestAlone()
    {
        var table = new TableEmbed(new Dictionary<string, float[]>
        {
            ["Which team?\npay me"] = [1f, 0f],
            ["billing"] = [0f, 1f], ["tech"] = [1f, 0f], ["sales"] = [0.9f, 0.1f], ["legal"] = [0f, 1f],
        });
        var questions = new QuestionSet()
            .Add("intent", Question.Choice("Which team?", "billing", "tech", "sales", "legal"))
            .Add("urgent", Question.Noul("Is it urgent?"))
            .Add("small", Question.Choice("Size?", "s", "m"));
        var engine = new RecordingEngine();

        var result = Shortlist.PredictShortlist(engine, "pay me", questions, table.Embed, k: 2);

        var asked = Assert.Single(engine.Calls);
        Assert.Equal(["tech", "sales"], asked["intent"].OptionKeys);
        Assert.Same(questions["urgent"], asked["urgent"]);
        Assert.Same(questions["small"], asked["small"]);
        Assert.Equal(["billing", "tech", "sales", "legal"], questions["intent"].OptionKeys);   // caller untouched

        Assert.Equal("tech", result["intent"].Choice);
        var meta = result.Shortlist!.ToDictionary(p => p.Key, p => p.Value);
        Assert.Equal(["tech", "sales"], meta["intent"].Labels);
        Assert.Equal((2, 4, false), (meta["intent"].K, meta["intent"].N, meta["intent"].Passthrough));
        Assert.True(meta["small"].Passthrough);
        Assert.Null(meta["small"].Scores);
        Assert.False(meta.ContainsKey("urgent"));
    }
}

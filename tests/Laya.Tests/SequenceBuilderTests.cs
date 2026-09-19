using Laya.Runtime;
using Laya.Tokenizers;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// Sequence layout and budget arithmetic. These need a real tokenizer (the budgets are counted in
/// tokens), so they use the English checkpoint's and skip when it is not on disk.
/// </summary>
public class SequenceBuilderTests
{
    private static HuggingFaceTokenizer Tokenizer()
        => HuggingFaceTokenizer.FromDirectory(
            Path.Combine(TestModels.CheckpointDirectory("english"), "tokenizer"));

    [ModelFact]
    public void LayoutIsClsHeadSepOptionsSepStateSep()
    {
        var tokenizer = Tokenizer();
        var question = Question.Choice("Which team?", "billing", "tech");
        var built = SequenceBuilder.Build(tokenizer, "I was charged twice", question);

        Assert.Equal(tokenizer.ClsTokenId, built.TokenIds[0]);
        Assert.Equal(tokenizer.SepTokenId, built.TokenIds[^1]);
        Assert.Equal(2, built.MarkerPositions.Length);

        // Every marker points at a [MASK], and the markers are in label order.
        foreach (int marker in built.MarkerPositions) Assert.Equal(tokenizer.MaskTokenId, built.TokenIds[marker]);
        Assert.True(built.MarkerPositions[0] < built.MarkerPositions[1]);
    }

    [ModelFact]
    public void MarkerCountAlwaysMatchesTheOptionCount()
    {
        var tokenizer = Tokenizer();
        foreach (var (_, question) in Presets.Triage())
        {
            var built = SequenceBuilder.Build(tokenizer, "some state", question);
            Assert.Equal(question.RenderOptions().Count, built.MarkerPositions.Length);
        }
    }

    [ModelFact]
    public void SequenceNeverExceedsMaxLength()
    {
        var tokenizer = Tokenizer();
        string huge = string.Join(' ', Enumerable.Repeat("the customer was charged twice", 500));
        var built = SequenceBuilder.Build(tokenizer, huge, Question.Noul("Is this urgent?"), maxLength: 512);
        Assert.Equal(512, built.TokenIds.Length);
    }

    [ModelFact]
    public void LeftTruncationKeepsTheEndOfTheState()
    {
        var tokenizer = Tokenizer();
        var question = Question.Noul("Is this urgent?");
        string state = string.Join(' ', Enumerable.Range(0, 400).Select(i => $"word{i}"));

        var right = SequenceBuilder.Build(tokenizer, state, question, truncateLeft: false);
        var left = SequenceBuilder.Build(tokenizer, state, question, truncateLeft: true);

        Assert.NotEqual(right.TokenIds, left.TokenIds);
        // The tail of a left-truncated sequence ends with the tail of the state, before the [SEP].
        var tail = tokenizer.Decode(left.TokenIds[^12..^1]);
        Assert.Contains("word399", tail, StringComparison.Ordinal);
    }

    [ModelFact]
    public void OversizedOptionsAreTrimmedToAnEqualShare()
    {
        var tokenizer = Tokenizer();
        string longText = string.Join(' ', Enumerable.Repeat("description", 80));
        var question = Question.Choice("Pick one",
            [.. Enumerable.Range(0, 12).Select(i =>
                new KeyValuePair<string, object?>($"option{i}", longText))]);

        var built = SequenceBuilder.Build(tokenizer, "state", question, headMaxLength: 192);

        // Every option still gets a marker; that is what the head needs to read them.
        Assert.Equal(12, built.MarkerPositions.Length);
        Assert.All(built.MarkerPositions, m => Assert.Equal(tokenizer.MaskTokenId, built.TokenIds[m]));
    }

    [ModelFact]
    public void MaskTokensInUserTextAreNeutralised()
    {
        var tokenizer = Tokenizer();
        // A state that contains the literal mask token must not create phantom option markers.
        var question = Question.Noul("Is this urgent?");
        var built = SequenceBuilder.Build(tokenizer, $"before {tokenizer.MaskToken} after", question);
        Assert.Equal(2, built.TokenIds.Count(id => id == tokenizer.MaskTokenId));
    }

    [ModelFact]
    public void StructuredStateIsSerializedAsJson()
    {
        var tokenizer = Tokenizer();
        var state = new List<KeyValuePair<string, object?>> { new("body", "charged twice") };
        var built = SequenceBuilder.Build(tokenizer, state, Question.Noul("Is this urgent?"));
        Assert.Contains("""{"body": "charged twice"}""", tokenizer.Decode(built.TokenIds), StringComparison.Ordinal);
    }
}

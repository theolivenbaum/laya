using Laya.Runtime;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// Criteria rendering: structured values must not crash and must not leak a runtime's own repr.
/// Ported from <c>.reference/tests/test_criteria.py</c>, which is itself a regression suite for a
/// bug where dict-valued criteria crashed <c>noul</c> and stringified as Python reprs elsewhere.
/// </summary>
public class CriteriaTests
{
    [Fact]
    public void StringCriterionPassesThrough()
        => Assert.Equal("phishing or scam", Question.RenderCriterion("phishing or scam"));

    [Fact]
    public void StructuredCriteriaBecomeJson()
    {
        Assert.Equal("""{"desc": "phishing"}""", Question.RenderCriterion(
            new List<KeyValuePair<string, object?>> { new("desc", "phishing") }));
        Assert.Equal("""["a", "b"]""", Question.RenderCriterion(new[] { "a", "b" }));
        Assert.Equal("3", Question.RenderCriterion(3));
        Assert.Equal("false", Question.RenderCriterion(false));
    }

    [Fact]
    public void NonAsciiIsKept()
        => Assert.Equal("""{"d": "münchen"}""", Question.RenderCriterion(
            new List<KeyValuePair<string, object?>> { new("d", "münchen") }));

    [Fact]
    public void NoulWithStructuredCriteriaRendersBothSides()
    {
        var question = Question.Noul("Is this phishing?",
            trueCriterion: new List<KeyValuePair<string, object?>> { new("desc", "phishing, scam or fraud") },
            falseCriterion: new List<KeyValuePair<string, object?>> { new("desc", "legitimate") });

        var options = question.RenderOptions();
        Assert.Equal(2, options.Count);
        Assert.Equal("""false: {"desc": "legitimate"}""", options[0]);
        Assert.Equal("""true: {"desc": "phishing, scam or fraud"}""", options[1]);
        Assert.DoesNotContain("'", string.Concat(options), StringComparison.Ordinal);
    }

    [Fact]
    public void NoulWithoutCriteriaUsesTheDefaultText()
    {
        var options = Question.Noul("Is the sky blue?").RenderOptions();
        Assert.Equal("false: no, the statement does not hold", options[0]);
        Assert.Equal("true: yes, the statement holds", options[1]);
    }

    [Fact]
    public void ChoiceRendersDescriptionsAndBareKeys()
    {
        var question = Question.Choice("x",
            ("billing", new List<KeyValuePair<string, object?>> { new("desc", "payments") }),
            ("tech", null),
            ("sales", ""));

        var options = question.RenderOptions();
        Assert.Equal("""billing: {"desc": "payments"}""", options[0]);
        Assert.Equal("tech", options[1]);         // only null/"" mean "no description"
        Assert.Equal("sales", options[2]);
    }

    [Fact]
    public void ZeroAndFalseAreRealDescriptions()
    {
        var options = Question.Choice("x", ("a", 0), ("b", false)).RenderOptions();
        Assert.Equal("a: 0", options[0]);
        Assert.Equal("b: false", options[1]);
    }

    [Fact]
    public void ScoreLevelsAreNumbered()
    {
        var options = Question.Score("How bad?", "fine", new[] { "a", "b" }).RenderOptions();
        Assert.Equal("level 0: fine", options[0]);
        Assert.Equal("""level 1: ["a", "b"]""", options[1]);
    }

    [Fact]
    public void ChoiceOptionOrderIsTheLabelOrder()
    {
        var question = Question.Choice("x", "first", "second", "third");
        Assert.Equal(["first", "second", "third"], question.OptionKeys);
    }

    [Fact]
    public void TemperatureBucketsMatchThePythonBoundaries()
    {
        Assert.Equal("noul:2", QuestionTypes.TemperatureBucket(QuestionType.Noul, 2));
        Assert.Equal("choice:3-5", QuestionTypes.TemperatureBucket(QuestionType.Choice, 5));
        Assert.Equal("choice:6-10", QuestionTypes.TemperatureBucket(QuestionType.Choice, 10));
        Assert.Equal("choice:11+", QuestionTypes.TemperatureBucket(QuestionType.Choice, 11));
        Assert.Equal("score:3-5", QuestionTypes.TemperatureBucket(QuestionType.Score, 4));
    }
}

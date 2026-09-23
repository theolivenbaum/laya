using Laya.Catalyst;
using Xunit;

namespace Laya.Tests;

/// <summary>The Catalyst-backed classifier, on the languages the heuristic cannot see.</summary>
public class CatalystClassifierTests
{
    private static readonly Lazy<CatalystLanguageClassifier> Classifier =
        new(() => CatalystLanguageClassifier.CreateAsync(trials: 21).GetAwaiter().GetResult());

    [Theory]
    [InlineData("Saya ditagih dua kali untuk langganan saya bulan ini dan saya ingin uang saya kembali", "id")]
    [InlineData("Nilitozwa mara mbili kwa usajili wangu mwezi huu na nataka pesa yangu irudishwe", "sw")]
    [InlineData("I was charged twice for my subscription this month and I want a refund", "en")]
    [InlineData("Ich wurde zweimal belastet, bitte erstatten Sie mir das Geld", "de")]
    public void IdentifiesLanguages(string text, string expected)
        => Assert.Equal(expected, Classifier.Value.Classify(text)?.Language);

    [Fact]
    public void ReportsIsoCodes()
    {
        Assert.Contains("en", Classifier.Value.Languages);
        Assert.Contains("id", Classifier.Value.Languages);
        Assert.Null(Classifier.Value.Classify("   "));
    }

    [Theory]
    [InlineData("Saya ditagih dua kali untuk langganan saya bulan ini dan saya ingin uang saya kembali")]
    [InlineData("Nilitozwa mara mbili kwa usajili wangu mwezi huu na nataka pesa yangu irudishwe")]
    public void RoutesPlainAsciiLanguagesTheHeuristicMisses(string text)
    {
        Assert.Equal("english", new Router().Route(text).Model);
        Assert.Equal("multilingual", new Router { LanguageClassifier = Classifier.Value }.Route(text).Model);
    }

    [Theory]
    [InlineData("Please refund the duplicate charge on invoice 4411 today.")]
    [InlineData("Server db-01 CPU at 98% since 02:00 UTC and the error rate is climbing")]
    [InlineData("refund me")]
    [InlineData("ok")]
    public void LeavesEnglishAlone(string text)
        => Assert.Equal("english", new Router { LanguageClassifier = Classifier.Value }.Route(text).Model);
}

using System.Text.Json;
using StructuredDecisions.Tokenizers;
using Xunit;

namespace StructuredDecisions.Tests;

/// <summary>The managed NFC against Python's <c>unicodedata.normalize("NFC", …)</c>.</summary>
public class UnicodeNfcTests
{
    [Fact]
    public void MatchesPythonNfc()
    {
        using var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(TestModels.FixtureRoot, "unicode-nfc.json")));
        var failures = new List<string>();
        foreach (var testCase in fixture.RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = testCase.GetProperty("text").GetString()!;
            string expected = testCase.GetProperty("nfc").GetString()!;
            string actual = UnicodeNfc.Normalize(text);
            if (actual != expected) failures.Add($"{JsonSerializer.Serialize(text)}: {JsonSerializer.Serialize(expected)} != {JsonSerializer.Serialize(actual)}");
        }
        Assert.True(failures.Count == 0, string.Join('\n', failures.Take(10)));
    }

    [Theory]
    [InlineData("é", "é")]
    [InlineData("각", "각")]
    [InlineData("Å", "Å")]
    [InlineData("Å", "Å")]              // singleton decomposition: never recomposed to the Angstrom sign
    [InlineData("plain ascii", "plain ascii")]
    public void ComposesCanonicalSequences(string input, string expected) => Assert.Equal(expected, UnicodeNfc.Normalize(input));
}

using System.Text.Json;
using Laya.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace Laya.Tests;

/// <summary>
/// Tokenizer parity against ids dumped from <c>transformers</c> by
/// <c>tools/dump_tokenizations.py</c>. A single divergent id shifts every downstream activation,
/// so this is checked exactly rather than within a tolerance.
///
/// <para>Both checkpoint families are covered because they use quite different tokenizers: the
/// English one is byte-level BPE over a GPT-2 style regex, the multilingual one is a
/// SentencePiece-style BPE with a metaspace pre-tokenizer and byte fallback.</para>
/// </summary>
public class TokenizerTests(ITestOutputHelper output)
{
    [ModelFact("english")]
    public void EnglishTokenizerMatchesTransformers() => Compare("english", "tokenizer-english.json");

    [ModelFact("multilingual")]
    public void MultilingualTokenizerMatchesTransformers()
        => Compare("multilingual", "tokenizer-multilingual.json");

    private void Compare(string checkpoint, string fixtureName)
    {
        string path = Path.Combine(TestModels.FixtureRoot, fixtureName);
        Assert.True(File.Exists(path), $"missing tokenizer fixture {path}");

        var tokenizer = HuggingFaceTokenizer.FromDirectory(
            Path.Combine(TestModels.CheckpointDirectory(checkpoint), "tokenizer"));

        using var fixture = JsonDocument.Parse(File.ReadAllText(path));
        var special = fixture.RootElement.GetProperty("special");
        Assert.Equal(special.GetProperty("cls").GetInt32(), tokenizer.ClsTokenId);
        Assert.Equal(special.GetProperty("sep").GetInt32(), tokenizer.SepTokenId);
        Assert.Equal(special.GetProperty("mask").GetInt32(), tokenizer.MaskTokenId);
        Assert.Equal(special.GetProperty("pad").GetInt32(), tokenizer.PadTokenId);

        int cases = 0;
        var failures = new List<string>();
        foreach (var testCase in fixture.RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = testCase.GetProperty("text").GetString()!;
            int[] expected = [.. testCase.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32())];
            int[] actual = [.. tokenizer.Encode(text)];
            if (!expected.AsSpan().SequenceEqual(actual))
            {
                failures.Add($"{JsonSerializer.Serialize(text)}\n    expected {string.Join(',', expected.Take(20))}"
                    + $"\n    actual   {string.Join(',', actual.Take(20))}");
            }
            cases++;
        }

        output.WriteLine($"{checkpoint}: {cases - failures.Count}/{cases} tokenizations matched");
        Assert.Empty(failures);
    }

    [ModelFact("english")]
    public void ByteLevelDecodeRoundTrips()
    {
        var tokenizer = HuggingFaceTokenizer.FromDirectory(
            Path.Combine(TestModels.CheckpointDirectory("english"), "tokenizer"));

        foreach (string text in (string[])["hello world", "Ünïcödé", "emoji 🚀 here", "  spaces  ", "日本語"])
        {
            Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text)));
        }
    }

    [Fact]
    public void ByteLevelAlphabetIsABijectionOverAllBytes()
    {
        // Every byte value must survive the round trip, or non-UTF-8-clean input silently changes.
        var bytes = Enumerable.Range(0, 256).Select(b => (byte)b).ToArray();
        string text = System.Text.Encoding.Latin1.GetString(bytes);

        // Latin1 keeps one byte per char, so encoding the UTF-8 of that string and decoding it back
        // has to reproduce the original characters.
        string encoded = ByteLevelAlphabet.Encode(text);
        Assert.Equal(text, ByteLevelAlphabet.Decode(encoded));
        Assert.DoesNotContain(encoded, c => char.IsWhiteSpace(c) || char.IsControl(c));
    }
}

using System.Text.Json;
using StructuredDecisions.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace StructuredDecisions.Tests;

/// <summary>
/// The Qwen3.5 tokenizer Mica reads with, against ids from <c>tokenizers</c>
/// (<c>tools/dump_mica_tokenizations.py</c>). Exact, like <see cref="TokenizerTests"/>.
/// </summary>
public class MicaTokenizerTests(ITestOutputHelper output)
{
    [MicaFact]
    public void QwenTokenizerMatchesTokenizers()
    {
        var tokenizer = HuggingFaceTokenizer.FromDirectory(MicaFactAttribute.Directory);
        using var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(TestModels.FixtureRoot, "tokenizer-mica.json")));

        int cases = 0;
        var failures = new List<string>();
        foreach (var testCase in fixture.RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = testCase.GetProperty("text").GetString()!;
            int[] expected = [.. testCase.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32())];
            int[] actual = [.. tokenizer.Encode(text)];
            if (!expected.AsSpan().SequenceEqual(actual))
            {
                int at = 0;
                while (at < Math.Min(expected.Length, actual.Length) && expected[at] == actual[at]) at++;
                failures.Add($"{JsonSerializer.Serialize(text[..Math.Min(text.Length, 200)])}\n    first difference at {at}: "
                    + $"expected {string.Join(',', expected.Skip(at).Take(8))}  actual {string.Join(',', actual.Skip(at).Take(8))}");
            }
            cases++;
        }

        output.WriteLine($"mica: {cases - failures.Count}/{cases} tokenizations matched");
        foreach (string failure in failures.Take(20)) output.WriteLine(failure);
        Assert.Empty(failures);
    }

    [MicaFact]
    public void ChatControlTokensRoundTrip()
    {
        var tokenizer = HuggingFaceTokenizer.FromDirectory(MicaFactAttribute.Directory);
        string text = "<|im_start|>user\n한국어 🙂 𠀀<|im_end|>\n";
        Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text)));
        Assert.Equal(248045, tokenizer.Encode("<|im_start|>")[0]);
    }
}

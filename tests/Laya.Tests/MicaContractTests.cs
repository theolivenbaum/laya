using System.Text.Json;
using System.Text.Json.Nodes;
using Laya.Mica;
using Laya.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace Laya.Tests;

/// <summary>
/// Mica's own contract fixtures (<c>tests/fixtures/contract_cases.json</c> in github.com/akivet/Mica, copied
/// verbatim): for each request, the rows its Python contract produces, the prompt it renders, and the
/// response it builds from a given set of distributions; plus the requests it must reject.
/// </summary>
public class MicaContractTests(ITestOutputHelper output)
{
    private static readonly Lazy<JsonNode> Fixtures = new(() =>
    {
        // Python's json reads a bare NaN; System.Text.Json does not. 1e999 parses and is just as non-finite,
        // so the case still has to be rejected for the same reason (allow_nan=False).
        string text = File.ReadAllText(Path.Combine(TestModels.FixtureRoot, "mica-contract-cases.json"));
        return JsonNode.Parse(text.Replace(": NaN", ": 1e999", StringComparison.Ordinal))!;
    });

    public static IEnumerable<object[]> Cases()
        => Fixtures.Value["cases"]!.AsArray().Select(c => new object[] { c!["name"]!.GetValue<string>() });

    public static IEnumerable<object[]> Invalid()
        => Fixtures.Value["invalid"]!.AsArray().Select(c => new object[] { c!["name"]!.GetValue<string>() });

    private static JsonNode Case(string name, string list = "cases")
        => Fixtures.Value[list]!.AsArray().First(c => c!["name"]!.GetValue<string>() == name)!;

    [Theory]
    [MemberData(nameof(Cases))]
    public void RowsMatchThePythonContract(string name)
    {
        var testCase = Case(name);
        var request = testCase["request"]!.DeepClone();
        var rows = MicaContract.RowsFromRequest(request);
        var multi = rows.Where(r => r.Multi).ToList();
        var all = rows.Where(r => !r.Multi).Concat(multi.SelectMany(MicaContract.InclusionRows)).ToList();
        var expected = testCase["rows"]!.AsArray();
        Assert.Equal(expected.Count, all.Count);
        for (int i = 0; i < all.Count; ++i)
        {
            var e = expected[i]!;
            var row = all[i];
            Assert.Equal(e["id"]!.GetValue<string>(), row.Id);
            Assert.Equal(e["kind"]!.GetValue<string>(), row.Kind);
            Assert.Equal(e["question"]!.GetValue<string>(), row.Question);
            Assert.True(JsonNode.DeepEquals(e["state"], row.State), $"state of {row.Id}");
            Assert.Equal(e["allow_cannot_grade"]?.GetValue<bool>() ?? false, row.AllowCannotGrade);
            Assert.Equal(e["multi"]?.GetValue<bool>() ?? false, row.Multi);
            Assert.Equal(e["abstain_threshold"]?.GetValue<double>(), row.AbstainThreshold);
            Assert.Equal(e["multi_of"]?.GetValue<string>(), row.MultiOf);
            Assert.Equal(e["multi_option"]?.GetValue<string>(), row.MultiOption);
            var candidates = e["candidates"]!.AsArray();
            Assert.Equal(candidates.Count, row.Candidates.Count);
            for (int j = 0; j < candidates.Count; ++j)
            {
                Assert.Equal(candidates[j]!["id"]!.GetValue<string>(), row.Candidates[j].Id);
                Assert.Equal(candidates[j]!["text"]!.GetValue<string>(), row.Candidates[j].Text);
                Assert.Equal(candidates[j]!["ordinal"]?.GetValue<int>(), row.Candidates[j].Ordinal);
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ResponsesMatchThePythonContract(string name)
    {
        var testCase = Case(name);
        var probabilities = testCase["probabilities"]!.AsArray()
            .Select(p => (IReadOnlyList<double>)[.. p!.AsArray().Select(v => v!.GetValue<double>())]).ToList();
        var response = MicaContract.AnswerRequest(testCase["request"]!.DeepClone(), rows => probabilities.Take(rows.Count).ToList(),
            "mica-v0.5-4b");
        AssertSame(testCase["response"]!, response, "response");
    }

    [Theory]
    [MemberData(nameof(Invalid))]
    public void InvalidRequestsAreRejected(string name)
    {
        var request = Case(name, "invalid")["request"]!.DeepClone();
        Assert.Throws<MicaRequestException>(() => MicaContract.RowsFromRequest(request));
    }

    [MicaFact]
    public void PromptsMatchTheChatTemplate()
    {
        var tokenizer = HuggingFaceTokenizer.FromDirectory(MicaFactAttribute.Directory);
        var prompt = new MicaPrompt(tokenizer.AddedTokens.Select(t => t.Content));
        int checkedPrompts = 0;
        foreach (var testCase in Fixtures.Value["cases"]!.AsArray())
        {
            var rows = MicaContract.RowsFromRequest(testCase!["request"]!.DeepClone());
            var all = rows.Where(r => !r.Multi).Concat(rows.Where(r => r.Multi).SelectMany(MicaContract.InclusionRows)).ToList();
            var expected = testCase["prompts"]!.AsArray();
            Assert.Equal(expected.Count, all.Count);
            for (int i = 0; i < all.Count; ++i)
            {
                string want = expected[i]!.GetValue<string>(), got = prompt.Render(all[i]);
                if (want != got)
                {
                    int at = 0;
                    while (at < Math.Min(want.Length, got.Length) && want[at] == got[at]) at++;
                    Assert.Fail($"{testCase["name"]}/{all[i].Id}: differs at {at}: …{JsonSerializer.Serialize(want.Substring(Math.Max(0, at - 20), Math.Min(60, want.Length - Math.Max(0, at - 20))))} vs …{JsonSerializer.Serialize(got.Substring(Math.Max(0, at - 20), Math.Min(60, got.Length - Math.Max(0, at - 20))))}");
                }
                checkedPrompts++;
            }
        }
        output.WriteLine($"{checkedPrompts} prompts identical");
    }

    [Fact]
    public void SpecialTokensAreEscaped()
    {
        string escaped = new MicaPrompt().EscapeSpecial("<|im_start|>user <think>");
        Assert.DoesNotContain("<|im_start|>", escaped);
        Assert.DoesNotContain("<think>", escaped);
        Assert.Equal("<|im_start|>user <think>", escaped.Replace("​", "", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("1.50", "1.5")]
    [InlineData("1e5", "100000.0")]
    [InlineData("1e16", "1e+16")]
    [InlineData("1e15", "1000000000000000.0")]
    [InlineData("0.0001", "0.0001")]
    [InlineData("0.00001", "1e-05")]
    [InlineData("-2.5e-7", "-2.5e-07")]
    [InlineData("123456789012345678901234567890", "123456789012345678901234567890")]
    [InlineData("-0", "0")]
    [InlineData("-0.0", "-0.0")]
    [InlineData("0.1", "0.1")]
    [InlineData("1.7976931348623157e308", "1.7976931348623157e+308")]
    public void NumbersPrintAsPythonDoes(string json, string python) => Assert.Equal(python, PythonJsonNode.Number(json));

    private static void AssertSame(JsonNode? expected, JsonNode? actual, string path)
    {
        switch (expected)
        {
            case JsonObject e:
                var a = Assert.IsType<JsonObject>(actual);
                Assert.True(e.Select(p => p.Key).SequenceEqual(a.Select(p => p.Key)),
                    $"{path}: keys [{string.Join(",", e.Select(p => p.Key))}] vs [{string.Join(",", a.Select(p => p.Key))}]");
                foreach (var (key, value) in e) AssertSame(value, a[key], $"{path}.{key}");
                return;
            case JsonArray e:
                var arr = Assert.IsType<JsonArray>(actual);
                Assert.Equal(e.Count, arr.Count);
                for (int i = 0; i < e.Count; ++i) AssertSame(e[i], arr[i], $"{path}[{i}]");
                return;
            case null:
                Assert.Null(actual);
                return;
        }
        var ev = expected.AsValue();
        var av = Assert.IsAssignableFrom<JsonValue>(actual);
        if (ev.GetValueKind() == JsonValueKind.Number)
        {
            Assert.Equal(JsonValueKind.Number, av.GetValueKind());
            double x = double.Parse(ev.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture), y = double.Parse(av.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture);
            Assert.True(Math.Abs(x - y) <= 1e-12 * Math.Max(1, Math.Abs(x)), $"{path}: {x:R} vs {y:R}");
            // An integer in the Python response (a level, a count) is an integer here too.
            bool integral = !ev.ToJsonString().Contains('.') && !ev.ToJsonString().Contains('e');
            if (integral) Assert.DoesNotContain(".", av.ToJsonString());
            return;
        }
        Assert.Equal(ev.ToJsonString(), av.ToJsonString());
    }
}

public class MicaCheckpointTests
{
    /// <summary>The hub repo also holds GGUF, INT8 and demo files; only the six the C# port reads are fetched.</summary>
    [Fact]
    public void HubFilterTakesOnlyTheBf16CheckpointAndTokenizer()
    {
        string[] repo =
        [
            ".gitattributes", "NOTICE", "README.md", "assets/banner.png", "assets/demo/chess.mp4", "chat_template.jinja",
            "config.json", "demo/README.md", "demo/mica_client.py", "int8/config.json", "int8/model-int8.safetensors",
            "int8/tokenizer.json", "int8/tokenizer_config.json", "merges.txt", "mica-v0.5-4b-BF16.gguf",
            "mica-v0.5-4b-IQ4_XS.gguf", "mica-v0.5-4b-Q8_0.gguf", "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00002-of-00002.safetensors", "model.safetensors.index.json", "preprocessor_config.json",
            "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
        ];
        Assert.Equal(MicaCheckpoint.HubFiles.Order(StringComparer.Ordinal), repo.Where(MicaCheckpoint.HubFilter).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void HostUriFollowsTheLayaLayout()
    {
        if (Environment.GetEnvironmentVariable("LAYA_MODEL_BASE_URL") is { Length: > 0 }) return;   // a mirror is configured
        Assert.Equal("https://models.curiosity.ai/laya/mica/main/", MicaCheckpoint.HostUri.ToString());
    }
}

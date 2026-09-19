using System.Text.Json;
using Laya.Runtime;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// The serialized state goes into the token sequence verbatim, so it has to be byte-identical to
/// what <c>json.dumps</c> produces — spaces after colons and commas included.
/// </summary>
public class PythonJsonTests
{
    [Fact]
    public void ObjectsUseThePythonSeparators()
    {
        var state = new List<KeyValuePair<string, object?>>
        {
            new("subject", "Invoice"),
            new("body", "please pay"),
        };
        Assert.Equal("""{"subject": "Invoice", "body": "please pay"}""", PythonJson.Dumps(state));
    }

    [Fact]
    public void ArraysUseThePythonSeparators()
        => Assert.Equal("[1, 2, 3]", PythonJson.Dumps(new[] { 1, 2, 3 }));

    [Fact]
    public void NonAsciiSurvivesByDefaultAndEscapesWhenAsked()
    {
        Assert.Equal("\"münchen\"", PythonJson.Dumps("münchen"));
        Assert.Equal("\"m\\u00fcnchen\"", PythonJson.Dumps("münchen", ensureAscii: true));
    }

    [Fact]
    public void ControlCharactersAreEscaped()
        => Assert.Equal("\"a\\nb\\tc\"", PythonJson.Dumps("a\nb\tc"));

    [Fact]
    public void JsonElementsRoundTripThroughThePythonSpelling()
    {
        using var document = JsonDocument.Parse("""{"a":1,"b":["x",null,true]}""");
        Assert.Equal("""{"a": 1, "b": ["x", null, true]}""", PythonJson.Dumps(document.RootElement));
    }

    [Fact]
    public void StringStatesPassThroughUnquoted()
        => Assert.Equal("hello", SequenceBuilder.SerializeState("hello"));

    [Fact]
    public void StructuredStatesAreSerialized()
        => Assert.Equal("""{"body": "hi"}""", SequenceBuilder.SerializeState(
            new List<KeyValuePair<string, object?>> { new("body", "hi") }));
}

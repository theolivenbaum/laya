using System.Text.Json;
using Laya.Runtime;

namespace Laya.Cli;

/// <summary>Reads question sets from the same JSON shape the Python API accepts, and from flags.</summary>
internal static class QuestionJson
{
    public static QuestionSet Parse(string json) => Runtime.QuestionJson.Parse(json);

    /// <summary>
    /// Parses the compact flag form:
    /// <c>[id=]type:instructions</c>, optionally followed by <c>|opt1,opt2</c> for a choice or
    /// score question.
    /// </summary>
    public static (string Id, Question Question) ParseInline(string spec, int index)
    {
        string id = $"q{index}";
        string rest = spec;

        int colon = spec.IndexOf(':', StringComparison.Ordinal);
        int equals = spec.IndexOf('=', StringComparison.Ordinal);
        if (equals >= 0 && (colon < 0 || equals < colon))
        {
            id = spec[..equals];
            rest = spec[(equals + 1)..];
        }

        colon = rest.IndexOf(':', StringComparison.Ordinal);
        if (colon < 0)
        {
            throw new ArgumentException($"--question '{spec}' should look like 'id=type:instructions[|a,b,c]'.");
        }

        var type = QuestionTypes.Parse(rest[..colon].Trim());
        string tail = rest[(colon + 1)..];
        string[] parts = tail.Split('|', 2);
        string instructions = parts[0].Trim();
        string[] options = parts.Length > 1
            ? [.. parts[1].Split(',').Select(o => o.Trim()).Where(o => o.Length > 0)]
            : [];

        return type switch
        {
            QuestionType.Choice when options.Length > 0 => (id, Question.Choice(instructions, options)),
            QuestionType.Choice => throw new ArgumentException($"--question '{spec}' is a choice but lists no options."),
            QuestionType.Score when options.Length > 0 => (id, Question.Score(instructions, [.. options])),
            QuestionType.Score => throw new ArgumentException($"--question '{spec}' is a score but lists no levels."),
            _ => (id, Question.Noul(instructions)),
        };
    }
}

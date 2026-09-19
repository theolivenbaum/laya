using System.Text.Json;
using Laya.Runtime;

namespace Laya.Cli;

/// <summary>Reads question sets from the same JSON shape the Python API accepts, and from flags.</summary>
internal static class QuestionJson
{
    /// <summary>
    /// Parses <c>{"id": {"type": "...", "instructions": "...", "criteria": …}}</c>, preserving the
    /// declaration order because that order is the label order the model was trained with.
    /// </summary>
    public static QuestionSet Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        var questions = new QuestionSet();
        foreach (var property in document.RootElement.EnumerateObject())
        {
            questions.Add(property.Name, ParseQuestion(property.Value));
        }
        return questions;
    }

    private static Question ParseQuestion(JsonElement element)
    {
        string type = element.GetProperty("type").GetString()
            ?? throw new ArgumentException("a question needs a \"type\".");
        object instructions = element.GetProperty("instructions").ValueKind == JsonValueKind.String
            ? element.GetProperty("instructions").GetString()!
            : element.GetProperty("instructions").Clone();

        bool hasCriteria = element.TryGetProperty("criteria", out var criteria)
            && criteria.ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined);

        switch (QuestionTypes.Parse(type))
        {
            case QuestionType.Choice:
            {
                if (!hasCriteria) throw new ArgumentException("a choice question needs \"criteria\".");
                if (criteria.ValueKind == JsonValueKind.Array)
                {
                    return Question.Choice(instructions, [.. criteria.EnumerateArray().Select(c => c.GetString()!)]);
                }
                return Question.Choice(instructions, criteria.EnumerateObject()
                    .Select(p => new KeyValuePair<string, object?>(p.Name, Unwrap(p.Value))));
            }
            case QuestionType.Score:
            {
                if (!hasCriteria) throw new ArgumentException("a score question needs \"criteria\".");
                return Question.Score(instructions, [.. criteria.EnumerateArray().Select(Unwrap)]);
            }
            default:
            {
                if (!hasCriteria) return Question.Noul(instructions);
                object? trueCriterion = criteria.TryGetProperty("true", out var t) ? Unwrap(t) : null;
                object? falseCriterion = criteria.TryGetProperty("false", out var f) ? Unwrap(f) : null;
                return Question.Noul(instructions, trueCriterion, falseCriterion);
            }
        }
    }

    private static object? Unwrap(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Null or JsonValueKind.Undefined => null,
        _ => element.Clone(),
    };

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

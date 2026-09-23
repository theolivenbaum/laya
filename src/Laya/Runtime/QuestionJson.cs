using System.Text.Json;

namespace Laya.Runtime;

/// <summary>Reads question sets from the JSON shape the Python API accepts.</summary>
public static class QuestionJson
{
    /// <summary>
    /// Parses <c>{"id": {"type": "...", "instructions": "...", "criteria": …}}</c>, preserving the
    /// declaration order because that order is the label order the model was trained with.
    /// </summary>
    public static QuestionSet Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        return Parse(document.RootElement);
    }

    /// <inheritdoc cref="Parse(string)"/>
    public static QuestionSet Parse(JsonElement root)
    {
        var questions = new QuestionSet();
        foreach (var property in root.EnumerateObject())
        {
            questions.Add(property.Name, ParseQuestion(property.Value));
        }
        return questions;
    }

    public static Question ParseQuestion(JsonElement element)
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
}

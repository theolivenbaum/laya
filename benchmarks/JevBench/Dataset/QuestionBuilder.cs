using System.Text.Json;
using Laya.Runtime;

namespace Laya.JevBench.Dataset;

/// <summary>
/// Turns a <see cref="JevQuestion"/> into the <see cref="Question"/> laya's own <c>Agent</c> takes —
/// the C# equivalent of JevBench's <c>adapters/base.py:build_question</c> feeding
/// <c>laya.load(...).predict(state, {"decision": question})</c> in <c>laya_local.py</c>.
///
/// <para>JevBench's frozen task files are written with <c>sort_keys=True</c> (see
/// <c>jevbench/tasks.py:Task.to_json</c>), so a "criteria" object's key order in the public JSONL
/// *is* the order the original scored run presented options to the model — there is no separate
/// "declared order" to recover. Reading it straight off <see cref="JsonElement.EnumerateObject"/>
/// (document order) reproduces that exactly, the same way laya's own <c>QuestionJson.Parse</c>
/// does for the CLI.</para>
/// </summary>
public static class QuestionBuilder
{
    public static Question Build(JevQuestion question)
    {
        object instructions = question.Instructions.ValueKind == JsonValueKind.String
            ? question.Instructions.GetString()!
            : question.Instructions.Clone();

        switch (question.Type)
        {
            case "choice":
            {
                if (question.Criteria is not { } criteria || criteria.ValueKind != JsonValueKind.Object)
                {
                    throw new InvalidDataException("a choice question needs an object 'criteria'.");
                }
                return Question.Choice(instructions, criteria.EnumerateObject()
                    .Select(p => new KeyValuePair<string, object?>(p.Name, Unwrap(p.Value))));
            }
            case "score":
            {
                if (question.Criteria is not { } criteria || criteria.ValueKind != JsonValueKind.Array)
                {
                    throw new InvalidDataException("a score question needs an array 'criteria'.");
                }
                return Question.Score(instructions, [.. criteria.EnumerateArray().Select(Unwrap)]);
            }
            default: // noul
            {
                if (question.Criteria is not { } criteria || criteria.ValueKind != JsonValueKind.Object)
                {
                    return Question.Noul(instructions);
                }
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

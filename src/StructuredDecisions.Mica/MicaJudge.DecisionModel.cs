using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace StructuredDecisions.Mica;

/// <summary>Mica behind the shared <see cref="IDecisionModel"/> contract.</summary>
public sealed partial class MicaJudge
{
    /// <summary>
    /// Answers a <see cref="QuestionSet"/> through Mica's contract: the questions become a <c>/v1/systemone</c>
    /// request, and each typed result an <see cref="Answer"/> (the full native result is on <see cref="Answer.Native"/>).
    /// Answers come back in question order.
    /// </summary>
    public DecisionResult SystemOne(object? state, QuestionSet questions) => SystemOne(state, questions, parallel: null);

    /// <inheritdoc cref="SystemOne(object?, QuestionSet)"/>
    public DecisionResult SystemOne(object? state, QuestionSet questions, ParallelOptions? parallel)
    {
        ArgumentNullException.ThrowIfNull(questions);
        if (questions.Count == 0) return new DecisionResult { Model = ModelName, Answers = [], Usage = new Usage(0, 0) };

        var response = SystemOne(ToRequest(state, questions), recorder: null, parallel);
        var native = response["answers"]!.AsObject();
        var answers = questions.Ids.Select(id => new KeyValuePair<string, Answer>(id, ToAnswer(native[id]!.AsObject(), questions[id]))).ToList();
        return new DecisionResult
        {
            Model = ModelName,
            Answers = answers,
            Usage = new Usage(response["usage"]!["input_tokens"]!.GetValue<int>(), 0),
        };
    }

    /// <summary>The <c>/v1/systemone</c> request a state and question set stand for.</summary>
    public static JsonObject ToRequest(object? state, QuestionSet questions)
    {
        var questionObject = new JsonObject();
        foreach (var (id, question) in questions)
        {
            var q = new JsonObject
            {
                ["type"] = QuestionTypes.Name(question.Type),
                ["instructions"] = ToNode(question.Instructions),
            };
            switch (question.Type)
            {
                case QuestionType.Choice:
                    var options = new JsonObject();
                    foreach (var (key, description) in question.NamedCriteria ?? []) options[key] = ToNode(description);
                    q["criteria"] = options;
                    break;
                case QuestionType.Score:
                    q["criteria"] = new JsonArray([.. (question.Levels ?? []).Select(ToNode)]);
                    break;
                default:
                    if (question.NamedCriteria is { Count: > 0 } criteria)
                    {
                        var noul = new JsonObject();
                        foreach (var (key, value) in criteria) noul[key] = ToNode(value);
                        q["criteria"] = noul;
                    }
                    break;
            }
            if (question.Multi) q["multi"] = true;
            if (question.AllowCannotGrade) q["allow_cannot_grade"] = true;
            if (question.AbstainThreshold is double threshold) q["abstain_threshold"] = threshold;
            questionObject[id] = q;
        }
        return new JsonObject { ["state"] = ToNode(state), ["questions"] = questionObject };
    }

    private static JsonNode? ToNode(object? value) => value switch
    {
        null => null,
        string text => JsonValue.Create(text),
        JsonNode node => node.DeepClone(),
        _ => JsonSerializer.SerializeToNode(value),
    };

    private static Answer ToAnswer(JsonObject native, Question question)
    {
        string type = native["type"]!.GetValue<string>();
        bool? abstain = native["abstain"]?.GetValue<bool>();
        if (type == "noul")
        {
            double p = native["noul"]!.GetValue<double>();
            return new Answer { Type = "noul", Noul = p, Confidence = Math.Max(p, 1 - p), Abstain = abstain, Native = native };
        }
        if (type == "choice" && native["multi"]?.GetValue<bool>() == true)
        {
            var inclusion = Pairs(native["inclusion"]!.AsObject());
            return new Answer
            {
                Type = "choice",
                Probabilities = inclusion,
                Selected = [.. native["selected"]!.AsArray().Select(n => n!.GetValue<string>())],
                // How sure the least certain inclusion is: every option is its own yes/no.
                Confidence = inclusion.Count == 0 ? 0 : inclusion.Min(p => Math.Max(p.Value, 1 - p.Value)),
                Native = native,
            };
        }
        if (type == "choice")
        {
            return new Answer
            {
                Type = "choice",
                Choice = native["choice"]!.GetValue<string>(),
                Probabilities = Pairs(native["probabilities"]!.AsObject()),
                Confidence = Number(native["confidence"]),
                Abstain = abstain,
                Native = native,
            };
        }
        return new Answer
        {
            Type = "score",
            Score = native["score"] is null ? null : Number(native["score"]),
            Level = native["level"] is null ? null : (int)Number(native["level"]),
            Legend = [.. native["legend"]!.AsObject().Select(p => new KeyValuePair<string, string>(p.Key, p.Value!.GetValue<string>()))],
            Probabilities = Pairs(native["probabilities"]!.AsObject()),
            Confidence = Number(native["confidence"]),
            CannotGradeProbability = native["cannot_grade_probability"] is null ? null : Number(native["cannot_grade_probability"]),
            Abstain = abstain,
            Native = native,
        };
    }

    private static List<KeyValuePair<string, double>> Pairs(JsonObject values)
        => [.. values.Select(p => new KeyValuePair<string, double>(p.Key, Number(p.Value)))];

    private static double Number(JsonNode? node) => double.Parse(node!.ToJsonString(), CultureInfo.InvariantCulture);

    /// <summary>Nothing to release: the weights are managed arrays.</summary>
    public void Dispose()
    {
    }
}

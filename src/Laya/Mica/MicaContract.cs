using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Laya.Mica;

/// <summary>A request Mica cannot answer: malformed, out of range, or longer than the context.</summary>
public sealed class MicaRequestException(string message) : ArgumentException(message);

/// <summary>One allowed answer of a question row.</summary>
public sealed record MicaCandidate(string Id, string Text, int? Ordinal = null, string? Description = null);

/// <summary>
/// One independent judgment: a question over the request's state with its candidates, read out of one
/// forward pass. A multi-answer choice becomes one inclusion row (a yes/no) per candidate.
/// </summary>
public sealed class MicaRow
{
    public required string Id { get; init; }
    public required string Kind { get; init; }            // "noul", "choice" or "score"
    public required JsonNode? State { get; init; }
    public required string Question { get; init; }
    public required IReadOnlyList<MicaCandidate> Candidates { get; init; }
    public bool AllowCannotGrade { get; init; }
    public bool Multi { get; init; }
    public double? AbstainThreshold { get; init; }
    public string? MultiOf { get; init; }
    public string? MultiOption { get; init; }
    public string? Lang { get; init; }
}

/// <summary>
/// The <c>/v1/systemone</c> contract of Mica, ported from <c>mica/contract.py</c> and <c>mica/native.py</c>:
/// request validation into rows, multi-answer expansion, and the typed results built from each row's
/// probability distribution. Validation rejects exactly what the Python rejects.
/// </summary>
public static class MicaContract
{
    public const int MaxCandidates = 255;

    private static readonly HashSet<string> TopFields = ["model", "state", "questions"];
    private static readonly HashSet<string> QuestionFields = ["type", "instructions", "criteria", "allow_cannot_grade", "multi", "abstain_threshold"];

    /// <summary>Parses a request; a JSON syntax error is reported as a <see cref="MicaRequestException"/>.</summary>
    public static JsonNode? Parse(string json)
    {
        try
        {
            return JsonNode.Parse(json);
        }
        catch (JsonException exception)
        {
            throw new MicaRequestException($"request is not valid JSON: {exception.Message}");
        }
    }

    /// <summary>Validates a request and creates one independent row per question (<c>rows_from_request</c>).</summary>
    public static List<MicaRow> RowsFromRequest(JsonNode? request)
    {
        if (request is not JsonObject top) throw new MicaRequestException("request must be an object");
        var unknown = top.Select(p => p.Key).Where(k => !TopFields.Contains(k)).ToList();
        if (unknown.Count > 0)
        {
            if (unknown.Contains("images")) throw new MicaRequestException("Image inputs are not supported by the local text runtime");
            throw new MicaRequestException($"Unknown request fields: {string.Join(", ", unknown.Order(StringComparer.Ordinal))}");
        }
        if (!top.TryGetPropertyValue("state", out var state) || state is null)
        {
            throw new MicaRequestException("request.state is required");
        }
        if (ContainsImagePayload(state)) throw new MicaRequestException("Image inputs are not supported by the local text runtime");
        if (!top.TryGetPropertyValue("questions", out var questionsNode) || questionsNode is not JsonObject questions || questions.Count == 0)
        {
            throw new MicaRequestException("request.questions must be a non-empty object");
        }

        var rows = new List<MicaRow>();
        foreach (var (questionId, questionNode) in questions)
        {
            if (string.IsNullOrEmpty(questionId)) throw new MicaRequestException("question IDs must be non-empty strings");
            if (questionNode is not JsonObject question) throw new MicaRequestException($"question '{questionId}' must be an object");
            var questionUnknown = question.Select(p => p.Key).Where(k => !QuestionFields.Contains(k)).ToList();
            if (questionUnknown.Count > 0)
            {
                throw new MicaRequestException($"question '{questionId}' has unknown fields: {string.Join(", ", questionUnknown)}");
            }

            string kind = question.TryGetPropertyValue("type", out var typeNode) && IsString(typeNode, out string? typeText)
                ? typeText!.ToLowerInvariant()
                : "";
            if (kind is not ("noul" or "choice" or "score"))
            {
                throw new MicaRequestException($"question '{questionId}' has unsupported type '{kind}'");
            }

            bool allowCannotGrade = false;
            if (question.TryGetPropertyValue("allow_cannot_grade", out var cannotGradeNode))
            {
                if (kind != "score" || !IsBool(cannotGradeNode, out allowCannotGrade))
                {
                    throw new MicaRequestException("allow_cannot_grade must be a boolean on a Score question");
                }
            }
            if (!question.TryGetPropertyValue("instructions", out var instructionsNode))
            {
                throw new MicaRequestException($"question '{questionId}' requires instructions");
            }
            string instructions = Text(instructionsNode);
            question.TryGetPropertyValue("criteria", out var criteria);

            List<MicaCandidate> candidates;
            if (kind == "noul")
            {
                if (criteria is null || criteria is JsonObject { Count: 0 })
                {
                    candidates = [new("false", "No"), new("true", "Yes")];
                }
                else
                {
                    if (criteria is not JsonObject noulCriteria
                        || !noulCriteria.Select(p => p.Key.ToLowerInvariant()).ToHashSet().SetEquals(["false", "true"]))
                    {
                        throw new MicaRequestException($"Noul question '{questionId}' criteria must contain false and true");
                    }
                    // A dict comprehension: the first spelling of a key fixes its position, the last its value.
                    var order = new List<string>();
                    var byId = new Dictionary<string, JsonNode?>(StringComparer.Ordinal);
                    foreach (var (key, value) in noulCriteria)
                    {
                        string id = key.ToLowerInvariant();
                        if (!byId.ContainsKey(id)) order.Add(id);
                        byId[id] = value;
                    }
                    candidates = [.. order.Select(id => new MicaCandidate(id, Text(byId[id], fallback: id == "false" ? "No" : "Yes")))];
                }
            }
            else if (kind == "choice")
            {
                if (criteria is not JsonObject choiceCriteria || choiceCriteria.Count < 2 || choiceCriteria.Count > MaxCandidates)
                {
                    throw new MicaRequestException($"Choice question '{questionId}' requires 2..{MaxCandidates} criteria");
                }
                candidates = [];
                foreach (var (candidateId, description) in choiceCriteria)
                {
                    if (string.IsNullOrEmpty(candidateId))
                    {
                        throw new MicaRequestException($"Choice question '{questionId}' candidate IDs must be non-empty strings");
                    }
                    candidates.Add(new MicaCandidate(candidateId, Text(description, fallback: candidateId)));
                }
            }
            else
            {
                int limit = MaxCandidates - (allowCannotGrade ? 1 : 0);
                if (criteria is not JsonArray levels || levels.Count < 2 || levels.Count > limit)
                {
                    throw new MicaRequestException($"Score question '{questionId}' requires 2..{limit} ordered criteria");
                }
                candidates = [.. levels.Select((level, index) => new MicaCandidate(index.ToString(System.Globalization.CultureInfo.InvariantCulture), Text(level), index))];
                if (allowCannotGrade) candidates.Add(new MicaCandidate("cannot_grade", "Cannot grade from the supplied state"));
            }

            bool multi = false;
            if (question.TryGetPropertyValue("multi", out var multiNode))
            {
                if (kind != "choice" || !IsBool(multiNode, out multi)) throw new MicaRequestException("multi must be a boolean on a Choice question");
            }
            double? abstain = null;
            if (question.TryGetPropertyValue("abstain_threshold", out var abstainNode))
            {
                if (abstainNode is not JsonValue v || v.GetValueKind() != JsonValueKind.Number) throw new MicaRequestException("abstain_threshold must be a number in [0, 1]");
                double t = v.GetValue<JsonElement>().GetDouble();
                if (!(t >= 0 && t <= 1)) throw new MicaRequestException("abstain_threshold must be a number in [0, 1]");
                abstain = t;
            }

            rows.Add(new MicaRow
            {
                Id = questionId,
                Kind = kind,
                State = state.DeepClone(),
                Question = instructions,
                Candidates = candidates,
                AllowCannotGrade = allowCannotGrade,
                Multi = multi,
                AbstainThreshold = abstain,
            });
        }
        return rows;
    }

    /// <summary>
    /// <c>_text</c>: a non-blank string as is, a non-empty object or array as compact sorted JSON, the
    /// fallback for a null, and an error for anything else.
    /// </summary>
    private static string Text(JsonNode? value, string? fallback = null)
    {
        if (value is null && fallback is not null) return fallback;
        if (IsString(value, out string? text))
        {
            if (PythonText.Strip(text!).Length == 0) throw new MicaRequestException("instructions and criteria descriptions must not be empty");
            return text!;
        }
        if (value is not (JsonObject { Count: > 0 } or JsonArray { Count: > 0 }))
        {
            throw new MicaRequestException("instructions and criteria must be strings, non-empty objects, or non-empty arrays");
        }
        try
        {
            return PythonJsonNode.Dumps(value, sortKeys: true, compact: true);
        }
        catch (ArgumentException)
        {
            throw new MicaRequestException("instructions or criteria are not valid JSON");
        }
    }

    private static bool ContainsImagePayload(JsonNode? value)
    {
        switch (value)
        {
            case JsonObject obj:
                if (StringOf(obj["modality"]).Equals("image", StringComparison.OrdinalIgnoreCase)
                    || StringOf(obj["type"]).Equals("image", StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
                if (obj["image_url"] is not null || obj["image_bytes"] is not null) return true;
                if (obj["images"] is JsonArray { Count: > 0 }) return true;
                return obj.Any(p => ContainsImagePayload(p.Value));
            case JsonArray array:
                return array.Any(ContainsImagePayload);
            default:
                return false;
        }
    }

    private static string StringOf(JsonNode? node) => IsString(node, out string? text) ? text! : "";

    private static bool IsString(JsonNode? node, out string? text)
    {
        text = null;
        if (node is JsonValue value && value.GetValueKind() == JsonValueKind.String)
        {
            text = value.GetValue<string>();
            return true;
        }
        return false;
    }

    private static bool IsBool(JsonNode? node, out bool flag)
    {
        flag = false;
        if (node is JsonValue value && value.GetValueKind() is JsonValueKind.True or JsonValueKind.False)
        {
            flag = value.GetValue<bool>();
            return true;
        }
        return false;
    }

    // ---- multi-answer choice: one inclusion noul per candidate ----

    private static readonly Dictionary<string, string> MultiTemplates = new(StringComparer.Ordinal)
    {
        ["en"] = "{q}\nOptions: {menu}\nIs this option one of the correct answers? -> {opt}",
        ["ko"] = "{q}\n보기: {menu}\n이 보기가 정답에 포함됩니까? -> {opt}",
        ["zh"] = "{q}\n选项：{menu}\n这个选项是正确答案之一吗？-> {opt}",
        ["ja"] = "{q}\n選択肢：{menu}\nこの選択肢は正解に含まれますか？-> {opt}",
    };

    /// <summary>The inclusion rows of one multi-answer choice row: "is this one of the correct answers?".</summary>
    public static List<MicaRow> InclusionRows(MicaRow row)
    {
        string template = row.Lang is not null && MultiTemplates.TryGetValue(row.Lang, out var t) ? t : MultiTemplates["en"];
        string menu = string.Join(" | ", row.Candidates.Select(c => c.Text));
        return [.. row.Candidates.Select(c => new MicaRow
        {
            Id = $"{row.Id}~{c.Id}",
            Kind = "noul",
            State = row.State,
            MultiOf = row.Id,
            MultiOption = c.Id,
            Question = template.Replace("{q}", row.Question, StringComparison.Ordinal)
                .Replace("{menu}", menu, StringComparison.Ordinal).Replace("{opt}", c.Text, StringComparison.Ordinal),
            Candidates = [new("true", "True"), new("false", "False")],
            Lang = row.Lang,
        })];
    }

    /// <summary>The probability of each count of correct answers: the Poisson-binomial of the inclusions.</summary>
    public static List<double> CountDistribution(IReadOnlyList<double> probabilities)
    {
        var distribution = new List<double> { 1.0 };
        foreach (double q in probabilities)
        {
            var next = new List<double>(distribution.Count + 1);
            for (int i = 0; i <= distribution.Count; ++i)
            {
                double a = i < distribution.Count ? distribution[i] : 0.0;
                next.Add(a * (1 - q) + (i > 0 ? distribution[i - 1] * q : 0.0));
            }
            distribution = next;
        }
        return distribution;
    }

    private static JsonObject Assemble(MicaRow row, IReadOnlyDictionary<string, double> inclusion)
    {
        var ids = row.Candidates.Select(c => c.Id).ToList();
        var p = ids.Select(i => inclusion[i]).ToList();
        var counts = CountDistribution(p);
        var inclusionObject = new JsonObject();
        for (int i = 0; i < ids.Count; ++i) inclusionObject[ids[i]] = p[i];
        return new JsonObject
        {
            ["type"] = "choice",
            ["multi"] = true,
            ["inclusion"] = inclusionObject,
            ["selected"] = new JsonArray([.. ids.Where((_, i) => p[i] >= .5).Select(id => (JsonNode)id)]),
            ["count_distribution"] = new JsonArray([.. counts.Select(c => (JsonNode)c)]),
            ["expected_count"] = PythonText.Sum(counts.Select((v, k) => k * v)),
        };
    }

    // ---- typed results ----

    /// <summary>Positions of false and true among a noul row's candidates; probabilities are always [false, true].</summary>
    public static (int False, int True) NoulCandidateOrder(IReadOnlyList<MicaCandidate> candidates)
    {
        if (candidates.Count != 2) throw new MicaRequestException("Noul candidates must contain false and true exactly once");
        var ids = candidates.Select(c => c.Id).ToList();
        if (!ids.Order(StringComparer.Ordinal).SequenceEqual(["false", "true"]))
        {
            throw new MicaRequestException("Noul candidates must contain false and true exactly once");
        }
        return (ids.IndexOf("false"), ids.IndexOf("true"));
    }

    /// <summary>
    /// The public typed shape of a row's distribution over its labels (<c>typed_result</c>). Choice keeps the
    /// caller's ids, Score keys levels by ordinal, a Noul's distribution is [false, true].
    /// </summary>
    public static JsonObject TypedResult(MicaRow row, IReadOnlyList<double> probabilities)
    {
        var candidates = row.Candidates;
        if (row.Kind == "noul") NoulCandidateOrder(candidates);
        else if (candidates.Count < 2) throw new MicaRequestException("Choice/Score requires at least two candidates");
        else if (candidates.Count > MaxCandidates) throw new MicaRequestException($"{row.Kind} has too many candidates");
        int expected = row.Kind == "noul" ? 2 : candidates.Count;
        if (probabilities.Count != expected) throw new ArgumentException("Probability count does not match the judgment");
        if (probabilities.Any(v => !double.IsFinite(v) || v < 0)) throw new ArgumentException("Invalid probability distribution");
        double total = PythonText.Sum(probabilities);
        if (Math.Abs(total - 1.0) > 1e-5) throw new ArgumentException("Invalid probability distribution");
        var p = probabilities.Select(v => v / total).ToList();

        if (row.Kind == "noul") return new JsonObject { ["type"] = "noul", ["noul"] = p[1] };

        int n = p.Count;
        if (row.Kind == "choice")
        {
            int mode = ArgMax(p);
            var probabilityObject = new JsonObject();
            for (int i = 0; i < n; ++i) probabilityObject[candidates[i].Id] = p[i];
            return new JsonObject
            {
                ["type"] = "choice",
                ["choice"] = candidates[mode].Id,
                ["probabilities"] = probabilityObject,
                ["confidence"] = (p[mode] - 1.0 / n) / (1 - 1.0 / n),
            };
        }

        bool cannotGrade = row.AllowCannotGrade;
        if (cannotGrade && (n < 3 || candidates[^1].Id != "cannot_grade"))
        {
            throw new ArgumentException("Score cannot-grade option must be final, after at least two levels");
        }
        var levels = cannotGrade ? candidates.Take(n - 1).ToList() : [.. candidates];
        var levelP = cannotGrade ? p.Take(n - 1).ToList() : p;
        int count = levels.Count;
        var ordered = levels.Select((level, i) => (Ordinal: level.Ordinal ?? i, P: levelP[i], Level: level))
            .OrderBy(x => x.Ordinal).ToList();
        if (!ordered.Select(x => x.Ordinal).SequenceEqual(Enumerable.Range(0, count)))
        {
            throw new ArgumentException("Score ordinals must be a complete 0-based sequence");
        }
        double mass = PythonText.Sum(ordered.Select(x => x.P));
        var graded = cannotGrade && mass != 0 ? ordered.Select(x => x.P / mass).ToList() : ordered.Select(x => x.P).ToList();
        int best = ArgMax(graded);
        double weighted = PythonText.Sum(ordered.Select(x => x.Ordinal * x.P));
        double? score = mass != 0 ? (cannotGrade ? weighted / mass : weighted) : null;
        double center = (count - 1) / 2.0;
        double deviation = PythonText.Sum(Enumerable.Range(0, count).Select(i => Math.Abs(i - center))) / count;
        double confidence = mass != 0
            ? Math.Max(0.0, 1 - PythonText.Sum(graded.Select((v, i) => Math.Abs(i - best) * v)) / deviation)
            : 0.0;
        var ranked = p.OrderByDescending(v => v).ToList();

        var legend = new JsonObject();
        var levelProbabilities = new JsonObject();
        for (int i = 0; i < count; ++i)
        {
            string key = i.ToString(System.Globalization.CultureInfo.InvariantCulture);
            legend[key] = ordered[i].Level.Text;
            levelProbabilities[key] = ordered[i].P;
        }
        bool hasLevel = mass != 0 && (!cannotGrade || ordered[best].P >= p[^1]);
        var result = new JsonObject
        {
            ["type"] = "score",
            ["score"] = score,
            ["legend"] = legend,
            ["probabilities"] = levelProbabilities,
            ["confidence"] = confidence,
            ["level"] = hasLevel ? best : null,
            ["expected_level"] = score,
            ["margin"] = ranked[0] - ranked[1],
        };
        if (cannotGrade) result["cannot_grade_probability"] = p[^1];
        return result;
    }

    private static int ArgMax(IReadOnlyList<double> values)
    {
        int best = 0;
        for (int i = 1; i < values.Count; ++i) if (values[i] > values[best]) best = i;
        return best;
    }

    /// <summary>
    /// Runs a validated request through a batch probability backend (<c>answer_request</c>): plain rows
    /// first, then the inclusion rows of multi-answer questions, one distribution per row.
    /// </summary>
    public static JsonObject AnswerRequest(JsonNode? request, Func<IReadOnlyList<MicaRow>, IReadOnlyList<IReadOnlyList<double>>> predict,
        string model, JsonObject? usage = null)
    {
        var rows = RowsFromRequest(request);
        var multi = rows.Where(r => r.Multi).ToList();
        var plain = rows.Where(r => !r.Multi).ToList();
        var expanded = multi.SelectMany(InclusionRows).ToList();
        var all = plain.Concat(expanded).ToList();
        var predicted = all.Count > 0 ? predict(all) : [];
        if (predicted.Count != all.Count) throw new InvalidOperationException("predict must return one distribution per row");

        var answers = new JsonObject();
        for (int i = 0; i < plain.Count; ++i)
        {
            var distribution = predicted[i];
            var result = TypedResult(plain[i], distribution);
            if (plain[i].AbstainThreshold is double threshold)
            {
                double confidence = distribution.Max();
                result["confidence_pmax"] = confidence;
                result["abstain"] = confidence < threshold;
            }
            answers[plain[i].Id] = result;
        }
        if (expanded.Count > 0)
        {
            var inclusion = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
            for (int i = 0; i < expanded.Count; ++i)
            {
                var r = expanded[i];
                if (!inclusion.TryGetValue(r.MultiOf!, out var map)) inclusion[r.MultiOf!] = map = new(StringComparer.Ordinal);
                map[r.MultiOption!] = predicted[plain.Count + i][1];    // noul order is [false, true]
            }
            foreach (var row in multi) answers[row.Id] = Assemble(row, inclusion[row.Id]);
        }

        var response = new JsonObject { ["model"] = model, ["answers"] = answers };
        if (usage is not null) response["usage"] = usage;
        return response;
    }
}

/// <summary>The bits of Python's string and float semantics the contract depends on.</summary>
internal static class PythonText
{
    /// <summary><c>str.strip()</c>: Python's whitespace includes the four ASCII separators U+001C–U+001F.</summary>
    public static string Strip(string text)
    {
        static bool Space(char c) => char.IsWhiteSpace(c) || c is >= '\u001c' and <= '\u001f';
        int start = 0, end = text.Length;
        while (start < end && Space(text[start])) start++;
        while (end > start && Space(text[end - 1])) end--;
        return start == 0 && end == text.Length ? text : text[start..end];
    }

    /// <summary><c>sum()</c> over floats as CPython 3.12+ computes it: Neumaier-compensated.</summary>
    public static double Sum(IEnumerable<double> values)
    {
        double sum = 0, compensation = 0;
        foreach (double x in values)
        {
            double t = sum + x;
            compensation += Math.Abs(sum) >= Math.Abs(x) ? (sum - t) + x : (x - t) + sum;
            sum = t;
        }
        return compensation != 0 && double.IsFinite(compensation) ? sum + compensation : sum;
    }
}

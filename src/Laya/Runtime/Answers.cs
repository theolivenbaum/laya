using System.Text.Json.Serialization;

namespace Laya.Runtime;

/// <summary>What the action head predicts alongside every answer.</summary>
public sealed record ActionEstimate(
    [property: JsonPropertyName("act_probability")] double ActProbability);

/// <summary>One answer. The populated fields depend on <see cref="Type"/>.</summary>
public sealed record Answer
{
    [JsonPropertyName("type")]
    public required string Type { get; init; }

    /// <summary>The winning option key of a <c>choice</c> question.</summary>
    [JsonPropertyName("choice")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Choice { get; init; }

    /// <summary>The expected level of a <c>score</c> question: <c>Σ i · p(i)</c>.</summary>
    [JsonPropertyName("score")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Score { get; init; }

    /// <summary>The probability that a <c>noul</c> statement holds.</summary>
    [JsonPropertyName("noul")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Noul { get; init; }

    /// <summary>Per-option probabilities, keyed by option name or level index.</summary>
    [JsonPropertyName("probabilities")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyList<KeyValuePair<string, double>>? Probabilities { get; init; }

    /// <summary>The level texts of a <c>score</c> question, keyed by index.</summary>
    [JsonPropertyName("legend")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyList<KeyValuePair<string, string>>? Legend { get; init; }

    [JsonPropertyName("confidence")]
    public required double Confidence { get; init; }

    [JsonPropertyName("action")]
    public required ActionEstimate Action { get; init; }

    /// <summary>Probability of a named option, or 0 when the question has no such option.</summary>
    public double ProbabilityOf(string option)
    {
        if (Probabilities is null) return 0d;
        foreach (var (key, value) in Probabilities)
        {
            if (key == option) return value;
        }
        return 0d;
    }
}

/// <summary>Token accounting, mirroring the Python payload's <c>usage</c> block.</summary>
public sealed record Usage(
    [property: JsonPropertyName("input_tokens")] int InputTokens,
    [property: JsonPropertyName("output_tokens")] int OutputTokens);

/// <summary>The result of one <c>system_one</c> call.</summary>
public sealed record DecisionResult
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("answers")]
    public required IReadOnlyList<KeyValuePair<string, Answer>> Answers { get; init; }

    [JsonPropertyName("usage")]
    public required Usage Usage { get; init; }

    /// <summary>Present when the call went through a <see cref="Router"/>.</summary>
    [JsonPropertyName("routing")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public RouteDecision? Routing { get; init; }

    public Answer this[string id]
    {
        get
        {
            foreach (var (key, answer) in Answers)
            {
                if (key == id) return answer;
            }
            throw new KeyNotFoundException($"no answer with id '{id}'.");
        }
    }

    public bool TryGet(string id, out Answer answer)
    {
        foreach (var (key, value) in Answers)
        {
            if (key == id)
            {
                answer = value;
                return true;
            }
        }
        answer = null!;
        return false;
    }
}

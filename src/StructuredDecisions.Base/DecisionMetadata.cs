using System.Text.Json.Serialization;

namespace StructuredDecisions;

// Metadata a result can carry about how it was produced: which model a router picked and why, and what
// a shortlist did to a large choice question. Plain data, shared by every model family.

/// <summary>
/// What Laya's <c>LanguageDetector.Analyse</c> found in a state. The JSON names match the
/// Python payload, because a routing decision is serialized straight into API responses.
/// </summary>
/// <param name="LanguageUndecided">
/// True when no language could be identified. Undecided is not English: it is reported separately
/// so a caller can tell "this is English" from "nothing here says what this is".
/// </param>
public sealed record LanguageAnalysis(
    [property: JsonPropertyName("script")] string Script,
    [property: JsonPropertyName("script_profile")] IReadOnlyDictionary<string, double> ScriptProfile,
    [property: JsonPropertyName("language")] string? Language,
    [property: JsonPropertyName("is_english")] bool IsEnglish,
    [property: JsonPropertyName("language_undecided")] bool LanguageUndecided,
    [property: JsonPropertyName("diacritic_rate")] double DiacriticRate,
    [property: JsonPropertyName("non_latin_fraction")] double NonLatinFraction)
{
    /// <summary>What an language classifier said, when one was consulted.</summary>
    [JsonPropertyName("classifier_language")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ClassifierLanguage { get; init; }

    [JsonPropertyName("classifier_probability")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? ClassifierProbability { get; init; }
}

/// <summary>The routing outcome: which checkpoint, why, and what was detected.</summary>
public sealed record RouteDecision
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("repo")]
    public required string Repo { get; init; }

    [JsonPropertyName("reason")]
    public required string Reason { get; init; }

    [JsonPropertyName("detection")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LanguageAnalysis? Detection { get; init; }

    [JsonPropertyName("workflow")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Workflow { get; init; }

    public override string ToString() => $"RouteDecision(model='{Model}', reason='{Reason}')";
}

/// <summary>What a shortlist pass (<c>Shortlist.PredictShortlist</c>) did to one choice question.</summary>
public sealed record ShortlistEntry
{
    /// <summary>The kept labels, best first — or every label in its original order on a passthrough.</summary>
    [JsonPropertyName("labels")]
    public required IReadOnlyList<string> Labels { get; init; }

    /// <summary>Cosine similarity per kept label, or null when nothing was dropped.</summary>
    [JsonPropertyName("scores")]
    public IReadOnlyList<double>? Scores { get; init; }

    [JsonPropertyName("k")]
    public required int K { get; init; }

    /// <summary>How many labels the question had before shortlisting.</summary>
    [JsonPropertyName("n")]
    public required int N { get; init; }

    /// <summary>True when the question had at most <see cref="K"/> labels and was forwarded unchanged.</summary>
    [JsonPropertyName("passthrough")]
    public required bool Passthrough { get; init; }
}

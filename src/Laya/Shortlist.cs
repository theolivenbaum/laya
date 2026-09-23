using System.Text.Json.Serialization;
using Laya.Models;
using Laya.Runtime;

namespace Laya;

/// <summary>Maps texts to one embedding row each: <c>[texts.Count][dim]</c>.</summary>
public delegate float[][] EmbedFunction(IReadOnlyList<string> texts);

/// <summary>What <see cref="Shortlist.PredictShortlist"/> did to one choice question.</summary>
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

/// <summary>
/// Opt-in embedding shortlist for high-cardinality <c>choice</c> questions — the port of
/// <c>.reference/laya/shortlist.py</c>.
///
/// <para>Choice options share one <c>head_max_len</c> budget, so a large label set leaves each label
/// only a few tokens. <see cref="PredictShortlist"/> embeds the state and every option, keeps the
/// top <c>k</c> by cosine similarity, and runs a single decision pass over the reduced set. The
/// decision model itself is untouched: it still scores every criterion it is given.</para>
/// </summary>
public static class Shortlist
{
    public const int DefaultK = 20;

    /// <summary>
    /// The top-<paramref name="k"/> labels for <paramref name="state"/>. <paramref name="embed"/> is
    /// called once, with the query first and then one string per option in criteria order; when
    /// <paramref name="k"/> covers every label it is not called at all. Ties keep the earlier label.
    /// </summary>
    public static IReadOnlyList<string> ShortlistChoice(object? state, Question question, EmbedFunction embed,
        int k = DefaultK)
        => Rank(state, question, embed, k).Labels;

    /// <summary>
    /// Shortlists each choice question, then answers everything in one pass. Non-choice questions and
    /// choices with at most <paramref name="k"/> labels are forwarded unchanged. Probabilities on a
    /// shortlisted choice are over the kept labels only.
    /// </summary>
    public static DecisionResult PredictShortlist(IDecisionEngine engine, object? state, QuestionSet questions,
        EmbedFunction embed, int k = DefaultK, ParallelOptions? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(engine);
        var (reduced, meta) = Reduce(state, questions, embed, k);
        return engine.SystemOne(state, reduced, parallel) with { Shortlist = meta };
    }

    /// <summary>The same through a router, so the reduced questions are routed like any others.</summary>
    public static DecisionResult PredictShortlist(Router router, object? state, QuestionSet questions,
        EmbedFunction embed, int k = DefaultK, string? model = null, string? task = null, string? lang = null,
        ParallelOptions? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(router);
        var (reduced, meta) = Reduce(state, questions, embed, k);
        return router.Predict(state, reduced, model, task, lang, parallel) with { Shortlist = meta };
    }

    private static (QuestionSet Reduced, IReadOnlyList<KeyValuePair<string, ShortlistEntry>> Meta) Reduce(
        object? state, QuestionSet questions, EmbedFunction embed, int k)
    {
        ArgumentNullException.ThrowIfNull(questions);
        CheckK(k);
        var reduced = new QuestionSet();
        var meta = new List<KeyValuePair<string, ShortlistEntry>>();
        foreach (var (id, question) in questions)
        {
            if (question.Type != QuestionType.Choice)
            {
                reduced.Add(id, question);
                continue;
            }

            var entry = Rank(state, question, embed, k);
            meta.Add(new KeyValuePair<string, ShortlistEntry>(id, entry));
            if (entry.Passthrough)
            {
                reduced.Add(id, question);
                continue;
            }

            var byKey = question.NamedCriteria!.ToDictionary(c => c.Key, c => c.Value, StringComparer.Ordinal);
            reduced.Add(id, Question.Choice(question.Instructions,
                entry.Labels.Select(label => new KeyValuePair<string, object?>(label, byKey[label]))));
        }
        return (reduced, meta);
    }

    private static ShortlistEntry Rank(object? state, Question question, EmbedFunction embed, int k)
    {
        CheckK(k);
        if (question.Type != QuestionType.Choice)
        {
            throw new ArgumentException("only choice questions can be shortlisted.", nameof(question));
        }
        var criteria = question.NamedCriteria;
        if (criteria is null || criteria.Count == 0)
        {
            throw new ArgumentException("choice criteria must contain at least one option.", nameof(question));
        }
        var keys = criteria.Select(c => c.Key).ToList();
        if (keys.Distinct(StringComparer.Ordinal).Count() != keys.Count)
        {
            throw new ArgumentException("choice criteria labels must be unique.", nameof(question));
        }
        if (k >= keys.Count)
        {
            return new ShortlistEntry { Labels = keys, Scores = null, K = k, N = keys.Count, Passthrough = true };
        }

        ArgumentNullException.ThrowIfNull(embed);
        var texts = new List<string>(keys.Count + 1) { QueryText(state, question.Instructions) };
        texts.AddRange(question.RenderOptions());
        var matrix = embed(texts);
        if (matrix is null || matrix.Length != texts.Count || matrix.Any(r => r is null || r.Length < 1)
            || matrix.Any(r => r.Length != matrix[0].Length))
        {
            throw new InvalidOperationException(
                $"the embed function must return {texts.Count} rows of equal, non-zero width.");
        }

        var similarities = new double[keys.Count];
        for (int i = 0; i < keys.Count; ++i) similarities[i] = Cosine(matrix[0], matrix[i + 1]);

        // A stable sort on the negated score is what keeps the earlier label on a tie.
        var order = Enumerable.Range(0, keys.Count).OrderBy(i => -similarities[i]).Take(k).ToList();
        return new ShortlistEntry
        {
            Labels = [.. order.Select(i => keys[i])],
            Scores = [.. order.Select(i => similarities[i])],
            K = k,
            N = keys.Count,
            Passthrough = false,
        };
    }

    private static void CheckK(int k)
    {
        if (k < 1) throw new ArgumentOutOfRangeException(nameof(k), k, "k must be a positive integer.");
    }

    private static string QueryText(object? state, object instructions)
    {
        string body = SequenceBuilder.SerializeState(state);
        string text = instructions as string ?? PythonJson.Dumps(instructions);
        return text.Length == 0 ? body : text + "\n" + body;
    }

    /// <summary>Cosine similarity; a zero vector (or a NaN/infinite component, read as zero) scores 0.</summary>
    private static double Cosine(float[] query, float[] document)
    {
        double dot = 0d, queryNorm = 0d, documentNorm = 0d;
        for (int i = 0; i < query.Length; ++i)
        {
            double q = float.IsFinite(query[i]) ? query[i] : 0d;
            double d = float.IsFinite(document[i]) ? document[i] : 0d;
            dot += q * d;
            queryNorm += q * q;
            documentNorm += d * d;
        }
        double denominator = Math.Sqrt(queryNorm) * Math.Sqrt(documentNorm);
        return denominator > 0d ? dot / denominator : 0d;
    }

    /// <summary>
    /// Mean-pools the encoder the agent already holds: each text becomes <c>[CLS] text [SEP]</c>,
    /// truncated to <paramref name="maxLength"/>, and its last hidden states are averaged. A dedicated
    /// bi-encoder will usually shortlist better; this exists for callers that only have the Laya
    /// checkpoint in memory. The decision head is not run and nothing is downloaded.
    /// </summary>
    public static EmbedFunction EmbedFunctionFromAgent(Agent agent, int maxLength = 512, int batchSize = 32,
        ParallelOptions? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(agent);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxLength, 3);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);

        var tokenizer = agent.Tokenizer;
        var encoder = agent.Model.Encoder;
        int hidden = encoder.Config.HiddenSize;

        return texts =>
        {
            var rows = new float[texts.Count][];
            for (int start = 0; start < texts.Count; start += batchSize)
            {
                int count = Math.Min(batchSize, texts.Count - start);
                var ids = new List<int>();
                var segments = new Segment[count];
                for (int i = 0; i < count; ++i)
                {
                    var encoded = tokenizer.Encode(texts[start + i] ?? string.Empty);
                    int body = Math.Min(encoded.Count, maxLength - 2);
                    segments[i] = new Segment(ids.Count, body + 2);
                    ids.Add(tokenizer.ClsTokenId);
                    ids.AddRange(encoded.Take(body));
                    ids.Add(tokenizer.SepTokenId);
                }

                float[] states = encoder.Forward([.. ids], segments, recorder: null, parallel);
                for (int i = 0; i < count; ++i)
                {
                    var pooled = new float[hidden];
                    var segment = segments[i];
                    for (int t = segment.Start; t < segment.End; ++t)
                    {
                        Numerics.SimdOps.Add(pooled, states.AsSpan(t * hidden, hidden));
                    }
                    Numerics.SimdOps.Scale(pooled, 1f / segment.Length);
                    rows[start + i] = pooled;
                }
            }
            return rows;
        };
    }
}

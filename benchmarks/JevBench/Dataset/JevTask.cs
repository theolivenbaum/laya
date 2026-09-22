using System.Text.Json;

namespace Laya.JevBench.Dataset;

/// <summary>
/// One typed decision record, straight from JevBench's public JSONL — the C# mirror of
/// <c>jevbench/tasks.py</c>'s <c>Task</c> dataclass. Only the fields the scorer and the adapter
/// need are kept; everything else (provenance, exact rationale) stays as raw JSON so nothing is
/// silently dropped.
/// </summary>
public sealed class JevTask
{
    public required string Id { get; init; }
    public required string Family { get; init; }
    public required JsonElement State { get; init; }
    public required JevQuestion Question { get; init; }
    public required IReadOnlyList<string> Labels { get; init; }

    /// <summary>
    /// Ground truth: a label string for noul/choice, an integer level index for score, or null
    /// when unmeasured (never present in the public split, kept for parity with the schema).
    /// </summary>
    public required JsonElement? Expected { get; init; }
    public required string Split { get; init; }
    public string? Group { get; init; }
    public JsonElement Provenance { get; init; }

    /// <summary>Which JevBench tier this record belongs to — assigned by the loader from the source file, since the public JSONL itself carries only "split".</summary>
    public required string Tier { get; init; }

    /// <summary>The exact ground-truth label, in the same string form <c>Labels</c> and a scored prediction use.</summary>
    public string? ExpectedLabel => Expected switch
    {
        null => null,
        { ValueKind: JsonValueKind.Null } => null,
        { ValueKind: JsonValueKind.String } e => e.GetString(),
        { ValueKind: JsonValueKind.Number } e => e.GetInt32().ToString(System.Globalization.CultureInfo.InvariantCulture),
        _ => throw new InvalidDataException($"{Id}: unexpected 'expected' value kind {Expected.Value.ValueKind}"),
    };

    /// <summary>
    /// The exact gold probability distribution over labels, present only on the hard tier's
    /// "probability" family (<c>provenance.gold_probs</c>) — used for the probability-fidelity
    /// calibration sub-score, never for accuracy.
    /// </summary>
    public IReadOnlyDictionary<string, double>? GoldProbabilities
    {
        get
        {
            if (Provenance.ValueKind != JsonValueKind.Object
                || !Provenance.TryGetProperty("gold_probs", out var gold)
                || gold.ValueKind != JsonValueKind.Object)
            {
                return null;
            }
            var dict = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (var p in gold.EnumerateObject()) dict[p.Name] = p.Value.GetDouble();
            return dict;
        }
    }
}

/// <summary>The typed question spec: type, instructions, and the ordered rubric ("criteria").</summary>
public sealed class JevQuestion
{
    public required string Type { get; init; }
    public required JsonElement Instructions { get; init; }
    public JsonElement? Criteria { get; init; }
}

/// <summary>Loads and validates JevBench's public JSONL task files.</summary>
public static class JevTaskLoader
{
    /// <summary>
    /// Reads one JSONL file and tags every record with <paramref name="tier"/> — the JevBench
    /// tier this file's items score into (easy / standard / hard). "judge" has no public items:
    /// both of its source cohorts (router, judge) are held out.
    /// </summary>
    public static List<JevTask> Load(string path, string tier)
    {
        var tasks = new List<JevTask>();
        foreach (string line in File.ReadLines(path))
        {
            string trimmed = line.Trim();
            if (trimmed.Length == 0) continue;
            using var document = JsonDocument.Parse(trimmed);
            tasks.Add(Parse(document.RootElement, tier, path));
        }
        return tasks;
    }

    private static JevTask Parse(JsonElement root, string tier, string path)
    {
        string id = root.GetProperty("id").GetString() ?? throw new InvalidDataException($"{path}: task with no id");
        var questionElement = root.GetProperty("question");
        var question = new JevQuestion
        {
            Type = questionElement.GetProperty("type").GetString()
                ?? throw new InvalidDataException($"{id}: question has no type"),
            Instructions = questionElement.GetProperty("instructions").Clone(),
            Criteria = questionElement.TryGetProperty("criteria", out var criteria)
                && criteria.ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined)
                ? criteria.Clone() : null,
        };

        var labels = new List<string>();
        foreach (var label in root.GetProperty("labels").EnumerateArray()) labels.Add(label.GetString()!);

        var task = new JevTask
        {
            Id = id,
            Family = root.GetProperty("family").GetString() ?? "",
            State = root.GetProperty("state").Clone(),
            Question = question,
            Labels = labels,
            Expected = root.TryGetProperty("expected", out var expected)
                && expected.ValueKind is not JsonValueKind.Undefined ? expected.Clone() : null,
            Split = root.GetProperty("split").GetString() ?? "",
            Group = root.TryGetProperty("group", out var g) && g.ValueKind == JsonValueKind.String ? g.GetString() : null,
            Provenance = root.TryGetProperty("provenance", out var prov) ? prov.Clone() : default,
            Tier = tier,
        };
        Validate(task, path);
        return task;
    }

    private static void Validate(JevTask task, string path)
    {
        if (task.Question.Type is not ("noul" or "choice" or "score"))
        {
            throw new InvalidDataException($"{path}/{task.Id}: bad question type '{task.Question.Type}'");
        }
        if (task.Labels.Count == 0) throw new InvalidDataException($"{path}/{task.Id}: empty labels");
        string? expected = task.ExpectedLabel;
        if (expected is not null && !task.Labels.Contains(expected))
        {
            throw new InvalidDataException($"{path}/{task.Id}: expected '{expected}' not in labels");
        }
    }
}

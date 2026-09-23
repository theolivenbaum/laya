using System.Globalization;
using System.Net.Http.Json;
using System.Text.Json;
using Laya.Runtime;
using Laya.Tokenizers;

namespace Laya.Training;

/// <summary>The reference answer to one question: its label and soft target distribution.</summary>
public sealed record GoldAnswer(string? Label, IReadOnlyDictionary<string, double> Probabilities, double? Score, double? Noul);

/// <summary>
/// One case of a typed-decisions dataset: a state, the questions asked about it, and the gold
/// answers. The shape is <c>LocalLLaMA/typed-decisions</c>': <c>state</c>, <c>questions</c> and
/// <c>gold</c> are JSON, either inline or JSON-encoded as strings the way the Hub stores them.
/// </summary>
public sealed record DecisionCase(string Id, string? Workflow, JsonElement State, QuestionSet Questions,
    IReadOnlyDictionary<string, GoldAnswer> Gold);

/// <summary>A training sequence plus where it came from, for reporting.</summary>
public sealed record LabelledItem(TrainingItem Item, string CaseId, string QuestionId, string? Workflow);

/// <summary>Reads, downloads and converts typed-decisions data.</summary>
public static class DecisionDataset
{
    /// <summary>Reads one case per line. Blank lines are skipped.</summary>
    public static List<DecisionCase> ReadJsonLines(string path)
    {
        var cases = new List<DecisionCase>();
        int line = 0;
        foreach (string text in File.ReadLines(path))
        {
            line++;
            if (string.IsNullOrWhiteSpace(text)) continue;
            using var document = JsonDocument.Parse(text);
            try
            {
                cases.Add(ParseCase(document.RootElement));
            }
            catch (Exception exception) when (exception is KeyNotFoundException or InvalidOperationException or ArgumentException or JsonException)
            {
                throw new InvalidDataException($"{path}:{line}: {exception.Message}", exception);
            }
        }
        return cases;
    }

    public static DecisionCase ParseCase(JsonElement row)
    {
        string id = row.TryGetProperty("id", out var idElement) ? idElement.ToString() : Guid.NewGuid().ToString("N");
        string? workflow = row.TryGetProperty("workflow", out var w) && w.ValueKind == JsonValueKind.String ? w.GetString() : null;
        var state = Decode(row.GetProperty("state"));
        var questions = QuestionJson.Parse(Decode(row.GetProperty("questions")));

        var gold = new Dictionary<string, GoldAnswer>(StringComparer.Ordinal);
        foreach (var property in Decode(row.GetProperty("gold")).EnumerateObject())
        {
            var value = property.Value;
            var probabilities = new Dictionary<string, double>(StringComparer.Ordinal);
            if (value.TryGetProperty("probabilities", out var p) && p.ValueKind == JsonValueKind.Object)
            {
                foreach (var entry in p.EnumerateObject()) probabilities[entry.Name] = entry.Value.GetDouble();
            }
            gold[property.Name] = new GoldAnswer(
                value.TryGetProperty("label", out var label) && label.ValueKind != JsonValueKind.Null ? label.ToString() : null,
                probabilities,
                value.TryGetProperty("score", out var score) && score.ValueKind == JsonValueKind.Number ? score.GetDouble() : null,
                value.TryGetProperty("noul", out var noul) && noul.ValueKind == JsonValueKind.Number ? noul.GetDouble() : null);
        }
        return new DecisionCase(id, workflow, state, questions, gold);
    }

    /// <summary>A JSON value, or the JSON a string value encodes.</summary>
    private static JsonElement Decode(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.String) return element.Clone();
        using var document = JsonDocument.Parse(element.GetString()!);
        return document.RootElement.Clone();
    }

    /// <summary>
    /// Downloads a split through the Hugging Face datasets-server rows API into a JSON-lines file and
    /// returns its path, reusing a previous download. No parquet reader is needed.
    /// </summary>
    public static async Task<string> DownloadAsync(string dataset, string config, string split, string directory,
        string? token = null, IProgress<string>? progress = null, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, $"{dataset.Replace('/', '_')}.{config}.{split}.jsonl");
        if (File.Exists(path)) return path;

        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
        token ??= Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token)) client.DefaultRequestHeaders.Authorization = new("Bearer", token);

        string temporary = path + ".tmp";
        await using (var writer = new StreamWriter(temporary))
        {
            int offset = 0, total = int.MaxValue;
            const int page = 100;
            while (offset < total)
            {
                string url = string.Create(CultureInfo.InvariantCulture,
                    $"https://datasets-server.huggingface.co/rows?dataset={Uri.EscapeDataString(dataset)}&config={Uri.EscapeDataString(config)}&split={Uri.EscapeDataString(split)}&offset={offset}&length={page}");
                using var document = JsonDocument.Parse(await client.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
                var root = document.RootElement;
                total = root.GetProperty("num_rows_total").GetInt32();
                int received = 0;
                foreach (var row in root.GetProperty("rows").EnumerateArray())
                {
                    await writer.WriteLineAsync(row.GetProperty("row").GetRawText().AsMemory(), cancellationToken).ConfigureAwait(false);
                    received++;
                }
                if (received == 0) break;
                offset += received;
                progress?.Report($"{dataset} [{config}/{split}]: {offset} / {total} rows");
            }
        }
        File.Move(temporary, path, overwrite: true);
        return path;
    }

    /// <summary>
    /// The soft target of one question in label order — the port of <c>build_training_item</c>:
    /// choice options in criteria order, <c>[false, true]</c> for noul, levels <c>0..n-1</c> for score;
    /// missing entries are 0 (0.5 for noul), and the result is normalised (uniform if it sums to 0).
    /// </summary>
    public static float[] Target(Question question, GoldAnswer gold)
    {
        double[] raw = question.Type switch
        {
            QuestionType.Choice => [.. question.OptionKeys.Select(k => gold.Probabilities.GetValueOrDefault(k, 0d))],
            QuestionType.Noul => [gold.Probabilities.GetValueOrDefault("false", 0.5), gold.Probabilities.GetValueOrDefault("true", 0.5)],
            _ => [.. Enumerable.Range(0, question.Levels?.Count ?? 4)
                .Select(i => gold.Probabilities.GetValueOrDefault(i.ToString(CultureInfo.InvariantCulture), 0d))],
        };
        double sum = raw.Sum();
        return sum > 0 ? [.. raw.Select(v => (float)(v / sum))] : [.. raw.Select(_ => 1f / raw.Length)];
    }

    /// <summary>
    /// Every (case, question) pair with a gold answer, tokenized as inference will see it. A question
    /// whose options do not fit <paramref name="headMaxLength"/> is skipped, as the reference does.
    /// </summary>
    public static List<LabelledItem> BuildItems(HuggingFaceTokenizer tokenizer, IEnumerable<DecisionCase> cases,
        int maxLength, int headMaxLength)
    {
        var items = new List<LabelledItem>();
        foreach (var @case in cases)
        {
            foreach (var (id, question) in @case.Questions)
            {
                if (!@case.Gold.TryGetValue(id, out var gold)) continue;
                var target = Target(question, gold);
                var sequence = SequenceBuilder.Build(tokenizer, @case.State, question, maxLength, headMaxLength);
                if (sequence.MarkerPositions.Length != question.RenderOptions().Count) continue;
                items.Add(new LabelledItem(
                    new TrainingItem(sequence.TokenIds, sequence.MarkerPositions, (int)question.Type, target),
                    @case.Id, id, @case.Workflow));
            }
        }
        return items;
    }
}

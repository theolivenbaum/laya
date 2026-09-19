using System.Globalization;
using System.Text.Json;

namespace Laya.Diagnostics;

/// <summary>Receives intermediate tensors so a run can be compared against the PyTorch reference.</summary>
public interface IStateRecorder
{
    void Record(string name, ReadOnlySpan<float> values, ReadOnlySpan<int> shape);
}

/// <summary>A captured tensor.</summary>
public sealed record RecordedState(string Name, int[] Shape, float[] Values)
{
    public float Mean => Values.Length == 0 ? 0f : Values.Average();

    public float MaxAbs
    {
        get
        {
            float max = 0f;
            foreach (float value in Values) max = MathF.Max(max, MathF.Abs(value));
            return max;
        }
    }
}

/// <summary>Keeps every recorded tensor in memory and can write them out as JSON.</summary>
public sealed class StateRecorder : IStateRecorder
{
    private readonly List<RecordedState> _states = [];
    private readonly Func<string, bool>? _filter;

    public StateRecorder(Func<string, bool>? filter = null) => _filter = filter;

    public IReadOnlyList<RecordedState> States => _states;

    public void Record(string name, ReadOnlySpan<float> values, ReadOnlySpan<int> shape)
    {
        if (_filter is not null && !_filter(name)) return;
        _states.Add(new RecordedState(name, shape.ToArray(), values.ToArray()));
    }

    public RecordedState? Find(string name) => _states.FirstOrDefault(s => s.Name == name);

    /// <summary>
    /// Writes the dump as JSON. Full tensors get large fast, so by default only a deterministic
    /// slice is written — the first <paramref name="sampleSize"/> values plus per-tensor
    /// statistics, which is enough to localise where two implementations diverge.
    /// <paramref name="answersJson"/> is embedded verbatim under an "answers" key.
    /// </summary>
    public void Save(string path, int sampleSize = 64, bool full = false, string? answersJson = null)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var stream = File.Create(path);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false });

        writer.WriteStartObject();
        writer.WriteString("format", "laya-state-dump/1");
        writer.WriteStartArray("tensors");
        foreach (var state in _states)
        {
            writer.WriteStartObject();
            writer.WriteString("name", state.Name);
            writer.WriteStartArray("shape");
            foreach (int dimension in state.Shape) writer.WriteNumberValue(dimension);
            writer.WriteEndArray();
            writer.WriteNumber("count", state.Values.Length);
            writer.WriteNumber("mean", state.Mean);
            writer.WriteNumber("max_abs", state.MaxAbs);
            writer.WriteStartArray("values");
            int limit = full ? state.Values.Length : Math.Min(sampleSize, state.Values.Length);
            for (int i = 0; i < limit; ++i) writer.WriteNumberValue(state.Values[i]);
            writer.WriteEndArray();
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
        if (answersJson is not null)
        {
            // Written alongside the tensors so one file carries both what the layers produced and
            // what the runtime made of it; the comparison tool checks the answers too.
            using var answers = JsonDocument.Parse(answersJson);
            writer.WritePropertyName("answers");
            answers.RootElement.WriteTo(writer);
        }
        writer.WriteEndObject();
    }

    public string Describe()
    {
        var lines = new List<string>();
        foreach (var state in _states)
        {
            lines.Add(string.Format(CultureInfo.InvariantCulture,
                "{0,-44} [{1,-12}] mean={2,12:F6} max|x|={3,12:F6} first={4:F6}",
                state.Name, string.Join("x", state.Shape), state.Mean, state.MaxAbs,
                state.Values.Length > 0 ? state.Values[0] : 0f));
        }
        return string.Join(Environment.NewLine, lines);
    }
}

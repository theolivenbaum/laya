using System.Text.Json;

namespace StructuredDecisions.Io;

/// <summary>
/// The weights of a checkpoint that may be split over several safetensors shards
/// (<c>model.safetensors.index.json</c> plus <c>model-0000k-of-0000n.safetensors</c>), or held in a
/// single <c>model.safetensors</c>. Every tensor read is routed to the shard that holds it.
/// </summary>
public sealed class SafetensorsCheckpoint : IDisposable
{
    private readonly List<SafetensorsFile> _files = [];
    private readonly Dictionary<string, SafetensorsFile> _byTensor = new(StringComparer.Ordinal);

    public string Directory { get; }

    public SafetensorsCheckpoint(string directory)
    {
        Directory = directory;
        string single = Path.Combine(directory, "model.safetensors");
        string index = Path.Combine(directory, "model.safetensors.index.json");
        IEnumerable<string> shards;
        if (File.Exists(single))
        {
            shards = [single];
        }
        else if (File.Exists(index))
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(index));
            shards = document.RootElement.GetProperty("weight_map").EnumerateObject()
                .Select(p => p.Value.GetString()!).Distinct(StringComparer.Ordinal)
                .Select(name => Path.Combine(directory, name)).ToList();
        }
        else
        {
            throw new FileNotFoundException($"No model.safetensors or model.safetensors.index.json in '{directory}'.");
        }

        foreach (string shard in shards)
        {
            var file = new SafetensorsFile(shard);
            _files.Add(file);
            foreach (string name in file.Entries.Keys) _byTensor[name] = file;
        }
    }

    public IEnumerable<string> Names => _byTensor.Keys;

    public bool Contains(string name) => _byTensor.ContainsKey(name);

    public SafetensorsEntry Entry(string name) => Shard(name).Entry(name);

    public float[] ReadFloat32(string name) => Shard(name).ReadFloat32(name);

    public ushort[] ReadBFloat16Bits(string name) => Shard(name).ReadBFloat16Bits(name);

    private SafetensorsFile Shard(string name) => _byTensor.TryGetValue(name, out var file)
        ? file
        : throw new KeyNotFoundException($"{Directory}: no tensor named '{name}'.");

    public void Dispose()
    {
        foreach (var file in _files) file.Dispose();
    }
}

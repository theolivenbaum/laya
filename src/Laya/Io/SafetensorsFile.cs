using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Laya.Io;

/// <summary>One tensor entry in a safetensors header.</summary>
public sealed record SafetensorsEntry(string Name, string DType, int[] Shape, long Start, long End)
{
    public long ElementCount
    {
        get
        {
            long count = 1;
            foreach (int dimension in Shape) count *= dimension;
            return count;
        }
    }
}

/// <summary>
/// Memory-mapped reader for <c>model.safetensors</c>.
///
/// <para>Laya checkpoints store the weights as fp16 with a couple of fp32 scalars, so every read
/// widens to fp32: PyTorch also runs this model in fp32 on CPU, and that is what the parity dumps
/// are measured against. Widening happens once, at load, into arrays the model keeps.</para>
/// </summary>
public sealed class SafetensorsFile : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly long _dataStart;

    public IReadOnlyDictionary<string, SafetensorsEntry> Entries { get; }
    public IReadOnlyDictionary<string, string> Metadata { get; }
    public string Path { get; }

    public SafetensorsFile(string path)
    {
        Path = path;
        using (var header = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            Span<byte> lengthBytes = stackalloc byte[8];
            header.ReadExactly(lengthBytes);
            long headerLength = BinaryPrimitives.ReadInt64LittleEndian(lengthBytes);
            if (headerLength <= 0 || headerLength > 512L * 1024 * 1024)
            {
                throw new InvalidDataException($"{path}: implausible safetensors header length {headerLength}.");
            }

            byte[] json = new byte[headerLength];
            header.ReadExactly(json);
            _dataStart = 8 + headerLength;

            var entries = new Dictionary<string, SafetensorsEntry>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
            using var document = JsonDocument.Parse(json);
            foreach (var property in document.RootElement.EnumerateObject())
            {
                if (property.Name == "__metadata__")
                {
                    foreach (var meta in property.Value.EnumerateObject())
                    {
                        metadata[meta.Name] = meta.Value.ToString();
                    }
                    continue;
                }

                string dtype = property.Value.GetProperty("dtype").GetString()
                    ?? throw new InvalidDataException($"{path}: tensor {property.Name} has no dtype.");
                var shapeElement = property.Value.GetProperty("shape");
                int[] shape = new int[shapeElement.GetArrayLength()];
                int index = 0;
                foreach (var dimension in shapeElement.EnumerateArray()) shape[index++] = dimension.GetInt32();
                var offsets = property.Value.GetProperty("data_offsets");
                long start = offsets[0].GetInt64();
                long end = offsets[1].GetInt64();
                entries[property.Name] = new SafetensorsEntry(property.Name, dtype, shape, start, end);
            }

            Entries = entries;
            Metadata = metadata;
        }

        _file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        _view = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
    }

    public bool Contains(string name) => Entries.ContainsKey(name);

    public SafetensorsEntry Entry(string name) => Entries.TryGetValue(name, out var entry)
        ? entry
        : throw new KeyNotFoundException($"{Path}: no tensor named '{name}'. " +
            $"Nearby names: {string.Join(", ", Entries.Keys.Where(k => k.Contains(name.Split('.')[0], StringComparison.Ordinal)).Take(5))}");

    /// <summary>Reads a tensor and widens it to fp32.</summary>
    public float[] ReadFloat32(string name)
    {
        var entry = Entry(name);
        long count = entry.ElementCount;
        if (count > int.MaxValue) throw new NotSupportedException($"{name}: tensor has {count} elements.");

        float[] result = new float[count];
        Read(entry, result);
        return result;
    }

    /// <summary>Reads a tensor into an existing fp32 buffer.</summary>
    public unsafe void Read(SafetensorsEntry entry, Span<float> destination)
    {
        long count = entry.ElementCount;
        if (destination.Length < count) throw new ArgumentException($"{entry.Name}: destination too small.", nameof(destination));

        byte* basePointer = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);
        try
        {
            byte* start = basePointer + _dataStart + entry.Start;
            long bytes = entry.End - entry.Start;
            switch (entry.DType)
            {
                case "F32":
                    Expect(bytes, count * 4, entry);
                    new ReadOnlySpan<float>(start, (int)count).CopyTo(destination);
                    break;
                case "F16":
                    Expect(bytes, count * 2, entry);
                    WidenHalf(new ReadOnlySpan<Half>(start, (int)count), destination);
                    break;
                case "BF16":
                    Expect(bytes, count * 2, entry);
                    WidenBFloat16(new ReadOnlySpan<ushort>(start, (int)count), destination);
                    break;
                case "F64":
                    Expect(bytes, count * 8, entry);
                    var doubles = new ReadOnlySpan<double>(start, (int)count);
                    for (int i = 0; i < count; ++i) destination[i] = (float)doubles[i];
                    break;
                case "I64":
                    Expect(bytes, count * 8, entry);
                    var longs = new ReadOnlySpan<long>(start, (int)count);
                    for (int i = 0; i < count; ++i) destination[i] = longs[i];
                    break;
                default:
                    throw new NotSupportedException($"{entry.Name}: unsupported safetensors dtype '{entry.DType}'.");
            }
        }
        finally
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    private static void Expect(long actual, long expected, SafetensorsEntry entry)
    {
        if (actual != expected)
        {
            throw new InvalidDataException(
                $"{entry.Name}: {entry.DType} tensor of shape [{string.Join(", ", entry.Shape)}] should occupy " +
                $"{expected} bytes but the header spans {actual}.");
        }
    }

    private static void WidenHalf(ReadOnlySpan<Half> source, Span<float> destination)
    {
        for (int i = 0; i < source.Length; ++i) destination[i] = (float)source[i];
    }

    private static void WidenBFloat16(ReadOnlySpan<ushort> source, Span<float> destination)
    {
        // bf16 is simply the top half of an fp32, so widening is a 16-bit shift.
        var asUInt = MemoryMarshal.Cast<float, uint>(destination);
        for (int i = 0; i < source.Length; ++i) asUInt[i] = (uint)source[i] << 16;
    }

    public void Dispose()
    {
        _view.Dispose();
        _file.Dispose();
    }
}

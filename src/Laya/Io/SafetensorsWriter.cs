using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace Laya.Io;

/// <summary>One tensor to write: its name, shape, fp32 values and the dtype it is stored as.</summary>
public sealed record SafetensorsTensor(string Name, int[] Shape, float[] Values, string DType = "F16");

/// <summary>
/// Writes <c>model.safetensors</c>: an 8-byte little-endian header length, a JSON header of
/// <c>{name: {dtype, shape, data_offsets}}</c> padded with spaces to an 8-byte boundary, then the raw
/// little-endian tensor bytes in header order. <c>F16</c>, <c>BF16</c> and <c>F32</c> are supported,
/// which covers every Laya checkpoint.
/// </summary>
public static class SafetensorsWriter
{
    public static void Write(string path, IReadOnlyList<SafetensorsTensor> tensors,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        long offset = 0;
        var header = new Dictionary<string, object>(StringComparer.Ordinal);
        if (metadata is { Count: > 0 }) header["__metadata__"] = metadata;
        foreach (var tensor in tensors)
        {
            long count = 1;
            foreach (int dimension in tensor.Shape) count *= dimension;
            if (count != tensor.Values.Length)
            {
                throw new ArgumentException(
                    $"{tensor.Name}: shape [{string.Join(", ", tensor.Shape)}] needs {count} values, got {tensor.Values.Length}.");
            }
            long bytes = count * ElementSize(tensor.DType);
            header[tensor.Name] = new Dictionary<string, object>
            {
                ["dtype"] = tensor.DType,
                ["shape"] = tensor.Shape,
                ["data_offsets"] = new[] { offset, offset + bytes },
            };
            offset += bytes;
        }

        byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
        int padded = (json.Length + 7) / 8 * 8;

        string temporary = path + ".tmp";
        using (var stream = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
        {
            Span<byte> length = stackalloc byte[8];
            BinaryPrimitives.WriteInt64LittleEndian(length, padded);
            stream.Write(length);
            stream.Write(json);
            for (int i = json.Length; i < padded; ++i) stream.WriteByte((byte)' ');

            byte[] buffer = new byte[1 << 20];
            foreach (var tensor in tensors) WriteValues(stream, tensor, buffer);
        }
        File.Move(temporary, path, overwrite: true);
    }

    private static int ElementSize(string dtype) => dtype switch
    {
        "F32" => 4,
        "F16" or "BF16" => 2,
        _ => throw new NotSupportedException($"cannot write safetensors dtype '{dtype}'."),
    };

    private static void WriteValues(Stream stream, SafetensorsTensor tensor, byte[] buffer)
    {
        int size = ElementSize(tensor.DType);
        int perChunk = buffer.Length / size;
        for (int start = 0; start < tensor.Values.Length; start += perChunk)
        {
            int count = Math.Min(perChunk, tensor.Values.Length - start);
            var span = buffer.AsSpan(0, count * size);
            for (int i = 0; i < count; ++i)
            {
                float value = tensor.Values[start + i];
                switch (tensor.DType)
                {
                    case "F32": BinaryPrimitives.WriteSingleLittleEndian(span[(i * 4)..], value); break;
                    case "F16": BinaryPrimitives.WriteHalfLittleEndian(span[(i * 2)..], (Half)value); break;
                    default: BinaryPrimitives.WriteUInt16LittleEndian(span[(i * 2)..], ToBFloat16(value)); break;
                }
            }
            stream.Write(span);
        }
    }

    /// <summary>Round-to-nearest-even truncation of an fp32 to its top 16 bits.</summary>
    private static ushort ToBFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value)) return (ushort)((bits >> 16) | 0x40);
        uint rounding = 0x7FFF + ((bits >> 16) & 1);
        return (ushort)((bits + rounding) >> 16);
    }
}

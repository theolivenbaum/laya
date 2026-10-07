using System.Buffers.Binary;
using System.IO.Compression;

namespace Laya.D1;

/// <summary>
/// A small PNG reader for <see cref="RgbImage.Load"/>: 8-bit greyscale, grey + alpha, RGB, RGBA and
/// palette images, non-interlaced, as Pillow's <c>convert("RGB")</c> sees them (alpha is dropped,
/// not composited). Enough for the CLI and tests without an imaging dependency.
/// </summary>
internal static class PngDecoder
{
    public static RgbImage Decode(byte[] data)
    {
        int position = 8;
        int width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
        byte[]? palette = null;
        using var compressed = new MemoryStream();
        while (position + 8 <= data.Length)
        {
            int length = BinaryPrimitives.ReadInt32BigEndian(data.AsSpan(position));
            string type = System.Text.Encoding.ASCII.GetString(data, position + 4, 4);
            var chunk = data.AsSpan(position + 8, length);
            switch (type)
            {
                case "IHDR":
                    width = BinaryPrimitives.ReadInt32BigEndian(chunk);
                    height = BinaryPrimitives.ReadInt32BigEndian(chunk[4..]);
                    bitDepth = chunk[8];
                    colorType = chunk[9];
                    interlace = chunk[12];
                    break;
                case "PLTE":
                    palette = chunk.ToArray();
                    break;
                case "IDAT":
                    compressed.Write(chunk);
                    break;
            }
            position += 12 + length;
            if (type == "IEND") break;
        }

        if (bitDepth != 8 || interlace != 0)
        {
            throw new NotSupportedException($"PNG with bit depth {bitDepth}{(interlace != 0 ? ", interlaced" : "")} is not read.");
        }
        int channels = colorType switch
        {
            0 => 1,
            2 => 3,
            3 => 1,
            4 => 2,
            6 => 4,
            _ => throw new NotSupportedException($"PNG color type {colorType} is not read."),
        };

        compressed.Position = 0;
        using var inflater = new ZLibStream(compressed, CompressionMode.Decompress);
        int stride = width * channels;
        var raw = new byte[(stride + 1) * height];
        inflater.ReadExactly(raw);

        var previous = new byte[stride];
        var current = new byte[stride];
        var pixels = new byte[width * height * 3];
        for (int y = 0; y < height; ++y)
        {
            int filter = raw[y * (stride + 1)];
            var line = raw.AsSpan(y * (stride + 1) + 1, stride);
            for (int i = 0; i < stride; ++i)
            {
                int left = i >= channels ? current[i - channels] : 0;
                int up = previous[i];
                int upLeft = i >= channels ? previous[i - channels] : 0;
                current[i] = (byte)(line[i] + filter switch
                {
                    0 => 0,
                    1 => left,
                    2 => up,
                    3 => (left + up) / 2,
                    4 => Paeth(left, up, upLeft),
                    _ => throw new InvalidDataException($"PNG filter {filter}."),
                });
            }
            for (int x = 0; x < width; ++x)
            {
                int o = (y * width + x) * 3;
                switch (colorType)
                {
                    case 0:
                    case 4:
                        pixels[o] = pixels[o + 1] = pixels[o + 2] = current[x * channels];
                        break;
                    case 3:
                        int index = current[x] * 3;
                        pixels[o] = palette![index];
                        pixels[o + 1] = palette[index + 1];
                        pixels[o + 2] = palette[index + 2];
                        break;
                    default:
                        pixels[o] = current[x * channels];
                        pixels[o + 1] = current[x * channels + 1];
                        pixels[o + 2] = current[x * channels + 2];
                        break;
                }
            }
            (previous, current) = (current, previous);
        }
        return new RgbImage(width, height, pixels);
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
}

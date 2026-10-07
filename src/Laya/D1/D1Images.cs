using System.Text;
using System.Text.Json;
using Laya.Models;

namespace Laya.D1;

/// <summary>An 8-bit RGB picture, row-major and interleaved: <c>Pixels[(y · Width + x) · 3 + c]</c>.</summary>
public sealed record RgbImage(int Width, int Height, byte[] Pixels)
{
    /// <summary>
    /// Reads a binary PPM (<c>P6</c>, 8-bit) or a PNG file. Other formats need decoding by the caller
    /// (any imaging library gives RGB bytes), since the base library takes no imaging dependency.
    /// </summary>
    public static RgbImage Load(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length > 2 && bytes[0] == 'P' && bytes[1] == '6') return ReadPpm(bytes);
        if (bytes.Length > 8 && bytes[0] == 0x89 && bytes[1] == (byte)'P' && bytes[2] == (byte)'N' && bytes[3] == (byte)'G')
        {
            return PngDecoder.Decode(bytes);
        }
        throw new NotSupportedException($"'{path}': only PPM (P6) and PNG are read; decode other formats to RGB bytes first.");
    }

    private static RgbImage ReadPpm(byte[] bytes)
    {
        int position = 2;
        int Next()
        {
            while (true)
            {
                while (char.IsWhiteSpace((char)bytes[position])) position++;
                if (bytes[position] != '#') break;
                while (bytes[position] != '\n') position++;
            }
            int value = 0;
            while (char.IsAsciiDigit((char)bytes[position])) value = value * 10 + (bytes[position++] - '0');
            return value;
        }
        int width = Next(), height = Next(), max = Next();
        if (max != 255) throw new NotSupportedException("only 8-bit PPM is read.");
        position++;   // the single whitespace after the header
        return new RgbImage(width, height, bytes.AsSpan(position, width * height * 3).ToArray()).Validate();
    }

    public RgbImage Validate()
    {
        if (Width <= 0 || Height <= 0) throw new ArgumentException("an image needs a positive size.");
        if (Pixels.Length != (long)Width * Height * 3)
        {
            throw new ArgumentException($"expected {Width} x {Height} x 3 = {(long)Width * Height * 3} bytes, got {Pixels.Length}.");
        }
        return this;
    }
}

/// <summary>One tile ready for the vision tower: normalized patches over a <c>Rows x Columns</c> patch grid.</summary>
public sealed record ImageTile(float[] Patches, int Rows, int Columns)
{
    /// <summary>Language-model tokens the tile becomes after the 2 x 2 pixel unshuffle.</summary>
    public int Tokens(int factor) => (Rows + factor - 1) / factor * ((Columns + factor - 1) / factor);
}

/// <summary>A processed picture: its tiles (the grid, row-major, then the thumbnail) and its placeholder text.</summary>
public sealed record ProcessedImage(IReadOnlyList<ImageTile> Tiles, int GridRows, int GridColumns, string Placeholder);

/// <summary>
/// <c>Lfm2VlImageProcessor</c> and <c>Lfm2VlProcessor</c>'s placeholder expansion, plus the pixel cap
/// <c>runner.cap_pixels</c> applies before them.
///
/// <para>A picture that fits is resized so both sides are multiples of 32 and it holds 64 to 256
/// image tokens. A larger one is cut into a grid of 512-pixel tiles (2 to 10 of them, the grid
/// closest to its aspect ratio) with a thumbnail of the whole appended. Each tile is cut into
/// 16-pixel patches, normalized to [-1, 1]; after the vision tower, 2 x 2 patches become one
/// language-model token.</para>
/// </summary>
public sealed class D1ImageProcessor
{
    public int DownsampleFactor { get; init; } = 2;
    public bool DoImageSplitting { get; init; } = true;
    public int MinTiles { get; init; } = 2;
    public int MaxTiles { get; init; } = 10;
    public bool UseThumbnail { get; init; } = true;
    public int MinImageTokens { get; init; } = 64;
    public int MaxImageTokens { get; init; } = 256;
    public int PatchSize { get; init; } = 16;
    public int TileSize { get; init; } = 512;
    public double MaxPixelsTolerance { get; init; } = 2.0;
    public float[] Mean { get; init; } = [0.5f, 0.5f, 0.5f];
    public float[] Std { get; init; } = [0.5f, 0.5f, 0.5f];
    public float RescaleFactor { get; init; } = 1f / 255f;
    public bool UseImageSpecialTokens { get; init; } = true;

    /// <summary><c>runner.VISION_MAX_PIXELS</c>: pictures are first capped at this many pixels.</summary>
    public long MaxPixels { get; init; } = 1024 * 1024;

    public const string ImageToken = "<image>";
    public const string ImageStart = "<|image_start|>";
    public const string ImageEnd = "<|image_end|>";
    public const string ImageThumbnail = "<|img_thumbnail|>";

    /// <summary>Reads <c>processor_config.json</c>'s <c>image_processor</c> block, or the model config.</summary>
    public static D1ImageProcessor FromDirectory(string directory)
    {
        string path = Path.Combine(directory, "processor_config.json");
        if (!File.Exists(path)) return new D1ImageProcessor();
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = document.RootElement.TryGetProperty("image_processor", out var ip) ? ip : document.RootElement;
        var defaults = new D1ImageProcessor();
        int Int(string name, int fallback) => root.TryGetProperty(name, out var v) ? v.GetInt32() : fallback;
        bool Bool(string name, bool fallback) => root.TryGetProperty(name, out var v) ? v.GetBoolean() : fallback;
        float[] Floats(string name, float[] fallback) => root.TryGetProperty(name, out var v)
            ? [.. v.EnumerateArray().Select(e => (float)e.GetDouble())]
            : fallback;
        return new D1ImageProcessor
        {
            DownsampleFactor = Int("downsample_factor", defaults.DownsampleFactor),
            DoImageSplitting = Bool("do_image_splitting", defaults.DoImageSplitting),
            MinTiles = Int("min_tiles", defaults.MinTiles),
            MaxTiles = Int("max_tiles", defaults.MaxTiles),
            UseThumbnail = Bool("use_thumbnail", defaults.UseThumbnail),
            MinImageTokens = Int("min_image_tokens", defaults.MinImageTokens),
            MaxImageTokens = Int("max_image_tokens", defaults.MaxImageTokens),
            PatchSize = Int("encoder_patch_size", defaults.PatchSize),
            TileSize = Int("tile_size", defaults.TileSize),
            MaxPixelsTolerance = root.TryGetProperty("max_pixels_tolerance", out var tol) ? tol.GetDouble() : defaults.MaxPixelsTolerance,
            Mean = Floats("image_mean", defaults.Mean),
            Std = Floats("image_std", defaults.Std),
            RescaleFactor = root.TryGetProperty("rescale_factor", out var rf) ? (float)rf.GetDouble() : defaults.RescaleFactor,
        };
    }

    /// <summary><c>cap_pixels</c>: a picture over <see cref="MaxPixels"/> is downscaled (Pillow bicubic) to fit.</summary>
    public RgbImage CapPixels(RgbImage image)
    {
        long pixels = (long)image.Width * image.Height;
        if (pixels <= MaxPixels) return image;
        double scale = Math.Sqrt(MaxPixels / (double)pixels);
        int width = Math.Max(1, (int)(image.Width * scale));
        int height = Math.Max(1, (int)(image.Height * scale));
        return new RgbImage(width, height,
            Resampling.ResizeBicubicPillow(image.Pixels, image.Height, image.Width, height, width));
    }

    /// <summary>Caps, resizes or tiles, normalizes and patchifies one picture, and builds its placeholder.</summary>
    public ProcessedImage Process(RgbImage image)
    {
        image = CapPixels(image.Validate());
        int height = image.Height, width = image.Width;
        int minTiles = DoImageSplitting ? MinTiles : 1;
        int maxTiles = DoImageSplitting ? MaxTiles : 1;
        bool splitting = !(minTiles == 1 && maxTiles == 1);
        var (newWidth, newHeight) = SmartResize(height, width);

        var tiles = new List<ImageTile>();
        int gridRows = 1, gridColumns = 1;
        if (IsTooLarge(height, width) && splitting)
        {
            (gridColumns, gridRows) = GridLayout(height, width, minTiles, maxTiles);
            int targetHeight = TileSize * gridRows, targetWidth = TileSize * gridColumns;
            byte[] resized = Resampling.ResizeBicubicTorch(image.Pixels, height, width, targetHeight, targetWidth);
            for (int r = 0; r < gridRows; ++r)
            {
                for (int c = 0; c < gridColumns; ++c)
                {
                    tiles.Add(Patchify(resized, targetWidth, r * TileSize, c * TileSize, TileSize, TileSize));
                }
            }
            if (UseThumbnail && gridRows * gridColumns != 1)
            {
                byte[] thumbnail = Resampling.ResizeBicubicTorch(image.Pixels, height, width, newHeight, newWidth);
                tiles.Add(Patchify(thumbnail, newWidth, 0, 0, newHeight, newWidth));
            }
        }
        else
        {
            byte[] resized = Resampling.ResizeBicubicTorch(image.Pixels, height, width, newHeight, newWidth);
            tiles.Add(Patchify(resized, newWidth, 0, 0, newHeight, newWidth));
        }

        return new ProcessedImage(tiles, gridRows, gridColumns, Placeholder(gridRows, gridColumns, newHeight, newWidth));
    }

    /// <summary><c>_build_image_tokens</c>: the expanded text a single <c>&lt;image&gt;</c> becomes.</summary>
    private string Placeholder(int rows, int columns, int imageHeight, int imageWidth)
    {
        int perTile = (TileSize / PatchSize + DownsampleFactor - 1) / DownsampleFactor;
        perTile *= perTile;
        int forImage = (imageHeight / PatchSize + DownsampleFactor - 1) / DownsampleFactor
            * ((imageWidth / PatchSize + DownsampleFactor - 1) / DownsampleFactor);

        var text = new StringBuilder();
        if (UseImageSpecialTokens) text.Append(ImageStart);
        if (rows > 1 || columns > 1)
        {
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < columns; ++c)
                {
                    if (UseImageSpecialTokens) text.Append($"<|img_row_{r + 1}_col_{c + 1}|>");
                    text.Insert(text.Length, ImageToken, perTile);
                }
            }
            if (UseThumbnail)
            {
                if (UseImageSpecialTokens) text.Append(ImageThumbnail);
                text.Insert(text.Length, ImageToken, forImage);
            }
        }
        else
        {
            text.Insert(text.Length, ImageToken, forImage);
        }
        if (UseImageSpecialTokens) text.Append(ImageEnd);
        return text.ToString();
    }

    /// <summary>Python's <c>round</c>: half to even.</summary>
    private static int RoundByFactor(double number, int factor) => (int)Math.Round(number / factor, MidpointRounding.ToEven) * factor;

    /// <summary><c>smart_resize</c>: sides multiples of patch · factor, area within the token budget; returns (width, height).</summary>
    public (int Width, int Height) SmartResize(int height, int width)
    {
        int factor = PatchSize * DownsampleFactor;
        long minPixels = (long)MinImageTokens * PatchSize * PatchSize * DownsampleFactor * DownsampleFactor;
        long maxPixels = (long)MaxImageTokens * PatchSize * PatchSize * DownsampleFactor * DownsampleFactor;
        int hBar = Math.Max(factor, RoundByFactor(height, factor));
        int wBar = Math.Max(factor, RoundByFactor(width, factor));
        if ((long)hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt((double)height * width / maxPixels);
            hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if ((long)hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt(minPixels / ((double)height * width));
            hBar = (int)Math.Ceiling(height * beta / factor) * factor;
            wBar = (int)Math.Ceiling(width * beta / factor) * factor;
        }
        return (wBar, hBar);
    }

    private bool IsTooLarge(int height, int width)
    {
        int factor = PatchSize * DownsampleFactor;
        int hBar = Math.Max(PatchSize, RoundByFactor(height, factor));
        int wBar = Math.Max(PatchSize, RoundByFactor(width, factor));
        return (double)hBar * wBar > (double)MaxImageTokens * PatchSize * PatchSize * DownsampleFactor * DownsampleFactor * MaxPixelsTolerance;
    }

    /// <summary><c>_get_grid_layout</c> / <c>find_closest_aspect_ratio</c>: returns (columns, rows).</summary>
    private (int Columns, int Rows) GridLayout(int height, int width, int minTiles, int maxTiles)
    {
        double aspect = (double)width / height;
        var ratios = new SortedSet<(int W, int H)>();
        for (int n = minTiles; n <= maxTiles; ++n)
        {
            for (int w = 1; w <= n; ++w)
            {
                for (int h = 1; h <= n; ++h)
                {
                    if (w * h >= minTiles && w * h <= maxTiles) ratios.Add((w, h));
                }
            }
        }

        double bestDifference = double.PositiveInfinity;
        var best = (W: 1, H: 1);
        long area = (long)width * height;
        foreach (var ratio in ratios.OrderBy(r => r.W * r.H))
        {
            double difference = Math.Abs(aspect - (double)ratio.W / ratio.H);
            if (difference < bestDifference)
            {
                bestDifference = difference;
                best = ratio;
            }
            else if (difference == bestDifference && area > 0.5 * TileSize * TileSize * ratio.W * ratio.H)
            {
                best = ratio;
            }
        }
        return best;
    }

    /// <summary>
    /// Rescale, normalize and cut one region into patches: patch-grid row-major, each patch
    /// <c>[py, px, channel]</c> (<c>convert_image_to_patches</c>).
    /// </summary>
    private ImageTile Patchify(byte[] rgb, int stride, int top, int left, int height, int width)
    {
        int p = PatchSize;
        int rows = height / p, columns = width / p;
        var patches = new float[rows * columns * p * p * 3];
        // transformers fuses the rescale into the normalization: (x - mean / rescale) / (std / rescale),
        // i.e. (x - 127.5) / 127.5 for d1, computed in fp32.
        Span<float> mean = stackalloc float[3];
        Span<float> std = stackalloc float[3];
        for (int c = 0; c < 3; ++c)
        {
            mean[c] = (float)(Mean[c] * (1.0 / RescaleFactor));
            std[c] = (float)(Std[c] * (1.0 / RescaleFactor));
        }
        int index = 0;
        for (int r = 0; r < rows; ++r)
        {
            for (int q = 0; q < columns; ++q)
            {
                for (int y = 0; y < p; ++y)
                {
                    int offset = ((top + r * p + y) * stride + left + q * p) * 3;
                    for (int x = 0; x < p * 3; ++x)
                    {
                        int c = x % 3;
                        patches[index++] = (rgb[offset + x] - mean[c]) / std[c];
                    }
                }
            }
        }
        return new ImageTile(patches, rows, columns);
    }
}

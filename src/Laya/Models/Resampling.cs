namespace Laya.Models;

/// <summary>
/// The separable antialiased resamplers the d1 vision path needs, matched to the libraries the
/// reference calls: PyTorch's float <c>interpolate(..., antialias=True)</c> (bilinear, for the
/// SigLIP2 position grid), torchvision's uint8 bicubic resize (PyTorch's native int16 fixed-point
/// kernel, for tiles and thumbnails) and Pillow's uint8 <c>Image.resize</c> (bicubic, 22-bit fixed
/// point, for the pixel cap <c>runner.cap_pixels</c> applies first). The two uint8 resamplers were
/// matched bit for bit against torchvision 0.29 and Pillow 12.
///
/// <para>All three share one scheme: an output sample at <c>i</c> is centred on
/// <c>(i + 0.5) · in / out</c> in input coordinates, and the filter is stretched by the scale when
/// shrinking, so every input pixel contributes (that is the antialiasing). The image is resampled
/// along the width first and then along the height.</para>
/// </summary>
public static class Resampling
{
    /// <summary>The filter a resampler applies.</summary>
    public enum Filter
    {
        /// <summary>Triangle, support 1.</summary>
        Bilinear,
        /// <summary>Keys cubic with <c>a = -0.5</c>, support 2 — the antialiased kernels of both PyTorch and Pillow.</summary>
        Bicubic,
    }

    private static double Support(Filter filter) => filter == Filter.Bilinear ? 1.0 : 2.0;

    private static double Evaluate(Filter filter, double x)
    {
        x = Math.Abs(x);
        if (filter == Filter.Bilinear) return x < 1.0 ? 1.0 - x : 0.0;
        const double a = -0.5;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    }

    private static float EvaluateSingle(Filter filter, float x)
    {
        x = MathF.Abs(x);
        if (filter == Filter.Bilinear) return x < 1f ? 1f - x : 0f;
        const float a = -0.5f;
        if (x < 1f) return ((a + 2f) * x - (a + 3f)) * x * x + 1f;
        if (x < 2f) return (((x - 5f) * x + 8f) * x - 4f) * a;
        return 0f;
    }

    /// <summary>One axis' taps: for each output index the first input index, the tap count and the weights.</summary>
    public sealed record Taps(int[] Start, int[] Count, float[] Weights, int Stride);

    /// <summary>
    /// The taps PyTorch's antialiased CPU kernels compute (<c>_compute_index_ranges_weights</c>), in
    /// fp32 as they are for a float tensor.
    /// </summary>
    public static Taps TorchTaps(int input, int output, Filter filter)
    {
        float scale = (float)input / output;
        float support = scale >= 1f ? (float)Support(filter) * scale : (float)Support(filter);
        float invScale = scale >= 1f ? 1f / scale : 1f;
        int stride = (int)Math.Ceiling(support) * 2 + 1;
        var start = new int[output];
        var count = new int[output];
        var weights = new float[output * stride];
        for (int i = 0; i < output; ++i)
        {
            float center = scale * (i + 0.5f);
            int min = Math.Max((int)(center - support + 0.5f), 0);
            int size = Math.Min((int)(center + support + 0.5f), input) - min;
            float total = 0f;
            for (int j = 0; j < size; ++j)
            {
                float w = EvaluateSingle(filter, (j + min - center + 0.5f) * invScale);
                weights[i * stride + j] = w;
                total += w;
            }
            if (total != 0f)
            {
                for (int j = 0; j < size; ++j) weights[i * stride + j] /= total;
            }
            start[i] = min;
            count[i] = size;
        }
        return new Taps(start, count, weights, stride);
    }

    /// <summary>
    /// <c>F.interpolate(grid, (height, width), mode="bilinear", align_corners=False, antialias=True)</c>
    /// over a <c>[grid, grid, channels]</c> table, returned as <c>[height · width, channels]</c>.
    /// </summary>
    public static float[] ResizePositionGrid(float[] table, int grid, int channels, int height, int width)
    {
        if (height == grid && width == grid) return (float[])table.Clone();
        var across = TorchTaps(grid, width, Filter.Bilinear);
        var down = TorchTaps(grid, height, Filter.Bilinear);

        // width first: [grid, width, channels]
        var horizontal = new float[grid * width * channels];
        for (int y = 0; y < grid; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                var output = horizontal.AsSpan((y * width + x) * channels, channels);
                for (int j = 0; j < across.Count[x]; ++j)
                {
                    float w = across.Weights[x * across.Stride + j];
                    var input = table.AsSpan((y * grid + across.Start[x] + j) * channels, channels);
                    for (int c = 0; c < channels; ++c) output[c] += w * input[c];
                }
            }
        }

        var result = new float[height * width * channels];
        for (int y = 0; y < height; ++y)
        {
            for (int j = 0; j < down.Count[y]; ++j)
            {
                float w = down.Weights[y * down.Stride + j];
                var input = horizontal.AsSpan((down.Start[y] + j) * width * channels, width * channels);
                var output = result.AsSpan(y * width * channels, width * channels);
                for (int c = 0; c < output.Length; ++c) output[c] += w * input[c];
            }
        }
        return result;
    }

    /// <summary>
    /// torchvision's <c>resize(uint8 image, (height, width), BICUBIC, antialias=True)</c> on CPU,
    /// which runs PyTorch's native uint8 kernel (the Pillow-SIMD algorithm): double-precision taps
    /// quantized to int16 with the largest precision that keeps the biggest tap below 2^15, a
    /// rounding width pass to bytes, then the height pass. Matched to torchvision 0.29 bit for bit.
    /// <paramref name="rgb"/> is interleaved <c>[height, width, 3]</c>.
    /// </summary>
    public static byte[] ResizeBicubicTorch(ReadOnlySpan<byte> rgb, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        byte[] current = rgb.ToArray();
        if (outWidth != inWidth)
        {
            var (start, count, taps, stride, precision) = Int16Taps(inWidth, outWidth);
            current = Horizontal(current, inHeight, inWidth, outWidth, start, count, taps, stride, precision);
        }
        if (outHeight != inHeight)
        {
            var (start, count, taps, stride, precision) = Int16Taps(inHeight, outHeight);
            current = Vertical(current, outWidth, outHeight, start, count, taps, stride, precision);
        }
        return current;
    }

    /// <summary>PyTorch's <c>_compute_index_ranges_int16_weights</c> for the antialiased bicubic filter.</summary>
    private static (int[] Start, int[] Count, int[] Taps, int Stride, int Precision) Int16Taps(int input, int output)
    {
        double scale = (double)input / output;
        double support = scale >= 1.0 ? Support(Filter.Bicubic) * scale : Support(Filter.Bicubic);
        double invScale = scale >= 1.0 ? 1.0 / scale : 1.0;
        int stride = (int)Math.Ceiling(support) * 2 + 1;
        var start = new int[output];
        var count = new int[output];
        var weights = new double[output * stride];
        double maxWeight = double.NegativeInfinity;
        for (int i = 0; i < output; ++i)
        {
            double center = scale * (i + 0.5);
            int min = Math.Max((int)(center - support + 0.5), 0);
            int size = Math.Min((int)(center + support + 0.5), input) - min;
            double total = 0;
            for (int j = 0; j < size; ++j)
            {
                double w = Evaluate(Filter.Bicubic, (j + min - center + 0.5) * invScale);
                weights[i * stride + j] = w;
                total += w;
            }
            for (int j = 0; j < size; ++j)
            {
                if (total != 0) weights[i * stride + j] /= total;
                maxWeight = Math.Max(maxWeight, weights[i * stride + j]);
            }
            start[i] = min;
            count[i] = size;
        }

        int precision = 0;
        while (precision < 22 && (int)(0.5 + maxWeight * (1 << (precision + 1))) < (1 << 15)) precision++;
        var taps = new int[output * stride];
        for (int i = 0; i < taps.Length; ++i)
        {
            double v = weights[i] * (1 << precision);
            taps[i] = (short)(v < 0 ? (int)(-0.5 + v) : (int)(0.5 + v));
        }
        return (start, count, taps, stride, precision);
    }

    private static byte[] Horizontal(byte[] source, int height, int inWidth, int outWidth, int[] start, int[] count, int[] taps,
        int stride, int precision)
    {
        const int channels = 3;
        var result = new byte[height * outWidth * channels];
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < outWidth; ++x)
            {
                for (int c = 0; c < channels; ++c)
                {
                    long sum = 1L << (precision - 1);
                    for (int j = 0; j < count[x]; ++j)
                    {
                        sum += source[(y * inWidth + start[x] + j) * channels + c] * (long)taps[x * stride + j];
                    }
                    result[(y * outWidth + x) * channels + c] = (byte)Math.Clamp(sum >> precision, 0, 255);
                }
            }
        }
        return result;
    }

    private static byte[] Vertical(byte[] source, int width, int outHeight, int[] start, int[] count, int[] taps, int stride,
        int precision)
    {
        const int channels = 3;
        var result = new byte[outHeight * width * channels];
        for (int y = 0; y < outHeight; ++y)
        {
            for (int i = 0; i < width * channels; ++i)
            {
                long sum = 1L << (precision - 1);
                for (int j = 0; j < count[y]; ++j)
                {
                    sum += source[(start[y] + j) * width * channels + i] * (long)taps[y * stride + j];
                }
                result[y * width * channels + i] = (byte)Math.Clamp(sum >> precision, 0, 255);
            }
        }
        return result;
    }

    /// <summary>
    /// Pillow's <c>Image.resize(size, BICUBIC)</c> for an 8-bit RGB image: double-precision taps,
    /// quantized to 22-bit fixed point, a rounding horizontal pass to bytes, then the vertical pass.
    /// </summary>
    public static byte[] ResizeBicubicPillow(ReadOnlySpan<byte> rgb, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        const int channels = 3;
        byte[] current = rgb.ToArray();
        int width = inWidth;
        if (outWidth != inWidth)
        {
            var (start, count, taps, stride) = PillowTaps(inWidth, outWidth);
            var next = new byte[inHeight * outWidth * channels];
            for (int y = 0; y < inHeight; ++y)
            {
                for (int x = 0; x < outWidth; ++x)
                {
                    for (int c = 0; c < channels; ++c)
                    {
                        long sum = 1L << (PrecisionBits - 1);
                        for (int j = 0; j < count[x]; ++j)
                        {
                            sum += current[(y * width + start[x] + j) * channels + c] * (long)taps[x * stride + j];
                        }
                        next[(y * outWidth + x) * channels + c] = Clip8(sum);
                    }
                }
            }
            current = next;
            width = outWidth;
        }
        if (outHeight != inHeight)
        {
            var (start, count, taps, stride) = PillowTaps(inHeight, outHeight);
            var next = new byte[outHeight * width * channels];
            for (int y = 0; y < outHeight; ++y)
            {
                for (int i = 0; i < width * channels; ++i)
                {
                    long sum = 1L << (PrecisionBits - 1);
                    for (int j = 0; j < count[y]; ++j)
                    {
                        sum += current[(start[y] + j) * width * channels + i] * (long)taps[y * stride + j];
                    }
                    next[y * width * channels + i] = Clip8(sum);
                }
            }
            current = next;
        }
        return current;
    }

    private const int PrecisionBits = 32 - 8 - 2;

    private static byte Clip8(long value)
    {
        long shifted = value >> PrecisionBits;
        return (byte)Math.Clamp(shifted, 0, 255);
    }

    /// <summary>Pillow's <c>precompute_coeffs</c> + <c>normalize_coeffs_8bpc</c> for its bicubic filter.</summary>
    private static (int[] Start, int[] Count, int[] Taps, int Stride) PillowTaps(int input, int output)
    {
        double scale = (double)input / output;
        double filterScale = Math.Max(scale, 1.0);
        double support = Support(Filter.Bicubic) * filterScale;
        int stride = (int)Math.Ceiling(support) * 2 + 1;
        var start = new int[output];
        var count = new int[output];
        var taps = new int[output * stride];
        var weights = new double[stride];
        for (int i = 0; i < output; ++i)
        {
            double center = (i + 0.5) * scale;
            double ss = 1.0 / filterScale;
            int min = Math.Max((int)(center - support + 0.5), 0);
            int max = Math.Min((int)(center + support + 0.5), input) - min;
            double total = 0;
            for (int j = 0; j < max; ++j)
            {
                double w = Evaluate(Filter.Bicubic, (j + min - center + 0.5) * ss);
                weights[j] = w;
                total += w;
            }
            for (int j = 0; j < max; ++j)
            {
                double w = total != 0 ? weights[j] / total : 0;
                taps[i * stride + j] = (int)(w * (1 << PrecisionBits) + (w < 0 ? -0.5 : 0.5));
            }
            start[i] = min;
            count[i] = max;
        }
        return (start, count, taps, stride);
    }
}

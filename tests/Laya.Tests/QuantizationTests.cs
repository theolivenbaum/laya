using Laya.Numerics;
using Xunit;
using Xunit.Abstractions;

namespace Laya.Tests;

/// <summary>
/// The int8 projection against the float one it replaces. Parity with PyTorch is measured on the
/// float path only (see <see cref="ParityTests"/>); what has to hold here is that the quantized
/// kernel computes the product it claims to, to the accuracy 8-bit arithmetic allows.
/// </summary>
public class QuantizationTests(ITestOutputHelper output)
{
    private static (float[] Weights, float[] Input) Random(int outFeatures, int inFeatures, int rows, int seed)
    {
        var random = new Random(seed);
        var weights = new float[outFeatures * inFeatures];
        // Per-output-channel scales spread over two orders of magnitude, since that is what a real
        // projection looks like and it is exactly what a per-channel scale is meant to absorb.
        for (int o = 0; o < outFeatures; ++o)
        {
            float channel = 0.02f * MathF.Exp((float)random.NextDouble() * 4 - 2);
            for (int i = 0; i < inFeatures; ++i)
            {
                weights[o * inFeatures + i] = channel * (float)(random.NextDouble() * 2 - 1);
            }
        }
        var input = new float[rows * inFeatures];
        for (int i = 0; i < input.Length; ++i) input[i] = (float)(random.NextDouble() * 2 - 1);
        return (weights, input);
    }

    private static double RelativeError(float[] reference, float[] candidate)
    {
        double numerator = 0, denominator = 0;
        for (int i = 0; i < reference.Length; ++i)
        {
            double difference = reference[i] - candidate[i];
            numerator += difference * difference;
            denominator += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(numerator / denominator);
    }

    [Theory]
    [InlineData(64, 128, 8)]
    [InlineData(1024, 1024, 17)]     // attn_out
    [InlineData(3072, 1024, 6)]      // qkv
    [InlineData(5248, 1024, 13)]     // mlp_in
    [InlineData(1024, 2624, 6)]      // mlp_out
    [InlineData(100, 260, 5)]        // ragged panel and a reduction dimension that is not a multiple of 4
    public void Int8ProjectionTracksTheFloatOne(int outFeatures, int inFeatures, int rows)
    {
        var (weights, input) = Random(outFeatures, inFeatures, rows, seed: 7);
        var bias = new float[outFeatures];
        for (int o = 0; o < outFeatures; ++o) bias[o] = 0.1f * MathF.Sin(o);

        var reference = new PackedMatrix(weights, outFeatures, inFeatures);
        var quantized = new PackedInt8Matrix(weights, outFeatures, inFeatures);

        var expected = new float[rows * outFeatures];
        var actual = new float[rows * outFeatures];
        reference.Multiply(input, rows, bias, expected);
        quantized.Multiply(input, rows, bias, actual);

        double error = RelativeError(expected, actual);
        output.WriteLine($"[{outFeatures}x{inFeatures}] rows={rows} kernel={PackedInt8Matrix.Kernel} relative error {error:F5}");
        Assert.True(error < 0.02, $"relative error {error:F5} exceeded 0.02");
    }

    [Fact]
    public void Int8ProjectionHandlesNoBiasAndStridedRows()
    {
        const int outFeatures = 1024, inFeatures = 1024, rows = 9, inputStride = 1152, outputStride = 1088;
        var (weights, dense) = Random(outFeatures, inFeatures, rows, seed: 11);

        var input = new float[rows * inputStride];
        for (int r = 0; r < rows; ++r)
        {
            dense.AsSpan(r * inFeatures, inFeatures).CopyTo(input.AsSpan(r * inputStride, inFeatures));
        }

        var reference = new PackedMatrix(weights, outFeatures, inFeatures);
        var quantized = new PackedInt8Matrix(weights, outFeatures, inFeatures);

        var expected = new float[rows * outputStride];
        var actual = new float[rows * outputStride];
        reference.Multiply(input, rows, inputStride, [], expected, outputStride);
        quantized.Multiply(input, rows, inputStride, [], actual, outputStride);

        Assert.True(RelativeError(expected, actual) < 0.02);
    }

    /// <summary>
    /// The rows past the register tile take a different code path, so a row count that is not a
    /// multiple of the tile has to produce the same answer as one that is. The held-out channels are
    /// chosen from whatever batch is presented, so they are switched off here — this is a test of
    /// the tail of the register tile, not of the channel selection.
    /// </summary>
    [Fact]
    public void Int8ProjectionAgreesAcrossTheRowTileBoundary()
    {
        const int outFeatures = 1024, inFeatures = 1024;
        var (weights, input) = Random(outFeatures, inFeatures, rows: 13, seed: 3);
        var quantized = new PackedInt8Matrix(weights, outFeatures, inFeatures, outlierChannels: 0);

        var all = new float[13 * outFeatures];
        quantized.Multiply(input, 13, [], all);

        for (int rows = 1; rows <= 13; ++rows)
        {
            var some = new float[rows * outFeatures];
            quantized.Multiply(input, rows, [], some);
            for (int i = 0; i < some.Length; ++i)
            {
                Assert.Equal(all[i], some[i], 4);
            }
        }
    }

    /// <summary>
    /// The quantization has to be exact when the values fit the grid: a weight matrix of whole
    /// multiples of its own channel scale, against activations that are whole multiples of theirs,
    /// should come back to within float round-off. This is what catches a wrong zero-point
    /// correction or a transposed panel index, which a statistical tolerance would hide.
    /// </summary>
    [Fact]
    public void Int8ProjectionIsExactOnTheQuantizationGrid()
    {
        const int outFeatures = 128, inFeatures = 64, rows = 5;
        var random = new Random(23);
        var weights = new float[outFeatures * inFeatures];
        for (int o = 0; o < outFeatures; ++o)
        {
            // Channel scale 1/32; every weight lands exactly on a grid point, and one weight per
            // channel is at full scale so the scale itself is what we think it is.
            weights[o * inFeatures] = 63 / 32f;
            for (int i = 1; i < inFeatures; ++i) weights[o * inFeatures + i] = random.Next(-63, 64) / 32f;
        }
        var input = new float[rows * inFeatures];
        for (int r = 0; r < rows; ++r)
        {
            // More channels at full scale than are ever held out, so holding some out cannot move
            // the activation scale and the whole product stays exactly on the grid.
            for (int i = 0; i < 40; ++i) input[r * inFeatures + i] = 127 / 64f;
            for (int i = 40; i < inFeatures; ++i) input[r * inFeatures + i] = random.Next(-127, 128) / 64f;
        }

        var expected = new float[rows * outFeatures];
        var actual = new float[rows * outFeatures];
        new PackedMatrix(weights, outFeatures, inFeatures).Multiply(input, rows, [], expected);
        new PackedInt8Matrix(weights, outFeatures, inFeatures).Multiply(input, rows, [], actual);

        for (int i = 0; i < expected.Length; ++i)
        {
            Assert.Equal(expected[i], actual[i], 3);
        }
    }
}

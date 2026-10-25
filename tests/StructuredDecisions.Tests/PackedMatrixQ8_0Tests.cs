using StructuredDecisions.Numerics;
using Xunit;

namespace StructuredDecisions.Tests;

public class PackedMatrixQ8_0Tests
{
    /// <summary>
    /// ggml's two quantizers round differently, and both are kept: weights (<c>quantize_row_q8_0_ref</c>) round half
    /// away from zero, activations (x86 <c>quantize_row_q8_0</c>) half to even; both store <c>d = amax/127</c> as fp16.
    /// </summary>
    [Fact]
    public void QuantizersRoundAsGgmlDoes()
    {
        var x = new float[Q8_0.BlockSize];
        x[0] = 127f;           // amax = 127: d = 1, id = 1
        x[1] = 2.5f;
        x[2] = -2.5f;
        x[3] = 3.5f;
        x[4] = -0.5f;
        x[5] = 126.49f;
        var weights = new sbyte[Q8_0.BlockSize];
        Assert.Equal(1f, Q8_0.QuantizeWeightBlock(x, weights));
        Assert.Equal([127, 3, -3, 4, -1, 126], weights[..6].Select(v => (int)v));

        var activations = new byte[Q8_0.BlockSize];
        var scales = new float[1];
        Q8_0.QuantizeActivations(x, 1, Q8_0.BlockSize, Q8_0.BlockSize, activations, scales);
        Assert.Equal(1f, scales[0]);
        Assert.Equal([127, 2, -2, 4, 0, 126], activations[..6].Select(v => v - 128));
        Assert.All(activations[6..], v => Assert.Equal(128, v));

        // d is kept as the fp16 it is stored as: 1/127 is not an fp16.
        Array.Clear(x);
        x[0] = 1f;
        float d = Q8_0.QuantizeWeightBlock(x, weights);
        Assert.Equal((float)(Half)(1f / 127f), d);
        Assert.NotEqual(1f / 127f, d);

        // An all-zero block has d = 0 and all-zero values (no division by zero).
        Array.Clear(x);
        Assert.Equal(0f, Q8_0.QuantizeWeightBlock(x, weights));
        Assert.All(weights, v => Assert.Equal(0, v));
    }

    public static TheoryData<int, int, int, bool> Shapes => new()
    {
        { 1, 64, 32, false },
        { 2, 64, 64, true },
        { 3, 128, 96, false },
        { 7, 130, 160, true },       // a ragged last panel, a one-row tail
        { 13, 192, 2560, false },
        { 32, 100, 288, true },      // only a ragged panel; enough rows to split across workers
        { 219, 256, 9216, false },
    };

    /// <summary>
    /// Every kernel gives exactly the bits of a scalar reference written from the definition — the activations
    /// quantized, each block's integer dot <c>Σ q_x · q_w</c>, then <c>acc = fma(Σ, d_w · d_x, acc)</c> block by block.
    /// The integer sums are exact and the fp32 operations are the same ones in the same order, so this is equality.
    /// </summary>
    [Theory]
    [MemberData(nameof(Shapes))]
    public void EveryKernelIsExactlyTheDefinition(int rows, int outFeatures, int inFeatures, bool withBias)
    {
        var random = new Random(rows * 131 + outFeatures * 7 + inFeatures);
        var weights = new float[outFeatures * inFeatures];
        for (int i = 0; i < weights.Length; ++i) weights[i] = (float)(random.NextDouble() * 2 - 1) * (i % 97 == 0 ? 8 : 1);
        const int inputPad = 5, outputPad = 3;
        int inputStride = inFeatures + inputPad, outputStride = outFeatures + outputPad;
        var input = new float[rows * inputStride];
        for (int i = 0; i < input.Length; ++i) input[i] = (float)(random.NextDouble() * 2 - 1);
        var bias = withBias ? Enumerable.Range(0, outFeatures).Select(_ => (float)random.NextDouble()).ToArray() : [];

        var expected = Reference(weights, outFeatures, inFeatures, input, rows, inputStride, bias);
        var matrix = new PackedMatrixQ8_0(weights, outFeatures, inFeatures);
        var kernels = new[] { Q8Kernel.Portable, Q8Kernel.Vnni256, Q8Kernel.Vnni512 }.Where(PackedMatrixQ8_0.IsSupported).ToList();
        foreach (var kernel in kernels)
        {
            foreach (var parallel in new ParallelOptions?[] { new() { MaxDegreeOfParallelism = 1 }, null })
            {
                matrix.Kernel = kernel;
                var actual = Enumerable.Repeat(float.NaN, rows * outputStride).ToArray();
                matrix.Multiply(input, rows, inputStride, bias, actual, outputStride, parallel);
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < outFeatures; ++c)
                    {
                        float want = expected[r * outFeatures + c], got = actual[r * outputStride + c];
                        if (BitConverter.SingleToInt32Bits(want) != BitConverter.SingleToInt32Bits(got))
                        {
                            Assert.Fail($"{kernel}, row {r}, column {c}: {got:R} != {want:R}");
                        }
                    }
                    for (int c = outFeatures; c < outputStride && r < rows - 1; ++c) Assert.True(float.IsNaN(actual[r * outputStride + c]), "wrote past the row");
                }
            }
        }
    }

    /// <summary>W8A8 stays close to the fp32 product: about half a percent per output on random data.</summary>
    [Fact]
    public void IsCloseToTheFp32Product()
    {
        const int rows = 9, outFeatures = 192, inFeatures = 2560;
        var random = new Random(3);
        var weights = Enumerable.Range(0, outFeatures * inFeatures).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var input = Enumerable.Range(0, rows * inFeatures).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var exact = new float[rows * outFeatures];
        var quantized = new float[rows * outFeatures];
        new PackedMatrix(weights, outFeatures, inFeatures).Multiply(input, rows, [], exact);
        new PackedMatrixQ8_0(weights, outFeatures, inFeatures).Multiply(input, rows, [], quantized);
        double error = 0, norm = 0;
        for (int i = 0; i < exact.Length; ++i)
        {
            error += (exact[i] - quantized[i]) * (double)(exact[i] - quantized[i]);
            norm += exact[i] * (double)exact[i];
        }
        Assert.InRange(Math.Sqrt(error / norm), 1e-4, 1e-2);
    }

    /// <summary>
    /// Dequantizing gives back <c>q · d</c>: within half a step of the original weight, plus what storing <c>d</c> as fp16
    /// moves the largest value (127 steps × 2⁻¹¹).
    /// </summary>
    [Fact]
    public void DequantizeIsWithinHalfAStep()
    {
        var random = new Random(5);
        var weights = Enumerable.Range(0, 70 * 64).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var matrix = new PackedMatrixQ8_0(weights, 70, 64);
        var column = new float[64];
        for (int c = 0; c < 70; ++c)
        {
            matrix.Dequantize(c, column);
            for (int b = 0; b < 2; ++b)
            {
                var block = weights.AsSpan(c * 64 + b * 32, 32);
                float step = (float)(Half)(block.ToArray().Max(MathF.Abs) / 127f);
                for (int t = 0; t < 32; ++t) Assert.True(MathF.Abs(column[b * 32 + t] - block[t]) <= step * (0.5f + 127f / 2048f) + 1e-7f);
            }
        }
    }

    private static float[] Reference(float[] weights, int outFeatures, int inFeatures, float[] input, int rows, int inputStride, float[] bias)
    {
        int blocks = inFeatures / Q8_0.BlockSize;
        var qw = new sbyte[weights.Length];
        var dw = new float[outFeatures * blocks];
        for (int c = 0; c < outFeatures; ++c)
        {
            for (int b = 0; b < blocks; ++b)
            {
                int at = c * inFeatures + b * Q8_0.BlockSize;
                dw[c * blocks + b] = Q8_0.QuantizeWeightBlock(weights.AsSpan(at, Q8_0.BlockSize), qw.AsSpan(at, Q8_0.BlockSize));
            }
        }
        var qx = new byte[rows * inFeatures];
        var dx = new float[rows * blocks];
        Q8_0.QuantizeActivations(input, rows, inputStride, inFeatures, qx, dx);

        var output = new float[rows * outFeatures];
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < outFeatures; ++c)
            {
                float acc = bias.Length > 0 ? bias[c] : 0f;
                for (int b = 0; b < blocks; ++b)
                {
                    int sum = 0;
                    for (int t = 0; t < Q8_0.BlockSize; ++t)
                    {
                        int k = b * Q8_0.BlockSize + t;
                        sum += (qx[r * inFeatures + k] - 128) * qw[c * inFeatures + k];
                    }
                    acc = MathF.FusedMultiplyAdd(sum, dw[c * blocks + b] * dx[r * blocks + b], acc);
                }
                output[r * outFeatures + c] = acc;
            }
        }
        return output;
    }
}

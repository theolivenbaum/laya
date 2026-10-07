using Laya.Numerics;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// <see cref="BFloat16Matrix"/> against a plain fp32 reference: every edge of the tiling (row
/// counts around the 16- and 32-row groups, column counts that leave a partial 12-wide panel,
/// reduction lengths across block boundaries and off the 16-step transpose) and the subnormal flush.
/// </summary>
public class BFloat16MatrixTests
{
    private static ushort Bf16(float value) => (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);

    private static float Widen(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    [Theory]
    [InlineData(1, 12, 16)]
    [InlineData(5, 13, 31)]
    [InlineData(16, 24, 300)]
    [InlineData(17, 25, 257)]
    [InlineData(31, 12, 512)]
    [InlineData(32, 36, 513)]
    [InlineData(33, 7, 64)]
    [InlineData(61, 200, 1000)]
    [InlineData(97, 389, 129)]
    public void MatchesTheFp32Product(int rows, int outFeatures, int inFeatures)
    {
        var random = new Random(rows * 7919 + outFeatures * 31 + inFeatures);
        var weights = new ushort[outFeatures * inFeatures];
        for (int i = 0; i < weights.Length; ++i) weights[i] = Bf16((float)(random.NextDouble() - 0.5));
        int stride = inFeatures + 3;
        var input = new float[rows * stride];
        for (int i = 0; i < input.Length; ++i) input[i] = (float)(random.NextDouble() - 0.5);

        var matrix = new BFloat16Matrix(weights, outFeatures, inFeatures);
        int outputStride = outFeatures + 5;
        var output = new float[rows * outputStride];
        matrix.Multiply(input, rows, stride, output, outputStride, new ParallelOptions { MaxDegreeOfParallelism = 1 });

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < outFeatures; ++c)
            {
                // The same chain of fp32 FMAs in the same order: the result is expected to be identical.
                float expected = 0f;
                for (int k = 0; k < inFeatures; ++k)
                {
                    expected = MathF.FusedMultiplyAdd(input[r * stride + k], Widen(weights[c * inFeatures + k]), expected);
                }
                Assert.Equal(expected, output[r * outputStride + c]);
            }
        }
    }

    [Fact]
    public void ParallelBlocksGiveTheSameResult()
    {
        var random = new Random(5);
        int rows = 40, outFeatures = 1000, inFeatures = 300;
        var weights = new ushort[outFeatures * inFeatures];
        for (int i = 0; i < weights.Length; ++i) weights[i] = Bf16((float)(random.NextDouble() - 0.5));
        var input = new float[rows * inFeatures];
        for (int i = 0; i < input.Length; ++i) input[i] = (float)(random.NextDouble() - 0.5);
        var matrix = new BFloat16Matrix(weights, outFeatures, inFeatures);
        var single = new float[rows * outFeatures];
        var parallel = new float[rows * outFeatures];
        matrix.Multiply(input, rows, single, new ParallelOptions { MaxDegreeOfParallelism = 1 });
        matrix.Multiply(input, rows, parallel, new ParallelOptions { MaxDegreeOfParallelism = 4 });
        Assert.Equal(single, parallel);
    }

    [Fact]
    public void SubnormalsAreFlushedToZero()
    {
        Assert.Equal((ushort)0x0000, BFloat16Matrix.FlushSubnormal(0x0001));
        Assert.Equal((ushort)0x8000, BFloat16Matrix.FlushSubnormal(0x8042));
        Assert.Equal((ushort)0x0080, BFloat16Matrix.FlushSubnormal(0x0080));   // smallest normal stays
        Assert.Equal((ushort)0x3F80, BFloat16Matrix.FlushSubnormal(0x3F80));

        // A subnormal activation contributes nothing, not its (invisible) product.
        var matrix = new BFloat16Matrix([Bf16(1f), Bf16(1f)], 1, 2);
        var output = new float[16];
        var input = new float[16 * 2];
        for (int r = 0; r < 16; ++r)
        {
            input[r * 2] = 1e-39f;
            input[r * 2 + 1] = r;
        }
        matrix.Multiply(input, 16, output);
        for (int r = 0; r < 16; ++r) Assert.Equal((float)r, output[r]);
    }
}

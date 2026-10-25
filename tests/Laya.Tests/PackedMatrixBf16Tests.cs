using Laya.Numerics;
using Xunit;

namespace Laya.Tests;

public class PackedMatrixBf16Tests
{
    /// <summary>bf16-resident weights give the same products as widening them first, over ragged shapes.</summary>
    [Theory]
    [InlineData(1, 64, 64)]
    [InlineData(7, 130, 33)]
    [InlineData(13, 256, 100)]
    [InlineData(6, 12352, 64)]
    public void MatchesTheWidenedFp32Kernel(int rows, int outFeatures, int inFeatures)
    {
        var random = new Random(rows * 31 + outFeatures);
        var bits = new ushort[outFeatures * inFeatures];
        var widened = new float[bits.Length];
        for (int i = 0; i < bits.Length; ++i)
        {
            float value = (float)(random.NextDouble() * 2 - 1);
            bits[i] = (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);
            widened[i] = BitConverter.UInt32BitsToSingle((uint)bits[i] << 16);
        }
        var input = new float[rows * inFeatures];
        for (int i = 0; i < input.Length; ++i) input[i] = (float)(random.NextDouble() * 2 - 1);
        var bias = new float[outFeatures];
        for (int i = 0; i < bias.Length; ++i) bias[i] = (float)random.NextDouble();

        var expected = new float[rows * outFeatures];
        var actual = new float[rows * outFeatures];
        new PackedMatrix(widened, outFeatures, inFeatures).Multiply(input, rows, bias, expected);
        new PackedMatrixBf16(bits, outFeatures, inFeatures).Multiply(input, rows, bias, actual);

        for (int i = 0; i < expected.Length; ++i)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-5f * (1 + MathF.Abs(expected[i])), $"{i}: {expected[i]} vs {actual[i]}");
        }
    }
}

using System.Numerics;
using Laya.Models;
using Laya.Numerics;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// Every SIMD kernel is checked against a naive scalar implementation of the same formula, because
/// the blocked and parallel paths are where a port silently goes wrong: an off-by-one in a tail
/// loop shows up as a plausible-looking answer, not a crash.
/// </summary>
public class NumericsTests
{
    /// <summary>Absolute tolerance for a length-<paramref name="k"/> fp32 reduction.</summary>
    private static double Tolerance(double expected, int k)
        => Math.Max(1e-4, Math.Abs(expected) * 1e-5) + k * 1e-6;

    private static float[] Random(int count, int seed)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; ++i) values[i] = (float)(random.NextDouble() * 4 - 2);
        return values;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(64)]
    [InlineData(1024)]
    [InlineData(1025)]
    public void DotMatchesTheScalarSum(int length)
    {
        var a = Random(length, 1);
        var b = Random(length, 2);

        double expected = 0;
        for (int i = 0; i < length; ++i) expected += (double)a[i] * b[i];

        Assert.Equal(expected, SimdOps.Dot(a, b), 3);
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 5, 7)]
    [InlineData(4, 4, 64)]
    [InlineData(6, 13, 129)]
    [InlineData(17, 33, 1024)]
    public void MatMulMatchesTheTripleLoop(int m, int n, int k)
    {
        var a = Random(m * k, 3);
        var b = Random(n * k, 4);
        var bias = Random(n, 5);
        var actual = new float[m * n];

        Gemm.MatMul(a, m, k, b, n, bias, actual);

        for (int row = 0; row < m; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                double expected = bias[col];
                for (int i = 0; i < k; ++i) expected += (double)a[row * k + i] * b[col * k + i];
                // fp32 accumulation order differs between the blocked kernel and this loop, so
                // the tolerance scales with the magnitude rather than fixing decimal places.
                Assert.Equal(expected, actual[row * n + col], Tolerance(expected, k));
            }
        }
    }

    [Fact]
    public void MatMulWithoutBiasIsTheSameAsWithAZeroBias()
    {
        var a = Random(5 * 32, 6);
        var b = Random(9 * 32, 7);
        var withZero = new float[5 * 9];
        var without = new float[5 * 9];

        Gemm.MatMul(a, 5, 32, b, 9, new float[9], withZero);
        Gemm.MatMul(a, 5, 32, b, 9, [], without);

        Assert.Equal(withZero, without);
    }

    [Fact]
    public void MatMulIsStableAcrossTheParallelThreshold()
    {
        // The parallel path only engages for large problems; both paths must agree exactly enough
        // that a forward pass does not depend on how many cores ran it.
        const int m = 64, n = 256, k = 256;
        var a = Random(m * k, 8);
        var b = Random(n * k, 9);
        var big = new float[m * n];
        Gemm.MatMul(a, m, k, b, n, [], big);

        for (int row = 0; row < m; ++row)
        {
            var single = new float[n];
            Gemm.MatMul(a.AsSpan(row * k, k), 1, k, b, n, [], single);
            for (int col = 0; col < n; ++col)
            {
                Assert.Equal(single[col], big[row * n + col], Tolerance(single[col], k));
            }
        }
    }

    [Fact]
    public void LayerNormMatchesTorchsBiasedVariance()
    {
        var input = Random(257, 10);
        var weight = Random(257, 11);
        var bias = Random(257, 12);
        var actual = new float[257];

        SimdOps.LayerNorm(input, weight, bias, 1e-5f, actual);

        double mean = input.Select(v => (double)v).Average();
        double variance = input.Select(v => (v - mean) * (v - mean)).Average();
        double inverse = 1 / Math.Sqrt(variance + 1e-5);
        for (int i = 0; i < input.Length; ++i)
        {
            Assert.Equal((input[i] - mean) * inverse * weight[i] + bias[i], actual[i], 3);
        }
    }

    [Fact]
    public void LayerNormWithoutBiasIsSupported()
    {
        var input = Random(64, 13);
        var weight = Random(64, 14);
        var withBias = new float[64];
        var withoutBias = new float[64];

        SimdOps.LayerNorm(input, weight, new float[64], 1e-5f, withBias);
        SimdOps.LayerNorm(input, weight, [], 1e-5f, withoutBias);

        Assert.Equal(withBias, withoutBias);
    }

    [Fact]
    public void SoftmaxSumsToOneAndIsShiftInvariant()
    {
        var values = Random(37, 15);
        var shifted = values.Select(v => v + 100f).ToArray();

        SimdOps.Softmax(values);
        SimdOps.Softmax(shifted);

        Assert.Equal(1f, values.Sum(), 4);
        for (int i = 0; i < values.Length; ++i) Assert.Equal(values[i], shifted[i], 5);
    }

    [Theory]
    [InlineData(0f, 0f)]
    [InlineData(1f, 0.8413447f)]
    [InlineData(-1f, -0.15865529f)]
    [InlineData(3f, 2.9959502f)]
    [InlineData(-3f, -0.00404969f)]
    public void GeluIsTheExactErfForm(float input, float expected)
    {
        var values = new[] { input };
        SimdOps.Gelu(values);
        Assert.Equal(expected, values[0], 5);
    }

    [Fact]
    public void VectorExpMatchesTheLibraryCall()
    {
        // The vectorized exp is only as good as its range reduction; check it across the span erf
        // asks for, including the saturating ends.
        for (double x = -87.0; x <= 88.0; x += 0.37)
        {
            var actual = SimdOps.Exp(new System.Numerics.Vector<float>((float)x));
            float expected = MathF.Exp((float)x);
            float tolerance = MathF.Max(MathF.Abs(expected) * 2e-6f, float.Epsilon);
            Assert.True(MathF.Abs(actual[0] - expected) <= tolerance,
                $"exp({x}): expected {expected}, got {actual[0]}");
        }
    }

    [Fact]
    public void VectorErfAgreesWithTheScalarOne()
    {
        for (double x = -8.0; x <= 8.0; x += 0.013)
        {
            var actual = SimdOps.Erf(new System.Numerics.Vector<float>((float)x));
            float expected = SimdOps.Erf((float)x);
            Assert.True(MathF.Abs(actual[0] - expected) <= 3e-6f,
                $"erf({x}): scalar {expected}, vector {actual[0]}");
        }
    }

    [Fact]
    public void GeluIsTheSameForEveryLaneAlignment()
    {
        // The vector body and the scalar tail must agree, or a tensor's last few channels drift.
        var random = new Random(42);
        var values = new float[Vector<float>.Count * 3 + 5];
        for (int i = 0; i < values.Length; ++i) values[i] = (float)(random.NextDouble() * 16 - 8);

        var expected = values.Select(v => 0.5f * v * (1f + SimdOps.Erf(v * 0.70710678f))).ToArray();
        SimdOps.Gelu(values);

        for (int i = 0; i < values.Length; ++i) Assert.Equal(expected[i], values[i], 5);
    }

    [Fact]
    public void ErfIsAccurateAcrossTheRange()
    {
        // Reference values from scipy.special.erf.
        (double X, double Expected)[] cases =
        [
            (0.0, 0.0), (0.25, 0.2763263902), (0.5, 0.5204998778), (1.0, 0.8427007929),
            (2.0, 0.9953222650), (3.0, 0.9999779095), (-1.5, -0.9661051465),
        ];
        foreach (var (x, expected) in cases) Assert.Equal(expected, SimdOps.Erf((float)x), 6);
    }

    [Fact]
    public void ReluClampsAtZero()
    {
        var values = new[] { -3f, -0.5f, 0f, 0.5f, 3f, -1e9f };
        SimdOps.Relu(values);
        Assert.Equal([0f, 0f, 0f, 0.5f, 3f, 0f], values);
    }

    [Fact]
    public void ArgMaxResolvesTiesToTheLowestIndex()
    {
        Assert.Equal(1, SimdOps.ArgMax([0f, 5f, 5f, 1f]));
        Assert.Equal(0, SimdOps.ArgMax([1f]));
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 5, 7)]
    [InlineData(4, 32, 64)]
    [InlineData(7, 33, 129)]
    [InlineData(9, 1024, 1024)]
    public void PackedMatrixMatchesTheTripleLoop(int rows, int outFeatures, int inFeatures)
    {
        // The packed kernel has three code paths — portable vectors, explicit AVX-512, and the
        // ragged tail — and a projection weight is repacked, so a layout mistake would show up as
        // plausible numbers rather than a crash.
        var weight = Random(outFeatures * inFeatures, 20);
        var input = Random(rows * inFeatures, 21);
        var bias = Random(outFeatures, 22);
        var actual = new float[rows * outFeatures];

        new PackedMatrix(weight, outFeatures, inFeatures).Multiply(input, rows, bias, actual);

        for (int row = 0; row < rows; ++row)
        {
            for (int o = 0; o < outFeatures; ++o)
            {
                double expected = bias[o];
                for (int i = 0; i < inFeatures; ++i) expected += (double)input[row * inFeatures + i] * weight[o * inFeatures + i];
                Assert.Equal(expected, actual[row * outFeatures + o], Tolerance(expected, inFeatures));
            }
        }
    }

    [Fact]
    public void ParallelOptionsDoNotChangeTheProduct()
    {
        // Large enough to cross the kernels' parallel thresholds. Each panel and each row block is
        // computed by exactly one worker whatever the degree, so the results are bit-identical, not
        // merely close.
        const int rows = 64, outFeatures = 256, inFeatures = 128;
        var weight = Random(outFeatures * inFeatures, 25);
        var input = Random(rows * inFeatures, 26);
        var bias = Random(outFeatures, 27);
        var single = new ParallelOptions { MaxDegreeOfParallelism = 1 };
        var several = new ParallelOptions { MaxDegreeOfParallelism = 3 };

        var packed = new PackedMatrix(weight, outFeatures, inFeatures);
        var packedDefault = new float[rows * outFeatures];
        var packedSingle = new float[rows * outFeatures];
        var packedSeveral = new float[rows * outFeatures];
        packed.Multiply(input, rows, bias, packedDefault);
        packed.Multiply(input, rows, bias, packedSingle, single);
        packed.Multiply(input, rows, bias, packedSeveral, several);
        Assert.Equal(packedDefault, packedSingle);
        Assert.Equal(packedDefault, packedSeveral);

        var gemmDefault = new float[rows * outFeatures];
        var gemmSingle = new float[rows * outFeatures];
        var gemmSeveral = new float[rows * outFeatures];
        Gemm.MatMul(input, rows, inFeatures, weight, outFeatures, bias, gemmDefault);
        Gemm.MatMul(input, rows, inFeatures, weight, outFeatures, bias, gemmSingle, single);
        Gemm.MatMul(input, rows, inFeatures, weight, outFeatures, bias, gemmSeveral, several);
        Assert.Equal(gemmDefault, gemmSingle);
        Assert.Equal(gemmDefault, gemmSeveral);
    }

    [Fact]
    public void WorkersOfResolvesUnboundedToEveryCore()
    {
        Assert.Equal(Environment.ProcessorCount, LayaRuntime.WorkersOf(new ParallelOptions()));
        Assert.Equal(1, LayaRuntime.WorkersOf(new ParallelOptions { MaxDegreeOfParallelism = 1 }));
        Assert.Equal(LayaRuntime.MaxDegreeOfParallelism, LayaRuntime.WorkersOf(null));
    }

    [Fact]
    public void PackedMatrixAgreesWithTheReferenceGemm()
    {
        const int rows = 11, outFeatures = 96, inFeatures = 128;
        var weight = Random(outFeatures * inFeatures, 23);
        var input = Random(rows * inFeatures, 24);
        var packed = new float[rows * outFeatures];
        var reference = new float[rows * outFeatures];

        new PackedMatrix(weight, outFeatures, inFeatures).Multiply(input, rows, [], packed);
        Gemm.MatMul(input, rows, inFeatures, weight, outFeatures, [], reference);

        for (int i = 0; i < packed.Length; ++i) Assert.Equal(reference[i], packed[i], Tolerance(reference[i], inFeatures));
    }

    [Fact]
    public void PackedMatrixIsRowCountIndependent()
    {
        // Rows are batched together in one forward pass, so a row must get the same answer however
        // many other rows travel with it.
        const int outFeatures = 64, inFeatures = 96;
        var weight = Random(outFeatures * inFeatures, 25);
        var input = Random(6 * inFeatures, 26);
        var packed = new PackedMatrix(weight, outFeatures, inFeatures);

        var batched = new float[6 * outFeatures];
        packed.Multiply(input, 6, [], batched);

        for (int row = 0; row < 6; ++row)
        {
            var single = new float[outFeatures];
            packed.Multiply(input.AsSpan(row * inFeatures, inFeatures), 1, [], single);
            for (int o = 0; o < outFeatures; ++o) Assert.Equal(single[o], batched[row * outFeatures + o], 4);
        }
    }

    [Fact]
    public void RopeRotatesByHalvesAndPreservesLength()
    {
        const int headDim = 8;
        var rope = new RopeCache(headDim, 10000d);
        var vector = Random(headDim, 16);
        double before = vector.Sum(v => (double)v * v);

        var rotated = vector.ToArray();
        rope.Apply(rotated, position: 5);

        Assert.Equal(before, rotated.Sum(v => (double)v * v), 3);

        // Position 0 is the identity rotation: cos(0) = 1, sin(0) = 0.
        var atZero = vector.ToArray();
        rope.Apply(atZero, position: 0);
        Assert.Equal(vector, atZero);
    }

    [Fact]
    public void RopeGrowsBeyondItsInitialCache()
    {
        var rope = new RopeCache(4, 10000d);
        var vector = new[] { 1f, 0f, 0f, 0f };
        rope.Apply(vector, position: 5000);     // past the pre-computed 512 positions
        Assert.Equal(1.0, vector.Sum(v => (double)v * v), 3);
    }
}

using StructuredDecisions.Numerics;
using Xunit;

namespace StructuredDecisions.Tests;

public class DeltaRuleTests
{
    /// <summary>The fused one-sweep kernel against the two-sweep form it replaces: bit for bit.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(37)]
    public unsafe void FusedSweepIsBitIdenticalToTwoSweeps(int rows)
    {
        if (!SimdOps.UseVector512) return;      // the kernel is AVX-512 only; the model falls back to the two-sweep form
        const int dk = 128, dv = 128, stride = 3 * 128 + 5;
        var random = new Random(rows);
        float Next() => (float)(random.NextDouble() * 2 - 1);
        var tokens = new float[rows * stride];
        for (int i = 0; i < tokens.Length; ++i) tokens[i] = Next() * 0.2f;
        var decay = new float[rows * 3];
        var beta = new float[rows * 3];
        for (int i = 0; i < decay.Length; ++i) { decay[i] = 0.5f + 0.5f * (float)random.NextDouble(); beta[i] = (float)random.NextDouble(); }
        var initial = new float[dk * dv];
        for (int i = 0; i < initial.Length; ++i) initial[i] = Next() * 0.1f;

        // Two sweeps per token, exactly as the generic path in Qwen35Model.
        var expectedState = initial.ToArray();
        var expectedOut = new float[rows * dv];
        var memory = new float[dv];
        var delta = new float[dv];
        for (int t = 0; t < rows; ++t)
        {
            var q = tokens.AsSpan(t * stride, dk);
            var k = tokens.AsSpan(t * stride + dk, dk);
            var v = tokens.AsSpan(t * stride + 2 * dk, dv);
            float g = decay[t * 3 + 1], b = beta[t * 3 + 1];
            Array.Clear(memory);
            for (int i = 0; i < dk; ++i)
            {
                var si = expectedState.AsSpan(i * dv, dv);
                SimdOps.Scale(si, g);
                SimdOps.AddScaled(memory, si, k[i]);
            }
            for (int j = 0; j < dv; ++j) delta[j] = (v[j] - memory[j]) * b;
            var o = expectedOut.AsSpan(t * dv, dv);
            for (int i = 0; i < dk; ++i)
            {
                var si = expectedState.AsSpan(i * dv, dv);
                SimdOps.AddScaled(si, delta, k[i]);
                SimdOps.AddScaled(o, si, q[i]);
            }
        }

        var state = initial.ToArray();
        var output = new float[rows * dv];
        fixed (float* sp = state, tp = tokens, dp = decay, bp = beta, op = output)
        {
            DecoderKernels.GatedDeltaRule128(sp, dk, rows, tp, tp + dk, tp + 2 * dk, stride, dp + 1, bp + 1, 3, op, dv);
        }
        Assert.Equal(expectedState, state);
        Assert.Equal(expectedOut, output);
    }
}

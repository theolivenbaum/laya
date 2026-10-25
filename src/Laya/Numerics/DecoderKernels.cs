using System.Numerics;

namespace Laya.Numerics;

/// <summary>
/// Elementwise and per-row kernels of the Qwen3.5 decoder: RMSNorm, the gated RMSNorm of the
/// DeltaNet output, SwiGLU, L2 normalisation and the logistic helpers. All fp32, all vectorized
/// through <see cref="SimdOps.Exp(Vector{float})"/> where an exponential is involved.
/// </summary>
public static class DecoderKernels
{
    /// <summary>Sum of squares, vectorized.</summary>
    public static float SumOfSquares(ReadOnlySpan<float> values)
    {
        int width = Vector<float>.Count;
        var acc = Vector<float>.Zero;
        int i = 0;
        for (; i <= values.Length - width; i += width)
        {
            var v = Vector.LoadUnsafe(in values[i]);
            acc = Vector.FusedMultiplyAdd(v, v, acc);
        }
        float sum = Vector.Sum(acc);
        for (; i < values.Length; ++i) sum += values[i] * values[i];
        return sum;
    }

    /// <summary>
    /// <c>x * rsqrt(mean(x²) + eps) * scale</c>. Qwen3.5's RMSNorm stores its weight zero-centred and
    /// multiplies by <c>1 + weight</c>; callers pass that sum, computed once in fp32 as transformers does.
    /// </summary>
    public static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> scale, float epsilon, Span<float> output)
    {
        int n = input.Length;
        float inv = 1f / MathF.Sqrt(SumOfSquares(input) / n + epsilon);
        int width = Vector<float>.Count;
        var invVec = new Vector<float>(inv);
        int i = 0;
        for (; i <= n - width; i += width)
        {
            (Vector.LoadUnsafe(in input[i]) * invVec * Vector.LoadUnsafe(in scale[i])).StoreUnsafe(ref output[i]);
        }
        for (; i < n; ++i) output[i] = input[i] * inv * scale[i];
    }

    /// <summary>
    /// The DeltaNet output norm, <c>weight * rmsnorm(x) * silu(gate)</c> (<c>Qwen3_5RMSNormGated</c>; its
    /// weight is a plain one-centred scale, unlike the decoder's).
    /// </summary>
    public static void RmsNormGated(ReadOnlySpan<float> input, ReadOnlySpan<float> gate, ReadOnlySpan<float> weight,
        float epsilon, Span<float> output)
    {
        int n = input.Length;
        float inv = 1f / MathF.Sqrt(SumOfSquares(input) / n + epsilon);
        int width = Vector<float>.Count;
        var invVec = new Vector<float>(inv);
        int i = 0;
        for (; i <= n - width; i += width)
        {
            var normed = Vector.LoadUnsafe(in weight[i]) * (Vector.LoadUnsafe(in input[i]) * invVec);
            (normed * Silu(Vector.LoadUnsafe(in gate[i]))).StoreUnsafe(ref output[i]);
        }
        for (; i < n; ++i) output[i] = weight[i] * (input[i] * inv) * Silu(gate[i]);
    }

    /// <summary>In place: <c>x * rsqrt(sum(x²) + eps)</c>, the FLA-style L2 norm DeltaNet applies to q and k.</summary>
    public static void L2Normalize(Span<float> values, float epsilon, float extraScale = 1f)
    {
        float inv = 1f / MathF.Sqrt(SumOfSquares(values) + epsilon);
        int width = Vector<float>.Count;
        var invVec = new Vector<float>(inv);
        int i = 0;
        for (; i <= values.Length - width; i += width)
        {
            var v = Vector.LoadUnsafe(in values[i]) * invVec;
            if (extraScale != 1f) v *= new Vector<float>(extraScale);
            v.StoreUnsafe(ref values[i]);
        }
        for (; i < values.Length; ++i)
        {
            float v = values[i] * inv;
            values[i] = extraScale != 1f ? v * extraScale : v;
        }
    }

    /// <summary>SwiGLU: <c>output = silu(gate) * up</c>.</summary>
    public static void SiluMultiply(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> output)
    {
        int width = Vector<float>.Count;
        int i = 0;
        for (; i <= gate.Length - width; i += width)
        {
            (Silu(Vector.LoadUnsafe(in gate[i])) * Vector.LoadUnsafe(in up[i])).StoreUnsafe(ref output[i]);
        }
        for (; i < gate.Length; ++i) output[i] = Silu(gate[i]) * up[i];
    }

    /// <summary>In place SiLU.</summary>
    public static void Silu(Span<float> values)
    {
        int width = Vector<float>.Count;
        int i = 0;
        for (; i <= values.Length - width; i += width)
        {
            Silu(Vector.LoadUnsafe(in values[i])).StoreUnsafe(ref values[i]);
        }
        for (; i < values.Length; ++i) values[i] = Silu(values[i]);
    }

    /// <summary>In place: <c>values *= sigmoid(gate)</c>, the full-attention output gate.</summary>
    public static void MultiplySigmoid(Span<float> values, ReadOnlySpan<float> gate)
    {
        int width = Vector<float>.Count;
        int i = 0;
        for (; i <= values.Length - width; i += width)
        {
            (Vector.LoadUnsafe(in values[i]) * Sigmoid(Vector.LoadUnsafe(in gate[i]))).StoreUnsafe(ref values[i]);
        }
        for (; i < values.Length; ++i) values[i] *= Sigmoid(gate[i]);
    }

    public static Vector<float> Sigmoid(Vector<float> x) => Vector<float>.One / (Vector<float>.One + SimdOps.Exp(-x));

    public static Vector<float> Silu(Vector<float> x) => x * Sigmoid(x);

    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    public static float Silu(float x) => x * Sigmoid(x);

    /// <summary><c>torch.nn.functional.softplus</c> with its defaults (beta 1, threshold 20).</summary>
    public static float Softplus(float x) => x > 20f ? x : (float)Math.Log(1d + Math.Exp(x));
}

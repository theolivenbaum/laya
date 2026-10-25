using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace StructuredDecisions.Numerics;

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

    /// <summary>
    /// The gated delta rule over <paramref name="rows"/> tokens for one value head with <c>dv = 128</c>, in place on
    /// the head's <c>[dk, 128]</c> state <paramref name="state"/>:
    /// <code>
    /// S ← S·g_t;  m = Sᵀk_t;  δ = (v_t − m)·β_t;  S ← S + k_t ⊗ δ;  o_t = Sᵀq_t
    /// </code>
    /// <para>Written naively this sweeps the 64 KB state twice per token (decay and read, then update and
    /// output), each sweep a call per state row: 28 GFLOP/s, 15% of Mica's whole forward pass. Here the
    /// update and output of token <c>t</c> and the decay and read of token <c>t + 1</c> share one sweep, and
    /// δ, o and m live in 24 zmm registers for the whole sweep: one load and one store of each state row per
    /// token. Every value is computed by the same operations in the same order as the two-sweep form, so the
    /// result is bit-identical to it.</para>
    /// </summary>
    /// <param name="q">Token 0's query; token <c>t</c>'s is at <c>q + t * stride</c>. Likewise <paramref name="k"/> and <paramref name="v"/>.</param>
    /// <param name="decay">Token 0's <c>exp(g)</c>; token <c>t</c>'s is at <c>decay[t * gateStride]</c>. Likewise <paramref name="beta"/>.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void GatedDeltaRule128(float* state, int dk, int rows, float* q, float* k, float* v, int stride,
        float* decay, float* beta, int gateStride, float* output, int outputStride)
    {
        if (rows <= 0) return;

        // Token 0's decay and read: S ← S·g_0, m = Sᵀk_0.
        var m0 = Vector512<float>.Zero;
        var m1 = Vector512<float>.Zero;
        var m2 = Vector512<float>.Zero;
        var m3 = Vector512<float>.Zero;
        var m4 = Vector512<float>.Zero;
        var m5 = Vector512<float>.Zero;
        var m6 = Vector512<float>.Zero;
        var m7 = Vector512<float>.Zero;
        {
            var g = Vector512.Create(decay[0]);
            float* s = state;
            for (int i = 0; i < dk; ++i, s += 128)
            {
                var ki = Vector512.Create(k[i]);
                Vector512<float> x;
                x = Vector512.Load(s + 0) * g; x.Store(s + 0); m0 = Vector512.FusedMultiplyAdd(x, ki, m0);
                x = Vector512.Load(s + 16) * g; x.Store(s + 16); m1 = Vector512.FusedMultiplyAdd(x, ki, m1);
                x = Vector512.Load(s + 32) * g; x.Store(s + 32); m2 = Vector512.FusedMultiplyAdd(x, ki, m2);
                x = Vector512.Load(s + 48) * g; x.Store(s + 48); m3 = Vector512.FusedMultiplyAdd(x, ki, m3);
                x = Vector512.Load(s + 64) * g; x.Store(s + 64); m4 = Vector512.FusedMultiplyAdd(x, ki, m4);
                x = Vector512.Load(s + 80) * g; x.Store(s + 80); m5 = Vector512.FusedMultiplyAdd(x, ki, m5);
                x = Vector512.Load(s + 96) * g; x.Store(s + 96); m6 = Vector512.FusedMultiplyAdd(x, ki, m6);
                x = Vector512.Load(s + 112) * g; x.Store(s + 112); m7 = Vector512.FusedMultiplyAdd(x, ki, m7);
            }
        }

        for (int t = 0; t < rows; ++t)
        {
            float* vt = v + (long)t * stride, kt = k + (long)t * stride, qt = q + (long)t * stride;
            var b = Vector512.Create(beta[(long)t * gateStride]);
            var d0 = (Vector512.Load(vt + 0) - m0) * b;
            var d1 = (Vector512.Load(vt + 16) - m1) * b;
            var d2 = (Vector512.Load(vt + 32) - m2) * b;
            var d3 = (Vector512.Load(vt + 48) - m3) * b;
            var d4 = (Vector512.Load(vt + 64) - m4) * b;
            var d5 = (Vector512.Load(vt + 80) - m5) * b;
            var d6 = (Vector512.Load(vt + 96) - m6) * b;
            var d7 = (Vector512.Load(vt + 112) - m7) * b;
            var o0 = Vector512<float>.Zero;
            var o1 = Vector512<float>.Zero;
            var o2 = Vector512<float>.Zero;
            var o3 = Vector512<float>.Zero;
            var o4 = Vector512<float>.Zero;
            var o5 = Vector512<float>.Zero;
            var o6 = Vector512<float>.Zero;
            var o7 = Vector512<float>.Zero;
            float* s = state;
            if (t + 1 < rows)
            {
                var g = Vector512.Create(decay[(long)(t + 1) * gateStride]);
                float* kn = k + (long)(t + 1) * stride;
                m0 = Vector512<float>.Zero;
                m1 = Vector512<float>.Zero;
                m2 = Vector512<float>.Zero;
                m3 = Vector512<float>.Zero;
                m4 = Vector512<float>.Zero;
                m5 = Vector512<float>.Zero;
                m6 = Vector512<float>.Zero;
                m7 = Vector512<float>.Zero;
                for (int i = 0; i < dk; ++i, s += 128)
                {
                    var ki = Vector512.Create(kt[i]);
                    var qi = Vector512.Create(qt[i]);
                    var kni = Vector512.Create(kn[i]);
                    Vector512<float> x;
                    x = Vector512.FusedMultiplyAdd(d0, ki, Vector512.Load(s + 0)); o0 = Vector512.FusedMultiplyAdd(x, qi, o0); x *= g; x.Store(s + 0); m0 = Vector512.FusedMultiplyAdd(x, kni, m0);
                    x = Vector512.FusedMultiplyAdd(d1, ki, Vector512.Load(s + 16)); o1 = Vector512.FusedMultiplyAdd(x, qi, o1); x *= g; x.Store(s + 16); m1 = Vector512.FusedMultiplyAdd(x, kni, m1);
                    x = Vector512.FusedMultiplyAdd(d2, ki, Vector512.Load(s + 32)); o2 = Vector512.FusedMultiplyAdd(x, qi, o2); x *= g; x.Store(s + 32); m2 = Vector512.FusedMultiplyAdd(x, kni, m2);
                    x = Vector512.FusedMultiplyAdd(d3, ki, Vector512.Load(s + 48)); o3 = Vector512.FusedMultiplyAdd(x, qi, o3); x *= g; x.Store(s + 48); m3 = Vector512.FusedMultiplyAdd(x, kni, m3);
                    x = Vector512.FusedMultiplyAdd(d4, ki, Vector512.Load(s + 64)); o4 = Vector512.FusedMultiplyAdd(x, qi, o4); x *= g; x.Store(s + 64); m4 = Vector512.FusedMultiplyAdd(x, kni, m4);
                    x = Vector512.FusedMultiplyAdd(d5, ki, Vector512.Load(s + 80)); o5 = Vector512.FusedMultiplyAdd(x, qi, o5); x *= g; x.Store(s + 80); m5 = Vector512.FusedMultiplyAdd(x, kni, m5);
                    x = Vector512.FusedMultiplyAdd(d6, ki, Vector512.Load(s + 96)); o6 = Vector512.FusedMultiplyAdd(x, qi, o6); x *= g; x.Store(s + 96); m6 = Vector512.FusedMultiplyAdd(x, kni, m6);
                    x = Vector512.FusedMultiplyAdd(d7, ki, Vector512.Load(s + 112)); o7 = Vector512.FusedMultiplyAdd(x, qi, o7); x *= g; x.Store(s + 112); m7 = Vector512.FusedMultiplyAdd(x, kni, m7);
                }
            }
            else
            {
                for (int i = 0; i < dk; ++i, s += 128)
                {
                    var ki = Vector512.Create(kt[i]);
                    var qi = Vector512.Create(qt[i]);
                    Vector512<float> x;
                    x = Vector512.FusedMultiplyAdd(d0, ki, Vector512.Load(s + 0)); o0 = Vector512.FusedMultiplyAdd(x, qi, o0); x.Store(s + 0);
                    x = Vector512.FusedMultiplyAdd(d1, ki, Vector512.Load(s + 16)); o1 = Vector512.FusedMultiplyAdd(x, qi, o1); x.Store(s + 16);
                    x = Vector512.FusedMultiplyAdd(d2, ki, Vector512.Load(s + 32)); o2 = Vector512.FusedMultiplyAdd(x, qi, o2); x.Store(s + 32);
                    x = Vector512.FusedMultiplyAdd(d3, ki, Vector512.Load(s + 48)); o3 = Vector512.FusedMultiplyAdd(x, qi, o3); x.Store(s + 48);
                    x = Vector512.FusedMultiplyAdd(d4, ki, Vector512.Load(s + 64)); o4 = Vector512.FusedMultiplyAdd(x, qi, o4); x.Store(s + 64);
                    x = Vector512.FusedMultiplyAdd(d5, ki, Vector512.Load(s + 80)); o5 = Vector512.FusedMultiplyAdd(x, qi, o5); x.Store(s + 80);
                    x = Vector512.FusedMultiplyAdd(d6, ki, Vector512.Load(s + 96)); o6 = Vector512.FusedMultiplyAdd(x, qi, o6); x.Store(s + 96);
                    x = Vector512.FusedMultiplyAdd(d7, ki, Vector512.Load(s + 112)); o7 = Vector512.FusedMultiplyAdd(x, qi, o7); x.Store(s + 112);
                }
            }
            float* ot = output + (long)t * outputStride;
            o0.Store(ot + 0);
            o1.Store(ot + 16);
            o2.Store(ot + 32);
            o3.Store(ot + 48);
            o4.Store(ot + 64);
            o5.Store(ot + 80);
            o6.Store(ot + 96);
            o7.Store(ot + 112);
        }
    }
}

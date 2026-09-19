using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Laya.Numerics;

/// <summary>
/// Span-based SIMD kernels used by the encoder and the decision head.
///
/// The shapes here follow PyTorch's: activations are row-major <c>[tokens, channels]</c> and
/// <see cref="Linear"/> weights are <c>[out, in]</c>, so one output channel's weights are
/// contiguous and every projection is a dot product over a contiguous run.
/// </summary>
public static class SimdOps
{
    /// <summary>Lane count of the portable float vector on this machine.</summary>
    public static int VectorWidth => Vector<float>.Count;

    /// <summary>
    /// True when the GEMM kernel should use explicit 512-bit vectors.
    ///
    /// <para><c>Vector&lt;T&gt;</c> stays 256 bits on AVX-512 hardware unless the whole process
    /// opts in, because on some server parts a short 512-bit loop downclocks the core and loses
    /// more than the width gains. A transformer projection is the opposite case — long, sustained,
    /// throughput-bound — so the kernel uses the explicit intrinsics whenever the hardware has
    /// them: on the Xeon this port was developed on that is worth about 20%. Set
    /// <c>LAYA_VECTOR_BITS=256</c> to opt out, or <c>512</c> to force it on.</para>
    /// </summary>
    public static bool UseVector512 { get; } = ResolveVector512();

    private static bool ResolveVector512()
    {
        string? requested = Environment.GetEnvironmentVariable("LAYA_VECTOR_BITS");
        if (requested == "512") return Avx512F.IsSupported;
        if (requested == "256") return false;
        return Avx512F.IsSupported;
    }

    /// <summary>Lane count the GEMM kernel uses.</summary>
    public static int PreferredLaneCount => UseVector512 ? 16 : Vector<float>.Count;

    /// <summary>One line describing what the kernels will actually use, for benchmarks and bug reports.</summary>
    public static string Capabilities =>
        $"lanes={PreferredLaneCount} (portable Vector<float>={Vector<float>.Count}), fma={HasFma}, " +
        $"avx512={UseVector512} (available={Avx512F.IsSupported}), cores={Environment.ProcessorCount}, " +
        $"tfm={TargetFramework}";

    private const string TargetFramework =
#if NET11_0_OR_GREATER
        "net11.0";
#else
        "net10.0";
#endif

    /// <summary>True when the hardware has fused multiply-add for the portable vector width.</summary>
    public static bool HasFma => System.Runtime.Intrinsics.X86.Fma.IsSupported || AdvSimd.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> FusedAdd(Vector<float> a, Vector<float> b, Vector<float> acc)
        => HasFma ? Vector.FusedMultiplyAdd(a, b, acc) : acc + a * b;

    /// <summary>Dot product of two equally long spans.</summary>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new ArgumentException("length mismatch", nameof(b));
        int width = Vector<float>.Count;
        var s0 = Vector<float>.Zero;
        var s1 = Vector<float>.Zero;
        int i = 0;
        // Two accumulators hide the FMA latency; the loop is memory bound either way.
        for (; i <= a.Length - 2 * width; i += 2 * width)
        {
            s0 = FusedAdd(Vector.LoadUnsafe(in a[i]), Vector.LoadUnsafe(in b[i]), s0);
            s1 = FusedAdd(Vector.LoadUnsafe(in a[i + width]), Vector.LoadUnsafe(in b[i + width]), s1);
        }
        for (; i <= a.Length - width; i += width)
        {
            s0 = FusedAdd(Vector.LoadUnsafe(in a[i]), Vector.LoadUnsafe(in b[i]), s0);
        }
        float sum = Vector.Sum(s0) + Vector.Sum(s1);
        for (; i < a.Length; ++i) sum += a[i] * b[i];
        return sum;
    }

    /// <summary><c>destination += source</c>.</summary>
    public static void Add(Span<float> destination, ReadOnlySpan<float> source)
    {
        int width = Vector<float>.Count;
        int i = 0;
        for (; i <= destination.Length - width; i += width)
        {
            (Vector.LoadUnsafe(in destination[i]) + Vector.LoadUnsafe(in source[i])).StoreUnsafe(ref destination[i]);
        }
        for (; i < destination.Length; ++i) destination[i] += source[i];
    }

    /// <summary>
    /// Dot product over <paramref name="length"/> floats at raw addresses — the attention inner
    /// loop, where re-deriving a bounds-checked span per call costs more than the arithmetic.
    /// </summary>
    public static unsafe float Dot(float* a, float* b, int length)
    {
        int width = Vector<float>.Count;
        var s0 = Vector<float>.Zero;
        var s1 = Vector<float>.Zero;
        int i = 0;
        for (; i <= length - 2 * width; i += 2 * width)
        {
            s0 = FusedAdd(Vector.Load(a + i), Vector.Load(b + i), s0);
            s1 = FusedAdd(Vector.Load(a + i + width), Vector.Load(b + i + width), s1);
        }
        for (; i <= length - width; i += width)
        {
            s0 = FusedAdd(Vector.Load(a + i), Vector.Load(b + i), s0);
        }
        float sum = Vector.Sum(s0) + Vector.Sum(s1);
        for (; i < length; ++i) sum += a[i] * b[i];
        return sum;
    }

    /// <summary><c>destination += source * scale</c> at raw addresses.</summary>
    public static unsafe void AddScaled(float* destination, float* source, int length, float scale)
    {
        int width = Vector<float>.Count;
        var factor = new Vector<float>(scale);
        int i = 0;
        for (; i <= length - width; i += width)
        {
            FusedAdd(Vector.Load(source + i), factor, Vector.Load(destination + i)).Store(destination + i);
        }
        for (; i < length; ++i) destination[i] += source[i] * scale;
    }

    /// <summary>
    /// <c>destination += source * scale</c>.
    ///
    /// <para>This is the attention value accumulation. Written as a scalar loop it is one of the
    /// slowest things in the forward pass; the head dimension is only 64 floats, so the loop
    /// overhead dominates unless it is vectorized.</para>
    /// </summary>
    public static void AddScaled(Span<float> destination, ReadOnlySpan<float> source, float scale)
    {
        int width = Vector<float>.Count;
        var factor = new Vector<float>(scale);
        int i = 0;
        for (; i <= destination.Length - width; i += width)
        {
            var accumulated = FusedAdd(Vector.LoadUnsafe(in source[i]), factor, Vector.LoadUnsafe(in destination[i]));
            accumulated.StoreUnsafe(ref destination[i]);
        }
        for (; i < destination.Length; ++i) destination[i] += source[i] * scale;
    }

    /// <summary><c>destination *= source</c>, elementwise.</summary>
    public static void Multiply(Span<float> destination, ReadOnlySpan<float> source)
    {
        int width = Vector<float>.Count;
        int i = 0;
        for (; i <= destination.Length - width; i += width)
        {
            (Vector.LoadUnsafe(in destination[i]) * Vector.LoadUnsafe(in source[i])).StoreUnsafe(ref destination[i]);
        }
        for (; i < destination.Length; ++i) destination[i] *= source[i];
    }

    /// <summary><c>destination *= scale</c>.</summary>
    public static void Scale(Span<float> destination, float scale)
    {
        int width = Vector<float>.Count;
        var v = new Vector<float>(scale);
        int i = 0;
        for (; i <= destination.Length - width; i += width)
        {
            (Vector.LoadUnsafe(in destination[i]) * v).StoreUnsafe(ref destination[i]);
        }
        for (; i < destination.Length; ++i) destination[i] *= scale;
    }

    /// <summary>Sum of a span, pairwise-free but vectorized.</summary>
    public static float Sum(ReadOnlySpan<float> values)
    {
        int width = Vector<float>.Count;
        var acc = Vector<float>.Zero;
        int i = 0;
        for (; i <= values.Length - width; i += width) acc += Vector.LoadUnsafe(in values[i]);
        float sum = Vector.Sum(acc);
        for (; i < values.Length; ++i) sum += values[i];
        return sum;
    }

    /// <summary>Largest value in a span; returns <see cref="float.NegativeInfinity"/> when empty.</summary>
    public static float Max(ReadOnlySpan<float> values)
    {
        if (values.Length == 0) return float.NegativeInfinity;
        int width = Vector<float>.Count;
        int i = 0;
        float best;
        if (values.Length >= width)
        {
            var acc = Vector.LoadUnsafe(in values[0]);
            for (i = width; i <= values.Length - width; i += width)
            {
                acc = Vector.Max(acc, Vector.LoadUnsafe(in values[i]));
            }
            best = acc[0];
            for (int lane = 1; lane < width; ++lane) best = MathF.Max(best, acc[lane]);
        }
        else
        {
            best = values[0];
            i = 1;
        }
        for (; i < values.Length; ++i) best = MathF.Max(best, values[i]);
        return best;
    }

    /// <summary>Index of the largest value; ties resolve to the lowest index, as <c>argmax</c> does.</summary>
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; ++i)
        {
            if (values[i] > values[best]) best = i;
        }
        return best;
    }

    /// <summary>
    /// LayerNorm over the last dimension, matching <c>torch.nn.LayerNorm</c>: the variance is
    /// biased (divided by n, not n-1) and epsilon is added inside the square root.
    /// </summary>
    public static void LayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias, float epsilon, Span<float> output)
    {
        int n = input.Length;
        float mean = Sum(input) / n;

        int width = Vector<float>.Count;
        var meanVec = new Vector<float>(mean);
        var varAcc = Vector<float>.Zero;
        int i = 0;
        for (; i <= n - width; i += width)
        {
            var d = Vector.LoadUnsafe(in input[i]) - meanVec;
            varAcc = FusedAdd(d, d, varAcc);
        }
        float variance = Vector.Sum(varAcc);
        for (; i < n; ++i)
        {
            float d = input[i] - mean;
            variance += d * d;
        }
        variance /= n;

        float inv = 1f / MathF.Sqrt(variance + epsilon);
        var invVec = new Vector<float>(inv);
        bool hasBias = bias.Length == n;
        i = 0;
        for (; i <= n - width; i += width)
        {
            var normalized = (Vector.LoadUnsafe(in input[i]) - meanVec) * invVec * Vector.LoadUnsafe(in weight[i]);
            if (hasBias) normalized += Vector.LoadUnsafe(in bias[i]);
            normalized.StoreUnsafe(ref output[i]);
        }
        for (; i < n; ++i)
        {
            float v = (input[i] - mean) * inv * weight[i];
            output[i] = hasBias ? v + bias[i] : v;
        }
    }

    /// <summary>In-place softmax over a span, with the usual max subtraction.</summary>
    public static void Softmax(Span<float> values)
    {
        float max = Max(values);
        if (float.IsNegativeInfinity(max))
        {
            values.Clear();
            return;
        }
        float total = 0f;
        for (int i = 0; i < values.Length; ++i)
        {
            float e = MathF.Exp(values[i] - max);
            values[i] = e;
            total += e;
        }
        Scale(values, total > 0f ? 1f / total : 0f);
    }

    private const float InvSqrt2 = 0.70710678118654752f;

    /// <summary>
    /// Exact GELU, <c>x * 0.5 * (1 + erf(x / sqrt(2)))</c>. This is what <c>nn.GELU()</c> and
    /// transformers' <c>GELUActivation</c> compute by default, so the tanh approximation is
    /// deliberately not used: it differs by up to ~1e-3 and that shows up in parity dumps.
    ///
    /// <para>The encoder evaluates this about 30 million times per call, so it is vectorized
    /// end to end — including the exponential inside <c>erf</c>. A scalar
    /// <see cref="Math.Exp(double)"/> per element made this single elementwise operation
    /// one of the most expensive stages in the whole forward pass.</para>
    /// </summary>
    public static void Gelu(Span<float> values)
    {
        int width = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var invSqrt2 = new Vector<float>(InvSqrt2);

        int i = 0;
        for (; i <= values.Length - width; i += width)
        {
            var x = Vector.LoadUnsafe(in values[i]);
            (x * half * (one + Erf(x * invSqrt2))).StoreUnsafe(ref values[i]);
        }
        for (; i < values.Length; ++i)
        {
            float x = values[i];
            values[i] = 0.5f * x * (1f + Erf(x * InvSqrt2));
        }
    }

    /// <summary>
    /// Vectorized error function, the same Numerical Recipes rational form as the scalar
    /// <see cref="Erf(float)"/> and accurate to about 1.2e-7 relative.
    /// </summary>
    public static Vector<float> Erf(Vector<float> value)
    {
        var one = Vector<float>.One;
        var sign = Vector.ConditionalSelect(Vector.LessThan(value, Vector<float>.Zero),
            new Vector<float>(-1f), one);
        var ax = Vector.Abs(value);

        var t = one / (one + new Vector<float>(0.5f) * ax);

        // Horner, outermost coefficient last — the same nesting as the scalar version.
        var poly = new Vector<float>(0.17087277f);
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(-0.82215223f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(1.48851587f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(-1.13520398f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(0.27886807f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(-0.18628806f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(0.09678418f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(0.37409196f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(1.00002368f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(-1.26551223f));
        poly -= ax * ax;

        var y = t * Exp(poly);

        // Beyond |x| ~ 6 the series underflows anyway; clamping keeps the exponential in range.
        var saturated = Vector.GreaterThan(ax, new Vector<float>(6f));
        return sign * Vector.ConditionalSelect(saturated, one, one - y);
    }

    private const float Log2E = 1.4426950408889634f;
    private const float Ln2High = 0.693359375f;
    private const float Ln2Low = -2.12194440e-4f;

    /// <summary>
    /// Vectorized <c>exp</c> for float lanes: range-reduce to <c>2^n · e^r</c> with
    /// <c>|r| &lt;= ln2/2</c>, evaluate a degree-6 Taylor series on <c>r</c>, and build
    /// <c>2^n</c> by writing the exponent field directly. Accurate to about 1 ulp over the range
    /// <c>erf</c> asks for, and roughly twenty times faster than a scalar library call per lane.
    /// </summary>
    public static Vector<float> Exp(Vector<float> value)
    {
        // exp overflows float at ~88.7 and flushes to zero at ~-87.3.
        var x = Vector.Min(Vector.Max(value, new Vector<float>(-87.3f)), new Vector<float>(88.7f));

        var n = Vector.Floor(Vector.FusedMultiplyAdd(x, new Vector<float>(Log2E), new Vector<float>(0.5f)));
        var r = Vector.FusedMultiplyAdd(n, new Vector<float>(-Ln2High), x);
        r = Vector.FusedMultiplyAdd(n, new Vector<float>(-Ln2Low), r);

        // e^r = 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720
        var p = new Vector<float>(1f / 720f);
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 120f));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 24f));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 6f));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(0.5f));
        p = Vector.FusedMultiplyAdd(p, r, Vector<float>.One);
        p = Vector.FusedMultiplyAdd(p, r, Vector<float>.One);

        // 2^n by placing n + 127 into the exponent field.
        var exponent = Vector.ConvertToInt32(n) + new Vector<int>(127);
        var scale = Vector.ShiftLeft(exponent, 23).As<int, float>();
        return p * scale;
    }

    /// <summary>Elementwise ReLU — the default activation of <c>nn.TransformerEncoderLayer</c>.</summary>
    public static void Relu(Span<float> values)
    {
        int width = Vector<float>.Count;
        var zero = Vector<float>.Zero;
        int i = 0;
        for (; i <= values.Length - width; i += width)
        {
            Vector.Max(Vector.LoadUnsafe(in values[i]), zero).StoreUnsafe(ref values[i]);
        }
        for (; i < values.Length; ++i) values[i] = MathF.Max(values[i], 0f);
    }

    /// <summary>
    /// Scalar error function. Abramowitz &amp; Stegun 7.1.26 is not accurate enough for parity work
    /// (~1.5e-7 absolute, but biased), so this is the Numerical Recipes rational form: max error
    /// below 1e-7 with no bias, computed in double. <see cref="Erf(Vector{float})"/> is the
    /// vectorized float equivalent and is what the hot paths use.
    /// </summary>
    public static float Erf(float value)
    {
        double x = value;
        double ax = Math.Abs(x);
        if (ax > 6.0) return MathF.Sign(value);

        // erf(x) = 1 - erfc(x), erfc via the Numerical Recipes rational approximation.
        double t = 1.0 / (1.0 + 0.5 * ax);
        double y = t * Math.Exp(-ax * ax - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                   t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
                   t * (-0.82215223 + t * 0.17087277)))))))));
        double erf = 1.0 - y;
        return (float)(x >= 0 ? erf : -erf);
    }

    /// <summary>
    /// <c>output = input · weightᵀ + bias</c> for a single token row, where <paramref name="weight"/>
    /// is the PyTorch <c>[out, in]</c> layout.
    /// </summary>
    public static void Linear(ReadOnlySpan<float> input, ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias, Span<float> output)
    {
        int inFeatures = input.Length;
        for (int o = 0; o < output.Length; ++o)
        {
            float v = Dot(input, weight.Slice(o * inFeatures, inFeatures));
            output[o] = bias.IsEmpty ? v : v + bias[o];
        }
    }
}

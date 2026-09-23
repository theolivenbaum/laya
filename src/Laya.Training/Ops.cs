using System.Buffers;
using System.Numerics;
using Laya.Models;
using Laya.Numerics;

namespace Laya.Training;

/// <summary>
/// The differentiable operations the decision model is built from, each as a forward and a backward
/// pass over row-major activations <c>[rows, features]</c>.
///
/// <para>Conventions: a forward writes its output; <see cref="LinearBackward"/> <em>overwrites</em> its
/// input gradient; every other backward <em>adds</em> into its input gradient, because those feed
/// residual streams whose gradient already carries the skip path. Parameter gradients always
/// accumulate, so a step can span several micro-batches.</para>
/// </summary>
internal static class Ops
{
    // ------------------------------------------------------------------------------------ linear

    /// <summary><c>y = x · Wᵀ + b</c>.</summary>
    public static void Linear(Parameter weight, Parameter? bias, float[] x, int rows, float[] y, ParallelOptions parallel)
    {
        int outFeatures = weight.Shape[0], inFeatures = weight.Shape[1];
        weight.Packed().Multiply(x.AsSpan(0, rows * inFeatures), rows, bias is null ? default : bias.Data,
            y.AsSpan(0, rows * outFeatures), parallel);
    }

    /// <summary>
    /// <c>dx = dy · W</c> (overwritten, skipped when <paramref name="dx"/> is null),
    /// <c>dW += dyᵀ · x</c> and <c>db += Σ dy</c> for whichever of the two trains.
    /// </summary>
    public static void LinearBackward(Parameter weight, Parameter? bias, float[] x, float[] dy, int rows,
        float[]? dx, ParallelOptions parallel)
    {
        int outFeatures = weight.Shape[0], inFeatures = weight.Shape[1];
        if (dx is not null)
        {
            weight.PackedTransposed().Multiply(dy.AsSpan(0, rows * outFeatures), rows, default,
                dx.AsSpan(0, rows * inFeatures), parallel);
        }

        if (weight.Trainable)
        {
            // dW[out, in] = dyᵀ[out, rows] · x[rows, in]: the activations, read [in = rows, out = in_features],
            // are the packed operand, and dyᵀ supplies one "token row" per output feature.
            float[] transposed = ArrayPool<float>.Shared.Rent(outFeatures * rows);
            float[] product = ArrayPool<float>.Shared.Rent(outFeatures * inFeatures);
            try
            {
                Transpose(dy, rows, outFeatures, transposed, parallel);
                var activations = PackedMatrix.FromInputMajor(x.AsSpan(0, rows * inFeatures), inFeatures, rows);
                activations.Multiply(transposed.AsSpan(0, outFeatures * rows), outFeatures, default,
                    product.AsSpan(0, outFeatures * inFeatures), parallel);
                AddParallel(weight.GradientBuffer(), product, outFeatures * inFeatures, parallel);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(transposed);
                ArrayPool<float>.Shared.Return(product);
            }
        }

        if (bias is { Trainable: true })
        {
            var gradient = bias.GradientBuffer();
            for (int r = 0; r < rows; ++r) SimdOps.Add(gradient, dy.AsSpan(r * outFeatures, outFeatures));
        }
    }

    public static void Transpose(float[] source, int rows, int columns, float[] destination, ParallelOptions parallel)
    {
        const int block = 32;
        int rowBlocks = (rows + block - 1) / block;
        Parallel.For(0, rowBlocks, parallel, rb =>
        {
            int r0 = rb * block, r1 = Math.Min(rows, r0 + block);
            for (int c0 = 0; c0 < columns; c0 += block)
            {
                int c1 = Math.Min(columns, c0 + block);
                for (int r = r0; r < r1; ++r)
                {
                    for (int c = c0; c < c1; ++c) destination[c * rows + r] = source[r * columns + c];
                }
            }
        });
    }

    private static void AddParallel(float[] destination, float[] source, int length, ParallelOptions parallel)
    {
        const int chunk = 1 << 16;
        int chunks = (length + chunk - 1) / chunk;
        Parallel.For(0, chunks, parallel, c =>
        {
            int start = c * chunk;
            int count = Math.Min(chunk, length - start);
            SimdOps.Add(destination.AsSpan(start, count), source.AsSpan(start, count));
        });
    }

    // --------------------------------------------------------------------------------- layer norm

    public static void LayerNorm(float[] x, int rows, int n, Parameter weight, Parameter? bias, float epsilon,
        float[] y, ParallelOptions parallel)
    {
        float[] w = weight.Data;
        float[] b = bias?.Data ?? [];
        Parallel.For(0, rows, parallel, r =>
            SimdOps.LayerNorm(x.AsSpan(r * n, n), w, b, epsilon, y.AsSpan(r * n, n)));
    }

    /// <summary>
    /// <c>dx += rstd · (g − mean(g) − x̂ · mean(g · x̂))</c> with <c>g = dy · γ</c>; <c>dγ += Σ dy · x̂</c>,
    /// <c>dβ += Σ dy</c>. The row statistics are recomputed in double, as they are cheap and the
    /// forward pass does not keep them.
    /// </summary>
    public static void LayerNormBackward(float[] x, int rows, int n, Parameter weight, Parameter? bias,
        float epsilon, float[] dy, float[] dx, ParallelOptions parallel)
    {
        var mean = new float[rows];
        var rstd = new float[rows];
        float[] gamma = weight.Data;

        Parallel.For(0, rows, parallel, r =>
        {
            var row = x.AsSpan(r * n, n);
            double sum = 0d;
            foreach (float v in row) sum += v;
            double mu = sum / n;
            double variance = 0d;
            foreach (float v in row) variance += (v - mu) * (v - mu);
            variance /= n;
            float m = (float)mu;
            float inv = (float)(1d / Math.Sqrt(variance + epsilon));
            mean[r] = m;
            rstd[r] = inv;

            var g = dy.AsSpan(r * n, n);
            var output = dx.AsSpan(r * n, n);
            double sumG = 0d, sumGx = 0d;
            for (int i = 0; i < n; ++i)
            {
                double gi = g[i] * gamma[i];
                double xhat = (row[i] - m) * inv;
                sumG += gi;
                sumGx += gi * xhat;
            }
            float meanG = (float)(sumG / n);
            float meanGx = (float)(sumGx / n);
            for (int i = 0; i < n; ++i)
            {
                float xhat = (row[i] - m) * inv;
                output[i] += inv * (g[i] * gamma[i] - meanG - xhat * meanGx);
            }
        });

        if (!weight.Trainable && bias is not { Trainable: true }) return;

        float[]? dGamma = weight.Trainable ? weight.GradientBuffer() : null;
        float[]? dBeta = bias is { Trainable: true } ? bias.GradientBuffer() : null;
        const int columnBlock = 64;
        int blocks = (n + columnBlock - 1) / columnBlock;
        Parallel.For(0, blocks, parallel, block =>
        {
            int c0 = block * columnBlock, c1 = Math.Min(n, c0 + columnBlock);
            Span<double> gammaAccumulator = stackalloc double[columnBlock];
            Span<double> betaAccumulator = stackalloc double[columnBlock];
            gammaAccumulator.Clear();
            betaAccumulator.Clear();
            for (int r = 0; r < rows; ++r)
            {
                int offset = r * n;
                for (int c = c0; c < c1; ++c)
                {
                    float g = dy[offset + c];
                    gammaAccumulator[c - c0] += g * (x[offset + c] - mean[r]) * rstd[r];
                    betaAccumulator[c - c0] += g;
                }
            }
            for (int c = c0; c < c1; ++c)
            {
                if (dGamma is not null) dGamma[c] += (float)gammaAccumulator[c - c0];
                if (dBeta is not null) dBeta[c] += (float)betaAccumulator[c - c0];
            }
        });
    }

    // -------------------------------------------------------------------------------- activations

    private const float InvSqrt2 = 0.70710678118654752f;
    private const float InvSqrt2Pi = 0.39894228040143268f;

    /// <summary><c>du = dg · (Φ(u) + u · φ(u))</c>, the derivative of exact GELU. Overwrites <paramref name="grad"/>.</summary>
    public static void GeluBackward(ReadOnlySpan<float> u, Span<float> grad)
    {
        int width = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var invSqrt2 = new Vector<float>(InvSqrt2);
        var invSqrt2Pi = new Vector<float>(InvSqrt2Pi);
        var minusHalf = new Vector<float>(-0.5f);

        int i = 0;
        for (; i <= u.Length - width; i += width)
        {
            var x = Vector.LoadUnsafe(in u[i]);
            var cdf = half * (one + SimdOps.Erf(x * invSqrt2));
            var pdf = invSqrt2Pi * SimdOps.Exp(minusHalf * x * x);
            (Vector.LoadUnsafe(in grad[i]) * (cdf + x * pdf)).StoreUnsafe(ref grad[i]);
        }
        for (; i < u.Length; ++i)
        {
            float x = u[i];
            float cdf = 0.5f * (1f + SimdOps.Erf(x * InvSqrt2));
            float pdf = InvSqrt2Pi * MathF.Exp(-0.5f * x * x);
            grad[i] *= cdf + x * pdf;
        }
    }

    // ------------------------------------------------------------------------------------ dropout

    /// <summary>
    /// A counter-based uniform in [0, 1): the same (seed, index) always draws the same number, so a
    /// checkpointed layer regenerates the exact dropout mask of its first forward pass when it is
    /// recomputed for backpropagation, and nothing has to be stored.
    /// </summary>
    public static float Uniform(ulong seed, ulong index)
    {
        ulong z = seed + (index + 1) * 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (z >> 40) * (1f / (1 << 24));
    }

    public static ulong Mix(ulong seed, ulong salt)
    {
        ulong z = seed ^ (salt * 0xD1B54A32D192ED03UL + 0x8CB92BA72F3D8DD7UL);
        z = (z ^ (z >> 33)) * 0xFF51AFD7ED558CCDUL;
        return z ^ (z >> 33);
    }

    /// <summary>Inverted dropout in place: <c>x · keep / (1 − p)</c>. Also its own backward.</summary>
    public static void Dropout(float[] values, int length, float probability, ulong seed, ParallelOptions parallel)
    {
        if (probability <= 0f) return;
        float scale = 1f / (1f - probability);
        const int chunk = 1 << 14;
        int chunks = (length + chunk - 1) / chunk;
        Parallel.For(0, chunks, parallel, c =>
        {
            int start = c * chunk, end = Math.Min(length, start + chunk);
            for (int i = start; i < end; ++i)
            {
                values[i] = Uniform(seed, (ulong)i) < probability ? 0f : values[i] * scale;
            }
        });
    }

    // ---------------------------------------------------------------------------------- attention

    /// <summary>How one attention block is shaped.</summary>
    public sealed record AttentionShape(int Hidden, int Heads, float Scale, int HalfWindow, Rotary? Rotary,
        float Dropout, ulong Seed)
    {
        public int HeadDim => Hidden / Heads;
    }

    /// <summary>
    /// Multi-head attention per segment, reading <c>qkv</c> rows laid out <c>[q | k | v]</c> with each
    /// head's slice at <c>head · headDim</c>. RoPE (if any) is applied with positions restarting at
    /// every segment; a finite <see cref="AttentionShape.HalfWindow"/> limits each query to
    /// <c>[i − w, i + w]</c>. Dropout, when set, acts on the attention probabilities.
    /// </summary>
    public static void AttentionForward(float[] qkv, IReadOnlyList<Segment> segments, AttentionShape shape,
        float[] context, ParallelOptions parallel)
        => RunUnits(segments, shape, parallel, (segment, head) => ForwardUnit(qkv, segment, head, shape, context));

    /// <summary>Gradients of <see cref="AttentionForward"/> with respect to q, k and v, written (not added) into <paramref name="dqkv"/>.</summary>
    public static void AttentionBackward(float[] qkv, IReadOnlyList<Segment> segments, AttentionShape shape,
        float[] dContext, float[] dqkv, ParallelOptions parallel)
        => RunUnits(segments, shape, parallel, (segment, head) => BackwardUnit(qkv, segment, head, shape, dContext, dqkv));

    private static void RunUnits(IReadOnlyList<Segment> segments, AttentionShape shape, ParallelOptions parallel,
        Action<Segment, int> body)
    {
        var units = new (Segment, int)[segments.Count * shape.Heads];
        int index = 0;
        foreach (var segment in segments)
        {
            for (int head = 0; head < shape.Heads; ++head) units[index++] = (segment, head);
        }
        Parallel.For(0, units.Length, parallel, u => body(units[u].Item1, units[u].Item2));
    }

    /// <summary>Gathers one head of one segment into contiguous <c>[L, D]</c> query, key and value arrays.</summary>
    private static void Gather(float[] qkv, Segment segment, int head, AttentionShape shape,
        Span<float> q, Span<float> k, Span<float> v)
    {
        int hidden = shape.Hidden, d = shape.HeadDim, stride = 3 * hidden;
        for (int t = 0; t < segment.Length; ++t)
        {
            int row = (segment.Start + t) * stride + head * d;
            qkv.AsSpan(row, d).CopyTo(q.Slice(t * d, d));
            qkv.AsSpan(row + hidden, d).CopyTo(k.Slice(t * d, d));
            qkv.AsSpan(row + 2 * hidden, d).CopyTo(v.Slice(t * d, d));
            if (shape.Rotary is not null)
            {
                shape.Rotary.Apply(q.Slice(t * d, d), t, inverse: false);
                shape.Rotary.Apply(k.Slice(t * d, d), t, inverse: false);
            }
        }
    }

    private static (int First, int Last) Window(int t, int length, int halfWindow)
        => halfWindow == int.MaxValue ? (0, length - 1) : (Math.Max(0, t - halfWindow), Math.Min(length - 1, t + halfWindow));

    private static ulong DropoutIndex(int head, int row, int key) => ((ulong)head << 42) | ((ulong)row << 21) | (uint)key;

    /// <summary>Softmax probabilities of query <paramref name="t"/> over its window, into <paramref name="p"/>.</summary>
    private static void Probabilities(ReadOnlySpan<float> q, ReadOnlySpan<float> k, int t, int first, int count,
        int d, float scale, Span<float> p)
    {
        var query = q.Slice(t * d, d);
        for (int j = 0; j < count; ++j) p[j] = scale * SimdOps.Dot(query, k.Slice((first + j) * d, d));
        SimdOps.Softmax(p[..count]);
    }

    private static void ForwardUnit(float[] qkv, Segment segment, int head, AttentionShape shape, float[] context)
    {
        int length = segment.Length, d = shape.HeadDim, hidden = shape.Hidden;
        float[] rented = ArrayPool<float>.Shared.Rent(3 * length * d + length);
        try
        {
            var q = rented.AsSpan(0, length * d);
            var k = rented.AsSpan(length * d, length * d);
            var v = rented.AsSpan(2 * length * d, length * d);
            var p = rented.AsSpan(3 * length * d, length);
            Gather(qkv, segment, head, shape, q, k, v);
            float keep = 1f / (1f - shape.Dropout);

            for (int t = 0; t < length; ++t)
            {
                var (first, last) = Window(t, length, shape.HalfWindow);
                int count = last - first + 1;
                Probabilities(q, k, t, first, count, d, shape.Scale, p);

                var output = context.AsSpan((segment.Start + t) * hidden + head * d, d);
                output.Clear();
                for (int j = 0; j < count; ++j)
                {
                    float weight = p[j];
                    if (shape.Dropout > 0f)
                    {
                        weight = Uniform(shape.Seed, DropoutIndex(head, segment.Start + t, first + j)) < shape.Dropout
                            ? 0f : weight * keep;
                    }
                    if (weight != 0f) SimdOps.AddScaled(output, v.Slice((first + j) * d, d), weight);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    private static void BackwardUnit(float[] qkv, Segment segment, int head, AttentionShape shape,
        float[] dContext, float[] dqkv)
    {
        int length = segment.Length, d = shape.HeadDim, hidden = shape.Hidden, stride = 3 * hidden;
        float[] rented = ArrayPool<float>.Shared.Rent(6 * length * d + 3 * length);
        try
        {
            var q = rented.AsSpan(0, length * d);
            var k = rented.AsSpan(length * d, length * d);
            var v = rented.AsSpan(2 * length * d, length * d);
            var dq = rented.AsSpan(3 * length * d, length * d);
            var dk = rented.AsSpan(4 * length * d, length * d);
            var dv = rented.AsSpan(5 * length * d, length * d);
            var p = rented.AsSpan(6 * length * d, length);
            var dp = rented.AsSpan(6 * length * d + length, length);
            var mask = rented.AsSpan(6 * length * d + 2 * length, length);
            dq.Clear();
            dk.Clear();
            dv.Clear();
            Gather(qkv, segment, head, shape, q, k, v);
            float keep = 1f / (1f - shape.Dropout);

            for (int t = 0; t < length; ++t)
            {
                var (first, last) = Window(t, length, shape.HalfWindow);
                int count = last - first + 1;
                Probabilities(q, k, t, first, count, d, shape.Scale, p);
                var upstream = dContext.AsSpan((segment.Start + t) * hidden + head * d, d);

                // With dropout the context used P' = P ⊙ mask; dV sees P', and dP = dP' ⊙ mask.
                for (int j = 0; j < count; ++j)
                {
                    mask[j] = shape.Dropout > 0f
                        ? (Uniform(shape.Seed, DropoutIndex(head, segment.Start + t, first + j)) < shape.Dropout ? 0f : keep)
                        : 1f;
                    float dropped = p[j] * mask[j];
                    if (dropped != 0f) SimdOps.AddScaled(dv.Slice((first + j) * d, d), upstream, dropped);
                    dp[j] = SimdOps.Dot(upstream, v.Slice((first + j) * d, d)) * mask[j];
                }

                float c = 0f;
                for (int j = 0; j < count; ++j) c += p[j] * dp[j];
                var dQuery = dq.Slice(t * d, d);
                var query = q.Slice(t * d, d);
                for (int j = 0; j < count; ++j)
                {
                    float ds = shape.Scale * p[j] * (dp[j] - c);
                    if (ds == 0f) continue;
                    SimdOps.AddScaled(dQuery, k.Slice((first + j) * d, d), ds);
                    SimdOps.AddScaled(dk.Slice((first + j) * d, d), query, ds);
                }
            }

            for (int t = 0; t < length; ++t)
            {
                if (shape.Rotary is not null)
                {
                    shape.Rotary.Apply(dq.Slice(t * d, d), t, inverse: true);
                    shape.Rotary.Apply(dk.Slice(t * d, d), t, inverse: true);
                }
                int row = (segment.Start + t) * stride + head * d;
                dq.Slice(t * d, d).CopyTo(dqkv.AsSpan(row, d));
                dk.Slice(t * d, d).CopyTo(dqkv.AsSpan(row + hidden, d));
                dv.Slice(t * d, d).CopyTo(dqkv.AsSpan(row + 2 * hidden, d));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }
}

/// <summary>
/// Rotary position embeddings, "rotate half" convention, with the same table construction as the
/// inference <see cref="RopeCache"/> so the two agree bit for bit. <c>inverse</c> rotates by the
/// negative angle, which is the transpose of the rotation — what backpropagation needs.
/// </summary>
internal sealed class Rotary
{
    private readonly int _headDim;
    private readonly double _theta;
    private float[] _cos = [];
    private float[] _sin = [];
    private int _positions;
    private readonly Lock _gate = new();

    public Rotary(int headDim, double theta)
    {
        _headDim = headDim;
        _theta = theta;
        Grow(1024);
    }

    private void Grow(int positions)
    {
        lock (_gate)
        {
            if (positions <= _positions) return;
            int half = _headDim / 2;
            var cos = new float[positions * half];
            var sin = new float[positions * half];
            for (int i = 0; i < half; ++i)
            {
                double inverseFrequency = 1.0 / Math.Pow(_theta, 2.0 * i / _headDim);
                for (int p = 0; p < positions; ++p)
                {
                    double angle = p * inverseFrequency;
                    cos[p * half + i] = (float)Math.Cos(angle);
                    sin[p * half + i] = (float)Math.Sin(angle);
                }
            }
            _cos = cos;
            _sin = sin;
            _positions = positions;
        }
    }

    /// <summary>Grows the tables up front, so parallel callers never race a resize.</summary>
    public void Ensure(int positions) => Grow(positions);

    public void Apply(Span<float> vector, int position, bool inverse)
    {
        if (position >= _positions) Grow(Math.Max(position + 1, _positions * 2));
        int half = _headDim / 2;
        var cos = _cos.AsSpan(position * half, half);
        var sin = _sin.AsSpan(position * half, half);
        for (int i = 0; i < half; ++i)
        {
            float a = vector[i];
            float b = vector[i + half];
            float s = inverse ? -sin[i] : sin[i];
            vector[i] = a * cos[i] - b * s;
            vector[i + half] = b * cos[i] + a * s;
        }
    }
}

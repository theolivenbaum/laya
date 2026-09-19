using System.Numerics;
using System.Runtime.CompilerServices;

namespace Laya.Numerics;

/// <summary>
/// The two inner loops of attention, written so that neither of them ends in a horizontal
/// reduction.
///
/// <para>The obvious formulation — a dot product per query-key pair, then an accumulate per
/// value row — spends most of its time reducing vectors to scalars. A forward pass does about
/// 14 million of those pairs, and <c>Vector.Sum</c> is a chain of shuffles and adds that the
/// FMA units sit idle through.</para>
///
/// <para>Both kernels here put the thing being summed over in the <em>lanes</em> instead:
/// scores keep adjacent keys in lanes, so a score vector is finished by a plain store, and the
/// value accumulation keeps the head dimension in lanes and holds the running total in
/// registers across the whole key loop.</para>
/// </summary>
public static class AttentionKernels
{
    /// <summary>Accumulator vectors held across the reduction; 4 keeps the register file comfortable.</summary>
    private const int Tiles = 4;

    /// <summary>
    /// Scores for one query against <paramref name="count"/> keys, where the keys are stored
    /// transposed: <paramref name="keysTransposed"/> is <c>[headDim][keyStride]</c>, so all the
    /// keys for one dimension are contiguous and adjacent keys land in adjacent lanes.
    /// </summary>
    public static unsafe void Scores(float* query, float* keysTransposed, int keyStride, int headDim,
        int count, float scale, float* scores)
    {
        int width = Vector<float>.Count;
        int block = Tiles * width;
        var scaleVector = new Vector<float>(scale);

        int key = 0;
        for (; key + block <= count; key += block)
        {
            var s0 = Vector<float>.Zero;
            var s1 = Vector<float>.Zero;
            var s2 = Vector<float>.Zero;
            var s3 = Vector<float>.Zero;
            float* k = keysTransposed + key;
            for (int d = 0; d < headDim; ++d, k += keyStride)
            {
                var q = new Vector<float>(query[d]);
                s0 = Vector.FusedMultiplyAdd(q, Vector.Load(k), s0);
                s1 = Vector.FusedMultiplyAdd(q, Vector.Load(k + width), s1);
                s2 = Vector.FusedMultiplyAdd(q, Vector.Load(k + 2 * width), s2);
                s3 = Vector.FusedMultiplyAdd(q, Vector.Load(k + 3 * width), s3);
            }
            (s0 * scaleVector).Store(scores + key);
            (s1 * scaleVector).Store(scores + key + width);
            (s2 * scaleVector).Store(scores + key + 2 * width);
            (s3 * scaleVector).Store(scores + key + 3 * width);
        }

        // The transposed keys are padded to a whole block, so the tail can still run full width
        // and simply write scores the caller will not look at.
        for (; key < count; key += width)
        {
            var s0 = Vector<float>.Zero;
            float* k = keysTransposed + key;
            for (int d = 0; d < headDim; ++d, k += keyStride)
            {
                s0 = Vector.FusedMultiplyAdd(new Vector<float>(query[d]), Vector.Load(k), s0);
            }
            (s0 * scaleVector).Store(scores + key);
        }
    }

    /// <summary>
    /// <c>destination[0..headDim) = Σ weights[j] · values[j][0..headDim)</c>, with the values
    /// stored contiguously as <c>[count][headDim]</c>.
    ///
    /// <para>The running total stays in registers for the whole key loop, so the output is
    /// touched once rather than once per key.</para>
    /// </summary>
    public static unsafe void WeightedSum(float* values, float* weights, int count, int headDim,
        float* destination)
    {
        int width = Vector<float>.Count;
        int block = Tiles * width;

        int dim = 0;
        for (; dim + block <= headDim; dim += block)
        {
            var a0 = Vector<float>.Zero;
            var a1 = Vector<float>.Zero;
            var a2 = Vector<float>.Zero;
            var a3 = Vector<float>.Zero;
            float* v = values + dim;
            for (int j = 0; j < count; ++j, v += headDim)
            {
                var w = new Vector<float>(weights[j]);
                a0 = Vector.FusedMultiplyAdd(w, Vector.Load(v), a0);
                a1 = Vector.FusedMultiplyAdd(w, Vector.Load(v + width), a1);
                a2 = Vector.FusedMultiplyAdd(w, Vector.Load(v + 2 * width), a2);
                a3 = Vector.FusedMultiplyAdd(w, Vector.Load(v + 3 * width), a3);
            }
            a0.Store(destination + dim);
            a1.Store(destination + dim + width);
            a2.Store(destination + dim + 2 * width);
            a3.Store(destination + dim + 3 * width);
        }

        for (; dim + width <= headDim; dim += width)
        {
            var a0 = Vector<float>.Zero;
            float* v = values + dim;
            for (int j = 0; j < count; ++j, v += headDim)
            {
                a0 = Vector.FusedMultiplyAdd(new Vector<float>(weights[j]), Vector.Load(v), a0);
            }
            a0.Store(destination + dim);
        }

        for (; dim < headDim; ++dim)
        {
            float sum = 0f;
            for (int j = 0; j < count; ++j) sum += weights[j] * values[j * headDim + dim];
            destination[dim] = sum;
        }
    }

    /// <summary>Key rows are padded to a whole accumulator block so the score kernel never masks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int PaddedKeyStride(int keys)
    {
        int block = Tiles * Vector<float>.Count;
        return (keys + block - 1) / block * block;
    }
}

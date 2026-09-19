using System.Numerics;
using System.Runtime.CompilerServices;

namespace Laya.Numerics;

/// <summary>
/// The matrix multiply every projection in the model goes through:
/// <c>C[M,N] = A[M,K] · B[N,K]ᵀ (+ bias[N])</c>.
///
/// <para>B is the PyTorch <c>nn.Linear</c> weight layout <c>[out_features, in_features]</c>, so the
/// reduction runs over contiguous memory in both operands. The kernel blocks 4 rows of A against
/// 4 rows of B, which keeps 16 FMA accumulators live off 8 vector loads — enough arithmetic
/// intensity that the inner loop is compute rather than load bound on AVX2 and AVX-512 alike.</para>
/// </summary>
public static class Gemm
{
    private const int RowBlock = 4;
    private const int ColBlock = 4;

    /// <summary>Rows below this run single-threaded; the work does not pay for the fork.</summary>
    private const int ParallelRowThreshold = 2;

    public static void MatMul(ReadOnlySpan<float> a, int m, int k, ReadOnlySpan<float> b, int n,
        ReadOnlySpan<float> bias, Span<float> c)
    {
        if (a.Length < (long)m * k) throw new ArgumentException("A is too small", nameof(a));
        if (b.Length < (long)n * k) throw new ArgumentException("B is too small", nameof(b));
        if (c.Length < (long)m * n) throw new ArgumentException("C is too small", nameof(c));

        int workers = Environment.ProcessorCount;
        if (m >= ParallelRowThreshold && workers > 1 && (long)m * n * k > 1_000_000)
        {
            RunParallel(a, m, k, b, n, bias, c, workers);
            return;
        }

        for (int row = 0; row < m; row += RowBlock)
        {
            Block(a, m, k, b, n, bias, c, row, Math.Min(RowBlock, m - row));
        }
    }

    private static unsafe void RunParallel(ReadOnlySpan<float> a, int m, int k, ReadOnlySpan<float> b, int n,
        ReadOnlySpan<float> bias, Span<float> c, int workers)
    {
        // Row blocks are independent and write disjoint, contiguous ranges of C, so the only
        // thing crossing threads is the shared read-only weight matrix.
        fixed (float* pa = a, pb = b, pc = c, pbias = bias)
        {
            nint aPtr = (nint)pa, bPtr = (nint)pb, cPtr = (nint)pc, biasPtr = (nint)pbias;
            int biasLength = bias.Length;
            int blocks = (m + RowBlock - 1) / RowBlock;
            int chunk = Math.Max(1, blocks / (workers * 4));
            int chunks = (blocks + chunk - 1) / chunk;
            Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = workers }, index =>
            {
                var av = new ReadOnlySpan<float>((void*)aPtr, m * k);
                var bv = new ReadOnlySpan<float>((void*)bPtr, n * k);
                var cv = new Span<float>((void*)cPtr, m * n);
                var biasv = biasLength == 0 ? ReadOnlySpan<float>.Empty : new ReadOnlySpan<float>((void*)biasPtr, biasLength);
                int first = index * chunk;
                int last = Math.Min(blocks, first + chunk);
                for (int block = first; block < last; ++block)
                {
                    int row = block * RowBlock;
                    Block(av, m, k, bv, n, biasv, cv, row, Math.Min(RowBlock, m - row));
                }
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Block(ReadOnlySpan<float> a, int m, int k, ReadOnlySpan<float> b, int n,
        ReadOnlySpan<float> bias, Span<float> c, int row, int rows)
    {
        int width = Vector<float>.Count;
        bool fma = SimdOps.HasFma;

        for (int col = 0; col < n; col += ColBlock)
        {
            int cols = Math.Min(ColBlock, n - col);

            Vector<float> s00 = default, s01 = default, s02 = default, s03 = default;
            Vector<float> s10 = default, s11 = default, s12 = default, s13 = default;
            Vector<float> s20 = default, s21 = default, s22 = default, s23 = default;
            Vector<float> s30 = default, s31 = default, s32 = default, s33 = default;

            if (rows == RowBlock && cols == ColBlock)
            {
                ref readonly float a0 = ref a[row * k];
                ref readonly float a1 = ref a[(row + 1) * k];
                ref readonly float a2 = ref a[(row + 2) * k];
                ref readonly float a3 = ref a[(row + 3) * k];
                ref readonly float b0 = ref b[col * k];
                ref readonly float b1 = ref b[(col + 1) * k];
                ref readonly float b2 = ref b[(col + 2) * k];
                ref readonly float b3 = ref b[(col + 3) * k];

                int i = 0;
                for (; i <= k - width; i += width)
                {
                    var va0 = Vector.LoadUnsafe(in a0, (nuint)i);
                    var va1 = Vector.LoadUnsafe(in a1, (nuint)i);
                    var va2 = Vector.LoadUnsafe(in a2, (nuint)i);
                    var va3 = Vector.LoadUnsafe(in a3, (nuint)i);
                    var vb0 = Vector.LoadUnsafe(in b0, (nuint)i);
                    var vb1 = Vector.LoadUnsafe(in b1, (nuint)i);
                    var vb2 = Vector.LoadUnsafe(in b2, (nuint)i);
                    var vb3 = Vector.LoadUnsafe(in b3, (nuint)i);

                    if (fma)
                    {
                        s00 = Vector.FusedMultiplyAdd(va0, vb0, s00);
                        s01 = Vector.FusedMultiplyAdd(va0, vb1, s01);
                        s02 = Vector.FusedMultiplyAdd(va0, vb2, s02);
                        s03 = Vector.FusedMultiplyAdd(va0, vb3, s03);
                        s10 = Vector.FusedMultiplyAdd(va1, vb0, s10);
                        s11 = Vector.FusedMultiplyAdd(va1, vb1, s11);
                        s12 = Vector.FusedMultiplyAdd(va1, vb2, s12);
                        s13 = Vector.FusedMultiplyAdd(va1, vb3, s13);
                        s20 = Vector.FusedMultiplyAdd(va2, vb0, s20);
                        s21 = Vector.FusedMultiplyAdd(va2, vb1, s21);
                        s22 = Vector.FusedMultiplyAdd(va2, vb2, s22);
                        s23 = Vector.FusedMultiplyAdd(va2, vb3, s23);
                        s30 = Vector.FusedMultiplyAdd(va3, vb0, s30);
                        s31 = Vector.FusedMultiplyAdd(va3, vb1, s31);
                        s32 = Vector.FusedMultiplyAdd(va3, vb2, s32);
                        s33 = Vector.FusedMultiplyAdd(va3, vb3, s33);
                    }
                    else
                    {
                        s00 += va0 * vb0; s01 += va0 * vb1; s02 += va0 * vb2; s03 += va0 * vb3;
                        s10 += va1 * vb0; s11 += va1 * vb1; s12 += va1 * vb2; s13 += va1 * vb3;
                        s20 += va2 * vb0; s21 += va2 * vb1; s22 += va2 * vb2; s23 += va2 * vb3;
                        s30 += va3 * vb0; s31 += va3 * vb1; s32 += va3 * vb2; s33 += va3 * vb3;
                    }
                }

                Store(c, n, bias, row, col, 0, 0, Vector.Sum(s00), Tail(a, b, k, row, col, i));
                Store(c, n, bias, row, col, 0, 1, Vector.Sum(s01), Tail(a, b, k, row, col + 1, i));
                Store(c, n, bias, row, col, 0, 2, Vector.Sum(s02), Tail(a, b, k, row, col + 2, i));
                Store(c, n, bias, row, col, 0, 3, Vector.Sum(s03), Tail(a, b, k, row, col + 3, i));
                Store(c, n, bias, row, col, 1, 0, Vector.Sum(s10), Tail(a, b, k, row + 1, col, i));
                Store(c, n, bias, row, col, 1, 1, Vector.Sum(s11), Tail(a, b, k, row + 1, col + 1, i));
                Store(c, n, bias, row, col, 1, 2, Vector.Sum(s12), Tail(a, b, k, row + 1, col + 2, i));
                Store(c, n, bias, row, col, 1, 3, Vector.Sum(s13), Tail(a, b, k, row + 1, col + 3, i));
                Store(c, n, bias, row, col, 2, 0, Vector.Sum(s20), Tail(a, b, k, row + 2, col, i));
                Store(c, n, bias, row, col, 2, 1, Vector.Sum(s21), Tail(a, b, k, row + 2, col + 1, i));
                Store(c, n, bias, row, col, 2, 2, Vector.Sum(s22), Tail(a, b, k, row + 2, col + 2, i));
                Store(c, n, bias, row, col, 2, 3, Vector.Sum(s23), Tail(a, b, k, row + 2, col + 3, i));
                Store(c, n, bias, row, col, 3, 0, Vector.Sum(s30), Tail(a, b, k, row + 3, col, i));
                Store(c, n, bias, row, col, 3, 1, Vector.Sum(s31), Tail(a, b, k, row + 3, col + 1, i));
                Store(c, n, bias, row, col, 3, 2, Vector.Sum(s32), Tail(a, b, k, row + 3, col + 2, i));
                Store(c, n, bias, row, col, 3, 3, Vector.Sum(s33), Tail(a, b, k, row + 3, col + 3, i));
            }
            else
            {
                // Ragged edge: fall back to plain dot products.
                for (int r = 0; r < rows; ++r)
                {
                    for (int q = 0; q < cols; ++q)
                    {
                        float v = SimdOps.Dot(a.Slice((row + r) * k, k), b.Slice((col + q) * k, k));
                        c[(row + r) * n + col + q] = bias.IsEmpty ? v : v + bias[col + q];
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Tail(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int k, int row, int col, int from)
    {
        float sum = 0f;
        for (int i = from; i < k; ++i) sum += a[row * k + i] * b[col * k + i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(Span<float> c, int n, ReadOnlySpan<float> bias, int row, int col,
        int dr, int dc, float value, float tail)
    {
        int index = (row + dr) * n + col + dc;
        float v = value + tail;
        c[index] = bias.IsEmpty ? v : v + bias[col + dc];
    }
}

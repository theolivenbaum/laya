using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Laya.Numerics;

/// <summary>
/// The int8 dot-accumulate behind <see cref="PackedInt8Matrix"/>: four <c>uint8 × int8</c> products
/// summed into each int32 lane, so a 64-byte vector covers sixteen output columns times four
/// reduction steps.
///
/// <para>Three implementations, picked by JIT-time constants so the dead ones are eliminated:</para>
/// <list type="number">
/// <item><c>AvxVnniInt8.V512</c> — <c>vpdpbsud</c>, one instruction for all 64 products.</item>
/// <item><c>vpmaddubsw</c> + <c>vpmaddwd</c> — three uops, the pre-VNNI sequence. The int16
/// intermediate of <c>vpmaddubsw</c> saturates, so this path is only safe with a reduced weight
/// range; see <see cref="WeightQMax"/>.</item>
/// <item>widen to int16 + <c>vpmaddwd</c> — eight uops, no range restriction, used when the weights
/// were quantized at full range but no 512-bit int8 dot exists.</item>
/// </list>
///
/// <para>.NET 10 exposes no standalone <c>Avx512Vnni</c>, so a "classic" AVX-512 server that reports
/// the <c>avx512_vnni</c> CPUID flag still lands on path 2 or 3. That is the machine this port was
/// tuned on, and it is why the int8 kernel here is a memory win rather than an arithmetic one — see
/// the note on <see cref="PackedInt8Matrix"/>.</para>
/// </summary>
internal static class Vnni
{
    /// <summary>Whether a vectorized int8 path exists at all.</summary>
    public static bool IsSupported => Avx512BW.IsSupported || Avx2.IsSupported;

    /// <summary>Activation zero point: the symmetric int8 value is stored offset by +128 so it can
    /// feed the unsigned operand every one of these instructions expects.</summary>
    public const int ZeroPoint = 128;

    /// <summary>Max magnitude of a quantized activation.</summary>
    public const int ActivationQMax = 127;

    /// <summary>Whether the width actually in use has a real int8-VNNI instruction.</summary>
    private static bool TrueVnni => SimdOps.UseVector512
        ? AvxVnniInt8.V512.IsSupported
        : AvxVnni.IsSupported || AvxVnniInt8.IsSupported;

    /// <summary>
    /// Whether the <c>vpmaddubsw</c> sequence is used instead of the widening one. It is three uops
    /// against eight, but its int16 intermediate holds <c>a0*w0 + a1*w1</c> and saturates at 32767,
    /// so it needs the weight range capped (see <see cref="WeightQMax"/>). Set
    /// <c>LAYA_INT8_KERNEL=widen</c> to measure the other one.
    /// </summary>
    public static bool UseMaddubs { get; } =
        !TrueVnni && Environment.GetEnvironmentVariable("LAYA_INT8_KERNEL") != "widen";

    /// <summary>
    /// Max magnitude of a quantized weight. 127 wherever the accumulation cannot saturate; 63 on the
    /// <c>vpmaddubsw</c> path, where the worst case is <c>255 * 63 * 2 = 32130</c>, just inside int16.
    /// Weights carry a per-output-channel scale and are far better conditioned than the activations,
    /// so spending the bit here rather than on the activations is the cheaper trade.
    /// </summary>
    public static int WeightQMax => UseMaddubs ? 63 : 127;

    /// <summary>
    /// How many input channels are held out of the int8 path and accumulated in float instead.
    ///
    /// <para>ModernBERT carries massive activations in a handful of fixed channels: at this model's
    /// <c>Wqkv</c> input, 68 channels are ever the largest in their row and eight of them account
    /// for 96.8% of the time, with channel 379 alone at 67%. Those channels are not noise — they
    /// act as attention biases, and clipping them to a few RMS destroys the model outright (86% of
    /// categorical answers changed, measured). But leaving them in the range means the per-row amax
    /// is set by one channel and the other thousand share what is left.</para>
    ///
    /// <para>So they are neither clipped nor quantized: the largest few channels are pinned to the
    /// activation zero point, where the existing offset correction cancels them exactly, and their
    /// contribution is added back in float afterwards. The remaining channels then get a scale that
    /// reflects their own range. The held-out columns cost <c>k/inFeatures</c> of the multiply —
    /// about 1.6% at sixteen channels.</para>
    ///
    /// <para>Set <c>LAYA_INT8_OUTLIERS</c> to change the count; 0 restores plain amax scaling.</para>
    /// </summary>
    public static int OutlierChannels { get; } =
        int.TryParse(Environment.GetEnvironmentVariable("LAYA_INT8_OUTLIERS"), out int configured)
            ? Math.Max(0, configured)
            : 16;

    /// <summary>
    /// Clipping the activation range to this many RMS before quantizing. <b>Measured and rejected</b>
    /// — it is the textbook answer for a heavy-tailed activation and it is wrong for this model,
    /// because the tail is load-bearing. Kept only so the result can be reproduced:
    /// <c>LAYA_INT8_CLIP=3</c> flips 86% of categorical answers, against 6% for no clipping at all.
    /// </summary>
    public static float ClipSigmas { get; } =
        float.TryParse(Environment.GetEnvironmentVariable("LAYA_INT8_CLIP"),
            System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture,
            out float sigmas) ? sigmas : 0f;

    /// <summary>512-bit: 64 uint8×int8 products into 16 int32 lanes, four per lane.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<int> DotAccumulate512(Vector512<int> acc, Vector512<byte> a, Vector512<sbyte> w)
    {
        if (AvxVnniInt8.V512.IsSupported)
        {
            return AvxVnniInt8.V512.MultiplyWideningAndAdd(acc, w, a); // vpdpbsud
        }
        if (UseMaddubs)
        {
            // vpmaddubsw pairs adjacent products into int16; vpmaddwd against ones pairs those into
            // the int32 lane, so the lane ends up holding all four products of its dword.
            var pairs = Avx512BW.MultiplyAddAdjacent(a, w);
            return acc + Avx512BW.MultiplyAddAdjacent(pairs, Vector512<short>.One);
        }
        // Deinterleave within each dword rather than splitting the vector in half: lane m must end
        // up with the four products of bytes 4m..4m+3, and widening the low and high 256-bit halves
        // would instead pair byte 2m with byte 32+2m. Masking the even bytes and shifting down the
        // odd ones keeps every product inside the dword it belongs to.
        var aShort = a.AsInt16();
        var wShort = w.AsInt16();
        var aEven = aShort & Vector512.Create((short)0x00FF);
        var aOdd = Vector512.ShiftRightLogical(aShort, 8);
        var wEven = Vector512.ShiftRightArithmetic(Vector512.ShiftLeft(wShort, 8), 8);
        var wOdd = Vector512.ShiftRightArithmetic(wShort, 8);
        acc += Avx512BW.MultiplyAddAdjacent(aEven, wEven);
        acc += Avx512BW.MultiplyAddAdjacent(aOdd, wOdd);
        return acc;
    }

    /// <summary>256-bit counterpart, for AVX2 hosts and for the tail of the 512-bit path.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> DotAccumulate256(Vector256<int> acc, Vector256<byte> a, Vector256<sbyte> w)
    {
        if (AvxVnni.IsSupported)
        {
            return AvxVnni.MultiplyWideningAndAdd(acc, a, w);          // vpdpbusd
        }
        if (AvxVnniInt8.IsSupported)
        {
            return AvxVnniInt8.MultiplyWideningAndAdd(acc, w, a);      // vpdpbsud
        }
        if (UseMaddubs)
        {
            var pairs = Avx2.MultiplyAddAdjacent(a, w);
            return Avx2.Add(acc, Avx2.MultiplyAddAdjacent(pairs, Vector256<short>.One));
        }
        // Same deinterleave as the 512-bit path; see there for why the halves cannot simply widen.
        var aShort = a.AsInt16();
        var wShort = w.AsInt16();
        var aEven = aShort & Vector256.Create((short)0x00FF);
        var aOdd = Vector256.ShiftRightLogical(aShort, 8);
        var wEven = Vector256.ShiftRightArithmetic(Vector256.ShiftLeft(wShort, 8), 8);
        var wOdd = Vector256.ShiftRightArithmetic(wShort, 8);
        acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(aEven, wEven));
        acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(aOdd, wOdd));
        return acc;
    }

    /// <summary>Names the instruction sequence actually in use, for <c>SimdOps.Capabilities</c>.</summary>
    public static string Description =>
        AvxVnniInt8.V512.IsSupported ? "vpdpbsud/512"
        : AvxVnni.IsSupported ? "vpdpbusd/256"
        : AvxVnniInt8.IsSupported ? "vpdpbsud/256"
        : UseMaddubs ? $"vpmaddubsw+vpmaddwd (weights ±{WeightQMax})"
        : "widen+vpmaddwd";
}

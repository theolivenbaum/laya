namespace Laya.Numerics;

/// <summary>Weight storage and compute precision for the projection kernels.</summary>
public enum Quantization
{
    /// <summary>Full float32 weights. The only mode the parity tests are measured against.</summary>
    None = 0,

    /// <summary>
    /// 8-bit per-output-channel symmetric weights with dynamically quantized per-row activations
    /// (W8A8). Roughly a quarter of the weight bytes; see <see cref="PackedInt8Matrix"/> for what
    /// that does and does not buy.
    /// </summary>
    Int8 = 1,
}

/// <summary>
/// A projection weight that has been repacked for the kernel that will multiply it. Both
/// implementations present the same product — <c>output = input · weightᵀ + bias</c> over token rows
/// — and differ only in the precision they keep the weights at.
/// </summary>
public interface IProjection
{
    int InFeatures { get; }
    int OutFeatures { get; }

    /// <summary>Bytes this packed copy occupies, for load-time reporting.</summary>
    long Bytes { get; }

    void Multiply(ReadOnlySpan<float> input, int rows, ReadOnlySpan<float> bias, Span<float> output);

    void Multiply(ReadOnlySpan<float> input, int rows, int inputStride,
        ReadOnlySpan<float> bias, Span<float> output, int outputStride);

    /// <summary>
    /// Repacks a PyTorch <c>[out, in]</c> weight at the requested precision. Int8 falls back to
    /// float when the host has no vectorized int8 path at all, so the mode is always selectable.
    /// </summary>
    static IProjection Create(ReadOnlySpan<float> rowMajor, int outFeatures, int inFeatures,
        Quantization quantization)
        => quantization == Quantization.Int8 && Vnni.IsSupported
            ? new PackedInt8Matrix(rowMajor, outFeatures, inFeatures)
            : new PackedMatrix(rowMajor, outFeatures, inFeatures);
}

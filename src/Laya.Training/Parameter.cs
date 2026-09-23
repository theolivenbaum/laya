using Laya.Numerics;

namespace Laya.Training;

/// <summary>
/// One tensor of the checkpoint: its values, its gradient, and whether training updates it.
///
/// <para>Values stay in the PyTorch layout (<c>[out, in]</c> for a projection), which is what the
/// optimizer and the safetensors writer want. The GEMM kernel wants panels, so a projection keeps
/// two packed copies — the weight for the forward pass and its transpose for <c>dX = dY · W</c> —
/// rebuilt lazily after every optimizer step. That is two extra copies of every trainable
/// projection, the price of running all three products of backpropagation on the tuned kernel.</para>
/// </summary>
public sealed class Parameter
{
    private PackedMatrix? _packed;
    private PackedMatrix? _packedTransposed;
    private float[]? _gradient;

    public Parameter(string name, int[] shape, float[] data, string storedDType)
    {
        Name = name;
        Shape = shape;
        Data = data;
        StoredDType = storedDType;
    }

    /// <summary>The state-dict name, e.g. <c>encoder.layers.3.attn.Wqkv.weight</c>.</summary>
    public string Name { get; }

    public int[] Shape { get; }

    public float[] Data { get; }

    /// <summary>The dtype the checkpoint stored this tensor in, and the one it is written back as.</summary>
    public string StoredDType { get; }

    /// <summary>False for a frozen tensor: no gradient is kept and the optimizer leaves it alone.</summary>
    public bool Trainable { get; set; } = true;

    /// <summary>The accumulated gradient, allocated on first use; null while nothing has been accumulated.</summary>
    public float[]? Gradient => _gradient;

    public int Length => Data.Length;

    internal float[] GradientBuffer() => _gradient ??= new float[Data.Length];

    public void ZeroGradient()
    {
        if (_gradient is not null) Array.Clear(_gradient);
    }

    /// <summary>Drops the gradient buffer entirely (for a parameter that will never train).</summary>
    internal void ReleaseGradient() => _gradient = null;

    /// <summary>The weight packed for <c>X · Wᵀ</c>.</summary>
    internal PackedMatrix Packed() => _packed ??= new PackedMatrix(Data, Shape[0], Shape[1]);

    /// <summary>The weight packed for <c>dY · W</c>: <c>W</c> read as an <c>[in = out_features, out = in_features]</c> matrix.</summary>
    internal PackedMatrix PackedTransposed() => _packedTransposed ??= PackedMatrix.FromInputMajor(Data, Shape[1], Shape[0]);

    /// <summary>Called after <see cref="Data"/> changes, so the packed copies are rebuilt from it.</summary>
    public void Invalidate()
    {
        _packed = null;
        _packedTransposed = null;
    }

    public override string ToString() => $"{Name} [{string.Join(", ", Shape)}]{(Trainable ? "" : " (frozen)")}";
}

using System.Buffers;

namespace Laya.Numerics;

/// <summary>
/// The scratch a forward pass needs, rented once and handed back at the end.
///
/// <para>A pass through the encoder needs a handful of big <c>float[]</c>s — the residual stream,
/// a normalized copy, packed QKV, the MLP's doubled intermediate — and every layer reuses the same
/// ones. Allocating them per call put around 55 MiB per call on the GC, which is pure churn: the
/// buffers die immediately and are exactly the same size every time.</para>
///
/// <para>Rented arrays are <em>at least</em> the requested length and carry whatever the previous
/// tenant left behind, so every consumer works through an explicitly sized span and writes before
/// it reads.</para>
/// </summary>
public sealed class ScratchBuffers : IDisposable
{
    private readonly List<float[]> _rented = [];

    /// <summary>Rents a buffer of at least <paramref name="length"/> floats.</summary>
    public float[] Rent(int length)
    {
        float[] buffer = ArrayPool<float>.Shared.Rent(length);
        _rented.Add(buffer);
        return buffer;
    }

    /// <summary>Rents a zeroed buffer of at least <paramref name="length"/> floats.</summary>
    public float[] RentCleared(int length)
    {
        float[] buffer = Rent(length);
        Array.Clear(buffer, 0, length);
        return buffer;
    }

    public void Dispose()
    {
        foreach (float[] buffer in _rented) ArrayPool<float>.Shared.Return(buffer);
        _rented.Clear();
    }
}

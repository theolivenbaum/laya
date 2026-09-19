using Laya.Io;

namespace Laya;

/// <summary>
/// The typed-decisions Laya checkpoint (ModernBERT-large, 421M parameters, 1024 tokens), packaged for NuGet.
///
/// <code>
/// using var agent = LayaTypedDecisions.Load();
/// var result = agent.SystemOne("I was charged twice", Presets.Triage());
/// </code>
///
/// <para>Everything but the weights is embedded in this package; <c>model.safetensors</c> is
/// downloaded from <see cref="PackagedCheckpoint.DefaultWeightsBaseUrl"/> into the Laya cache
/// (<c>$LAYA_HOME</c>, <c>$HF_HOME/laya</c> or <c>~/.cache/laya</c>) the first time the
/// checkpoint is loaded, and reused from there afterwards. Set
/// <c>LAYA_MODEL_BASE_URL</c> to fetch it from a mirror instead.</para>
/// </summary>
public static class LayaTypedDecisions
{
    /// <summary>The checkpoint's name, and the directory it is cached under.</summary>
    public const string Name = "typed-decisions";

    /// <summary>This package's checkpoint: embedded files plus the weights download.</summary>
    public static PackagedCheckpoint Checkpoint { get; } = new(
        typeof(LayaTypedDecisions).Assembly, Name, PackagedCheckpoint.ResolveWeightsUrl("typed-decisions/model.safetensors"));

    /// <summary>Where the weights are fetched from, honouring <c>LAYA_MODEL_BASE_URL</c>.</summary>
    public static Uri WeightsUrl => Checkpoint.WeightsUrl;

    /// <summary>True when the weights are already in the cache, so <see cref="Load"/> is offline.</summary>
    public static bool IsDownloaded(string? cacheRoot = null) => Checkpoint.IsDownloaded(cacheRoot);

    /// <summary>
    /// Materialises the checkpoint (downloading the weights on first use) and returns its directory.
    /// </summary>
    public static string Prepare(string? cacheRoot = null, IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Checkpoint.Prepare(cacheRoot, progress, cancellationToken);

    /// <inheritdoc cref="Prepare"/>
    public static Task<string> PrepareAsync(string? cacheRoot = null, IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Checkpoint.PrepareAsync(cacheRoot, progress, cancellationToken);

    /// <summary>Prepares the checkpoint and loads an agent from it.</summary>
    public static Agent Load(string? cacheRoot = null, IProgress<DownloadProgress>? progress = null)
        => Checkpoint.Load(cacheRoot, progress);
}

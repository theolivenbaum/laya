using Laya.Io;

namespace Laya;

/// <summary>
/// The multilingual Laya checkpoint (mmBERT-base, 322M parameters, 1024 tokens, 100+ languages), packaged for NuGet.
///
/// <code>
/// using var agent = LayaMultilingual.Load();
/// var result = agent.SystemOne("Mir wurde zweimal abgebucht", Presets.Triage());
/// </code>
///
/// <para>Everything but the weights is embedded in this package; <c>model.safetensors</c> is
/// downloaded from <see cref="PackagedCheckpoint.DefaultWeightsBaseUrl"/> into the Laya cache
/// (<c>$LAYA_HOME</c>, <c>$HF_HOME/laya</c> or <c>~/.cache/laya</c>) the first time the
/// checkpoint is loaded, and reused from there afterwards. Set
/// <c>LAYA_MODEL_BASE_URL</c> to fetch it from a mirror instead.</para>
/// </summary>
public static class LayaMultilingual
{
    /// <summary>The checkpoint's name, and the directory it is cached under.</summary>
    public const string Name = "multilingual";

    /// <summary>This package's checkpoint: embedded files plus the weights download.</summary>
    public static PackagedCheckpoint Checkpoint { get; } = new(
        typeof(LayaMultilingual).Assembly, Name, PackagedCheckpoint.ResolveWeightsUrl("multilingual/model.safetensors"));

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

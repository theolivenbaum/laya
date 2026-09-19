using System.Globalization;
using System.Net;
using System.Reflection;

namespace Laya.Io;

/// <summary>
/// A checkpoint that ships inside a NuGet package: every file except the weights is an embedded
/// resource of the package assembly, and <c>model.safetensors</c> — far too large for a package —
/// is fetched once from a static host into the same cache the hub downloader uses.
///
/// <code>
/// using var agent = LayaEnglish.Load();   // from the Laya.Model.English package
/// </code>
///
/// <para>The materialised layout is exactly what <see cref="Agent.FromDirectory"/> expects, so a
/// packaged checkpoint and a hub snapshot are interchangeable:</para>
/// <code>
/// &lt;cache&gt;/packaged/&lt;name&gt;/rl_agent_config.json
///                            encoder/config.json
///                            tokenizer/tokenizer.json
///                            tokenizer/tokenizer_config.json
///                            model.safetensors        (downloaded on first use)
/// </code>
/// </summary>
public sealed class PackagedCheckpoint
{
    /// <summary>The prefix every embedded checkpoint file carries, as set by the model packages.</summary>
    public const string ResourcePrefix = "laya/checkpoint/";

    /// <summary>Where the model packages fetch their weights from unless a mirror is configured.</summary>
    public const string DefaultWeightsBaseUrl = "https://models.curiosity.ai/laya/";

    /// <summary>The environment variable that points the weights download at a mirror.</summary>
    public const string WeightsBaseUrlVariable = "LAYA_MODEL_BASE_URL";

    private readonly Assembly _assembly;

    /// <param name="assembly">The package assembly holding the embedded checkpoint files.</param>
    /// <param name="name">Checkpoint name, used as the cache directory: <c>english</c>, …</param>
    /// <param name="weightsUrl">Where <c>model.safetensors</c> is fetched from on first use.</param>
    /// <param name="weightsBytes">
    /// The published size of the weights file, or 0 when it is not known. A file already in the
    /// cache with a different size is treated as a failed download and fetched again.
    /// </param>
    public PackagedCheckpoint(Assembly assembly, string name, Uri weightsUrl, long weightsBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(weightsUrl);

        _assembly = assembly;
        Name = name;
        WeightsUrl = weightsUrl;
        WeightsBytes = weightsBytes;
    }

    /// <summary>
    /// Resolves a checkpoint-relative weights path (<c>model.safetensors</c>,
    /// <c>multilingual/model.safetensors</c>, …) against the weights host, which
    /// <c>LAYA_MODEL_BASE_URL</c> can repoint at a mirror.
    /// </summary>
    public static Uri ResolveWeightsUrl(string relativePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(relativePath);
        string root = Environment.GetEnvironmentVariable(WeightsBaseUrlVariable) is { Length: > 0 } custom
            ? custom
            : DefaultWeightsBaseUrl;
        if (!root.EndsWith('/')) root += "/";
        return new Uri(new Uri(root), relativePath);
    }

    /// <summary>Checkpoint name, e.g. <c>english</c>.</summary>
    public string Name { get; }

    /// <summary>Where the weights are fetched from on first use.</summary>
    public Uri WeightsUrl { get; }

    /// <summary>The published size of the weights, or 0 when unknown.</summary>
    public long WeightsBytes { get; }

    /// <summary>The directory this checkpoint materialises into.</summary>
    public string Directory(string? cacheRoot = null)
        => Path.Combine(cacheRoot ?? HuggingFaceDownloader.DefaultCacheRoot, "packaged", Name);

    /// <summary>
    /// Writes the embedded files out (when they are not already there) and downloads the weights
    /// (when they are not already there), then returns the checkpoint directory.
    /// </summary>
    public string Prepare(string? cacheRoot = null, IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => PrepareAsync(cacheRoot, progress, cancellationToken).GetAwaiter().GetResult();

    /// <inheritdoc cref="Prepare"/>
    public async Task<string> PrepareAsync(string? cacheRoot = null, IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        string directory = Directory(cacheRoot);
        System.IO.Directory.CreateDirectory(directory);

        using (await CacheLock.AcquireAsync(directory, cancellationToken).ConfigureAwait(false))
        {
            ExtractEmbeddedFiles(directory, progress);

            string weights = Path.Combine(directory, "model.safetensors");
            if (!IsComplete(weights))
            {
                await DownloadWeightsAsync(weights, progress, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                progress?.Report(new DownloadProgress("model.safetensors", WeightsBytes, WeightsBytes));
            }
        }
        return directory;
    }

    /// <summary>Prepares the checkpoint and loads it.</summary>
    public Agent Load(string? cacheRoot = null, IProgress<DownloadProgress>? progress = null)
        => Agent.FromDirectory(Prepare(cacheRoot, progress));

    /// <summary>True when the weights are already in the cache at their published size.</summary>
    public bool IsDownloaded(string? cacheRoot = null)
        => IsComplete(Path.Combine(Directory(cacheRoot), "model.safetensors"));

    private bool IsComplete(string weights)
    {
        if (!File.Exists(weights)) return false;
        long length = new FileInfo(weights).Length;
        return WeightsBytes > 0 ? length == WeightsBytes : length > 0;
    }

    /// <summary>
    /// Writes the embedded files (everything but the weights) into the cache and returns the
    /// checkpoint directory, without touching the network. Useful to stage a checkpoint whose
    /// weights are supplied some other way, and to check the package is intact.
    /// </summary>
    public string Extract(string? cacheRoot = null, IProgress<DownloadProgress>? progress = null)
    {
        string directory = Directory(cacheRoot);
        System.IO.Directory.CreateDirectory(directory);
        ExtractEmbeddedFiles(directory, progress);
        return directory;
    }

    private void ExtractEmbeddedFiles(string directory, IProgress<DownloadProgress>? progress)
    {
        var names = _assembly.GetManifestResourceNames()
            .Where(n => n.StartsWith(ResourcePrefix, StringComparison.Ordinal))
            .ToArray();
        if (names.Length == 0)
        {
            throw new InvalidOperationException(
                $"'{_assembly.GetName().Name}' carries no embedded checkpoint files (resources under " +
                $"'{ResourcePrefix}'). The package is built wrong.");
        }

        foreach (string name in names)
        {
            string relative = name[ResourcePrefix.Length..];
            string destination = Path.Combine(directory, relative.Replace('/', Path.DirectorySeparatorChar));
            System.IO.Directory.CreateDirectory(Path.GetDirectoryName(destination)!);

            using var source = _assembly.GetManifestResourceStream(name)
                ?? throw new InvalidOperationException($"Embedded resource '{name}' could not be opened.");
            if (File.Exists(destination) && new FileInfo(destination).Length == source.Length)
            {
                progress?.Report(new DownloadProgress(relative, source.Length, source.Length));
                continue;
            }

            // Write beside the destination and move into place: a half-written config must never
            // look like a usable one to another process that is loading the same checkpoint.
            string partial = destination + ".part";
            using (var output = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                source.CopyTo(output);
            }
            File.Move(partial, destination, overwrite: true);
            progress?.Report(new DownloadProgress(relative, source.Length, source.Length));
        }
    }

    private async Task DownloadWeightsAsync(string destination, IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var http = new HttpClient(new SocketsHttpHandler
        {
            AutomaticDecompression = DecompressionMethods.All,
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
        })
        {
            Timeout = TimeSpan.FromMinutes(60),
        };
        http.DefaultRequestHeaders.UserAgent.ParseAdd(
            "laya-dotnet/" + (typeof(PackagedCheckpoint).Assembly.GetName().Version?.ToString(3) ?? "0.0.0"));

        await HttpDownload.ToFileAsync(http, WeightsUrl.ToString(), destination, "model.safetensors",
            WeightsBytes, progress, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// A cross-process lock on one checkpoint directory, so two processes starting at the same time
    /// do not both download 800 MB into the same <c>.part</c> file.
    /// </summary>
    private sealed class CacheLock : IDisposable
    {
        private readonly FileStream _stream;

        private CacheLock(FileStream stream) => _stream = stream;

        public static async Task<CacheLock> AcquireAsync(string directory, CancellationToken cancellationToken)
        {
            string path = Path.Combine(directory, ".laya-lock");
            var delay = TimeSpan.FromMilliseconds(200);
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                try
                {
                    var stream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None,
                        bufferSize: 1, FileOptions.DeleteOnClose);
                    stream.Write(System.Text.Encoding.UTF8.GetBytes(
                        Environment.ProcessId.ToString(CultureInfo.InvariantCulture)));
                    stream.Flush();
                    return new CacheLock(stream);
                }
                catch (IOException)
                {
                    // Another process holds it; it is downloading the same files we want.
                    await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                    if (delay < TimeSpan.FromSeconds(5)) delay *= 2;
                }
                catch (UnauthorizedAccessException)
                {
                    // A read-only or otherwise unlockable cache directory: proceed unlocked rather
                    // than failing outright, since a single process is the common case.
                    return new CacheLock(null!);
                }
            }
        }

        public void Dispose() => _stream?.Dispose();
    }
}

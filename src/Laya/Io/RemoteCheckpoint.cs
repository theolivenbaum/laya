using System.Globalization;
using System.Net;

namespace Laya.Io;

/// <summary>
/// One checkpoint on the Laya model host, downloaded on demand.
///
/// <code>
/// using var agent = RemoteCheckpoint.English.Load();
/// </code>
///
/// <para>Nothing about a checkpoint ships in the package: all five files —
/// <c>rl_agent_config.json</c>, <c>encoder/config.json</c>, the two tokenizer files and the
/// 600-850 MB <c>model.safetensors</c> — are fetched from
/// <c>https://models.curiosity.ai/laya/&lt;checkpoint&gt;/&lt;revision&gt;/</c> into the same cache
/// the hub downloader uses, and reused from there afterwards. The layout it materialises is
/// exactly what <see cref="Agent.FromDirectory"/> expects, so a host download and a hub snapshot
/// are interchangeable.</para>
///
/// <para>Set <c>LAYA_MODEL_BASE_URL</c> to fetch from a mirror that follows the same layout.</para>
/// </summary>
public sealed class RemoteCheckpoint
{
    /// <summary>Where the checkpoints are fetched from unless a mirror is configured.</summary>
    public const string DefaultBaseUrl = "https://models.curiosity.ai/laya/";

    /// <summary>The environment variable that points the download at a mirror.</summary>
    public const string BaseUrlVariable = "LAYA_MODEL_BASE_URL";

    /// <summary>The five files a checkpoint is made of, as host- and cache-relative paths.</summary>
    public static readonly IReadOnlyList<string> Files =
    [
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "model.safetensors",
    ];

    /// <summary>English: ModernBERT-large, 421M parameters, 512 tokens.</summary>
    public static RemoteCheckpoint English { get; } = new("english", path: null);

    /// <summary>Multilingual: mmBERT-base, 322M parameters, 1024 tokens, 100+ languages.</summary>
    public static RemoteCheckpoint Multilingual { get; } = new("multilingual", "multilingual");

    /// <summary>Typed decisions: ModernBERT-large, 421M parameters, 1024 tokens.</summary>
    public static RemoteCheckpoint TypedDecisions { get; } = new("typed-decisions", "typed-decisions");

    /// <summary>Every checkpoint the host publishes.</summary>
    public static IReadOnlyList<RemoteCheckpoint> All { get; } = [English, Multilingual, TypedDecisions];

    private RemoteCheckpoint(string name, string? path, string revision = "main")
    {
        Name = name;
        Path = path;
        Revision = revision;
    }

    /// <summary>The checkpoint's name: <c>english</c>, <c>multilingual</c>, <c>typed-decisions</c>.</summary>
    public string Name { get; }

    /// <summary>The checkpoint's folder on the host, or null for English, which sits at the root.</summary>
    public string? Path { get; }

    /// <summary>The revision folder on the host. Only <c>main</c> is published today.</summary>
    public string Revision { get; }

    /// <summary>
    /// The one whose <see cref="Name"/> matches, for a command line or a configuration file.
    /// </summary>
    public static RemoteCheckpoint FromName(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        return All.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase))
            ?? throw new ArgumentException(
                $"Unknown checkpoint '{name}'. Known: {string.Join(", ", All.Select(c => c.Name))}.", nameof(name));
    }

    /// <summary>The directory this checkpoint's files are downloaded into.</summary>
    public string Directory(string? cacheRoot = null)
        => System.IO.Path.Combine(cacheRoot ?? HuggingFaceDownloader.DefaultCacheRoot,
            "models.curiosity.ai", Name, Revision);

    /// <summary>
    /// The checkpoint's folder on the host, honouring <c>LAYA_MODEL_BASE_URL</c>:
    /// <c>https://models.curiosity.ai/laya/multilingual/main/</c>, and
    /// <c>https://models.curiosity.ai/laya/main/</c> for English, which sits at the root.
    /// </summary>
    public Uri BaseUri
    {
        get
        {
            string root = Environment.GetEnvironmentVariable(BaseUrlVariable) is { Length: > 0 } custom
                ? custom
                : DefaultBaseUrl;
            if (!root.EndsWith('/')) root += "/";
            string prefix = Path is null ? string.Empty : Path + "/";
            return new Uri(new Uri(root), $"{prefix}{Revision}/");
        }
    }

    /// <summary>Where <paramref name="file"/> is fetched from.</summary>
    public Uri UrlFor(string file)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(file);
        return new Uri(BaseUri, file);
    }

    /// <summary>True when every file is already in the cache, so <see cref="Load"/> is offline.</summary>
    public bool IsDownloaded(string? cacheRoot = null)
    {
        string directory = Directory(cacheRoot);
        return Files.All(f => IsPresent(System.IO.Path.Combine(directory, ToLocalPath(f))));
    }

    /// <summary>
    /// Downloads whatever is missing and returns the checkpoint directory. A file already in the
    /// cache is left alone, so a second call costs nothing; an interrupted download resumes.
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

        using var _ = await CacheLock.AcquireAsync(directory, cancellationToken).ConfigureAwait(false);
        using var http = CreateClient();

        foreach (string file in Files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string destination = System.IO.Path.Combine(directory, ToLocalPath(file));
            if (IsPresent(destination))
            {
                long size = new FileInfo(destination).Length;
                progress?.Report(new DownloadProgress(file, size, size));
                continue;
            }

            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(destination)!);
            await HttpDownload.ToFileAsync(http, UrlFor(file).ToString(), destination, file,
                expectedSize: 0, progress, cancellationToken).ConfigureAwait(false);
        }
        return directory;
    }

    /// <summary>Downloads the checkpoint if it is not cached yet, then loads it.</summary>
    public Agent Load(string? cacheRoot = null, IProgress<DownloadProgress>? progress = null)
        => Agent.FromDirectory(Prepare(cacheRoot, progress));

    public override string ToString() => $"{Name} ({UrlFor("model.safetensors")})";

    private static bool IsPresent(string path) => File.Exists(path) && new FileInfo(path).Length > 0;

    private static string ToLocalPath(string file) => file.Replace('/', System.IO.Path.DirectorySeparatorChar);

    private static HttpClient CreateClient()
    {
        var http = new HttpClient(new SocketsHttpHandler
        {
            AutomaticDecompression = DecompressionMethods.All,
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
        })
        {
            // The weights are the better part of a gigabyte on whatever connection the caller has.
            Timeout = TimeSpan.FromMinutes(60),
        };
        http.DefaultRequestHeaders.UserAgent.ParseAdd(
            "laya-dotnet/" + (typeof(RemoteCheckpoint).Assembly.GetName().Version?.ToString(3) ?? "0.0.0"));
        return http;
    }

    /// <summary>
    /// A cross-process lock on one checkpoint directory, so two processes starting at the same
    /// time do not both download 800 MB into the same <c>.part</c> file.
    /// </summary>
    private sealed class CacheLock : IDisposable
    {
        private readonly FileStream? _stream;

        private CacheLock(FileStream? stream) => _stream = stream;

        public static async Task<CacheLock> AcquireAsync(string directory, CancellationToken cancellationToken)
        {
            string path = System.IO.Path.Combine(directory, ".laya-lock");
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
                    return new CacheLock(null);
                }
            }
        }

        public void Dispose() => _stream?.Dispose();
    }
}

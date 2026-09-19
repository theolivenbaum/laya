using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;

namespace Laya.Io;

/// <summary>Progress for one file being fetched from the hub.</summary>
public readonly record struct DownloadProgress(string File, long BytesRead, long TotalBytes)
{
    public double Fraction => TotalBytes > 0 ? (double)BytesRead / TotalBytes : 0d;
}

/// <summary>
/// Minimal replacement for <c>huggingface_hub.snapshot_download</c>: lists a repository, filters
/// to the checkpoint that is wanted, and downloads the files into a local cache directory with
/// resume support.
///
/// <para>Only what the port needs is implemented — public (or token-authenticated) model repos,
/// the <c>main</c> revision by default, and the "one subfolder of a bundle" filter that
/// <c>Agent(subfolder=…)</c> relies on so that loading the multilingual checkpoint does not drag
/// the other 1.2 GB along with it.</para>
/// </summary>
public sealed class HuggingFaceDownloader : IDisposable
{
    private const string DefaultEndpoint = "https://huggingface.co";

    private readonly HttpClient _http;
    private readonly string _endpoint;
    private readonly bool _ownsClient;

    public HuggingFaceDownloader(string? token = null, string? endpoint = null, HttpClient? client = null)
    {
        _endpoint = (endpoint ?? Environment.GetEnvironmentVariable("HF_ENDPOINT") ?? DefaultEndpoint).TrimEnd('/');
        _ownsClient = client is null;
        _http = client ?? new HttpClient(new SocketsHttpHandler
        {
            AutomaticDecompression = DecompressionMethods.All,
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
        })
        {
            Timeout = TimeSpan.FromMinutes(30),
        };

        string? resolved = token ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrWhiteSpace(resolved))
        {
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", resolved);
        }
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("laya-dotnet/0.3.3");
    }

    /// <summary>The default cache root: <c>$HF_HOME/laya</c>, <c>$LAYA_HOME</c> or <c>~/.cache/laya</c>.</summary>
    public static string DefaultCacheRoot
    {
        get
        {
            string? explicitRoot = Environment.GetEnvironmentVariable("LAYA_HOME");
            if (!string.IsNullOrWhiteSpace(explicitRoot)) return explicitRoot;
            string? hfHome = Environment.GetEnvironmentVariable("HF_HOME");
            if (!string.IsNullOrWhiteSpace(hfHome)) return Path.Combine(hfHome, "laya");
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, ".cache", "laya");
        }
    }

    /// <summary>Every file in the repository, as repo-relative paths with their sizes.</summary>
    public async Task<IReadOnlyList<(string Path, long Size)>> ListFilesAsync(
        string repoId, string revision = "main", CancellationToken cancellationToken = default)
    {
        string url = $"{_endpoint}/api/models/{repoId}/tree/{revision}?recursive=true";
        using var response = await _http.GetAsync(url, cancellationToken).ConfigureAwait(false);
        await EnsureSuccessAsync(response, repoId, cancellationToken).ConfigureAwait(false);

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken).ConfigureAwait(false);

        var files = new List<(string, long)>();
        foreach (var element in document.RootElement.EnumerateArray())
        {
            if (element.GetProperty("type").GetString() != "file") continue;
            string path = element.GetProperty("path").GetString()!;
            long size = element.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object
                ? lfs.GetProperty("size").GetInt64()
                : element.GetProperty("size").GetInt64();
            files.Add((path, size));
        }
        return files;
    }

    /// <summary>
    /// Downloads the files of <paramref name="repoId"/> that <paramref name="include"/> accepts and
    /// returns the local snapshot directory. Files already present with the expected size are left
    /// alone, so a second call is close to free.
    /// </summary>
    public async Task<string> SnapshotAsync(
        string repoId,
        string? cacheRoot = null,
        string revision = "main",
        Func<string, bool>? include = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        string root = Path.Combine(cacheRoot ?? DefaultCacheRoot, repoId.Replace('/', '_'), revision);
        Directory.CreateDirectory(root);

        var files = await ListFilesAsync(repoId, revision, cancellationToken).ConfigureAwait(false);
        foreach (var (path, size) in files)
        {
            if (include is not null && !include(path)) continue;

            string destination = Path.Combine(root, path.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            if (File.Exists(destination) && new FileInfo(destination).Length == size)
            {
                progress?.Report(new DownloadProgress(path, size, size));
                continue;
            }

            await DownloadFileAsync(repoId, path, destination, revision, size, progress, cancellationToken)
                .ConfigureAwait(false);
        }
        return root;
    }

    /// <summary>
    /// The file set one Laya checkpoint needs: its config, encoder config, weights and tokenizer.
    /// <paramref name="subfolder"/> is null for the English checkpoint at the repo root.
    /// </summary>
    public static Func<string, bool> CheckpointFilter(string? subfolder)
    {
        string[] wanted =
        [
            "rl_agent_config.json",
            "encoder/config.json",
            "model.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ];

        if (string.IsNullOrEmpty(subfolder))
        {
            // At the repo root every one of those names is also a prefix of a subfolder's copy,
            // so the match has to be exact or the bundle's other checkpoints come along.
            var set = new HashSet<string>(wanted, StringComparer.Ordinal);
            return path => set.Contains(path);
        }

        string prefix = subfolder.TrimEnd('/') + "/";
        var scoped = new HashSet<string>(wanted.Select(w => prefix + w), StringComparer.Ordinal);
        return path => scoped.Contains(path);
    }

    private Task DownloadFileAsync(string repoId, string path, string destination, string revision,
        long expectedSize, IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
    {
        string url = $"{_endpoint}/{repoId}/resolve/{revision}/{Uri.EscapeDataString(path).Replace("%2F", "/", StringComparison.Ordinal)}";
        return HttpDownload.ToFileAsync(_http, url, destination, path, expectedSize, progress, cancellationToken,
            authHint: " The repository may be gated or private; set HF_TOKEN to a token that can read it.");
    }

    private static Task EnsureSuccessAsync(HttpResponseMessage response, string what, CancellationToken cancellationToken)
        => HttpDownload.EnsureSuccessAsync(response, what, cancellationToken,
            authHint: " The repository may be gated or private; set HF_TOKEN to a token that can read it.");

    public void Dispose()
    {
        if (_ownsClient) _http.Dispose();
    }
}

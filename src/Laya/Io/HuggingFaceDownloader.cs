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

    private async Task DownloadFileAsync(string repoId, string path, string destination, string revision,
        long expectedSize, IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
    {
        string partial = destination + ".part";
        long resumeFrom = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        if (resumeFrom > expectedSize && expectedSize > 0)
        {
            File.Delete(partial);
            resumeFrom = 0;
        }

        string url = $"{_endpoint}/{repoId}/resolve/{revision}/{Uri.EscapeDataString(path).Replace("%2F", "/", StringComparison.Ordinal)}";
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (resumeFrom > 0) request.Headers.Range = new RangeHeaderValue(resumeFrom, null);

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);
        if (resumeFrom > 0 && response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            // The partial file is already the whole file.
            File.Move(partial, destination, overwrite: true);
            progress?.Report(new DownloadProgress(path, expectedSize, expectedSize));
            return;
        }
        await EnsureSuccessAsync(response, $"{repoId}/{path}", cancellationToken).ConfigureAwait(false);

        bool appending = resumeFrom > 0 && response.StatusCode == HttpStatusCode.PartialContent;
        if (!appending) resumeFrom = 0;

        long total = expectedSize > 0
            ? expectedSize
            : (response.Content.Headers.ContentLength ?? 0) + resumeFrom;

        await using (var input = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
        await using (var output = new FileStream(partial, appending ? FileMode.Append : FileMode.Create,
                         FileAccess.Write, FileShare.None, 1 << 20, useAsync: true))
        {
            byte[] buffer = new byte[1 << 20];
            long read = resumeFrom;
            int n;
            while ((n = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await output.WriteAsync(buffer.AsMemory(0, n), cancellationToken).ConfigureAwait(false);
                read += n;
                progress?.Report(new DownloadProgress(path, read, total));
            }
        }

        if (expectedSize > 0 && new FileInfo(partial).Length != expectedSize)
        {
            throw new IOException($"{path}: expected {expectedSize} bytes but received {new FileInfo(partial).Length}.");
        }
        File.Move(partial, destination, overwrite: true);
    }

    private static async Task EnsureSuccessAsync(HttpResponseMessage response, string what, CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode) return;
        string body = string.Empty;
        try
        {
            body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (HttpRequestException)
        {
            // The status code is the useful part; a failure to read the body must not mask it.
        }

        string hint = response.StatusCode switch
        {
            HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden =>
                " The repository may be gated or private; set HF_TOKEN to a token that can read it.",
            HttpStatusCode.NotFound => " Check the repository id and revision.",
            _ => string.Empty,
        };
        throw new HttpRequestException($"Hugging Face request for '{what}' failed: {(int)response.StatusCode} {response.ReasonPhrase}.{hint} {body}".TrimEnd());
    }

    public void Dispose()
    {
        if (_ownsClient) _http.Dispose();
    }
}

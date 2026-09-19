using System.Net;
using System.Net.Http.Headers;

namespace Laya.Io;

/// <summary>
/// One resumable file download, shared by the Hugging Face snapshot downloader and the packaged
/// checkpoints that fetch their weights from a plain HTTP host.
///
/// <para>The transfer lands in a <c>.part</c> file next to the destination and is only moved into
/// place once the whole body has arrived, so an interrupted download never looks like a finished
/// one, and a retry resumes with a <c>Range</c> request instead of starting over.</para>
/// </summary>
internal static class HttpDownload
{
    /// <summary>
    /// Fetches <paramref name="url"/> into <paramref name="destination"/>, resuming a previous
    /// partial transfer when one is present.
    /// </summary>
    /// <param name="label">The name reported through <paramref name="progress"/>.</param>
    /// <param name="expectedSize">The size the file is known to have, or 0 when it is unknown.</param>
    public static async Task ToFileAsync(
        HttpClient http,
        string url,
        string destination,
        string label,
        long expectedSize,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken,
        string? authHint = null)
    {
        string partial = destination + ".part";
        long resumeFrom = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        if (resumeFrom > expectedSize && expectedSize > 0)
        {
            File.Delete(partial);
            resumeFrom = 0;
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (resumeFrom > 0) request.Headers.Range = new RangeHeaderValue(resumeFrom, null);

        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);
        if (resumeFrom > 0 && response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            // The partial file is already the whole file.
            File.Move(partial, destination, overwrite: true);
            progress?.Report(new DownloadProgress(label, expectedSize, expectedSize));
            return;
        }
        await EnsureSuccessAsync(response, label, cancellationToken, authHint).ConfigureAwait(false);

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
                progress?.Report(new DownloadProgress(label, read, total));
            }
        }

        long received = new FileInfo(partial).Length;
        if (total > 0 && received != total)
        {
            throw new IOException($"{label}: expected {total} bytes but received {received}.");
        }
        File.Move(partial, destination, overwrite: true);
    }

    /// <summary>Turns a failed response into an exception that names what was being fetched.</summary>
    /// <param name="authHint">Appended to the message on 401/403, where the caller knows how to authenticate.</param>
    public static async Task EnsureSuccessAsync(HttpResponseMessage response, string what,
        CancellationToken cancellationToken, string? authHint = null)
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
            HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden => authHint ?? string.Empty,
            HttpStatusCode.NotFound => " Check the address; the file may have been moved.",
            _ => string.Empty,
        };
        throw new HttpRequestException(
            $"Request for '{what}' failed: {(int)response.StatusCode} {response.ReasonPhrase}.{hint} {body}".TrimEnd(),
            inner: null, response.StatusCode);
    }
}

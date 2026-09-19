using System.Collections.Concurrent;
using System.Net;
using System.Text;
using Laya.Io;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// The on-demand checkpoint downloader: the URLs it builds are the ones the model host publishes,
/// and a download lands the five files where <see cref="Agent.FromDirectory"/> looks for them.
/// </summary>
public class RemoteCheckpointTests
{
    [Fact]
    public void UrlsMatchTheModelHostLayout()
    {
        Assert.Equal("https://models.curiosity.ai/laya/main/model.safetensors",
            RemoteCheckpoint.English.UrlFor("model.safetensors").ToString());
        Assert.Equal("https://models.curiosity.ai/laya/main/encoder/config.json",
            RemoteCheckpoint.English.UrlFor("encoder/config.json").ToString());
        Assert.Equal("https://models.curiosity.ai/laya/multilingual/main/tokenizer/tokenizer.json",
            RemoteCheckpoint.Multilingual.UrlFor("tokenizer/tokenizer.json").ToString());
        Assert.Equal("https://models.curiosity.ai/laya/typed-decisions/main/rl_agent_config.json",
            RemoteCheckpoint.TypedDecisions.UrlFor("rl_agent_config.json").ToString());
    }

    [Fact]
    public void CheckpointsAreLookedUpByName()
    {
        Assert.Same(RemoteCheckpoint.English, RemoteCheckpoint.FromName("english"));
        Assert.Same(RemoteCheckpoint.TypedDecisions, RemoteCheckpoint.FromName("Typed-Decisions"));
        Assert.Throws<ArgumentException>(() => RemoteCheckpoint.FromName("french"));
    }

    [Fact]
    public void MirrorOverrideRedirectsEveryFile()
    {
        string? previous = Environment.GetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable);
        try
        {
            Environment.SetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable, "https://mirror.example/laya");
            Assert.Equal("https://mirror.example/laya/multilingual/main/model.safetensors",
                RemoteCheckpoint.Multilingual.UrlFor("model.safetensors").ToString());
        }
        finally
        {
            Environment.SetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable, previous);
        }
    }

    /// <summary>
    /// The whole path against a local stand-in for the host: every file is fetched once, into the
    /// layout the loader expects, and a second call is served from the cache.
    /// </summary>
    [Fact]
    public async Task DownloadsEveryFileOnceAndReusesTheCache()
    {
        var served = new ConcurrentDictionary<string, int>();
        using var listener = new HttpListener();
        string prefix = StartOnAFreePort(listener);
        var serving = Serve(listener, served);

        string cacheRoot = Path.Combine(Path.GetTempPath(), "laya-tests", Path.GetRandomFileName());
        string? previous = Environment.GetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable);
        try
        {
            Environment.SetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable, prefix);
            var checkpoint = RemoteCheckpoint.Multilingual;

            Assert.False(checkpoint.IsDownloaded(cacheRoot));
            string directory = await checkpoint.PrepareAsync(cacheRoot);

            Assert.Equal(Path.Combine(cacheRoot, "models.curiosity.ai", "multilingual", "main"), directory);
            foreach (string file in RemoteCheckpoint.Files)
            {
                string local = Path.Combine(directory, file.Replace('/', Path.DirectorySeparatorChar));
                Assert.True(File.Exists(local), $"{file} was not downloaded");
                Assert.Equal(Body(file), File.ReadAllText(local));
            }

            // The host's own layout, one request per file, nothing half-written left behind.
            Assert.Equal(
                ["/laya/multilingual/main/encoder/config.json",
                 "/laya/multilingual/main/model.safetensors",
                 "/laya/multilingual/main/rl_agent_config.json",
                 "/laya/multilingual/main/tokenizer/tokenizer.json",
                 "/laya/multilingual/main/tokenizer/tokenizer_config.json"],
                served.Keys.Order().ToArray());
            Assert.All(served.Values, count => Assert.Equal(1, count));
            Assert.Empty(Directory.GetFiles(directory, "*.part", SearchOption.AllDirectories));
            Assert.True(checkpoint.IsDownloaded(cacheRoot));

            // Cached: nothing is fetched twice.
            Assert.Equal(directory, await checkpoint.PrepareAsync(cacheRoot));
            Assert.All(served.Values, count => Assert.Equal(1, count));
        }
        finally
        {
            Environment.SetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable, previous);
            listener.Stop();
            await serving;
            if (Directory.Exists(cacheRoot)) Directory.Delete(cacheRoot, recursive: true);
        }
    }

    [Fact]
    public async Task AMissingFileFailsWithTheUrlInTheMessage()
    {
        using var listener = new HttpListener();
        string prefix = StartOnAFreePort(listener);
        var serving = Serve(listener, new ConcurrentDictionary<string, int>(), notFound: true);

        string cacheRoot = Path.Combine(Path.GetTempPath(), "laya-tests", Path.GetRandomFileName());
        string? previous = Environment.GetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable);
        try
        {
            Environment.SetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable, prefix);
            var failure = await Assert.ThrowsAsync<HttpRequestException>(
                () => RemoteCheckpoint.English.PrepareAsync(cacheRoot));
            Assert.Equal(HttpStatusCode.NotFound, failure.StatusCode);
            Assert.Contains("rl_agent_config.json", failure.Message, StringComparison.Ordinal);
        }
        finally
        {
            Environment.SetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable, previous);
            listener.Stop();
            await serving;
            if (Directory.Exists(cacheRoot)) Directory.Delete(cacheRoot, recursive: true);
        }
    }

    private static string Body(string file) => $"contents of {file}";

    private static Task Serve(HttpListener listener, ConcurrentDictionary<string, int> served, bool notFound = false)
        => Task.Run(async () =>
        {
            while (listener.IsListening)
            {
                HttpListenerContext context;
                try
                {
                    context = await listener.GetContextAsync();
                }
                catch (HttpListenerException)
                {
                    return;         // stopped; the test is over
                }
                catch (ObjectDisposedException)
                {
                    return;
                }

                string path = context.Request.Url!.AbsolutePath;
                served.AddOrUpdate(path, 1, (_, count) => count + 1);

                if (notFound)
                {
                    context.Response.StatusCode = (int)HttpStatusCode.NotFound;
                    context.Response.Close();
                    continue;
                }

                byte[] body = Encoding.UTF8.GetBytes(Body(path[(path.LastIndexOf("/main/", StringComparison.Ordinal) + 6)..]));
                context.Response.ContentLength64 = body.Length;
                await context.Response.OutputStream.WriteAsync(body);
                context.Response.Close();
            }
        });

    /// <summary>Binds the listener to a free loopback port and returns its base address.</summary>
    private static string StartOnAFreePort(HttpListener listener)
    {
        for (int attempt = 0; attempt < 16; ++attempt)
        {
            string prefix = $"http://127.0.0.1:{Random.Shared.Next(20000, 60000)}/";
            listener.Prefixes.Clear();
            listener.Prefixes.Add(prefix);
            try
            {
                listener.Start();
                return prefix + "laya/";
            }
            catch (HttpListenerException)
            {
                // Port taken; try another.
            }
        }
        throw new InvalidOperationException("No free loopback port for the test server.");
    }
}

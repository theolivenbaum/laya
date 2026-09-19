using System.Net;
using System.Text;
using Laya.Io;
using Laya.Tokenizers;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// The NuGet checkpoint packages. Everything but <c>model.safetensors</c> is embedded, so the
/// package's payload can be checked without the 800 MB download: extract it and load the pieces
/// the runtime loads.
/// </summary>
public class PackagedCheckpointTests
{
    public static TheoryData<PackagedCheckpoint, string> Checkpoints => new()
    {
        { LayaEnglish.Checkpoint, "english" },
        { LayaMultilingual.Checkpoint, "multilingual" },
        { LayaTypedDecisions.Checkpoint, "typed-decisions" },
    };

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void ExtractsEveryFileButTheWeights(PackagedCheckpoint checkpoint, string name)
    {
        string cacheRoot = Path.Combine(Path.GetTempPath(), "laya-tests", Path.GetRandomFileName());
        try
        {
            string directory = checkpoint.Extract(cacheRoot);

            Assert.Equal(name, checkpoint.Name);
            Assert.Equal(Path.Combine(cacheRoot, "packaged", name), directory);
            Assert.True(File.Exists(Path.Combine(directory, "rl_agent_config.json")));
            Assert.True(File.Exists(Path.Combine(directory, "encoder", "config.json")));
            Assert.True(File.Exists(Path.Combine(directory, "tokenizer", "tokenizer.json")));
            Assert.True(File.Exists(Path.Combine(directory, "tokenizer", "tokenizer_config.json")));

            // The weights are the one thing that must not be in the package.
            Assert.False(File.Exists(Path.Combine(directory, "model.safetensors")));
            Assert.False(checkpoint.IsDownloaded(cacheRoot));

            // And the embedded copies are the real thing: the runtime's own loaders read them.
            var config = Models.LayaConfig.Load(Path.Combine(directory, "rl_agent_config.json"));
            Assert.True(config.MaxLength > 0);
            var encoder = Models.ModernBertConfig.Load(Path.Combine(directory, "encoder", "config.json"));
            Assert.True(encoder.NumHiddenLayers > 0);
            var tokenizer = HuggingFaceTokenizer.FromDirectory(Path.Combine(directory, "tokenizer"));
            Assert.True(tokenizer.MaskTokenId >= 0 && tokenizer.ClsTokenId >= 0 && tokenizer.SepTokenId >= 0);

            // Extracting twice is a no-op, not a re-write.
            Assert.Equal(directory, checkpoint.Extract(cacheRoot));
        }
        finally
        {
            if (Directory.Exists(cacheRoot)) Directory.Delete(cacheRoot, recursive: true);
        }
    }

    [Fact]
    public void WeightsUrlsPointAtTheModelHost()
    {
        Assert.Equal("https://models.curiosity.ai/laya/model.safetensors", LayaEnglish.WeightsUrl.ToString());
        Assert.Equal("https://models.curiosity.ai/laya/multilingual/model.safetensors",
            LayaMultilingual.WeightsUrl.ToString());
        Assert.Equal("https://models.curiosity.ai/laya/typed-decisions/model.safetensors",
            LayaTypedDecisions.WeightsUrl.ToString());
    }

    [Fact]
    public void MirrorOverrideRedirectsTheWeightsDownload()
    {
        string? previous = Environment.GetEnvironmentVariable(PackagedCheckpoint.WeightsBaseUrlVariable);
        try
        {
            Environment.SetEnvironmentVariable(PackagedCheckpoint.WeightsBaseUrlVariable, "https://mirror.example/models");
            Assert.Equal("https://mirror.example/models/multilingual/model.safetensors",
                PackagedCheckpoint.ResolveWeightsUrl("multilingual/model.safetensors").ToString());
        }
        finally
        {
            Environment.SetEnvironmentVariable(PackagedCheckpoint.WeightsBaseUrlVariable, previous);
        }
    }

    /// <summary>
    /// The whole on-demand path, against this assembly's stand-in payload and a local server:
    /// the embedded files are written out, the weights are fetched once, and a second call is
    /// served from the cache without touching the network.
    /// </summary>
    [Fact]
    public async Task DownloadsTheWeightsOnceAndReusesThem()
    {
        byte[] weights = Encoding.UTF8.GetBytes("not really safetensors, but the bytes have to arrive intact");
        int requests = 0;

        using var listener = new HttpListener();
        string prefix = StartOnAFreePort(listener);
        var serving = Task.Run(async () =>
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
                    return;         // the listener was stopped; the test is over
                }
                catch (ObjectDisposedException)
                {
                    return;
                }

                Interlocked.Increment(ref requests);
                context.Response.ContentLength64 = weights.Length;
                await context.Response.OutputStream.WriteAsync(weights);
                context.Response.Close();
            }
        });

        string cacheRoot = Path.Combine(Path.GetTempPath(), "laya-tests", Path.GetRandomFileName());
        var checkpoint = new PackagedCheckpoint(typeof(PackagedCheckpointTests).Assembly, "fixture",
            new Uri(prefix + "model.safetensors"));
        try
        {
            string directory = await checkpoint.PrepareAsync(cacheRoot);

            Assert.Equal("{\"fixture\":\"root\"}", File.ReadAllText(Path.Combine(directory, "rl_agent_config.json")).Trim());
            Assert.Equal("{\"fixture\":\"tokenizer\"}",
                File.ReadAllText(Path.Combine(directory, "tokenizer", "tokenizer.json")).Trim());
            Assert.Equal(weights, File.ReadAllBytes(Path.Combine(directory, "model.safetensors")));
            Assert.True(checkpoint.IsDownloaded(cacheRoot));
            Assert.Equal(1, requests);

            // Cached: the second call must not fetch the weights again.
            await checkpoint.PrepareAsync(cacheRoot);
            Assert.Equal(1, requests);

            // And nothing is left behind half-written.
            Assert.Empty(Directory.GetFiles(directory, "*.part", SearchOption.AllDirectories));
        }
        finally
        {
            listener.Stop();
            await serving;
            if (Directory.Exists(cacheRoot)) Directory.Delete(cacheRoot, recursive: true);
        }
    }

    /// <summary>Binds the listener to a free loopback port and returns its prefix.</summary>
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
                return prefix;
            }
            catch (HttpListenerException)
            {
                // Port taken; try another.
            }
        }
        throw new InvalidOperationException("No free loopback port for the test server.");
    }
}

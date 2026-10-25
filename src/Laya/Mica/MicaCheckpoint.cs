using Laya.Io;

namespace Laya.Mica;

/// <summary>Where a Mica checkpoint is fetched from.</summary>
public enum MicaSource
{
    /// <summary>The upstream Hugging Face repository, <c>sky7350/Mica-v0.5-4B</c>: the full multimodal checkpoint in two shards.</summary>
    Hub,

    /// <summary>
    /// The Laya model host, <c>https://models.curiosity.ai/laya/mica/main/</c>: the text decoder only, one
    /// <c>model.safetensors</c>, as <c>laya mica-convert</c> writes it.
    /// </summary>
    Host,
}

/// <summary>
/// Downloads Mica on demand. Both sources materialise a directory <see cref="MicaJudge.FromDirectory"/>
/// loads. Nothing of the model ships in the package.
/// </summary>
public static class MicaCheckpoint
{
    public const string HubRepo = "sky7350/Mica-v0.5-4B";

    /// <summary>The revision the C# port was validated against; <c>main</c> may move.</summary>
    public const string HubRevision = "5ea7aafe324defec353494417ba723195c119b59";

    /// <summary>The checkpoint's name on the Laya host and in the cache.</summary>
    public const string HostName = "mica";

    /// <summary>What a Mica checkpoint needs from the hub: the text config, both shards and their index, the tokenizer.</summary>
    public static readonly IReadOnlyList<string> HubFiles =
    [
        "config.json",
        "model.safetensors.index.json",
        "model.safetensors-00001-of-00002.safetensors",
        "model.safetensors-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    /// <summary>The converted checkpoint's files, as <c>laya mica-convert</c> writes them and the host serves them.</summary>
    public static readonly IReadOnlyList<string> HostFiles =
    [
        "config.json",
        "calibration.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ];

    /// <summary>
    /// The hub filter. Exact names: the repo also carries 15 GB of GGUF and INT8 copies whose names share
    /// prefixes with these.
    /// </summary>
    public static Func<string, bool> HubFilter { get; } = new HashSet<string>(HubFiles, StringComparer.Ordinal).Contains;

    /// <summary><c>https://models.curiosity.ai/laya/mica/main/</c>, or the same path under <c>LAYA_MODEL_BASE_URL</c>.</summary>
    public static Uri HostUri
    {
        get
        {
            string root = Environment.GetEnvironmentVariable(RemoteCheckpoint.BaseUrlVariable) is { Length: > 0 } custom
                ? custom
                : RemoteCheckpoint.DefaultBaseUrl;
            if (!root.EndsWith('/')) root += "/";
            return new Uri(new Uri(root), $"{HostName}/main/");
        }
    }

    /// <summary>Downloads whatever is missing and returns the checkpoint directory.</summary>
    public static string Prepare(MicaSource source = MicaSource.Hub, string? cacheRoot = null,
        IProgress<DownloadProgress>? progress = null, string? token = null, CancellationToken cancellationToken = default)
        => PrepareAsync(source, cacheRoot, progress, token, cancellationToken).GetAwaiter().GetResult();

    public static async Task<string> PrepareAsync(MicaSource source = MicaSource.Hub, string? cacheRoot = null,
        IProgress<DownloadProgress>? progress = null, string? token = null, CancellationToken cancellationToken = default)
    {
        if (source == MicaSource.Hub)
        {
            using var downloader = new HuggingFaceDownloader(token);
            return await downloader.SnapshotAsync(HubRepo, cacheRoot, HubRevision, HubFilter, progress, cancellationToken)
                .ConfigureAwait(false);
        }

        string directory = Path.Combine(cacheRoot ?? HuggingFaceDownloader.DefaultCacheRoot, "models.curiosity.ai", HostName, "main");
        Directory.CreateDirectory(directory);
        using var http = new HttpClient { Timeout = TimeSpan.FromHours(2) };
        foreach (string file in HostFiles)
        {
            string destination = Path.Combine(directory, file);
            if (File.Exists(destination) && new FileInfo(destination).Length > 0)
            {
                long size = new FileInfo(destination).Length;
                progress?.Report(new DownloadProgress(file, size, size));
                continue;
            }
            await HttpDownload.ToFileAsync(http, new Uri(HostUri, file).ToString(), destination, file, 0, progress, cancellationToken)
                .ConfigureAwait(false);
        }
        return directory;
    }

    /// <summary>Downloads Mica if it is not cached, then loads it (about 8.5 GB of memory).</summary>
    public static MicaJudge Load(MicaSource source = MicaSource.Hub, string? cacheRoot = null, IProgress<DownloadProgress>? progress = null)
        => MicaJudge.FromDirectory(Prepare(source, cacheRoot, progress));
}

/// <summary>
/// Turns the hub checkpoint into the host layout: one <c>model.safetensors</c> holding only the text
/// decoder (<c>model.language_model.*</c>; the vision tower and the MTP head are dropped), byte for byte in
/// the original bf16, plus the config, tokenizer and calibration files.
/// </summary>
public static class MicaConverter
{
    /// <summary>Converts <paramref name="source"/> into <paramref name="destination"/> and returns the files written.</summary>
    public static IReadOnlyList<string> Convert(string source, string destination, Action<string>? log = null)
    {
        Directory.CreateDirectory(destination);
        string index = Path.Combine(source, "model.safetensors.index.json");
        var shards = File.Exists(index)
            ? System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(index))!["weight_map"]!.AsObject()
                .Select(p => p.Value!.GetValue<string>()).Distinct().ToList()
            : ["model.safetensors"];

        var files = shards.Select(s => new SafetensorsFile(Path.Combine(source, s))).ToList();
        try
        {
            var tensors = files.SelectMany(f => f.Entries.Values.Select(e => (File: f, Entry: e)))
                .Where(t => t.Entry.Name.StartsWith("model.language_model.", StringComparison.Ordinal))
                .OrderBy(t => t.Entry.Name, StringComparer.Ordinal).ToList();

            var header = new System.Text.Json.Nodes.JsonObject
            {
                ["__metadata__"] = new System.Text.Json.Nodes.JsonObject
                {
                    ["format"] = "pt",
                    ["source"] = $"{MicaCheckpoint.HubRepo}@{MicaCheckpoint.HubRevision}",
                    ["converted_by"] = "laya mica-convert (text decoder only, bf16 copied unchanged)",
                },
            };
            long offset = 0;
            foreach (var (_, entry) in tensors)
            {
                long bytes = entry.End - entry.Start;
                header[entry.Name] = new System.Text.Json.Nodes.JsonObject
                {
                    ["dtype"] = entry.DType,
                    ["shape"] = new System.Text.Json.Nodes.JsonArray([.. entry.Shape.Select(d => (System.Text.Json.Nodes.JsonNode)d)]),
                    ["data_offsets"] = new System.Text.Json.Nodes.JsonArray(offset, offset + bytes),
                };
                offset += bytes;
            }
            byte[] json = System.Text.Encoding.UTF8.GetBytes(header.ToJsonString());
            int padded = (json.Length + 7) / 8 * 8;

            string model = Path.Combine(destination, "model.safetensors");
            log?.Invoke($"writing {tensors.Count} tensors, {offset / (1024.0 * 1024 * 1024):F2} GiB, to {model}");
            using (var output = new FileStream(model + ".tmp", FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
            {
                Span<byte> length = stackalloc byte[8];
                System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(length, padded);
                output.Write(length);
                output.Write(json);
                for (int i = json.Length; i < padded; ++i) output.WriteByte((byte)' ');
                var buffer = new byte[1 << 22];
                var streams = new Dictionary<SafetensorsFile, FileStream>();
                try
                {
                    foreach (var (file, entry) in tensors)
                    {
                        if (!streams.TryGetValue(file, out var input))
                        {
                            streams[file] = input = new FileStream(file.Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
                        }
                        input.Position = file.DataOffset + entry.Start;
                        for (long left = entry.End - entry.Start; left > 0;)
                        {
                            int read = input.Read(buffer, 0, (int)Math.Min(buffer.Length, left));
                            if (read <= 0) throw new EndOfStreamException($"{file.Path}: truncated at {entry.Name}");
                            output.Write(buffer, 0, read);
                            left -= read;
                        }
                    }
                }
                finally
                {
                    foreach (var stream in streams.Values) stream.Dispose();
                }
            }
            File.Move(model + ".tmp", model, overwrite: true);
        }
        finally
        {
            foreach (var file in files) file.Dispose();
        }

        foreach (string name in new[] { "config.json", "tokenizer.json", "tokenizer_config.json" })
        {
            File.Copy(Path.Combine(source, name), Path.Combine(destination, name), overwrite: true);
        }
        string calibration = Path.Combine(source, "calibration.json");
        if (File.Exists(calibration)) File.Copy(calibration, Path.Combine(destination, "calibration.json"), overwrite: true);
        else File.WriteAllText(Path.Combine(destination, "calibration.json"), "{\"temperature\": 1.0}\n");   // Mica v0.5's shipped calibration
        return MicaCheckpoint.HostFiles;
    }
}

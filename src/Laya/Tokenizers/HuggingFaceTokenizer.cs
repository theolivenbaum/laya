using System.Text;
using System.Text.Json;

namespace Laya.Tokenizers;

/// <summary>An entry of the <c>added_tokens</c> table.</summary>
public sealed record AddedToken(int Id, string Content, bool Special, bool Normalized, bool LStrip, bool RStrip);

/// <summary>
/// A <c>tokenizer.json</c> tokenizer, reimplemented for the pieces laya actually uses.
///
/// <para>Laya always tokenizes with <c>add_special_tokens=False</c> and assembles the
/// <c>[CLS] … [SEP]</c> layout itself, so the <c>post_processor</c> is deliberately not applied.
/// What does matter is that added tokens are matched before the model runs, that byte-level and
/// SentencePiece-style pre-tokenization both work (the English and multilingual checkpoints use one
/// each), and that decoding round-trips.</para>
/// </summary>
public sealed class HuggingFaceTokenizer
{
    private readonly INormalizer? _normalizer;
    private readonly IPreTokenizer? _preTokenizer;
    private readonly BpeModel _model;
    private readonly AddedTokenMatcher _rawAdded;
    private readonly AddedTokenMatcher _normalizedAdded;
    private readonly HashSet<int> _addedIds;
    private readonly bool _byteLevelDecoder;
    private readonly string? _metaspaceReplacement;

    public IReadOnlyDictionary<string, AddedToken> AddedTokensByContent { get; }
    public IReadOnlyList<AddedToken> AddedTokens { get; }

    // Special ids come from tokenizer_config.json, which names them rather than numbering them.
    public int ClsTokenId { get; private set; } = -1;
    public int SepTokenId { get; private set; } = -1;
    public int MaskTokenId { get; private set; } = -1;
    public int PadTokenId { get; private set; } = -1;
    public int UnkTokenId { get; private set; } = -1;
    public string ClsToken { get; private set; } = "[CLS]";
    public string SepToken { get; private set; } = "[SEP]";
    public string MaskToken { get; private set; } = "[MASK]";
    public string PadToken { get; private set; } = "[PAD]";

    private HuggingFaceTokenizer(INormalizer? normalizer, IPreTokenizer? preTokenizer, BpeModel model,
        IReadOnlyList<AddedToken> added, bool byteLevelDecoder, string? metaspaceReplacement)
    {
        _normalizer = normalizer;
        _preTokenizer = preTokenizer;
        _model = model;
        _byteLevelDecoder = byteLevelDecoder;
        _metaspaceReplacement = metaspaceReplacement;
        AddedTokens = added;
        AddedTokensByContent = added.ToDictionary(a => a.Content, a => a, StringComparer.Ordinal);
        _addedIds = [.. added.Select(a => a.Id)];

        // tokenizers matches added tokens in two passes: `normalized: false` entries are carved out
        // of the raw text, and `normalized: true` ones out of the normalized text. The distinction
        // is load-bearing for the multilingual checkpoint, whose "▁▁" runs are non-normalized and
        // so must never match the spaces that the Replace normalizer would turn into them.
        _rawAdded = new AddedTokenMatcher([.. added.Where(a => !a.Normalized)]);
        _normalizedAdded = new AddedTokenMatcher([.. added.Where(a => a.Normalized)]);
    }

    /// <summary>Loads <c>tokenizer/tokenizer.json</c> and <c>tokenizer/tokenizer_config.json</c>.</summary>
    public static HuggingFaceTokenizer FromDirectory(string directory)
    {
        string tokenizerPath = Path.Combine(directory, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            throw new FileNotFoundException($"No tokenizer.json in '{directory}'.", tokenizerPath);
        }

        var tokenizer = FromFile(tokenizerPath);
        string configPath = Path.Combine(directory, "tokenizer_config.json");
        if (File.Exists(configPath)) tokenizer.ApplyConfig(File.ReadAllBytes(configPath));
        return tokenizer;
    }

    public static HuggingFaceTokenizer FromFile(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        return FromJson(document.RootElement);
    }

    public static HuggingFaceTokenizer FromJson(JsonElement root)
    {
        var modelElement = root.GetProperty("model");
        string modelType = modelElement.GetProperty("type").GetString() ?? string.Empty;
        if (modelType != "BPE")
        {
            throw new NotSupportedException(
                $"tokenizer.json model '{modelType}' is not implemented; laya checkpoints all use BPE.");
        }

        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var entry in modelElement.GetProperty("vocab").EnumerateObject())
        {
            vocab[entry.Name] = entry.Value.GetInt32();
        }

        var merges = new List<(string, string)>();
        if (modelElement.TryGetProperty("merges", out var mergesElement))
        {
            foreach (var merge in mergesElement.EnumerateArray())
            {
                if (merge.ValueKind == JsonValueKind.Array)
                {
                    merges.Add((merge[0].GetString()!, merge[1].GetString()!));
                }
                else
                {
                    // Older exports store "a b"; the pair separator is the first space.
                    string text = merge.GetString()!;
                    int space = text.IndexOf(' ', StringComparison.Ordinal);
                    merges.Add((text[..space], text[(space + 1)..]));
                }
            }
        }

        var added = new List<AddedToken>();
        if (root.TryGetProperty("added_tokens", out var addedElement))
        {
            foreach (var entry in addedElement.EnumerateArray())
            {
                added.Add(new AddedToken(
                    entry.GetProperty("id").GetInt32(),
                    entry.GetProperty("content").GetString()!,
                    entry.TryGetProperty("special", out var sp) && sp.GetBoolean(),
                    !entry.TryGetProperty("normalized", out var nm) || nm.GetBoolean(),
                    entry.TryGetProperty("lstrip", out var ls) && ls.GetBoolean(),
                    entry.TryGetProperty("rstrip", out var rs) && rs.GetBoolean()));
            }
        }

        // The added tokens go in before the vocabulary is frozen, so the model is never mutated after
        // construction.
        var model = new BpeModel(
            vocab,
            merges,
            modelElement.TryGetProperty("unk_token", out var unk) && unk.ValueKind == JsonValueKind.String ? unk.GetString() : null,
            modelElement.TryGetProperty("byte_fallback", out var bf) && bf.GetBoolean(),
            modelElement.TryGetProperty("fuse_unk", out var fu) && fu.GetBoolean(),
            modelElement.TryGetProperty("ignore_merges", out var im) && im.GetBoolean(),
            [.. added.Select(a => (a.Id, a.Content))]);

        INormalizer? normalizer = root.TryGetProperty("normalizer", out var normalizerElement)
            ? NormalizerFactory.Create(normalizerElement)
            : null;
        IPreTokenizer? preTokenizer = root.TryGetProperty("pre_tokenizer", out var preTokenizerElement)
            ? PreTokenizerFactory.Create(preTokenizerElement)
            : null;

        var (byteLevelDecoder, metaspace) = ReadDecoder(root);
        return new HuggingFaceTokenizer(normalizer, preTokenizer, model, added, byteLevelDecoder, metaspace);
    }

    private static (bool ByteLevel, string? Metaspace) ReadDecoder(JsonElement root)
    {
        if (!root.TryGetProperty("decoder", out var decoder) || decoder.ValueKind != JsonValueKind.Object)
        {
            return (false, null);
        }

        string type = decoder.GetProperty("type").GetString() ?? string.Empty;
        switch (type)
        {
            case "ByteLevel":
                return (true, null);
            case "Metaspace":
                return (false, decoder.TryGetProperty("replacement", out var rep) ? rep.GetString() : "▁");
            case "Sequence":
                foreach (var child in decoder.GetProperty("decoders").EnumerateArray())
                {
                    string childType = child.GetProperty("type").GetString() ?? string.Empty;
                    if (childType == "ByteLevel") return (true, null);
                    if (childType == "Replace" && child.TryGetProperty("pattern", out var pattern)
                        && pattern.TryGetProperty("String", out var literal))
                    {
                        return (false, literal.GetString());
                    }
                }
                return (false, null);
            default:
                return (false, null);
        }
    }

    /// <summary>Applies the special-token names from <c>tokenizer_config.json</c>.</summary>
    public void ApplyConfig(ReadOnlySpan<byte> json)
    {
        var reader = JsonDocument.Parse(json.ToArray());
        var root = reader.RootElement;

        ClsToken = ReadSpecial(root, "cls_token") ?? ClsToken;
        SepToken = ReadSpecial(root, "sep_token") ?? SepToken;
        MaskToken = ReadSpecial(root, "mask_token") ?? MaskToken;
        PadToken = ReadSpecial(root, "pad_token") ?? PadToken;
        string? unkToken = ReadSpecial(root, "unk_token");

        ClsTokenId = IdOf(ClsToken);
        SepTokenId = IdOf(SepToken);
        MaskTokenId = IdOf(MaskToken);
        PadTokenId = IdOf(PadToken);
        UnkTokenId = unkToken is null ? -1 : IdOf(unkToken);
    }

    private static string? ReadSpecial(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var value)) return null;
        return value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            // Some exports store the full AddedToken object rather than its text.
            JsonValueKind.Object when value.TryGetProperty("content", out var content) => content.GetString(),
            _ => null,
        };
    }

    /// <summary>The id of a token string, or -1 when the vocabulary does not have it.</summary>
    public int IdOf(string token) => _model.TryGetId(token, out int id) ? id : -1;

    /// <summary>The string a token id renders as.</summary>
    public string TokenOf(int id) => _model.Render(id);

    /// <summary>Encodes text without any special tokens, which is the only mode laya uses.</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>(Math.Max(8, text.Length / 3));
        Encode(text, ids);
        return ids;
    }

    public void Encode(string text, List<int> destination)
    {
        if (string.IsNullOrEmpty(text)) return;

        var pieces = new List<string>();
        foreach (var (raw, rawId) in _rawAdded.Split(text))
        {
            if (rawId >= 0)
            {
                destination.Add(rawId);
                continue;
            }

            string normalized = _normalizer?.Normalize(raw) ?? raw;
            foreach (var (segment, addedId) in _normalizedAdded.Split(normalized))
            {
                if (addedId >= 0)
                {
                    destination.Add(addedId);
                    continue;
                }

                pieces.Clear();
                if (_preTokenizer is null) pieces.Add(segment);
                else _preTokenizer.Split(segment, pieces);

                foreach (string piece in pieces) _model.Encode(piece, destination);
            }
        }
    }

    /// <summary>Decodes ids back to text, inverting whichever byte or metaspace scheme applies.</summary>
    public string Decode(IEnumerable<int> ids, bool skipSpecialTokens = false)
    {
        // Added tokens carry literal text and must bypass the byte-level and metaspace decoders:
        // the whitespace-run tokens are real spaces, and a space is not in the byte-level alphabet,
        // so decoding them as model tokens would silently drop them.
        var output = new StringBuilder();
        var pending = new StringBuilder();

        void Flush()
        {
            if (pending.Length == 0) return;
            output.Append(DecodeModelTokens(pending.ToString()));
            pending.Clear();
        }

        foreach (int id in ids)
        {
            string token = _model.Render(id);
            if (_addedIds.Contains(id) && AddedTokensByContent.TryGetValue(token, out var added))
            {
                Flush();
                if (!(skipSpecialTokens && added.Special)) output.Append(added.Content);
                continue;
            }
            pending.Append(token);
        }
        Flush();
        return output.ToString();
    }

    private string DecodeModelTokens(string text)
    {
        if (_byteLevelDecoder) return ByteLevelAlphabet.Decode(text);
        if (_metaspaceReplacement is not null)
        {
            return DecodeByteFallback(text).Replace(_metaspaceReplacement, " ", StringComparison.Ordinal);
        }
        return text;
    }

    private static string DecodeByteFallback(string text)
    {
        if (!text.Contains("<0x", StringComparison.Ordinal)) return text;

        var bytes = new List<byte>();
        var output = new StringBuilder();
        for (int i = 0; i < text.Length;)
        {
            if (i + 6 <= text.Length && text[i] == '<' && text[i + 1] == '0' && text[i + 2] == 'x' && text[i + 5] == '>'
                && byte.TryParse(text.AsSpan(i + 3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte b))
            {
                bytes.Add(b);
                i += 6;
                continue;
            }

            if (bytes.Count > 0)
            {
                output.Append(Encoding.UTF8.GetString(bytes.ToArray()));
                bytes.Clear();
            }
            output.Append(text[i]);
            i++;
        }
        if (bytes.Count > 0) output.Append(Encoding.UTF8.GetString(bytes.ToArray()));
        return output.ToString();
    }
}

/// <summary>
/// Longest-match scanner for the added-token vocabulary.
///
/// <para><c>tokenizers</c> carves added tokens out of the text before the model ever sees it, so a
/// run of 24 spaces becomes the single id that was added for it rather than whatever BPE would
/// have produced. Matching is longest-first at each position, which is what the underlying regex
/// alternation (sorted by decreasing length) does.</para>
/// </summary>
internal sealed class AddedTokenMatcher
{
    private readonly AddedToken[] _tokens;
    private readonly Dictionary<char, List<AddedToken>> _byFirstChar = [];
    private readonly bool _empty;

    public AddedTokenMatcher(IReadOnlyList<AddedToken> tokens)
    {
        _tokens = [.. tokens.Where(t => t.Content.Length > 0).OrderByDescending(t => t.Content.Length)];
        _empty = _tokens.Length == 0;
        foreach (var token in _tokens)
        {
            if (!_byFirstChar.TryGetValue(token.Content[0], out var bucket))
            {
                bucket = [];
                _byFirstChar[token.Content[0]] = bucket;
            }
            bucket.Add(token);
        }
    }

    /// <summary>
    /// Splits text into alternating plain segments (id -1) and added-token hits (the token's id).
    /// </summary>
    public IEnumerable<(string Segment, int AddedId)> Split(string text)
    {
        if (_empty)
        {
            yield return (text, -1);
            yield break;
        }

        int plainStart = 0;
        for (int i = 0; i < text.Length;)
        {
            if (!_byFirstChar.TryGetValue(text[i], out var candidates))
            {
                i++;
                continue;
            }

            AddedToken? hit = null;
            foreach (var candidate in candidates)
            {
                if (i + candidate.Content.Length <= text.Length
                    && text.AsSpan(i, candidate.Content.Length).SequenceEqual(candidate.Content))
                {
                    hit = candidate;
                    break;      // candidates are longest-first
                }
            }

            if (hit is null)
            {
                i++;
                continue;
            }

            if (i > plainStart) yield return (text[plainStart..i], -1);
            yield return (hit.Content, hit.Id);
            i += hit.Content.Length;
            plainStart = i;
        }

        if (plainStart < text.Length) yield return (text[plainStart..], -1);
    }
}

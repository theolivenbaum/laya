using System.Collections.Concurrent;
using System.Globalization;
using System.Text;

namespace Laya.Tokenizers;

/// <summary>
/// The byte-pair-encoding model from <c>tokenizer.json</c>.
///
/// <para>Merges are applied lowest-rank first, exactly as <c>tokenizers</c> does: a word is a list
/// of symbols, and on each step the adjacent pair with the smallest merge rank is joined. Words are
/// short, so the straightforward scan beats a priority queue and keeps the code obviously correct.
/// A cache short-circuits the common case where the same word appears many times.</para>
/// </summary>
public sealed class BpeModel
{
    private readonly Dictionary<string, int> _vocab;
    private string[] _idToToken;
    private readonly Dictionary<(string, string), int> _ranks;
    private readonly ConcurrentDictionary<string, int[]> _cache = new(StringComparer.Ordinal);
    private readonly int[] _byteFallbackIds = new int[256];

    public bool ByteFallback { get; }
    public bool FuseUnk { get; }
    public bool IgnoreMerges { get; }
    public string? UnkToken { get; }
    public int UnkId { get; }
    public int Count => _idToToken.Length;

    public BpeModel(Dictionary<string, int> vocab, IReadOnlyList<(string Left, string Right)> merges,
        string? unkToken, bool byteFallback, bool fuseUnk, bool ignoreMerges)
    {
        _vocab = vocab;
        ByteFallback = byteFallback;
        FuseUnk = fuseUnk;
        IgnoreMerges = ignoreMerges;
        UnkToken = unkToken;
        UnkId = unkToken is not null && vocab.TryGetValue(unkToken, out int unk) ? unk : -1;

        int max = 0;
        foreach (int id in vocab.Values) max = Math.Max(max, id);
        _idToToken = new string[max + 1];
        foreach (var (token, id) in vocab) _idToToken[id] = token;

        _ranks = new Dictionary<(string, string), int>(merges.Count);
        for (int i = 0; i < merges.Count; ++i) _ranks.TryAdd(merges[i], i);

        for (int b = 0; b < 256; ++b)
        {
            _byteFallbackIds[b] = vocab.TryGetValue($"<0x{b:X2}>", out int id) ? id : -1;
        }
    }

    public bool TryGetId(string token, out int id) => _vocab.TryGetValue(token, out id);

    public string? TokenAt(int id) => (uint)id < (uint)_idToToken.Length ? _idToToken[id] : null;

    internal void EnsureIdCapacity(int maxId)
    {
        if (maxId < _idToToken.Length) return;
        Array.Resize(ref _idToToken, maxId + 1);
    }

    internal void SetToken(int id, string content)
    {
        EnsureIdCapacity(id);
        _idToToken[id] = content;
        _vocab.TryAdd(content, id);
    }

    /// <summary>Tokenizes one pre-token into vocabulary ids.</summary>
    public void Encode(string word, List<int> destination)
    {
        if (word.Length == 0) return;

        if (IgnoreMerges && _vocab.TryGetValue(word, out int direct))
        {
            destination.Add(direct);
            return;
        }

        if (_cache.TryGetValue(word, out int[]? cached))
        {
            destination.AddRange(cached);
            return;
        }

        int[] ids = Merge(word);
        if (word.Length <= 64) _cache.TryAdd(word, ids);
        destination.AddRange(ids);
    }

    private int[] Merge(string word)
    {
        // Symbols start as single characters (surrogate pairs stay together, which matters for the
        // byte-fallback vocabularies where a code point outside the vocab must be split into bytes).
        var symbols = new List<string>(word.Length);
        for (int i = 0; i < word.Length;)
        {
            int length = char.IsHighSurrogate(word[i]) && i + 1 < word.Length && char.IsLowSurrogate(word[i + 1]) ? 2 : 1;
            symbols.Add(word.Substring(i, length));
            i += length;
        }

        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIndex = -1;
            for (int i = 0; i + 1 < symbols.Count; ++i)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) break;

            symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1];
            symbols.RemoveAt(bestIndex + 1);
        }

        var ids = new List<int>(symbols.Count);
        bool previousWasUnk = false;
        foreach (string symbol in symbols)
        {
            if (_vocab.TryGetValue(symbol, out int id))
            {
                ids.Add(id);
                previousWasUnk = false;
                continue;
            }

            if (ByteFallback)
            {
                bool complete = true;
                int before = ids.Count;
                foreach (byte b in Encoding.UTF8.GetBytes(symbol))
                {
                    int fallback = _byteFallbackIds[b];
                    if (fallback < 0)
                    {
                        complete = false;
                        break;
                    }
                    ids.Add(fallback);
                }
                if (complete)
                {
                    previousWasUnk = false;
                    continue;
                }
                ids.RemoveRange(before, ids.Count - before);
            }

            if (UnkId >= 0)
            {
                if (!(FuseUnk && previousWasUnk)) ids.Add(UnkId);
                previousWasUnk = true;
            }
            // With no unk token there is nothing sensible to emit, and byte-level vocabularies
            // always cover every byte, so this is unreachable for the laya checkpoints.
        }
        return [.. ids];
    }

    /// <summary>Renders an id back to its vocabulary string, or an <c>&lt;id&gt;</c> placeholder.</summary>
    public string Render(int id) => TokenAt(id) ?? "<" + id.ToString(CultureInfo.InvariantCulture) + ">";
}

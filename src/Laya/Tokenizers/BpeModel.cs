using System.Collections.Concurrent;
using System.Collections.Frozen;
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
    // The vocabulary and the merge table are complete once the model is built and read on every
    // merge step after that, so they are frozen: a FrozenDictionary picks its hashing strategy for
    // the keys it holds and is read-only by construction. The word cache is the one map written at
    // run time, on every miss from whichever thread is encoding, so it stays concurrent.
    private readonly FrozenDictionary<string, int> _vocab;
    private readonly string[] _idToToken;
    private readonly FrozenDictionary<(string, string), int> _ranks;
    private readonly ConcurrentDictionary<string, int[]> _cache = new(StringComparer.Ordinal);
    private readonly int[] _byteFallbackIds = new int[256];

    public bool ByteFallback { get; }
    public bool FuseUnk { get; }
    public bool IgnoreMerges { get; }
    public string? UnkToken { get; }
    public int UnkId { get; }
    public int Count => _idToToken.Length;

    /// <param name="addedTokens">
    /// The tokenizer's <c>added_tokens</c>, which join the vocabulary under their own ids before it
    /// is frozen. An added token whose content the vocabulary already has keeps the vocabulary's id.
    /// </param>
    public BpeModel(IReadOnlyDictionary<string, int> vocab, IReadOnlyList<(string Left, string Right)> merges,
        string? unkToken, bool byteFallback, bool fuseUnk, bool ignoreMerges,
        IReadOnlyList<(int Id, string Content)>? addedTokens = null)
    {
        var complete = new Dictionary<string, int>(vocab.Count + (addedTokens?.Count ?? 0), StringComparer.Ordinal);
        foreach (var (token, id) in vocab) complete[token] = id;

        int max = -1;
        foreach (int id in complete.Values) max = Math.Max(max, id);
        if (addedTokens is not null)
        {
            foreach (var (id, _) in addedTokens) max = Math.Max(max, id);
        }

        _idToToken = new string[max + 1];
        foreach (var (token, id) in complete) _idToToken[id] = token;
        if (addedTokens is not null)
        {
            foreach (var (id, content) in addedTokens)
            {
                _idToToken[id] = content;
                complete.TryAdd(content, id);
            }
        }

        _vocab = complete.ToFrozenDictionary(StringComparer.Ordinal);
        ByteFallback = byteFallback;
        FuseUnk = fuseUnk;
        IgnoreMerges = ignoreMerges;
        UnkToken = unkToken;
        UnkId = unkToken is not null && _vocab.TryGetValue(unkToken, out int unk) ? unk : -1;

        var ranks = new Dictionary<(string, string), int>(merges.Count);
        for (int i = 0; i < merges.Count; ++i) ranks.TryAdd(merges[i], i);
        _ranks = ranks.ToFrozenDictionary();

        for (int b = 0; b < 256; ++b)
        {
            _byteFallbackIds[b] = vocab.TryGetValue($"<0x{b:X2}>", out int id) ? id : -1;
        }
    }

    public bool TryGetId(string token, out int id) => _vocab.TryGetValue(token, out id);

    public string? TokenAt(int id) => (uint)id < (uint)_idToToken.Length ? _idToToken[id] : null;

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

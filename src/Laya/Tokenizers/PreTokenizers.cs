using System.Buffers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Laya.Tokenizers;

/// <summary>A <c>tokenizer.json</c> pre-tokenizer: text in, word pieces out.</summary>
public interface IPreTokenizer
{
    void Split(string text, List<string> destination);
}

/// <summary>
/// GPT-2 style byte-level pre-tokenization: split on the classic regex, then map every UTF-8 byte
/// to a printable code point so that BPE never has to deal with raw bytes.
/// </summary>
public sealed partial class ByteLevelPreTokenizer(bool addPrefixSpace, bool useRegex) : IPreTokenizer
{
    // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.Compiled)]
    private static partial Regex SplitPattern();

    public void Split(string text, List<string> destination)
    {
        if (text.Length == 0) return;
        if (addPrefixSpace && text[0] != ' ') text = " " + text;

        if (!useRegex)
        {
            destination.Add(ByteLevelAlphabet.Encode(text));
            return;
        }

        foreach (var match in SplitPattern().EnumerateMatches(text))
        {
            if (match.Length == 0) continue;
            destination.Add(ByteLevelAlphabet.Encode(text.AsSpan(match.Index, match.Length)));
        }
    }
}

/// <summary>
/// SentencePiece-style pre-tokenization: spaces are represented by <c>▁</c> and each word keeps the
/// marker that precedes it.
/// </summary>
public sealed class MetaspacePreTokenizer(string replacement, string prependScheme, bool split) : IPreTokenizer
{
    public void Split(string text, List<string> destination)
    {
        if (text.Length == 0) return;

        string prepared = text.Replace(" ", replacement, StringComparison.Ordinal);
        if (prependScheme == "always" && !prepared.StartsWith(replacement, StringComparison.Ordinal))
        {
            prepared = replacement + prepared;
        }

        if (!split)
        {
            destination.Add(prepared);
            return;
        }

        // Split before every marker, keeping it attached to the piece that follows.
        int start = 0;
        for (int i = 0; i < prepared.Length; ++i)
        {
            if (i > start && prepared.AsSpan(i).StartsWith(replacement, StringComparison.Ordinal))
            {
                destination.Add(prepared[start..i]);
                start = i;
            }
        }
        if (start < prepared.Length) destination.Add(prepared[start..]);
    }
}

public sealed class WhitespaceSplitPreTokenizer : IPreTokenizer
{
    public void Split(string text, List<string> destination)
    {
        foreach (var range in text.AsSpan().Split(' '))
        {
            var piece = text.AsSpan()[range];
            if (!piece.IsEmpty) destination.Add(piece.ToString());
        }
    }
}

public sealed class SequencePreTokenizer(IReadOnlyList<IPreTokenizer> preTokenizers) : IPreTokenizer
{
    public void Split(string text, List<string> destination)
    {
        var current = new List<string> { text };
        var next = new List<string>();
        foreach (var preTokenizer in preTokenizers)
        {
            next.Clear();
            foreach (string piece in current) preTokenizer.Split(piece, next);
            (current, next) = (next, current);
        }
        destination.AddRange(current);
    }
}

internal static class PreTokenizerFactory
{
    public static IPreTokenizer? Create(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object) return null;
        string type = element.GetProperty("type").GetString() ?? string.Empty;
        return type switch
        {
            "ByteLevel" => new ByteLevelPreTokenizer(
                element.TryGetProperty("add_prefix_space", out var aps) && aps.GetBoolean(),
                !element.TryGetProperty("use_regex", out var ur) || ur.GetBoolean()),
            "Metaspace" => new MetaspacePreTokenizer(
                element.TryGetProperty("replacement", out var rep) ? rep.GetString() ?? "▁" : "▁",
                element.TryGetProperty("prepend_scheme", out var ps) ? ps.GetString() ?? "always" : "always",
                !element.TryGetProperty("split", out var sp) || sp.GetBoolean()),
            "WhitespaceSplit" => new WhitespaceSplitPreTokenizer(),
            "Sequence" => CreateSequence(element),
            _ => throw new NotSupportedException($"tokenizer.json pre_tokenizer '{type}' is not implemented."),
        };
    }

    private static IPreTokenizer CreateSequence(JsonElement element)
    {
        var parts = new List<IPreTokenizer>();
        foreach (var child in element.GetProperty("pretokenizers").EnumerateArray())
        {
            var created = Create(child);
            if (created is not null) parts.Add(created);
        }
        return new SequencePreTokenizer(parts);
    }
}

/// <summary>
/// GPT-2's <c>bytes_to_unicode</c> table. Every one of the 256 byte values gets a printable,
/// non-whitespace code point so the BPE vocabulary is pure text; the 188 already-printable bytes
/// map to themselves and the remaining 68 are shifted into U+0100…U+0143.
/// </summary>
public static class ByteLevelAlphabet
{
    private static readonly char[] ByteToChar = BuildByteToChar();
    private static readonly Dictionary<char, byte> CharToByte = BuildCharToByte();

    private static char[] BuildByteToChar()
    {
        var map = new char[256];
        var assigned = new bool[256];
        void Keep(int from, int to)
        {
            for (int b = from; b <= to; ++b)
            {
                map[b] = (char)b;
                assigned[b] = true;
            }
        }
        Keep('!', '~');
        Keep(0xA1, 0xAC);
        Keep(0xAE, 0xFF);

        int next = 0;
        for (int b = 0; b < 256; ++b)
        {
            if (assigned[b]) continue;
            map[b] = (char)(256 + next);
            next++;
        }
        return map;
    }

    private static Dictionary<char, byte> BuildCharToByte()
    {
        var reverse = new Dictionary<char, byte>(256);
        for (int b = 0; b < 256; ++b) reverse[ByteToChar[b]] = (byte)b;
        return reverse;
    }

    /// <summary>UTF-8 encodes <paramref name="text"/> and maps each byte into the alphabet.</summary>
    public static string Encode(ReadOnlySpan<char> text)
    {
        int byteCount = Encoding.UTF8.GetByteCount(text);
        byte[] bytes = ArrayPool<byte>.Shared.Rent(byteCount);
        char[] chars = ArrayPool<char>.Shared.Rent(byteCount);
        try
        {
            Encoding.UTF8.GetBytes(text, bytes);
            for (int i = 0; i < byteCount; ++i) chars[i] = ByteToChar[bytes[i]];
            return new string(chars, 0, byteCount);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
            ArrayPool<char>.Shared.Return(chars);
        }
    }

    /// <summary>Inverse of <see cref="Encode(ReadOnlySpan{char})"/>.</summary>
    public static string Decode(ReadOnlySpan<char> text)
    {
        var bytes = new byte[text.Length];
        int count = 0;
        foreach (char c in text)
        {
            if (CharToByte.TryGetValue(c, out byte b)) bytes[count++] = b;
        }
        return Encoding.UTF8.GetString(bytes.AsSpan(0, count));
    }
}

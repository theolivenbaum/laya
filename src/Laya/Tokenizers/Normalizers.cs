using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Laya.Tokenizers;

/// <summary>A <c>tokenizer.json</c> normalizer: text in, text out.</summary>
public interface INormalizer
{
    string Normalize(string text);
}

/// <summary>Unicode normalization form C, the ModernBERT tokenizer's normalizer.</summary>
public sealed class NfcNormalizer : INormalizer
{
    public string Normalize(string text) => text.IsNormalized(NormalizationForm.FormC)
        ? text
        : text.Normalize(NormalizationForm.FormC);
}

public sealed class NfkcNormalizer : INormalizer
{
    public string Normalize(string text) => text.Normalize(NormalizationForm.FormKC);
}

public sealed class NfdNormalizer : INormalizer
{
    public string Normalize(string text) => text.Normalize(NormalizationForm.FormD);
}

/// <summary>Literal or regex replacement, as used by the SentencePiece-style tokenizers.</summary>
public sealed class ReplaceNormalizer(string pattern, string content, bool isRegex) : INormalizer
{
    private readonly System.Text.RegularExpressions.Regex? _regex =
        isRegex ? new System.Text.RegularExpressions.Regex(pattern, System.Text.RegularExpressions.RegexOptions.Compiled) : null;

    public string Normalize(string text) => _regex is null
        ? text.Replace(pattern, content, StringComparison.Ordinal)
        : _regex.Replace(text, content);
}

public sealed class LowercaseNormalizer : INormalizer
{
    public string Normalize(string text) => text.ToLowerInvariant();
}

public sealed class StripNormalizer(bool left, bool right) : INormalizer
{
    public string Normalize(string text) => (left, right) switch
    {
        (true, true) => text.Trim(),
        (true, false) => text.TrimStart(),
        (false, true) => text.TrimEnd(),
        _ => text,
    };
}

public sealed class PrependNormalizer(string prepend) : INormalizer
{
    public string Normalize(string text) => prepend + text;
}

public sealed class SequenceNormalizer(IReadOnlyList<INormalizer> normalizers) : INormalizer
{
    public string Normalize(string text)
    {
        foreach (var normalizer in normalizers) text = normalizer.Normalize(text);
        return text;
    }
}

internal static class NormalizerFactory
{
    public static INormalizer? Create(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object) return null;
        string type = element.GetProperty("type").GetString() ?? string.Empty;
        switch (type)
        {
            case "NFC": return new NfcNormalizer();
            case "NFKC": return new NfkcNormalizer();
            case "NFD": return new NfdNormalizer();
            case "Lowercase": return new LowercaseNormalizer();
            case "Prepend": return new PrependNormalizer(element.GetProperty("prepend").GetString() ?? string.Empty);
            case "Strip":
                return new StripNormalizer(
                    element.TryGetProperty("strip_left", out var l) && l.GetBoolean(),
                    element.TryGetProperty("strip_right", out var r) && r.GetBoolean());
            case "Replace":
            {
                var pattern = element.GetProperty("pattern");
                string content = element.GetProperty("content").GetString() ?? string.Empty;
                if (pattern.TryGetProperty("String", out var literal))
                {
                    return new ReplaceNormalizer(literal.GetString() ?? string.Empty, content, isRegex: false);
                }
                return new ReplaceNormalizer(pattern.GetProperty("Regex").GetString() ?? string.Empty, content, isRegex: true);
            }
            case "Sequence":
            {
                var parts = new List<INormalizer>();
                foreach (var child in element.GetProperty("normalizers").EnumerateArray())
                {
                    var created = Create(child);
                    if (created is not null) parts.Add(created);
                }
                return parts.Count == 0 ? null : new SequenceNormalizer(parts);
            }
            case "Precompiled":
            case "BertNormalizer":
                // Not used by any laya checkpoint; failing loudly beats normalizing differently
                // from the trained tokenizer and silently shifting every token id.
                throw new NotSupportedException($"tokenizer.json normalizer '{type}' is not implemented.");
            default:
                throw new NotSupportedException($"tokenizer.json normalizer '{type}' is not implemented.");
        }
    }

    public static string Describe(INormalizer? normalizer) => normalizer switch
    {
        null => "none",
        SequenceNormalizer => "sequence",
        _ => normalizer.GetType().Name.Replace("Normalizer", string.Empty, StringComparison.Ordinal)
            .ToLower(CultureInfo.InvariantCulture),
    };
}

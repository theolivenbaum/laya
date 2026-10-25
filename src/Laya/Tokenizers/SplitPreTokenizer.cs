using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Laya.Tokenizers;

/// <summary>
/// The <c>Split</c> pre-tokenizer: cut the text on a regex, keeping matches and the gaps between them
/// as separate pieces (<c>behavior: "Isolated"</c>) or dropping the matches (<c>"Removed"</c>).
///
/// <para>tokenizers evaluates the pattern with Oniguruma on code points. .NET's regex engine works on
/// UTF-16 code units, so a class like <c>\p{L}</c> never matches a letter outside the BMP (each half of
/// the surrogate pair is category <c>Cs</c>) — CJK Extension B, mathematical alphanumerics and a few
/// scripts would split differently. The pattern Qwen3.5 ships is therefore matched by
/// <see cref="Qwen35Pattern"/>, a hand-written scanner over runes; any other pattern falls back to
/// <see cref="Regex"/>.</para>
/// </summary>
public sealed class SplitPreTokenizer : IPreTokenizer
{
    private readonly Func<string, int, int>? _scanner;     // (text, position) -> match length, 0 if none
    private readonly Regex? _regex;
    private readonly string? _literal;
    private readonly bool _removeMatches;

    private SplitPreTokenizer(Func<string, int, int>? scanner, Regex? regex, string? literal, bool removeMatches)
    {
        _scanner = scanner;
        _regex = regex;
        _literal = literal;
        _removeMatches = removeMatches;
    }

    internal static SplitPreTokenizer Create(JsonElement element)
    {
        string behavior = element.TryGetProperty("behavior", out var b) ? b.GetString() ?? "Isolated" : "Isolated";
        if (element.TryGetProperty("invert", out var invert) && invert.GetBoolean())
        {
            throw new NotSupportedException("tokenizer.json Split with invert: true is not implemented.");
        }
        bool remove = behavior switch
        {
            "Isolated" => false,
            "Removed" => true,
            _ => throw new NotSupportedException($"tokenizer.json Split behavior '{behavior}' is not implemented."),
        };

        var pattern = element.GetProperty("pattern");
        if (pattern.TryGetProperty("String", out var literal))
        {
            return new SplitPreTokenizer(null, null, literal.GetString() ?? string.Empty, remove);
        }

        string regex = pattern.GetProperty("Regex").GetString() ?? string.Empty;
        if (regex == Qwen35Pattern.Source) return new SplitPreTokenizer(Qwen35Pattern.Match, null, null, remove);
        return new SplitPreTokenizer(null, new Regex(regex, RegexOptions.Compiled | RegexOptions.CultureInvariant), null, remove);
    }

    public void Split(string text, List<string> destination)
    {
        if (text.Length == 0) return;
        int gapStart = 0;

        void Emit(int start, int length)
        {
            if (start > gapStart) destination.Add(text[gapStart..start]);
            if (!_removeMatches) destination.Add(text.Substring(start, length));
            gapStart = start + length;
        }

        if (_scanner is not null)
        {
            for (int i = 0; i < text.Length;)
            {
                int length = _scanner(text, i);
                if (length == 0)
                {
                    i += char.IsHighSurrogate(text[i]) && i + 1 < text.Length ? 2 : 1;
                    continue;
                }
                Emit(i, length);
                i += length;
            }
        }
        else if (_regex is not null)
        {
            foreach (var match in _regex.EnumerateMatches(text))
            {
                if (match.Length > 0) Emit(match.Index, match.Length);
            }
        }
        else if (!string.IsNullOrEmpty(_literal))
        {
            for (int i = text.IndexOf(_literal, StringComparison.Ordinal); i >= 0;
                 i = text.IndexOf(_literal, i + _literal.Length, StringComparison.Ordinal))
            {
                Emit(i, _literal.Length);
            }
        }

        if (gapStart < text.Length) destination.Add(text[gapStart..]);
    }
}

/// <summary>
/// The Qwen3.5 pre-tokenizer regex, matched by hand over code points:
/// <code>
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)            1. English contractions
/// |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+        2. a word, with at most one leading non-letter
/// |\p{N}                                  3. a single digit
/// | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*         4. punctuation and symbols, then any line breaks
/// |\s*[\r\n]+                             5. whitespace up to the last line break
/// |\s+(?!\S)                              6. whitespace, leaving the last one for the next word
/// |\s+                                    7. whatever whitespace is left
/// </code>
/// Each alternative is resolved the way a backtracking engine resolves it: the first that matches at
/// the position wins, and within it the greedy quantifiers give back only what the rest needs.
/// </summary>
internal static class Qwen35Pattern
{
    public const string Source =
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    private enum Kind : byte { Letter, Mark, Number, Space, LineBreak, Other }

    private static Kind Classify(Rune rune)
    {
        int value = rune.Value;
        if (value is '\r' or '\n') return Kind.LineBreak;
        switch (Rune.GetUnicodeCategory(rune))
        {
            case UnicodeCategory.UppercaseLetter:
            case UnicodeCategory.LowercaseLetter:
            case UnicodeCategory.TitlecaseLetter:
            case UnicodeCategory.ModifierLetter:
            case UnicodeCategory.OtherLetter:
                return Kind.Letter;
            case UnicodeCategory.NonSpacingMark:
            case UnicodeCategory.SpacingCombiningMark:
            case UnicodeCategory.EnclosingMark:
                return Kind.Mark;
            case UnicodeCategory.DecimalDigitNumber:
            case UnicodeCategory.LetterNumber:
            case UnicodeCategory.OtherNumber:
                return Kind.Number;
        }
        // \s in Oniguruma's Unicode mode is the White_Space property, which is exactly what
        // Rune.IsWhiteSpace tests (Zs, Zl, Zp, U+0009–U+000D, U+0085).
        return Rune.IsWhiteSpace(rune) ? Kind.Space : Kind.Other;
    }

    private static Kind At(string text, int index, out int width)
    {
        if (Rune.DecodeFromUtf16(text.AsSpan(index), out var rune, out width) != OperationStatus.Done)
        {
            width = 1;          // a lone surrogate: Oniguruma would have rejected the string; treat it as a symbol
            return Kind.Other;
        }
        return Classify(rune);
    }

    private static bool IsSpace(Kind kind) => kind is Kind.Space or Kind.LineBreak;

    /// <summary>Length of the match at <paramref name="start"/>; every code point matches something.</summary>
    public static int Match(string text, int start)
    {
        int n = text.Length;

        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (text[start] == '\'' && start + 1 < n)
        {
            char c1 = char.ToLowerInvariant(text[start + 1]);
            char c2 = start + 2 < n ? char.ToLowerInvariant(text[start + 2]) : '\0';
            if (c1 is 's' or 't' or 'm' or 'd' || text[start + 1] == 'ſ') return 2;  // ſ folds to s
            if ((c1 == 'r' || c1 == 'v') && c2 == 'e') return 3;
            if (c1 == 'l' && c2 == 'l') return 3;
        }

        var first = At(text, start, out int firstWidth);

        // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (first is not (Kind.LineBreak or Kind.Letter or Kind.Number))
        {
            int run = LetterRun(text, start + firstWidth);
            if (run > 0) return firstWidth + run;
        }
        if (first is Kind.Letter or Kind.Mark) return LetterRun(text, start);

        // 3. \p{N}
        if (first == Kind.Number) return firstWidth;

        // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        int symbols = start;
        if (text[start] == ' ' && start + 1 < n && At(text, start + 1, out _) == Kind.Other) symbols = start + 1;
        if (At(text, symbols, out _) == Kind.Other)
        {
            int i = symbols;
            while (i < n && At(text, i, out int w) == Kind.Other) i += w;
            while (i < n && text[i] is '\r' or '\n') i++;
            return i - start;
        }

        // Everything left starts with whitespace. Find the maximal whitespace run.
        int end = start, lastBreak = -1;
        while (end < n)
        {
            var kind = At(text, end, out int w);
            if (!IsSpace(kind)) break;
            if (kind == Kind.LineBreak) lastBreak = end;
            end += w;
        }

        // 5. \s*[\r\n]+ — \s* backs off to the last line break of the run, which [\r\n]+ then takes.
        if (lastBreak >= 0) return lastBreak + 1 - start;

        // 6. \s+(?!\S) — the whole run at the end of the text, otherwise all but its last code point.
        if (end == n) return end - start;
        int lastWidth = char.IsLowSurrogate(text[end - 1]) && end - 2 >= start ? 2 : 1;
        if (end - lastWidth > start) return end - lastWidth - start;

        // 7. \s+
        return end - start;
    }

    private static int LetterRun(string text, int start)
    {
        int i = start;
        while (i < text.Length)
        {
            var kind = At(text, i, out int w);
            if (kind is not (Kind.Letter or Kind.Mark)) break;
            i += w;
        }
        return i - start;
    }
}

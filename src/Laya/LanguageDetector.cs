using System.Collections;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace Laya;

/// <summary>
/// What <see cref="LanguageDetector.Analyse"/> found in a state. The JSON names match the Python
/// payload, because a routing decision is serialized straight into API responses.
/// </summary>
public sealed record LanguageAnalysis(
    [property: JsonPropertyName("script")] string Script,
    [property: JsonPropertyName("script_profile")] IReadOnlyDictionary<string, double> ScriptProfile,
    [property: JsonPropertyName("language")] string? Language,
    [property: JsonPropertyName("is_english")] bool IsEnglish,
    [property: JsonPropertyName("non_latin_fraction")] double NonLatinFraction);

/// <summary>
/// Dependency-free language and script detection, used to route between checkpoints.
///
/// <para>Routing only needs one decision: is this English Latin text, or something the English
/// checkpoint cannot read? On non-Latin scripts that checkpoint does not degrade gently, it
/// collapses — 0.100 on 20-option Hindi intent against 0.050 for random — so <em>script</em> is the
/// primary signal and "is this Latin text English?" is the secondary one. Script detection is
/// exact; the Latin-language guess is an explicitly best-effort stopword and diacritic
/// heuristic.</para>
/// </summary>
public static partial class LanguageDetector
{
    private static readonly (string Name, (int Low, int High)[] Ranges)[] ScriptRanges =
    [
        ("greek", [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
        ("cyrillic", [(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
        ("hebrew", [(0x0590, 0x05FF)]),
        ("arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
        ("devanagari", [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
        ("bengali", [(0x0980, 0x09FF)]),
        ("gurmukhi", [(0x0A00, 0x0A7F)]),
        ("gujarati", [(0x0A80, 0x0AFF)]),
        ("oriya", [(0x0B00, 0x0B7F)]),
        ("tamil", [(0x0B80, 0x0BFF)]),
        ("telugu", [(0x0C00, 0x0C7F)]),
        ("kannada", [(0x0C80, 0x0CFF)]),
        ("malayalam", [(0x0D00, 0x0D7F)]),
        ("sinhala", [(0x0D80, 0x0DFF)]),
        ("thai", [(0x0E00, 0x0E7F)]),
        ("lao", [(0x0E80, 0x0EFF)]),
        ("tibetan", [(0x0F00, 0x0FFF)]),
        ("myanmar", [(0x1000, 0x109F)]),
        ("georgian", [(0x10A0, 0x10FF)]),
        ("ethiopic", [(0x1200, 0x137F)]),
        ("khmer", [(0x1780, 0x17FF)]),
        ("hangul", [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]),
        ("kana", [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
        ("han", [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]),
    ];

    // Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
    // weighted and a margin is required before calling something non-English.
    private static readonly Dictionary<string, HashSet<string>> StopWords = new(StringComparer.Ordinal)
    {
        ["en"] = Set("the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i"),
        ["fr"] = Set("le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce"),
        ["de"] = Set("der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber"),
        ["es"] = Set("el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus"),
        ["pt"] = Set("os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela"),
        ["it"] = Set("il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "nella", "alla"),
        ["nl"] = Set("het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt"),
    };

    private static readonly HashSet<char> NonEnglishDiacritics =
        [.. "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö"];

    private static HashSet<string> Set(params string[] words) => [.. words];

    [GeneratedRegex(@"[^\W\d_]+", RegexOptions.Compiled)]
    private static partial Regex WordPattern();

    /// <summary>
    /// Flattens a state into the text detection sees. Keys are ignored: they are usually English
    /// even when the values are not.
    /// </summary>
    public static string StateText(object? state, int maxChars = 4000)
    {
        var parts = new List<string>();
        Collect(state, parts, 0);
        string joined = string.Join(' ', parts);
        return joined.Length <= maxChars ? joined : joined[..maxChars];
    }

    private static void Collect(object? state, List<string> destination, int depth)
    {
        if (depth > 6 || state is null) return;
        switch (state)
        {
            case string text:
                destination.Add(text);
                return;
            case JsonElement element:
                CollectJson(element, destination, depth);
                return;
            case IDictionary dictionary:
                foreach (DictionaryEntry entry in dictionary) Collect(entry.Value, destination, depth + 1);
                return;
            case IEnumerable<KeyValuePair<string, object?>> pairs:
                foreach (var pair in pairs) Collect(pair.Value, destination, depth + 1);
                return;
            case IEnumerable sequence:
                foreach (object? item in sequence) Collect(item, destination, depth + 1);
                return;
            default:
                return;
        }
    }

    private static void CollectJson(JsonElement element, List<string> destination, int depth)
    {
        if (depth > 6) return;
        switch (element.ValueKind)
        {
            case JsonValueKind.String:
                destination.Add(element.GetString() ?? string.Empty);
                return;
            case JsonValueKind.Object:
                foreach (var property in element.EnumerateObject()) CollectJson(property.Value, destination, depth + 1);
                return;
            case JsonValueKind.Array:
                foreach (var item in element.EnumerateArray()) CollectJson(item, destination, depth + 1);
                return;
            default:
                return;
        }
    }

    /// <summary>Dominant script of the text, or <c>"unknown"</c> when it contains no letters.</summary>
    public static string DetectScript(string text)
    {
        var counts = CountScripts(text);
        long total = 0;
        foreach (var entry in counts) total += entry.Value;
        if (total == 0) return "unknown";

        // Python's `max(counts.items(), ...)` keeps the first maximum in dictionary order, and
        // "latin" is assigned last there, so a tie goes to the non-Latin script. CountScripts
        // returns the same order for exactly that reason.
        string best = counts[0].Key;
        long bestCount = counts[0].Value;
        for (int i = 1; i < counts.Count; ++i)
        {
            if (counts[i].Value > bestCount)
            {
                best = counts[i].Key;
                bestCount = counts[i].Value;
            }
        }
        return best;
    }

    /// <summary>Fraction of alphabetic characters belonging to each detected script.</summary>
    public static IReadOnlyDictionary<string, double> ScriptProfile(string text)
    {
        var counts = CountScripts(text);
        long total = 0;
        foreach (var entry in counts) total += entry.Value;
        if (total == 0) return new Dictionary<string, double>(StringComparer.Ordinal);

        var profile = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var (name, count) in counts)
        {
            if (count > 0) profile[name] = (double)count / total;
        }
        return profile;
    }

    /// <summary>Letter counts per script, in first-encounter order with "latin" last.</summary>
    private static List<KeyValuePair<string, long>> CountScripts(string text)
    {
        var order = new List<string>();
        var counts = new Dictionary<string, long>(StringComparer.Ordinal);
        long latin = 0;

        foreach (var rune in text.EnumerateRunes())
        {
            if (!System.Text.Rune.IsLetter(rune)) continue;
            int codePoint = rune.Value;
            if (codePoint < 0x0250 || (codePoint >= 0x1E00 && codePoint <= 0x1EFF))
            {
                latin++;
                continue;
            }
            foreach (var (name, ranges) in ScriptRanges)
            {
                bool inRange = false;
                foreach (var (low, high) in ranges)
                {
                    if (codePoint >= low && codePoint <= high)
                    {
                        inRange = true;
                        break;
                    }
                }
                if (!inRange) continue;
                if (!counts.ContainsKey(name)) order.Add(name);
                counts[name] = counts.GetValueOrDefault(name) + 1;
                break;
            }
        }

        var result = new List<KeyValuePair<string, long>>(order.Count + 1);
        foreach (string name in order) result.Add(new KeyValuePair<string, long>(name, counts[name]));
        result.Add(new KeyValuePair<string, long>("latin", latin));
        return result;
    }

    /// <summary>
    /// Best-effort language code for Latin-script text, or null when undecided. A non-English
    /// language has to beat English by a margin, so ordinary English is never misrouted; short
    /// inputs return null on purpose.
    /// </summary>
    public static string? GuessLatinLanguage(string text)
    {
        var words = new List<string>();
        foreach (Match match in WordPattern().Matches(text)) words.Add(match.Value.ToLowerInvariant());
        if (words.Count < 4) return null;

        var scores = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (language, stopWords) in StopWords)
        {
            int hits = 0;
            foreach (string word in words)
            {
                if (stopWords.Contains(word)) hits++;
            }
            scores[language] = hits;
        }

        string lowered = text.ToLowerInvariant();
        int diacritics = 0;
        foreach (char c in lowered)
        {
            if (NonEnglishDiacritics.Contains(c)) diacritics++;
        }
        double diacriticRate = (double)diacritics / Math.Max(1, lowered.Length);

        int english = scores.GetValueOrDefault("en");
        string? bestLanguage = null;
        int best = 0;
        foreach (var (language, score) in scores)
        {
            if (language == "en") continue;
            if (bestLanguage is null || score > best)
            {
                bestLanguage = language;
                best = score;
            }
        }

        if (best == 0 && diacriticRate < 0.02) return english > 0 ? "en" : null;
        if (bestLanguage is not null && best >= Math.Max(2, english + 2)) return bestLanguage;
        if (diacriticRate >= 0.04 && bestLanguage is not null && best >= english) return bestLanguage;
        return english > 0 ? "en" : null;
    }

    /// <summary>Full detection result for a state.</summary>
    public static LanguageAnalysis Analyse(object? state)
    {
        string text = StateText(state);
        var profile = ScriptProfile(text);
        string script = DetectScript(text);
        double nonLatin = profile.Count > 0 ? Math.Round(1d - profile.GetValueOrDefault("latin"), 4) : 0d;

        if (script == "unknown")
        {
            return new LanguageAnalysis("unknown", profile, null, true, 0d);
        }
        if (script != "latin")
        {
            return new LanguageAnalysis(script, profile, null, false, nonLatin);
        }

        string? language = GuessLatinLanguage(text);
        return new LanguageAnalysis("latin", profile, language, language is null or "en", nonLatin);
    }

    /// <summary>True when the English checkpoint can be expected to read this state.</summary>
    public static bool IsEnglish(object? state) => Analyse(state).IsEnglish;
}

using System.Collections;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace Laya;

/// <summary>
/// What <see cref="LanguageDetector.Analyse(object?)"/> found in a state. The JSON names match the
/// Python payload, because a routing decision is serialized straight into API responses.
/// </summary>
/// <param name="LanguageUndecided">
/// True when no language could be identified. Undecided is not English: it is reported separately
/// so a caller can tell "this is English" from "nothing here says what this is".
/// </param>
public sealed record LanguageAnalysis(
    [property: JsonPropertyName("script")] string Script,
    [property: JsonPropertyName("script_profile")] IReadOnlyDictionary<string, double> ScriptProfile,
    [property: JsonPropertyName("language")] string? Language,
    [property: JsonPropertyName("is_english")] bool IsEnglish,
    [property: JsonPropertyName("language_undecided")] bool LanguageUndecided,
    [property: JsonPropertyName("diacritic_rate")] double DiacriticRate,
    [property: JsonPropertyName("non_latin_fraction")] double NonLatinFraction)
{
    /// <summary>What an <see cref="ILanguageClassifier"/> said, when one was consulted.</summary>
    [JsonPropertyName("classifier_language")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ClassifierLanguage { get; init; }

    [JsonPropertyName("classifier_probability")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? ClassifierProbability { get; init; }
}

/// <summary>The evidence behind <see cref="LanguageDetector.GuessLatinLanguage"/>.</summary>
public readonly record struct LatinProfile(string? Language, int EnglishHits, double DiacriticRate, bool LooksNonEnglish);

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
        ("armenian", [(0x0530, 0x058F)]),
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
    //
    // The Romance lists carry the *unaccented* function words as well as the accented ones: a state
    // that lost its accents (mail clients and ticket systems strip them) keeps no diacritic rate, so
    // `la`, `un`, `y`, `e`, `et` are the only evidence left. Kept in the Python's order, because a tie
    // between two languages goes to the one listed first.
    private static readonly (string Language, HashSet<string> Words)[] StopWords =
    [
        ("en", Set("the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i")),
        ("fr", Set("le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
            "et", "du", "au", "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton",
            "ma", "ta", "sa", "mes", "tes", "ses", "ces", "deux", "trois", "très", "bien",
            "tout", "tous", "toute", "fait", "veux", "veut", "peux", "peut", "dois", "doit",
            "merci", "bonjour", "jour", "jours", "mois", "fois", "quand", "comment", "pourquoi",
            "alors", "donc")),
        ("de", Set("der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber")),
        // `de`/`en` are Spanish too, but also common English tokens (`de facto`, `en route`), so they stay out.
        ("es", Set("el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
            "la", "un", "y", "al", "lo", "le", "les", "su", "mi", "tu", "nos",
            "ni", "dos", "tres", "fue", "fueron", "ser", "tiene", "tienen", "tengo", "puede",
            "pueden", "quiero", "necesito", "hemos", "han", "sobre", "entre", "cuando", "donde",
            "porque", "aunque", "también", "ya", "eso", "esto", "esa", "ese", "nada", "algo",
            "aquí", "hoy", "gracias")),
        // `no` is Portuguese too, and one of the most frequent English words, so it stays out.
        ("pt", Set("os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
            "o", "e", "na", "nas", "nos", "ao", "aos", "por", "foi", "era", "ser", "sou",
            "tem", "tenho", "pode", "podem", "quero", "preciso", "eu", "meu", "minha", "seu",
            "sua", "isso", "isto", "aqui", "ali", "como", "quando", "onde", "porque", "mais",
            "já", "ainda", "agora", "hoje", "ontem", "dois", "três", "tudo", "nada", "obrigado",
            "olá")),
        // The articulated prepositions are Italian-only, which lets a state of shared articles still name it.
        ("it", Set("il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "nella", "alla",
            "la", "le", "un", "uno", "una", "e", "ed", "o", "da", "su", "tra", "fra", "mi",
            "ci", "ne", "ho", "hai", "ha", "abbiamo", "avete", "hanno", "era", "stato", "stata",
            "devo", "deve", "devono", "voglio", "vorrei", "mio", "mia", "tuo", "sua", "quando",
            "dove", "perche", "molto", "poco", "sempre", "mai", "già", "ancora", "adesso", "oggi",
            "ieri", "grazie", "ciao", "scusa",
            "nel", "nell", "negli", "sul", "sulla", "sulle", "dal", "dalla", "dallo", "dagli", "dei",
            "delle", "dello", "degli", "agli", "alle", "col")),
        ("nl", Set("het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt")),
        // Romanian words its Romance neighbours do not share (`la`, `o`, `un`, `de`, `pe`, `ca` left out),
        // so adding it cannot steal a French/Spanish/Italian/Portuguese state.
        ("ro", Set("și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
            "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
            "acesta", "aceasta", "mi", "ți", "vă", "nu")),
    ];

    // Letters ordinary English does not use: the signal that catches a Latin-script language with no
    // stopword list (Polish, Czech, Turkish, Baltic, ...) before it is handed to the English checkpoint.
    private static readonly HashSet<char> NonEnglishDiacritics =
    [
        .. "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ",     // Western European
        .. "ăâîșțşţ",                              // Romanian
        .. "ąćęłńśźż",                             // Polish
        .. "čďěňřšťůž",                            // Czech / Slovak
        .. "őű",                                   // Hungarian
        .. "ğı",                                   // Turkish (text is lowercased before matching)
        .. "āēģīķļņūž",                            // Baltic
        .. "đ",                                    // Serbo-Croatian / Vietnamese
    ];

    // Words more than one list claims: matching one says "not English" without saying which language.
    private static readonly HashSet<string> SharedWords = BuildSharedWords();

    /// <summary>A diacritic rate at or above this is evidence the text is not English.</summary>
    public const double NonEnglishDiacriticRate = 0.02;

    private static HashSet<string> BuildSharedWords()
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (_, words) in StopWords)
        {
            foreach (string word in words) counts[word] = counts.GetValueOrDefault(word) + 1;
        }
        return [.. counts.Where(c => c.Value > 1).Select(c => c.Key)];
    }

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
    /// The evidence behind the Latin-script language guess. <see cref="Analyse"/> needs it rather
    /// than just the verdict, because "undecided" and "English" are different answers and only one of
    /// them is safe to send to the English checkpoint. A non-English language is only named when it
    /// matched at least one word no other list claims.
    /// </summary>
    public static LatinProfile LatinProfile(string text)
    {
        var words = new List<string>();
        foreach (Match match in WordPattern().Matches(text)) words.Add(match.Value.ToLowerInvariant());

        string lowered = text.ToLowerInvariant();
        int diacritics = 0;
        foreach (char c in lowered)
        {
            if (NonEnglishDiacritics.Contains(c)) diacritics++;
        }
        double diacriticRate = (double)diacritics / Math.Max(1, lowered.Length);
        bool nonEnglish = diacriticRate >= NonEnglishDiacriticRate;
        if (words.Count < 4) return new LatinProfile(null, 0, diacriticRate, nonEnglish);

        var distinct = new HashSet<string>(words, StringComparer.Ordinal);
        int english = 0;
        string? bestLanguage = null;
        int best = 0;
        foreach (var (language, stopWords) in StopWords)
        {
            int hits = 0;
            foreach (string word in words)
            {
                if (stopWords.Contains(word)) hits++;
            }
            if (language == "en")
            {
                english = hits;
                continue;
            }

            // Only a language that matched a word of its own may be named: the top score can otherwise
            // be pure overlap (`la` and `e` in Romanian text made Italian the winner). Such a language
            // drops out of the running rather than losing the tie, so a lesser score with real evidence
            // is still named.
            bool evidenced = false;
            foreach (string word in distinct)
            {
                if (stopWords.Contains(word) && !SharedWords.Contains(word))
                {
                    evidenced = true;
                    break;
                }
            }
            if (!evidenced) continue;
            if (bestLanguage is null || hits > best)
            {
                bestLanguage = language;
                best = hits;
            }
        }

        string? guess = null;
        if (bestLanguage is not null && best >= Math.Max(2, english + 2))
        {
            guess = bestLanguage;                   // a clear margin over English function words
        }
        else if (bestLanguage is not null && nonEnglish && best >= Math.Max(2, english))
        {
            guess = bestLanguage;                   // two hits here too: one shared word is a guess
        }
        else if (english > 0 && !nonEnglish)
        {
            guess = "en";
        }
        return new LatinProfile(guess, english, diacriticRate, nonEnglish);
    }

    private static int CountWords(string text) => WordPattern().Count(text);

    /// <summary>
    /// Best-effort language code for Latin-script text, or null when undecided. A non-English
    /// language has to beat English by a margin and match a word of its own, so ordinary English is
    /// never misrouted; short inputs return null on purpose.
    /// </summary>
    public static string? GuessLatinLanguage(string text) => LatinProfile(text).Language;

    /// <summary>Full detection result for a state.</summary>
    public static LanguageAnalysis Analyse(object? state) => Analyse(state, classifier: null);

    /// <summary>
    /// Full detection result for a state, consulting <paramref name="classifier"/> when the heuristic
    /// cannot identify a Latin-script language.
    /// </summary>
    public static LanguageAnalysis Analyse(object? state, ILanguageClassifier? classifier)
    {
        string text = StateText(state);
        var profile = ScriptProfile(text);
        string script = DetectScript(text);
        double nonLatin = profile.Count > 0 ? Math.Round(1d - profile.GetValueOrDefault("latin"), 4) : 0d;

        if (script == "unknown")
        {
            return new LanguageAnalysis("unknown", profile, null, true, true, 0d, 0d);
        }
        if (script != "latin")
        {
            return new LanguageAnalysis(script, profile, null, false, true, 0d, nonLatin);
        }

        var latin = LatinProfile(text);
        // Undecided is not English: non-English letters are enough to prefer the multilingual
        // checkpoint, while text without them (including short English) still goes to the English one.
        bool undecided = latin.Language is null;
        bool english = latin.Language == "en" || (undecided && !latin.LooksNonEnglish);
        var analysis = new LanguageAnalysis("latin", profile, latin.Language, english, undecided,
            Math.Round(latin.DiacriticRate, 4), nonLatin);

        // The heuristic admits what it cannot see: a plain-ASCII language it holds no stopwords for
        // (Indonesian, Swahili, Turkish without its letters) reads as undecided-therefore-English.
        // That is exactly the case a real classifier is for, and the only one it is asked about.
        if (undecided && english && classifier is not null && CountWords(text) >= classifier.MinimumWords)
        {
            var guess = classifier.Classify(text);
            if (guess is not null)
            {
                bool classifierEnglish = guess.Value.Language == "en";
                analysis = analysis with
                {
                    ClassifierLanguage = guess.Value.Language,
                    ClassifierProbability = Math.Round(guess.Value.Probability, 4),
                    IsEnglish = classifierEnglish || guess.Value.Probability < classifier.MinimumProbability,
                };
            }
        }
        return analysis;
    }

    /// <summary>True when the English checkpoint can be expected to read this state.</summary>
    public static bool IsEnglish(object? state) => Analyse(state).IsEnglish;

    /// <inheritdoc cref="IsEnglish(object?)"/>
    public static bool IsEnglish(object? state, ILanguageClassifier? classifier) => Analyse(state, classifier).IsEnglish;
}

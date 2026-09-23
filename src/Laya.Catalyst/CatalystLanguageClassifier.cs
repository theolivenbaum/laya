using CatalystDetector = Catalyst.Models.LanguageDetector;
using Mosaik.Core;
using CatalystLanguage = Mosaik.Core.Language;

namespace Laya.Catalyst;

/// <summary>
/// An <see cref="ILanguageClassifier"/> backed by Catalyst's CLD3-derived n-gram language detector,
/// which covers 53 languages and ships its model inside the Catalyst package.
///
/// <code>
/// var router = new Router { LanguageClassifier = await CatalystLanguageClassifier.CreateAsync() };
/// router.Route("Saya ditagih dua kali untuk langganan saya bulan ini").Model;   // "multilingual"
/// </code>
///
/// <para>The router only asks it about Latin-script text the built-in heuristic could not identify
/// and would otherwise send to the English checkpoint, so its weakness on very short text (where any
/// n-gram model guesses) is fenced off by <see cref="MinimumWords"/> and
/// <see cref="MinimumProbability"/> rather than trusted.</para>
///
/// <para>The detector samples n-grams at random, so a borderline text can come back with a slightly
/// different probability from one call to the next. <see cref="Trials"/> averages more samples.</para>
/// </summary>
public sealed class CatalystLanguageClassifier : ILanguageClassifier
{
    private readonly CatalystDetector _detector;

    private CatalystLanguageClassifier(CatalystDetector detector) => _detector = detector;

    /// <summary>Loads the embedded detector model (a few hundred milliseconds, once).</summary>
    public static async Task<CatalystLanguageClassifier> CreateAsync(int trials = 7)
    {
        var detector = await CatalystDetector.FromStoreAsync(CatalystLanguage.Any, Mosaik.Core.Version.Latest, "")
            .ConfigureAwait(false);
        detector.Trials = Math.Max(1, trials);
        return new CatalystLanguageClassifier(detector);
    }

    /// <inheritdoc/>
    public double MinimumProbability { get; init; } = 0.8;

    /// <inheritdoc/>
    public int MinimumWords { get; init; } = 4;

    /// <summary>Sampling rounds averaged per call; more is steadier and proportionally slower.</summary>
    public int Trials
    {
        get => _detector.Trials;
        init => _detector.Trials = Math.Max(1, value);
    }

    /// <summary>The ISO 639-1 codes the detector can return.</summary>
    public IReadOnlyList<string> Languages => [.. _detector.Data.Languages.Select(ToCode)];

    /// <inheritdoc/>
    public LanguageGuess? Classify(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        var detected = _detector.DetectAll(text);
        if (detected.Count == 0 || detected[0].Language is CatalystLanguage.Unknown or CatalystLanguage.Any) return null;
        double english = detected.Where(d => d.Language == CatalystLanguage.English).Sum(d => d.Probability);
        return new LanguageGuess(ToCode(detected[0].Language), detected[0].Probability, english);
    }

    /// <summary>Every candidate the detector considered, best first.</summary>
    public IReadOnlyList<LanguageGuess> ClassifyAll(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return [];
        var detected = _detector.DetectAll(text);
        double english = detected.Where(d => d.Language == CatalystLanguage.English).Sum(d => d.Probability);
        return [.. detected
            .Where(d => d.Language is not (CatalystLanguage.Unknown or CatalystLanguage.Any))
            .Select(d => new LanguageGuess(ToCode(d.Language), d.Probability, english))];
    }

    private static string ToCode(CatalystLanguage language) => Mosaik.Core.Languages.EnumToCode(language);
}

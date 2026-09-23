namespace Laya;

/// <summary>
/// A classifier's verdict: the most likely language as an ISO 639-1 code (<c>"en"</c>, <c>"id"</c>, ...),
/// its probability, and the probability the classifier gives English. Routing asks "is this not
/// English?", which a spread over close relatives (Indonesian / Malay / Tagalog) answers firmly even
/// when no single one of them is a clear winner — hence the third field.
/// </summary>
public readonly record struct LanguageGuess(string Language, double Probability, double EnglishProbability);

/// <summary>
/// A statistical language identifier the <see cref="Router"/> can consult.
///
/// <para>The built-in detector is exact about script and deliberately conservative about Latin-script
/// languages: it names one only on stopword evidence, and text it cannot identify that carries no
/// non-English letters is treated as English. That is right for short English and wrong for a
/// plain-ASCII language it holds no list for (Indonesian, Swahili, Tagalog, Turkish with its letters
/// stripped). A classifier is consulted for exactly that case and no other, so it can only move an
/// undecided state to the multilingual checkpoint, never an identified one away from it.</para>
///
/// <para><c>Laya.Catalyst</c> provides an implementation backed by Catalyst's language detector.</para>
/// </summary>
public interface ILanguageClassifier
{
    /// <summary>The most likely language of <paramref name="text"/>, or null when it cannot tell.</summary>
    LanguageGuess? Classify(string text);

    /// <summary>
    /// The state leaves the English checkpoint only when the classifier gives non-English languages at
    /// least this much probability in total (<c>1 - EnglishProbability</c>).
    /// </summary>
    double MinimumProbability => 0.8;

    /// <summary>Texts with fewer words than this are not classified at all: short text is where n-gram models guess.</summary>
    int MinimumWords => 4;
}

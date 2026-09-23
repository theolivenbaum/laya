namespace Laya;

/// <summary>A classifier's verdict: an ISO 639-1 code (<c>"en"</c>, <c>"id"</c>, ...) and how sure it is.</summary>
public readonly record struct LanguageGuess(string Language, double Probability);

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

    /// <summary>A non-English verdict below this probability is ignored and the state stays English.</summary>
    double MinimumProbability => 0.8;

    /// <summary>Texts with fewer words than this are not classified at all: short text is where n-gram models guess.</summary>
    int MinimumWords => 4;
}

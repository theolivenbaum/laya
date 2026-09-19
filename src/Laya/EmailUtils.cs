using System.Text.RegularExpressions;

namespace Laya;

/// <summary>Cleaning and structuring of email input, ported from <c>.reference/laya/email.py</c>.</summary>
public static partial class EmailUtils
{
    [GeneratedRegex(@"^\s*On .{0,300}wrote:\s*$", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex QuoteHeaderOnWrote();

    [GeneratedRegex(@"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex QuoteHeaderForwarded();

    [GeneratedRegex(@"^\s*_{8,}\s*$", RegexOptions.Compiled)]
    private static partial Regex QuoteHeaderRule();

    [GeneratedRegex(@"^\s*From:\s.+$", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex QuoteHeaderFrom();

    [GeneratedRegex(@"^\s*--\s*$", RegexOptions.Compiled)]
    private static partial Regex SignatureDashes();

    [GeneratedRegex(@"^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$",
        RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex SignatureClosing();

    [GeneratedRegex(@"^\s*sent from my (iphone|android|mobile|ipad)", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex SignatureSentFrom();

    [GeneratedRegex(@"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|" +
        @"if you (have )?received this (e-?mail|message) in error)", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex Disclaimer();

    [GeneratedRegex(@"\n\s*\n", RegexOptions.Compiled)]
    private static partial Regex ParagraphSplit();

    [GeneratedRegex(@"[ \t]+", RegexOptions.Compiled)]
    private static partial Regex HorizontalWhitespace();

    private static readonly Regex[] QuoteHeaders =
        [QuoteHeaderOnWrote(), QuoteHeaderForwarded(), QuoteHeaderRule(), QuoteHeaderFrom()];

    private static readonly Regex[] SignatureMarkers =
        [SignatureDashes(), SignatureClosing(), SignatureSentFrom()];

    /// <summary>Removes quoted history, signatures and legal disclaimers so the model sees the message.</summary>
    public static string CleanEmailBody(string? body, int maxChars = 3000)
    {
        string text = (body ?? string.Empty)
            .Replace("\r\n", "\n", StringComparison.Ordinal)
            .Replace("\r", "\n", StringComparison.Ordinal)
            .Replace("\\n", "\n", StringComparison.Ordinal);

        var lines = new List<string>();
        foreach (string line in text.Split('\n'))
        {
            if (lines.Count > 0 && QuoteHeaders.Any(p => p.IsMatch(line))) break;
            if (line.TrimStart().StartsWith('>')) continue;
            lines.Add(line.TrimEnd());
        }

        int cut = lines.Count;
        // Only look for a sign-off in the last 40% of the message, and never in the first 8 lines:
        // "Thanks" as the opening word of a request is not a signature.
        int from = Math.Max(1, Math.Min((int)(lines.Count * 0.6), lines.Count - 8));
        for (int i = from; i < lines.Count; ++i)
        {
            if (lines[i].Trim().Length <= 40 && SignatureMarkers.Any(p => p.IsMatch(lines[i])))
            {
                cut = i;
                break;
            }
        }
        lines = lines[..cut];

        var paragraphs = ParagraphSplit().Split(string.Join('\n', lines))
            .Where(p => !Disclaimer().IsMatch(p))
            .Select(p => p.Trim())
            .Where(p => p.Length > 0);

        string cleaned = HorizontalWhitespace().Replace(string.Join("\n\n", paragraphs), " ");
        return cleaned.Length <= maxChars ? cleaned : cleaned[..maxChars];
    }

    /// <summary>Builds the state dictionary for email classification.</summary>
    public static IReadOnlyList<KeyValuePair<string, object?>> EmailState(string? subject, string? body,
        string? sender = null, bool clean = true, IEnumerable<KeyValuePair<string, object?>>? extra = null)
    {
        var state = new List<KeyValuePair<string, object?>>
        {
            new("subject", (subject ?? string.Empty).Trim()),
            new("body", clean ? CleanEmailBody(body) : body ?? string.Empty),
        };
        if (!string.IsNullOrEmpty(sender)) state.Add(new KeyValuePair<string, object?>("from", sender));
        if (extra is not null)
        {
            foreach (var (key, value) in extra)
            {
                if (value is not null) state.Add(new KeyValuePair<string, object?>(key, value));
            }
        }
        return state;
    }
}

using Xunit;

namespace Laya.Tests;

/// <summary>Email cleaning: quoted history, signatures and disclaimers must not reach the model.</summary>
public class EmailTests
{
    [Fact]
    public void QuotedHistoryIsCut()
    {
        string cleaned = EmailUtils.CleanEmailBody(
            "Can you refund invoice 4411?\n\nOn Tue, 3 Jun 2025, Support wrote:\n> earlier message\n> more");
        Assert.Contains("refund invoice 4411", cleaned, StringComparison.Ordinal);
        Assert.DoesNotContain("earlier message", cleaned, StringComparison.Ordinal);
    }

    [Fact]
    public void ForwardedMarkersCut()
    {
        string cleaned = EmailUtils.CleanEmailBody("Please look at this.\n---- Forwarded Message ----\nold stuff");
        Assert.DoesNotContain("old stuff", cleaned, StringComparison.Ordinal);
    }

    [Fact]
    public void QuotedLinesAreDropped()
    {
        string cleaned = EmailUtils.CleanEmailBody("my reply\n> their text\nmore of mine");
        Assert.DoesNotContain("their text", cleaned, StringComparison.Ordinal);
        Assert.Contains("more of mine", cleaned, StringComparison.Ordinal);
    }

    [Fact]
    public void SignaturesAreCutOnlyNearTheEnd()
    {
        string body = string.Join('\n',
            ["Thanks for the quick reply.",                       // an opening "Thanks" is not a sign-off
             .. Enumerable.Repeat("Here is some more detail about the problem.", 12),
             "Regards,", "Alex"]);

        string cleaned = EmailUtils.CleanEmailBody(body);
        Assert.Contains("Thanks for the quick reply", cleaned, StringComparison.Ordinal);
        Assert.DoesNotContain("Alex", cleaned, StringComparison.Ordinal);
    }

    [Fact]
    public void DisclaimerParagraphsAreRemoved()
    {
        string cleaned = EmailUtils.CleanEmailBody(
            "Real content here.\n\nThis e-mail is confidential and intended solely for the use of the named addressee.");
        Assert.Contains("Real content", cleaned, StringComparison.Ordinal);
        Assert.DoesNotContain("confidential", cleaned, StringComparison.Ordinal);
    }

    [Fact]
    public void WhitespaceIsCollapsedAndLengthIsBounded()
    {
        string cleaned = EmailUtils.CleanEmailBody("a     b\t\tc", maxChars: 5);
        Assert.Equal("a b c", cleaned);
        Assert.Equal("aaaaa", EmailUtils.CleanEmailBody(new string('a', 100), maxChars: 5));
    }

    [Fact]
    public void NullAndEmptyBodiesAreSafe()
    {
        Assert.Equal(string.Empty, EmailUtils.CleanEmailBody(null));
        Assert.Equal(string.Empty, EmailUtils.CleanEmailBody(string.Empty));
    }

    [Fact]
    public void EscapedNewlinesFromJsonPayloadsAreHonoured()
    {
        string cleaned = EmailUtils.CleanEmailBody(@"line one\n> quoted\nline two");
        Assert.DoesNotContain("quoted", cleaned, StringComparison.Ordinal);
    }

    [Fact]
    public void EmailStateKeepsSubjectBodyAndSender()
    {
        var state = EmailUtils.EmailState("  Invoice 4411  ", "Please refund.", "a@example.com");
        Assert.Equal("subject", state[0].Key);
        Assert.Equal("Invoice 4411", state[0].Value);
        Assert.Equal("body", state[1].Key);
        Assert.Equal("from", state[2].Key);
    }

    [Fact]
    public void EmailStateDropsNullExtras()
    {
        var state = EmailUtils.EmailState("s", "b", extra:
        [
            new KeyValuePair<string, object?>("thread_id", "t-1"),
            new KeyValuePair<string, object?>("label", null),
        ]);
        Assert.Contains(state, e => e.Key == "thread_id");
        Assert.DoesNotContain(state, e => e.Key == "label");
    }
}

/// <summary>
/// A disclaimer footer must not delete the sender's request. Ported from
/// <c>.reference/tests/test_email.py</c> (upstream #94).
/// </summary>
public class EmailDisclaimerTests
{
    private const string Disclaimer = "This email is confidential and intended solely for the named addressee.";

    [Theory]
    [InlineData("My account is locked.\n" + Disclaimer + "\nPlease unlock it.", "My account is locked. Please unlock it.")]
    [InlineData("My account is locked\n" + Disclaimer + "\nPlease unlock it.", "Please unlock it.")]
    [InlineData("My account is locked. " + Disclaimer, "My account is locked.")]
    [InlineData("My account is locked.\n\n" + Disclaimer, "My account is locked.")]
    [InlineData("My account is locked.\n\nThis email and any files transmitted with it are\n" +
                "confidential and intended solely for the named addressee.", "My account is locked.")]
    [InlineData("Please reopen ticket 4411.\n\nIf you have received this message in error, delete it.", "Please reopen ticket 4411.")]
    [InlineData("Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text", "Thanks for the update.")]
    [InlineData("Hi team,\nCan you confirm the refund?\nRegards,\nAlice", "Hi team,\nCan you confirm the refund?")]
    [InlineData("", "")]
    public void KeepsTheRequest(string body, string expected)
        => Assert.Equal(expected, EmailUtils.CleanEmailBody(body));

    [Fact]
    public void EmailStateKeepsTheRequest()
    {
        var state = EmailUtils.EmailState("Locked out", "My account is locked. " + Disclaimer);
        Assert.Equal("My account is locked.", state.Single(p => p.Key == "body").Value);
    }
}

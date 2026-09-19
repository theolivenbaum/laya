using Xunit;

namespace Laya.Tests;

/// <summary>Language and script detection. Ported from <c>.reference/tests/test_router.py</c>.</summary>
public class LanguageTests
{
    [Theory]
    [InlineData("The customer was charged twice and wants a refund.", "latin")]
    [InlineData("Le client a été facturé deux fois et demande un remboursement.", "latin")]
    [InlineData("ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari")]
    [InlineData("お客様は二重に請求されたため返金を希望しています。", "kana")]
    [InlineData("客户被重复扣款要求退款", "han")]
    [InlineData("고객이 두 번 청구되어 환불을 원합니다", "hangul")]
    [InlineData("تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic")]
    [InlineData("வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil")]
    [InlineData("С клиента дважды сняли деньги и он хочет возврат", "cyrillic")]
    [InlineData("ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai")]
    [InlineData("Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek")]
    [InlineData("הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew")]
    [InlineData("", "unknown")]
    [InlineData("12345 6789", "unknown")]
    public void DetectsTheDominantScript(string text, string expected)
        => Assert.Equal(expected, LanguageDetector.DetectScript(text));

    [Theory]
    [InlineData("Please refund the duplicate charge on invoice 4411 today.", true)]
    [InlineData("refund me", true)]
    [InlineData("ग्राहक से दो बार शुल्क लिया गया", false)]
    [InlineData("お客様は二重に請求されました", false)]
    [InlineData("С клиента дважды сняли деньги", false)]
    [InlineData("Le client a été facturé deux fois et il demande un remboursement pour la " +
                "facture qui a été payée le mois dernier avec la carte de crédit", false)]
    [InlineData("Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die " +
                "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde", false)]
    public void RecognisesWhatTheEnglishCheckpointCanRead(string text, bool expected)
        => Assert.Equal(expected, LanguageDetector.IsEnglish(text));

    [Theory]
    [InlineData("The customer was charged twice and wants a refund for this invoice", "en")]
    [InlineData("Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr")]
    [InlineData("Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung", "de")]
    [InlineData("El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura", "es")]
    [InlineData("refund", null)]
    public void GuessesLatinLanguages(string text, string? expected)
        => Assert.Equal(expected, LanguageDetector.GuessLatinLanguage(text));

    [Fact]
    public void LongEnglishNeverGuessesAnotherLanguage()
        => Assert.Equal("en", LanguageDetector.GuessLatinLanguage(
            "Please refund the duplicate charge on invoice 4411 today because we have been " +
            "waiting for three days and nobody has replied to us"));

    [Fact]
    public void FlattensStructuredState()
    {
        Assert.Contains("charged twice", LanguageDetector.StateText(new Dictionary<string, object?>
        {
            ["body"] = "charged twice",
            ["n"] = 3,
        }), StringComparison.Ordinal);

        Assert.Contains("deep", LanguageDetector.StateText(new Dictionary<string, object?>
        {
            ["a"] = new Dictionary<string, object?> { ["b"] = new[] { "deep" } },
        }), StringComparison.Ordinal);

        Assert.Contains("x", LanguageDetector.StateText(new object[]
        {
            "x",
            new Dictionary<string, object?> { ["y"] = "z" },
        }), StringComparison.Ordinal);

        Assert.Equal(string.Empty, LanguageDetector.StateText(null));
    }

    [Fact]
    public void KeysDoNotDriveDetection()
    {
        // English keys around Hindi content must still route as Hindi.
        var state = new Dictionary<string, object?>
        {
            ["subject"] = "नमस्ते",
            ["body"] = "ग्राहक से दो बार शुल्क लिया गया",
        };
        Assert.False(LanguageDetector.Analyse(state).IsEnglish);
    }

    [Fact]
    public void ProfileSumsToOne()
    {
        var profile = LanguageDetector.ScriptProfile("hello мир");
        Assert.Equal(1.0, profile.Values.Sum(), 6);
        Assert.True(profile["latin"] > 0);
        Assert.True(profile["cyrillic"] > 0);
    }

    [Fact]
    public void UnknownScriptIsReportedAsEnglishSoItUsesTheDefault()
    {
        var analysis = LanguageDetector.Analyse("12345");
        Assert.Equal("unknown", analysis.Script);
        Assert.True(analysis.IsEnglish);
        Assert.Equal(0d, analysis.NonLatinFraction);
    }
}

using Laya.Runtime;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// Latin-script languages the English checkpoint cannot read, and the English it must keep reading.
/// Ported from the upstream additions to <c>.reference/tests/test_router.py</c> (#23, #35, #172).
/// </summary>
public class LatinRoutingTests
{
    private static readonly Router Router = new();

    [Theory]
    [InlineData("Հայերեն", "armenian")]
    [InlineData("ՀԱՅԵՐԵՆ", "armenian")]
    [InlineData("։֊", "unknown")]
    public void DetectsArmenian(string text, string expected)
        => Assert.Equal(expected, LanguageDetector.DetectScript(text));

    [Fact]
    public void ArmenianProfileAndRouting()
    {
        Assert.Equal(1.0, LanguageDetector.Analyse("Հայերեն").ScriptProfile["armenian"]);
        Assert.Equal(0.7, LanguageDetector.Analyse("Հայերեն abc").NonLatinFraction);
        Assert.Equal("multilingual", Router.Route(new Dictionary<string, object?> { ["body"] = "Հայերեն" }).Model);
        Assert.Equal("english", Router.Route(new Dictionary<string, object?> { ["body"] = "Հայերեն" }, model: "english").Model);
    }

    [Theory]
    [InlineData("Հայերեն", false)]
    [InlineData("Gătește-mi o rețetă de sarmale de post pentru mâine.", false)]
    [InlineData("Am fost taxat de două ori pentru factura din luna martie și vreau banii", false)]
    [InlineData("Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę", false)]
    [InlineData("Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz", false)]
    [InlineData("Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım", false)]
    [InlineData("Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay", false)]
    [InlineData("We visited a cafe in Zurich and the naive assumption about the invoice was wrong, " +
                "so please refund the duplicate charge", true)]
    public void AnUnidentifiedLanguageIsNeverAssumedEnglish(string text, bool expected)
        => Assert.Equal(expected, LanguageDetector.IsEnglish(text));

    [Fact]
    public void UndecidedIsReportedAsUndecided()
    {
        var turkish = LanguageDetector.Analyse("Müşteriden iki kez ücret alındı ve para iadesi istiyor");
        Assert.True(turkish.LanguageUndecided);
        Assert.Null(turkish.Language);
        Assert.False(LanguageDetector.Analyse("Please refund the duplicate charge on the invoice").LanguageUndecided);
        Assert.True(LanguageDetector.Analyse("Gătește-mi o rețetă de sarmale").DiacriticRate > 0.02);
        Assert.Equal(0d, LanguageDetector.Analyse("Please refund the duplicate charge today").DiacriticRate);
        Assert.Null(LanguageDetector.GuessLatinLanguage("Cât e ora acum la Tokyo"));
    }

    [Fact]
    public void KnownGapRomanianWithoutDiacriticsStillReadsAsEnglish()
        => Assert.True(LanguageDetector.IsEnglish("Care este ora in Tokyo?"));

    [Theory]
    [InlineData("Gătește-mi o rețetă de sarmale de post pentru mâine.")]
    [InlineData("Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez.")]
    [InlineData("Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę")]
    [InlineData("Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım")]
    public void UnknownLatinRoutesToMultilingual(string text)
        => Assert.Equal("multilingual", Router.Route(text).Model);

    [Fact]
    public void TheReasonSaysWhatItRoutedOn()
    {
        Assert.Contains("not identified", Router.Route("Müşteriden iki kez ücret alındı ve para iadesi istiyor").Reason,
            StringComparison.Ordinal);
        Assert.Equal("english", Router.Route("Please refund the duplicate charge on invoice 4411 today.").Model);
        Assert.Equal("english", Router.Route("refund me").Model);
    }

    [Theory]
    [InlineData("es", "El pedido llego roto y nadie responde cuando escribo al soporte")]
    [InlineData("es", "Quiero cancelar mi plan y pedir un reembolso")]
    [InlineData("es", "La factura tiene un error en el importe total")]
    [InlineData("es", "Necesito que me devuelvan el dinero de la compra duplicada")]
    [InlineData("it", "Il cliente e stato addebitato due volte e vuole un rimborso")]
    [InlineData("it", "Voglio cancellare il mio abbonamento e chiedere un rimborso")]
    [InlineData("it", "La fattura contiene un errore nell importo totale")]
    [InlineData("pt", "O cliente foi cobrado duas vezes e quer o dinheiro de volta")]
    [InlineData("fr", "Le client a ete facture deux fois et demande un remboursement")]
    [InlineData("fr", "Je ne peux pas acceder a mon compte et j ai besoin d aide")]
    public void PlainAsciiRomanceIsIdentified(string language, string text)
    {
        Assert.Equal(language, LanguageDetector.GuessLatinLanguage(text));
        Assert.False(LanguageDetector.IsEnglish(text));
        Assert.Equal("multilingual", Router.Route(text).Model);
    }

    [Theory]
    [InlineData("La facturación tiene un error y necesito una corrección urgente")]
    [InlineData("La fattura è sbagliata, devo avere un rimborso per il pagamento")]
    [InlineData("La commande est arrivée cassée et personne ne répond au support")]
    public void AccentedRomanceStillRoutesToMultilingual(string text)
        => Assert.Equal("multilingual", Router.Route(text).Model);

    [Theory]
    [InlineData("The customer was charged twice and wants a refund for this invoice")]
    [InlineData("Please cancel my subscription and refund the duplicate charge today")]
    [InlineData("The report by Smith et al. shows the de facto standard, e.g. the LA office and Rio")]
    [InlineData("Our MI5 and UN contacts discussed the DOS attack in LA last month")]
    [InlineData("No refund was issued, so I am writing to you again about invoice 4411")]
    [InlineData("no refund no reply")]
    [InlineData("The son of the director filed a complaint about the duplicate invoice")]
    public void EnglishControlsDoNotMove(string text)
    {
        Assert.True(LanguageDetector.IsEnglish(text));
        Assert.Equal("english", Router.Route(text).Model);
    }

    [Fact]
    public void SharedWordsAloneNameNoLanguage()
    {
        Assert.Null(LanguageDetector.Analyse("Cât e ora acum la Tokyo").Language);
        Assert.Equal("multilingual", Router.Route("Cât e ora acum la Tokyo").Model);
        Assert.Equal("it", LanguageDetector.GuessLatinLanguage("La fattura contiene un errore nell importo totale"));
        Assert.Equal("es", LanguageDetector.GuessLatinLanguage("La factura tiene un error en el importe total"));
    }

    /// <summary>A classifier that always answers the same thing, and counts how often it was asked.</summary>
    private sealed class FixedClassifier(string language, double probability, double? english = null) : ILanguageClassifier
    {
        public int Calls { get; private set; }

        public LanguageGuess? Classify(string text)
        {
            Calls++;
            return new LanguageGuess(language, probability, english ?? (language == "en" ? probability : 0d));
        }
    }

    [Fact]
    public void AClassifierDecidesOnlyWhatTheHeuristicCannot()
    {
        var indonesian = new FixedClassifier("id", 0.99);
        var router = new Router { LanguageClassifier = indonesian };

        // Plain-ASCII Indonesian: no stopword list, no diacritics — the heuristic says English.
        const string text = "Saya ditagih dua kali untuk langganan saya bulan ini";
        Assert.Equal("english", new Router().Route(text).Model);
        var decision = router.Route(text);
        Assert.Equal("multilingual", decision.Model);
        Assert.Equal("id", decision.Detection!.ClassifierLanguage);
        Assert.Contains("classifier", decision.Reason, StringComparison.Ordinal);

        // English the heuristic identified is never second-guessed, and neither is a named language.
        int before = indonesian.Calls;
        Assert.Equal("english", router.Route("Please refund the duplicate charge on invoice 4411 today.").Model);
        Assert.Equal("multilingual", router.Route("Quiero cancelar mi plan y pedir un reembolso").Model);
        Assert.Equal(before, indonesian.Calls);

        // Short text is not classified at all.
        Assert.Equal("english", router.Route("refund me").Model);
    }

    [Fact]
    public void AnUnsureOrEnglishClassifierKeepsTheStateEnglish()
    {
        const string text = "Saya ditagih dua kali untuk langganan saya bulan ini";
        Assert.Equal("english", new Router { LanguageClassifier = new FixedClassifier("id", 0.5, english: 0.4) }.Route(text).Model);
        // A spread over close relatives is still firmly "not English".
        Assert.Equal("multilingual", new Router { LanguageClassifier = new FixedClassifier("id", 0.5, english: 0.05) }.Route(text).Model);
        Assert.Equal("english", new Router { LanguageClassifier = new FixedClassifier("en", 0.99) }.Route(text).Model);
    }
}

/// <summary>The temperature clamp (upstream #35): a fitted temperature may soften, never sharpen hard.</summary>
public class TemperatureClampTests
{
    [Theory]
    [InlineData(0.1006f, 0.5f)]
    [InlineData(0.10058280825614929f, Calibration.TemperatureMin)]
    [InlineData(1.7601518630981445f, 1.7601518630981445f)]
    [InlineData(1.0f, 1.0f)]
    [InlineData(9.0f, Calibration.TemperatureMax)]
    [InlineData(0.0f, Calibration.TemperatureMin)]
    [InlineData(-3.0f, Calibration.TemperatureMin)]
    [InlineData(float.NaN, 1.0f)]
    [InlineData(float.PositiveInfinity, 1.0f)]
    public void Clamps(float raw, float expected)
        => Assert.Equal(expected, Calibration.ClampTemperature(raw));

    [Fact]
    public void ThirteenOptionsIsTheElevenPlusBucket()
    {
        Assert.True(Calibration.TemperatureMin <= 1f && 1f <= Calibration.TemperatureMax);
        Assert.Equal("choice:11+", QuestionTypes.TemperatureBucket(QuestionType.Choice, 13));
    }
}

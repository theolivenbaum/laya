using Laya.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace Laya.Tests;

/// <summary>
/// End-to-end behaviour on real weights: the presets have to produce sane answers, and routing has
/// to send non-English input somewhere that can read it. Ported in spirit from
/// <c>.reference/tests/test_local_e2e.py</c>.
/// </summary>
public class AgentTests(ITestOutputHelper output)
{
    [ModelFact]
    public void ParallelOptionsDoNotChangeTheAnswers()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("english"));
        var questions = Presets.Triage();
        const string state = "I was charged twice for invoice 4411 and I want my money back today.";

        var everyCore = agent.SystemOne(state, questions);
        var oneThread = agent.SystemOne(state, questions, new ParallelOptions { MaxDegreeOfParallelism = 1 });

        // The kernels split work by panel and by (segment, head), never by partial sums, so the
        // thread count cannot move a single float.
        foreach (var (id, answer) in everyCore.Answers)
        {
            var other = oneThread[id];
            Assert.Equal(answer.Choice, other.Choice);
            Assert.Equal(answer.Score, other.Score);
            Assert.Equal(answer.Noul, other.Noul);
            Assert.Equal(answer.Confidence, other.Confidence);
        }
    }

    [ModelFact]
    public void TheShippedElevenPlusBucketIsClamped()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("english"));
        Assert.Contains(agent.ClampedTemperatures, t => t.StartsWith("choice:11+=", StringComparison.Ordinal));
        Assert.True(agent.RawTemperatureFor(QuestionType.Choice, 13) < Calibration.TemperatureMin);
        Assert.Equal(Calibration.TemperatureMin, agent.TemperatureFor(QuestionType.Choice, 13));
    }

    [ModelFact]
    public void TheEncoderEmbedderShortlistsTheObviousLabel()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("english"));
        var embed = Shortlist.EmbedFunctionFromAgent(agent);
        var rows = embed(["refund my payment", "billing refund", ""]);
        Assert.Equal(3, rows.Length);
        Assert.All(rows, r => Assert.Equal(agent.EncoderConfig.HiddenSize, r.Length));

        var question = Question.Choice("Which team?", "billing and refunds", "hardware repair",
            "office catering", "legal contracts");
        var kept = Shortlist.ShortlistChoice("I was charged twice and want a refund", question, embed, k: 2);
        output.WriteLine(string.Join(", ", kept));
        Assert.Equal(2, kept.Count);
    }

    [ModelFact]
    public void TriagePresetAnswersEveryQuestion()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("english"));
        var questions = Presets.Triage();
        var result = agent.SystemOne(
            "I was charged twice for invoice 4411 and I want my money back today.", questions);

        Assert.Equal(questions.Count, result.Answers.Count);
        Assert.Equal(questions.Ids, result.Answers.Select(a => a.Key).ToArray());
        Assert.True(result.Usage.InputTokens > 0);

        foreach (var (id, answer) in result.Answers)
        {
            Assert.InRange(answer.Confidence, 0d, 1d);
            Assert.InRange(answer.Action.ActProbability, 0d, 1d);
            if (answer.Probabilities is not null)
            {
                Assert.Equal(1d, answer.Probabilities.Sum(p => p.Value), 2);
            }
            output.WriteLine($"{id}: {answer.Choice ?? answer.Score?.ToString() ?? answer.Noul?.ToString()}");
        }
    }

    [ModelFact]
    public void ARefundComplaintIsReadAsARefundComplaint()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("english"));
        var result = agent.SystemOne(
            new List<KeyValuePair<string, object?>>
            {
                new("message", "You charged my card twice for invoice 4411. Please refund the duplicate today."),
            },
            Presets.Triage());

        Assert.Equal("refund", result["intent"].Choice);
        Assert.True(result["refund_requested"].Noul > 0.5);
    }

    [ModelFact]
    public void PhishingIsSeparatedFromLegitimateMail()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("english"));
        var questions = Presets.Email();

        var phishing = agent.SystemOne(EmailUtils.EmailState(
            "Urgent: your account is locked",
            "Your account has been locked for security reasons. Verify immediately at " +
            "http://wellsfargo--verify.tj49.wsipv6.com or it will be closed permanently.",
            "security@wellsf-argo-verify.com"), questions);

        var legitimate = agent.SystemOne(EmailUtils.EmailState(
            "Invoice 4411 for May",
            "Hi, attaching the invoice for May as agreed. Let me know if anything looks wrong.",
            "accounts@supplier.example"), questions);

        output.WriteLine($"phishing={phishing["is_phishing"].Noul} legitimate={legitimate["is_phishing"].Noul}");
        Assert.True(phishing["is_phishing"].Noul > legitimate["is_phishing"].Noul);
    }

    [ModelFact]
    public void QuestionsAreIndependentOfTheOrderTheyAreAskedIn()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("english"));
        const string state = "The build has been broken since this morning and the release is tomorrow.";

        var forward = agent.SystemOne(state, new QuestionSet()
            .Add("urgent", Question.Noul("Is this urgent?"))
            .Add("blocked", Question.Noul("Is someone blocked?")));

        var backward = agent.SystemOne(state, new QuestionSet()
            .Add("blocked", Question.Noul("Is someone blocked?"))
            .Add("urgent", Question.Noul("Is this urgent?")));

        // Each question is its own forward pass, so asking them in a different order must not move
        // a single probability.
        Assert.Equal(forward["urgent"].Noul!.Value, backward["urgent"].Noul!.Value, 6);
        Assert.Equal(forward["blocked"].Noul!.Value, backward["blocked"].Noul!.Value, 6);
    }

    [ModelFact]
    public void BatchingAQuestionSetChangesNothing()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("english"));
        const string state = "I was charged twice for invoice 4411 and I want my money back today.";
        var questions = Presets.Triage();

        var batched = agent.SystemOne(state, questions);
        foreach (var (id, question) in questions)
        {
            var alone = agent.SystemOne(state, new QuestionSet().Add(id, question))[id];
            var together = batched[id];

            // The whole point of concatenating sequences rather than padding them is that nothing
            // crosses between questions; if it did, this is where it would show.
            Assert.Equal(alone.Choice, together.Choice);
            Assert.Equal(alone.Confidence, together.Confidence, 4);
            Assert.Equal(alone.Noul ?? 0, together.Noul ?? 0, 4);
            Assert.Equal(alone.Score ?? 0, together.Score ?? 0, 3);
            Assert.Equal(alone.Action.ActProbability, together.Action.ActProbability, 4);
        }
    }

    [ModelFact]
    public void EmptyQuestionSetsAreAnsweredWithNothing()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("english"));
        var result = agent.SystemOne("anything", new QuestionSet());
        Assert.Empty(result.Answers);
        Assert.Equal(0, result.Usage.InputTokens);
    }

    [ModelFact("multilingual")]
    public void TheMultilingualCheckpointReadsNonEnglishBilling()
    {
        using var agent = Agent.FromDirectory(TestModels.CheckpointDirectory("multilingual"));
        var questions = new QuestionSet()
            .Add("dept", Question.Choice("Which team should handle `message`?",
                ("billing", "invoices, payments, refunds"),
                ("technical", "bugs, outages, integrations"),
                ("sales", "pricing, demos, new purchases"),
                ("hr", "hiring, leave, payroll")));

        (string Language, string Text)[] cases =
        [
            ("english", "I was charged twice for invoice 4411, please refund it today."),
            ("german", "Ich wurde zweimal fuer Rechnung 4411 belastet, bitte erstatten Sie den Betrag."),
            ("french", "J'ai ete facture deux fois pour la facture 4411, remboursez-moi s'il vous plait."),
            ("spanish", "Me cobraron dos veces la factura 4411, por favor devuelvanme el dinero."),
            ("hindi", "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"),
            ("japanese", "請求書4411で二重に請求されました。返金してください。"),
            ("chinese", "发票4411被重复扣款，请退款。"),
            ("russian", "С меня дважды списали деньги по счёту 4411, верните деньги."),
        ];

        int correct = 0;
        foreach (var (language, text) in cases)
        {
            var answer = agent.SystemOne(
                new List<KeyValuePair<string, object?>> { new("message", text) }, questions)["dept"];
            if (answer.Choice == "billing") correct++;
            output.WriteLine($"{language,-9} {answer.Choice} ({answer.ProbabilityOf(answer.Choice!):F2})");
        }

        // The Python end-to-end suite holds this checkpoint to 6/8 on the same set.
        Assert.True(correct >= 6, $"only {correct}/8 billing intents were recognised");
    }

    [ModelFact]
    public void RouterRunsThroughToAnAnswerAndRecordsTheDecision()
    {
        var router = new Router(models:
        [
            new KeyValuePair<string, ModelSpec>("english", TestModels.CheckpointDirectory("english")),
        ]);

        var result = router.Predict("I was charged twice, please refund.", Presets.Triage());
        Assert.NotNull(result.Routing);
        Assert.Equal("english", result.Routing!.Model);
        Assert.Equal(Presets.Triage().Count, result.Answers.Count);
        router.Dispose();
    }
}

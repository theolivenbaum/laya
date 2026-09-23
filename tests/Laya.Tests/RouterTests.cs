using Laya.Runtime;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// Routing decisions and the LRU bookkeeping. No weights are loaded: <see cref="Router.Route"/> is
/// pure, and <see cref="Router.EngineFactory"/> stands in for the real checkpoints everywhere else.
/// Ported from <c>.reference/tests/test_router.py</c>.
/// </summary>
public class RouterTests
{
    private static QuestionSet Generic() => new QuestionSet()
        .Add("dept", Question.Choice("Which team?", "billing", "tech"));

    private static readonly string[][] TypedDecisionWorkflows =
    [
        ["action", "needs_review", "outcome", "risk", "urgency"],
        ["action", "category", "churn_risk", "needs_human", "urgency"],
        ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
        ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
    ];

    private static QuestionSet Workflow(string[] ids)
    {
        var questions = new QuestionSet();
        foreach (string id in ids) questions.Add(id, Question.Noul("x"));
        return questions;
    }

    /// <summary>A checkpoint that loads instantly and answers nothing.</summary>
    private sealed class StubEngine(string name) : IDecisionEngine
    {
        public string Name => name;
        public bool Disposed { get; private set; }

        public DecisionResult SystemOne(object? state, QuestionSet questions) => new()
        {
            Model = name,
            Answers = [],
            Usage = new Usage(0, 0),
        };

        public void Dispose() => Disposed = true;
    }

    private static Router Stubbed(int maxLoaded = 1, bool autoTaskDetection = false)
        => new(maxLoaded: maxLoaded, autoTaskDetection: autoTaskDetection)
        {
            EngineFactory = (name, _) => new StubEngine(name),
        };

    [Theory]
    [InlineData("en", "english")]
    [InlineData("laya", "english")]
    [InlineData("English", "english")]
    [InlineData("multi", "multilingual")]
    [InlineData("ML", "multilingual")]
    [InlineData("typed", "typed-decisions")]
    [InlineData("typed_decisions", "typed-decisions")]
    [InlineData("decisions", "typed-decisions")]
    public void NormalisesAliases(string alias, string expected)
        => Assert.Equal(expected, Router.Normalise(alias));

    [Fact]
    public void UnknownNameThrows()
        => Assert.Throws<ArgumentException>(() => Router.Normalise("nope"));

    [Fact]
    public void MatchesTypedDecisionWorkflowsExactly()
    {
        string[] expected =
            ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"];
        for (int i = 0; i < TypedDecisionWorkflows.Length; ++i)
        {
            Assert.Equal(expected[i], Router.MatchTypedDecisionsWorkflow(Workflow(TypedDecisionWorkflows[i])));
        }

        Assert.Null(Router.MatchTypedDecisionsWorkflow(new QuestionSet()
            .Add("urgency", Question.Noul("x")).Add("category", Question.Noul("x"))));
        Assert.Null(Router.MatchTypedDecisionsWorkflow(Workflow([.. TypedDecisionWorkflows[1], "extra"])));
        Assert.Null(Router.MatchTypedDecisionsWorkflow(new QuestionSet()));
        Assert.Null(Router.MatchTypedDecisionsWorkflow(null));
    }

    [Fact]
    public void RoutesOnScriptAndLanguage()
    {
        var router = new Router();
        var questions = Generic();

        Assert.Equal("english", router.Route(new Dictionary<string, object?> { ["body"] = "I was charged twice, please refund." }, questions).Model);
        Assert.Equal("multilingual", router.Route(new Dictionary<string, object?> { ["body"] = "मुझसे दो बार शुल्क लिया गया" }, questions).Model);
        Assert.Equal("multilingual", router.Route(new Dictionary<string, object?> { ["body"] = "二重に請求されました" }, questions).Model);
        Assert.Equal("multilingual", router.Route(new Dictionary<string, object?> { ["body"] = "두 번 청구되었습니다" }, questions).Model);
        Assert.Equal("multilingual", router.Route(new Dictionary<string, object?> { ["body"] = "تم خصم المبلغ مرتين" }, questions).Model);
        Assert.Equal("multilingual", router.Route(new Dictionary<string, object?>
        {
            ["body"] = "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung die nicht korrekt ist",
        }, questions).Model);

        Assert.Equal("english", router.Route(new Dictionary<string, object?>(), questions).Model);
        Assert.Equal("english", router.Route(null, questions).Model);
    }

    [Fact]
    public void ExplicitOverridesWin()
    {
        var router = new Router();
        var questions = Generic();
        var hindi = new Dictionary<string, object?> { ["body"] = "मुझसे दो बार" };

        Assert.Equal("multilingual", router.Route(hindi, questions, model: "multilingual").Model);
        Assert.Equal("english", router.Route(hindi, questions, model: "english").Model);
        Assert.Equal("typed-decisions", router.Route(hindi, questions, task: "typed_decisions").Model);
        Assert.Equal("english", router.Route(hindi, questions, lang: "en").Model);
        Assert.Equal("multilingual", router.Route("hello there", questions, lang: "de").Model);
    }

    [Fact]
    public void TypedDecisionDetectionIsOptIn()
    {
        var workflow = Workflow(TypedDecisionWorkflows[1]);
        var state = new Dictionary<string, object?> { ["body"] = "I was charged twice" };

        Assert.Equal("english", new Router().Route(state, workflow).Model);

        var auto = new Router(autoTaskDetection: true);
        Assert.Equal("typed-decisions", auto.Route(state, workflow).Model);
        Assert.Equal("english", auto.Route(state, Generic()).Model);
        Assert.Equal("multilingual", auto.Route(state, workflow, model: "multilingual").Model);
    }

    [Fact]
    public void DecisionCarriesRepoReasonAndDetection()
    {
        var decision = new Router().Route(new Dictionary<string, object?> { ["body"] = "मुझसे दो बार शुल्क लिया गया" }, Generic());
        Assert.Equal("convaiinnovations/laya/multilingual", decision.Repo);
        Assert.False(string.IsNullOrEmpty(decision.Reason));
        Assert.Equal("devanagari", decision.Detection!.Script);
        Assert.Equal("multilingual", decision.Model);
    }

    [Fact]
    public void DefaultCanBeOverridden()
        => Assert.Equal("multilingual", new Router(@default: "multilingual").Route("12345", Generic()).Model);

    [Fact]
    public void LruKeepsTheNewest()
    {
        var router = Stubbed(maxLoaded: 1);
        router.Load("english");
        router.Load("multilingual");
        Assert.Equal(["multilingual"], router.Loaded);
    }

    [Fact]
    public void LruEvictsTheOldest()
    {
        var router = Stubbed(maxLoaded: 2);
        router.Load("english");
        router.Load("multilingual");
        router.Load("typed-decisions");
        Assert.Equal(["multilingual", "typed-decisions"], router.Loaded);
    }

    [Fact]
    public void TouchingProtectsFromEviction()
    {
        var router = Stubbed(maxLoaded: 2);
        router.Load("english");
        router.Load("multilingual");
        router.Load("english");                 // touch
        router.Load("typed-decisions");
        Assert.Equal(["english", "typed-decisions"], router.Loaded.Order());
    }

    [Fact]
    public void UnloadClearsOneOrAll()
    {
        var router = Stubbed(maxLoaded: 2);
        router.Load("english");
        router.Load("multilingual");
        router.Unload("english");
        Assert.DoesNotContain("english", router.Loaded);
        router.Unload();
        Assert.Empty(router.Loaded);
    }

    [Fact]
    public void EvictionDisposesWhatItOwns()
    {
        var router = Stubbed(maxLoaded: 1);
        var first = (StubEngine)router.Load("english");
        router.Load("multilingual");
        Assert.True(first.Disposed);
    }

    [Fact]
    public void PreloadRaisesTheCapSoNothingIsEvicted()
    {
        var router = Stubbed(maxLoaded: 1);
        router.Preload();
        Assert.Equal(["english", "multilingual", "typed-decisions"], router.Loaded.Order());
        Assert.True(router.MaxLoaded >= 3);

        var subset = Stubbed(maxLoaded: 1).Preload(["english", "multilingual"]);
        Assert.Equal(["english", "multilingual"], subset.Loaded.Order());
        subset.Load("english");
        Assert.Equal(["english", "multilingual"], subset.Loaded.Order());
    }

    [Fact]
    public void AttachRegistersAnExistingEngineAndNeverDisposesIt()
    {
        var router = Stubbed(maxLoaded: 1);
        var sentinel = new StubEngine("already-built");
        router.Attach("english", sentinel);

        Assert.Contains("english", router.Loaded);
        Assert.True(router.MaxLoaded >= 1);
        Assert.Same(sentinel, router.Load("english"));

        // At cap 1 an attached engine is evicted like any other, so make room first — the same
        // thing the Python test does before loading a second checkpoint.
        router.MaxLoaded = 2;
        router.Load("multilingual");
        Assert.Equal(["english", "multilingual"], router.Loaded.Order());
        Assert.Same(sentinel, router.Load("english"));

        router.Unload();
        Assert.False(sentinel.Disposed);        // the caller still owns it
    }

    [Fact]
    public void AttachAcceptsAliases()
        => Assert.NotNull(Stubbed().Attach("en", new StubEngine("x")));

    [Fact]
    public void BundleAndStandaloneMapsAgree()
    {
        Assert.Equal(new ModelSpec(Router.BundleRepo, null), Router.DefaultModels["english"]);
        Assert.Equal(new ModelSpec(Router.BundleRepo, "multilingual"), Router.DefaultModels["multilingual"]);
        Assert.Equal(new ModelSpec(Router.BundleRepo, "typed-decisions"), Router.DefaultModels["typed-decisions"]);
        Assert.Equal(Router.DefaultModels.Keys.Order(), Router.StandaloneModels.Keys.Order());

        Assert.Equal("convaiinnovations/laya", new ModelSpec(Router.BundleRepo, null).ToString());
        Assert.Equal("convaiinnovations/laya/multilingual", new ModelSpec(Router.BundleRepo, "multilingual").ToString());
        Assert.Equal("some/repo", ((ModelSpec)"some/repo").ToString());
    }

    [Fact]
    public void StandaloneReposAreOptIn()
    {
        var hindi = new Dictionary<string, object?> { ["m"] = "मुझसे दो बार" };
        Assert.Equal("convaiinnovations/laya/multilingual", new Router().Route(hindi, Generic()).Repo);

        var standalone = new Router(standaloneRepos: true);
        Assert.Equal("convaiinnovations/laya-multilingual", standalone.Route(hindi, Generic()).Repo);
        Assert.Equal("convaiinnovations/laya", standalone.Route(
            new Dictionary<string, object?> { ["m"] = "I was charged twice" }, Generic()).Repo);
    }

    [Fact]
    public void LocalPathOverridesAreKept()
    {
        var router = new Router(models:
        [
            new KeyValuePair<string, ModelSpec>("english", "/tmp/en"),
            new KeyValuePair<string, ModelSpec>("multilingual", "/tmp/ml"),
        ]);
        Assert.Equal("/tmp/ml", router.Route(new Dictionary<string, object?> { ["m"] = "मुझसे दो बार" }, Generic()).Repo);
    }
}

/// <summary>Model lifecycle under concurrency (upstream #95).</summary>
public class RouterConcurrencyTests
{
    private sealed class SlowEngine : IDecisionEngine
    {
        public SlowEngine() => Thread.Sleep(50);    // widen the check-then-build window

        public DecisionResult SystemOne(object? state, QuestionSet questions)
            => new() { Model = "slow", Answers = [], Usage = new Usage(0, 0) };

        public void Dispose() { }
    }

    [Fact]
    public void ConcurrentLoadsShareOneEngine()
    {
        int built = 0;
        var router = new Router
        {
            EngineFactory = (_, _) =>
            {
                Interlocked.Increment(ref built);
                return new SlowEngine();
            },
        };

        var engines = new IDecisionEngine[8];
        Parallel.For(0, engines.Length, new ParallelOptions { MaxDegreeOfParallelism = 8 },
            i => engines[i] = router.Load("english"));

        Assert.Single(engines.Distinct());
        Assert.Equal(1, built);
        Assert.Equal(["english"], router.Loaded);
    }

    [Fact]
    public void HotPathLoadsKeepTheLruConsistent()
    {
        var router = new Router(maxLoaded: 3) { EngineFactory = (_, _) => new SlowEngine() };
        router.Load("english");
        Parallel.For(0, 20, _ => router.Load("en"));
        Assert.Equal(["english"], router.Loaded);
    }
}

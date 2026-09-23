using System.Globalization;
using System.Text.Json.Serialization;
using Laya.Io;
using Laya.Runtime;

namespace Laya;

/// <summary>The routing outcome: which checkpoint, why, and what was detected.</summary>
public sealed record RouteDecision
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("repo")]
    public required string Repo { get; init; }

    [JsonPropertyName("reason")]
    public required string Reason { get; init; }

    [JsonPropertyName("detection")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LanguageAnalysis? Detection { get; init; }

    [JsonPropertyName("workflow")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Workflow { get; init; }

    public override string ToString() => $"RouteDecision(model='{Model}', reason='{Reason}')";
}

/// <summary>Where one checkpoint lives: a repo id (or local path) and an optional subfolder.</summary>
public readonly record struct ModelSpec(string Repo, string? Subfolder)
{
    public override string ToString() => Subfolder is null ? Repo : $"{Repo}/{Subfolder}";

    /// <summary>A bare repo id or local directory, with no subfolder.</summary>
    public static implicit operator ModelSpec(string repo) => new(repo, null);
}

/// <summary>
/// Lazily loads Laya checkpoints and sends each request to the one best suited to it.
///
/// <para>Routing is driven by script detection, because the English checkpoint does not degrade
/// gently off English — it collapses, confidently. Loading a checkpoint costs seconds while
/// detection costs microseconds, so anything that alternates languages should
/// <see cref="Preload"/> rather than pay an eviction on every request.</para>
/// </summary>
public sealed class Router : IDisposable
{
    /// <summary>The hub repo that bundles all three checkpoints; only the requested one is downloaded.</summary>
    public const string BundleRepo = "convaiinnovations/laya";

    public static IReadOnlyDictionary<string, ModelSpec> DefaultModels { get; } =
        new Dictionary<string, ModelSpec>(StringComparer.Ordinal)
        {
            ["english"] = new(BundleRepo, null),
            ["multilingual"] = new(BundleRepo, "multilingual"),
            ["typed-decisions"] = new(BundleRepo, "typed-decisions"),
        };

    /// <summary>The same checkpoints in their own repos, for anyone who prefers them.</summary>
    public static IReadOnlyDictionary<string, ModelSpec> StandaloneModels { get; } =
        new Dictionary<string, ModelSpec>(StringComparer.Ordinal)
        {
            ["english"] = new("convaiinnovations/laya", null),
            ["multilingual"] = new("convaiinnovations/laya-multilingual", null),
            ["typed-decisions"] = new("convaiinnovations/laya-typed-decisions", null),
        };

    private static readonly Dictionary<string, string> Aliases = new(StringComparer.Ordinal)
    {
        ["en"] = "english", ["laya"] = "english", ["default"] = "english",
        ["multi"] = "multilingual", ["ml"] = "multilingual", ["laya-multilingual"] = "multilingual",
        ["typed"] = "typed-decisions", ["typed_decisions"] = "typed-decisions",
        ["laya-typed-decisions"] = "typed-decisions", ["decisions"] = "typed-decisions",
    };

    /// <summary>
    /// Question-id signatures of the four typed-decisions workflows. Used only when
    /// <see cref="AutoTaskDetection"/> is on, because that checkpoint is fine-tuned on those
    /// specific workflows and should not become a silent default.
    /// </summary>
    private static readonly (string Workflow, string[] Ids)[] TypedDecisionWorkflows =
    [
        ("agent_trace_observability", ["action", "needs_review", "outcome", "risk", "urgency"]),
        ("customer_service", ["action", "category", "churn_risk", "needs_human", "urgency"]),
        ("invoice_processing", ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
        ("security_incidents", ["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
    ];

    private readonly Dictionary<string, ModelSpec> _models;
    private readonly Dictionary<string, IDecisionEngine> _agents = new(StringComparer.Ordinal);
    private readonly List<string> _order = [];          // least-recently-used first
    private readonly HashSet<string> _attached = new(StringComparer.Ordinal);
    private readonly string? _token;
    private readonly string? _cacheRoot;

    // Guards the model lifecycle (load, evict, attach, preload, unload) and the LRU bookkeeping.
    // Inference is deliberately outside it, so concurrent predictions share a checkpoint without
    // serialising; a load holds it while building, so concurrent first calls build one agent, not eight.
    private readonly Lock _gate = new();
    private int _maxLoaded = 1;

    /// <summary>
    /// How many checkpoints stay resident; the least recently used is evicted beyond that. The
    /// default of 1 exists because all three together are ~1.16B parameters — raise it (or call
    /// <see cref="Preload"/>) when a server alternates between languages.
    /// </summary>
    public int MaxLoaded
    {
        get => _maxLoaded;
        set
        {
            lock (_gate)
            {
                _maxLoaded = Math.Max(1, value);
                Evict();
            }
        }
    }
    public string Default { get; }
    public bool AutoTaskDetection { get; }
    public IProgress<DownloadProgress>? DownloadProgress { get; init; }

    /// <summary>
    /// How a checkpoint is built. Defaults to downloading and loading a real <see cref="Agent"/>;
    /// tests substitute a stub so the LRU can be exercised without any weights.
    /// </summary>
    public Func<string, ModelSpec, IDecisionEngine>? EngineFactory { get; init; }

    /// <summary>
    /// A statistical language identifier consulted when the built-in heuristic cannot name a
    /// Latin-script language (see <see cref="ILanguageClassifier"/>). Null keeps routing exactly as
    /// the Python package does it.
    /// </summary>
    public ILanguageClassifier? LanguageClassifier { get; init; }

    public Router(IEnumerable<KeyValuePair<string, ModelSpec>>? models = null, string? token = null,
        int maxLoaded = 1, string @default = "english", bool autoTaskDetection = false,
        bool standaloneRepos = false, bool preload = false, string? cacheRoot = null)
    {
        _models = new Dictionary<string, ModelSpec>(standaloneRepos ? StandaloneModels : DefaultModels, StringComparer.Ordinal);
        if (models is not null)
        {
            foreach (var (name, spec) in models) _models[Normalise(name)] = spec;
        }
        _token = token ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        _cacheRoot = cacheRoot;
        MaxLoaded = Math.Max(1, maxLoaded);
        Default = Normalise(@default);
        AutoTaskDetection = autoTaskDetection;
        if (preload) Preload();
    }

    /// <summary>Resolves an alias or a checkpoint name to its canonical name.</summary>
    public static string Normalise(string name)
    {
        string key = name.Trim().ToLowerInvariant();
        if (Aliases.TryGetValue(key, out string? alias)) key = alias;
        if (!DefaultModels.ContainsKey(key))
        {
            throw new ArgumentException(
                $"unknown model '{name}'; choose one of {string.Join(", ", DefaultModels.Keys.Order())} " +
                $"(or an alias: {string.Join(", ", Aliases.Keys.Order())}).", nameof(name));
        }
        return key;
    }

    /// <summary>
    /// The typed-decisions workflow whose question ids these are, else null. The match has to be
    /// exact, so an unrelated schema that happens to contain "urgency" is never captured.
    /// </summary>
    public static string? MatchTypedDecisionsWorkflow(QuestionSet? questions)
    {
        if (questions is null || questions.Count == 0) return null;
        var ids = new HashSet<string>(questions.Ids, StringComparer.Ordinal);
        foreach (var (workflow, signature) in TypedDecisionWorkflows)
        {
            if (ids.Count == signature.Length && signature.All(ids.Contains)) return workflow;
        }
        return null;
    }

    /// <summary>The resident checkpoints, least recently used first. A snapshot, safe to enumerate.</summary>
    public IReadOnlyList<string> Loaded
    {
        get
        {
            lock (_gate) return [.. _order];
        }
    }

    /// <summary>Returns the engine for a checkpoint, downloading and building it on first use.</summary>
    public IDecisionEngine Load(string name)
    {
        string key = Normalise(name);
        lock (_gate)
        {
            if (_agents.TryGetValue(key, out var existing))
            {
                Touch(key);
                return existing;
            }

            var spec = _models[key];
            var agent = EngineFactory is not null
                ? EngineFactory(key, spec)
                : Agent.Load(spec.Repo, spec.Subfolder, _token, _cacheRoot, DownloadProgress);
            _agents[key] = agent;
            _order.Add(key);
            Evict();
            return agent;
        }
    }

    private void Touch(string key)
    {
        _order.Remove(key);
        _order.Add(key);
    }

    private void Evict()
    {
        while (_order.Count > MaxLoaded)
        {
            string victim = _order[0];
            _order.RemoveAt(0);
            if (_agents.Remove(victim, out var agent) && !_attached.Remove(victim)) agent.Dispose();
        }
    }

    /// <summary>
    /// Registers an already-built agent instead of loading a second copy. The router does not
    /// dispose an attached agent — whoever built it still owns it.
    /// </summary>
    public IDecisionEngine Attach(string name, IDecisionEngine agent)
    {
        string key = Normalise(name);
        lock (_gate)
        {
            if (_agents.Remove(key, out var previous) && !_attached.Contains(key) && !ReferenceEquals(previous, agent))
            {
                previous.Dispose();
            }
            _agents[key] = agent;
            _attached.Add(key);
            Touch(key);
            MaxLoaded = Math.Max(MaxLoaded, _agents.Count);
            return agent;
        }
    }

    /// <summary>
    /// Downloads and builds checkpoints up front so no request ever pays a model load.
    /// <see cref="MaxLoaded"/> is raised to fit whatever is preloaded, otherwise the LRU would
    /// immediately evict what this just built.
    /// </summary>
    public Router Preload(IEnumerable<string>? names = null)
    {
        var wanted = (names ?? _models.Keys).Select(Normalise).ToList();
        lock (_gate)
        {
            MaxLoaded = Math.Max(MaxLoaded, Math.Max(wanted.Count, _agents.Count));
            foreach (string name in wanted)
            {
                if (!_agents.ContainsKey(name)) Load(name);
            }
        }
        return this;
    }

    /// <summary>Frees one checkpoint, or all of them.</summary>
    public void Unload(string? name = null)
    {
        string? normalised = name is null ? null : Normalise(name);
        lock (_gate)
        {
            if (normalised is null)
            {
                foreach (var (key, agent) in _agents)
                {
                    if (!_attached.Contains(key)) agent.Dispose();
                }
                _agents.Clear();
                _order.Clear();
                _attached.Clear();
                return;
            }

            if (_agents.Remove(normalised, out var loaded) && !_attached.Remove(normalised)) loaded.Dispose();
            _order.Remove(normalised);
        }
    }

    /// <summary>
    /// Decides which checkpoint to use without loading or running anything.
    ///
    /// <para>Precedence: explicit model, explicit task, detected workflow (opt-in), explicit
    /// language, detected script/language, then the default.</para>
    /// </summary>
    public RouteDecision Route(object? state, QuestionSet? questions = null, string? model = null,
        string? task = null, string? lang = null)
    {
        if (model is not null)
        {
            string key = Normalise(model);
            return new RouteDecision { Model = key, Repo = _models[key].ToString(), Reason = $"explicit model='{model}'" };
        }

        if (task is not null)
        {
            string normalisedTask = task.ToLowerInvariant().Replace('-', '_');
            string key = Normalise(normalisedTask == "typed_decisions" ? "typed-decisions" : task);
            return new RouteDecision { Model = key, Repo = _models[key].ToString(), Reason = $"explicit task='{task}'" };
        }

        string? workflow = MatchTypedDecisionsWorkflow(questions);
        if (workflow is not null && AutoTaskDetection)
        {
            return new RouteDecision
            {
                Model = "typed-decisions",
                Repo = _models["typed-decisions"].ToString(),
                Reason = $"question ids match the '{workflow}' typed-decisions workflow",
                Workflow = workflow,
            };
        }

        if (lang is not null)
        {
            string prefix = lang.ToLowerInvariant().Split('-')[0];
            string key = prefix is "en" or "eng" or "english" ? "english" : "multilingual";
            return new RouteDecision
            {
                Model = key,
                Repo = _models[key].ToString(),
                Reason = $"explicit lang='{lang}'",
                Workflow = workflow,
            };
        }

        var detection = LanguageDetector.Analyse(state, LanguageClassifier);
        string chosen;
        string reason;
        if (detection.Script == "unknown")
        {
            chosen = Default;
            reason = $"no letters detected in state; using default ({chosen})";
        }
        else if (detection.Script != "latin")
        {
            chosen = "multilingual";
            reason = string.Format(CultureInfo.InvariantCulture,
                "non-Latin script ({0}, {1:F0}% of letters); the English checkpoint cannot read it",
                detection.Script, 100 * detection.NonLatinFraction);
        }
        else if (!detection.IsEnglish)
        {
            chosen = "multilingual";
            if (detection.Language is not null)
            {
                reason = $"Latin script but language looks like '{detection.Language}', not English";
            }
            else if (detection.ClassifierLanguage is not null && detection.ClassifierLanguage != "en")
            {
                reason = string.Format(CultureInfo.InvariantCulture,
                    "Latin script, no stopword evidence, but the language classifier says '{0}' (p={1:F2}); " +
                    "not safe for the English checkpoint", detection.ClassifierLanguage, detection.ClassifierProbability);
            }
            else
            {
                // Unidentified Latin-script language: routed on the non-English letters alone.
                reason = string.Format(CultureInfo.InvariantCulture,
                    "Latin script, language not identified but {0:F0}% non-English letters; " +
                    "not safe for the English checkpoint", 100 * detection.DiacriticRate);
            }
        }
        else
        {
            chosen = "english";
            reason = "English Latin text";
        }

        return new RouteDecision
        {
            Model = chosen,
            Repo = _models[chosen].ToString(),
            Reason = reason,
            Detection = detection,
            Workflow = workflow,
        };
    }

    /// <summary>Routes, then answers every question on the chosen checkpoint.</summary>
    /// <param name="parallel">The threads the pass may use; null means <see cref="LayaRuntime.ParallelOptions"/>.</param>
    public DecisionResult Predict(object? state, QuestionSet questions, string? model = null,
        string? task = null, string? lang = null, ParallelOptions? parallel = null)
    {
        var decision = Route(state, questions, model, task, lang);
        var agent = Load(decision.Model);
        var result = agent.SystemOne(state, questions, parallel);
        return result with { Routing = decision };
    }

    /// <summary>Alias matching the Python <c>system_one</c>.</summary>
    public DecisionResult SystemOne(object? state, QuestionSet questions, string? model = null,
        string? task = null, string? lang = null, ParallelOptions? parallel = null)
        => Predict(state, questions, model, task, lang, parallel);

    public override string ToString()
        => $"Router(loaded=[{string.Join(", ", Loaded)}], maxLoaded={MaxLoaded}, default='{Default}')";

    public void Dispose() => Unload();
}

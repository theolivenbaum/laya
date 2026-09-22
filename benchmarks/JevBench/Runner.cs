using System.Diagnostics;
using Laya.JevBench.Dataset;
using Laya.JevBench.Reporting;
using Laya.JevBench.Scoring;
using Laya.Runtime;

namespace Laya.JevBench;

/// <summary>
/// Runs one <see cref="JevTask"/> through laya's own <c>Agent</c> and scores the answer — the C#,
/// in-process equivalent of <c>jevbench/adapters/laya_local.py</c> (which does the same thing by
/// shelling out to the Python <c>laya</c> package) plus <c>jevbench/runner.py</c>'s per-item scoring
/// (minus the budget ledger and raw-response archiving, which exist there to gate paid API calls;
/// nothing here is billed).
/// </summary>
public static class Runner
{
    /// <summary>
    /// Builds the single "decision" question exactly as <c>adapters/base.py:build_question</c> does,
    /// runs it through the agent, and maps the answer back to the label-keyed probability map every
    /// JevBench record is scored against — laya's own three question types already answer in
    /// JevBench's shape (native, from the option-marker softmax), so the mapping is 1:1:
    /// noul -&gt; {"yes": p, "no": 1-p}; choice/score -&gt; probabilities, used as-is.
    /// </summary>
    public static TaskResult Run(Agent agent, JevTask task, ParallelOptions? parallel)
    {
        var question = QuestionBuilder.Build(task.Question);
        var questions = new QuestionSet().Add("decision", question);

        var stopwatch = Stopwatch.StartNew();
        DecisionResult result;
        try
        {
            result = agent.Predict(task.State, questions, parallel);
        }
        catch (Exception e)
        {
            stopwatch.Stop();
            return new TaskResult
            {
                Id = task.Id, Family = task.Family, Tier = task.Tier, Group = task.Group,
                Ok = false, Error = $"{e.GetType().Name}: {e.Message}",
                Scored = new ScoredOutcome { Valid = false, StrictValid = false, Renormalized = false, Correct = false, Predicted = null },
                LatencyS = stopwatch.Elapsed.TotalSeconds, InputTokens = 0,
            };
        }
        stopwatch.Stop();

        if (!result.TryGet("decision", out var answer) || answer.Type != task.Question.Type)
        {
            return new TaskResult
            {
                Id = task.Id, Family = task.Family, Tier = task.Tier, Group = task.Group,
                Ok = false, Error = "missing or mistyped answers.decision",
                Scored = new ScoredOutcome { Valid = false, StrictValid = false, Renormalized = false, Correct = false, Predicted = null },
                LatencyS = stopwatch.Elapsed.TotalSeconds, InputTokens = result.Usage.InputTokens,
            };
        }

        var probs = new Dictionary<string, double>(StringComparer.Ordinal);
        if (task.Question.Type == "noul")
        {
            double p = answer.Noul ?? throw new InvalidOperationException("noul answer has no probability");
            probs["yes"] = p;
            probs["no"] = 1.0 - p;
        }
        else
        {
            foreach (var (key, value) in answer.Probabilities ?? []) probs[key] = value;
        }

        var scored = Scoring.Scoring.ScoreTask(probs, task);
        return new TaskResult
        {
            Id = task.Id, Family = task.Family, Tier = task.Tier, Group = task.Group,
            Ok = true, Error = scored.Error, Scored = scored,
            LatencyS = stopwatch.Elapsed.TotalSeconds, InputTokens = result.Usage.InputTokens,
        };
    }
}

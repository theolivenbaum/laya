using System.Text.Json;
using Laya.JevBench.Dataset;

namespace Laya.JevBench.Reporting;

/// <summary>One task's baseline outcome, read from the copied <c>data/baseline/laya-per-task.json</c>.</summary>
public sealed record BaselineTask(string Outcome, double LatencyS);

/// <summary>
/// The previously published Laya row (JevBench v1.2.2, <c>results/v1.2/additions/laya.json</c> +
/// <c>laya-per-task.json</c>) restricted to the public items this tool can also run, so a fresh run
/// through the .NET port can be compared item-for-item against the original Python-package run.
///
/// <para><b>Deliberately does not surface <c>laya-per-task.json</c>'s own <c>by_tier</c> block</b>:
/// that block is accuracy over the *full* v1.2 tier (all 534 decisions, most held out — e.g.
/// "standard" there is 96 items, "hard" is 220), not the 231 public ones this tool can run. Diffing
/// this tool's public-only accuracy against that would silently compare different item sets.
/// <see cref="TierAccuracy"/> instead recomputes each tier's accuracy from only the public items in
/// <c>public_tasks</c>, using this run's own tier assignment (<see cref="JevTask.Tier"/>) so the two
/// numbers are the same item set by construction.</para>
/// </summary>
public sealed class Baseline
{
    public required IReadOnlyDictionary<string, double> TierAccuracy { get; init; }
    public required IReadOnlyDictionary<string, BaselineTask> Tasks { get; init; }
    public required string Display { get; init; }

    public static Baseline Load(string perTaskPath, string summaryPath, IReadOnlyDictionary<string, JevTask> tasksById)
    {
        using var perTaskDoc = JsonDocument.Parse(File.ReadAllText(perTaskPath));
        using var summaryDoc = JsonDocument.Parse(File.ReadAllText(summaryPath));

        var tasks = new Dictionary<string, BaselineTask>(StringComparer.Ordinal);
        foreach (var task in perTaskDoc.RootElement.GetProperty("public_tasks").EnumerateObject())
        {
            var arr = task.Value;
            tasks[task.Name] = new BaselineTask(arr[0].GetString()!, arr[1].GetDouble());
        }

        var byTier = new Dictionary<string, (int Correct, int Scorable)>(StringComparer.Ordinal);
        foreach (var (id, outcome) in tasks)
        {
            if (!tasksById.TryGetValue(id, out var task)) continue; // outside the tiers this run loaded
            if (outcome.Outcome == "n") continue; // no ground truth: excluded from accuracy, as in the Python scorer
            var (correct, scorable) = byTier.GetValueOrDefault(task.Tier);
            byTier[task.Tier] = (correct + (outcome.Outcome == "c" ? 1 : 0), scorable + 1);
        }
        var tierAccuracy = byTier.ToDictionary(kv => kv.Key, kv => kv.Value.Scorable > 0 ? (double)kv.Value.Correct / kv.Value.Scorable : 0.0,
            StringComparer.Ordinal);

        string display = summaryDoc.RootElement.GetProperty("display").GetString() ?? "Laya (baseline)";
        return new Baseline { TierAccuracy = tierAccuracy, Tasks = tasks, Display = display };
    }
}

public sealed record TaskDelta(string Id, string Baseline, string Current, bool Regressed, bool Improved);

public static class BaselineComparison
{
    /// <summary>Per-task outcome-letter changes ("c"/"w"/"f"/"n") between the baseline run and this one.</summary>
    public static IReadOnlyList<TaskDelta> Diff(Baseline baseline, IReadOnlyList<TaskResult> results)
    {
        var deltas = new List<TaskDelta>();
        foreach (var r in results)
        {
            if (!baseline.Tasks.TryGetValue(r.Id, out var b)) continue;
            string current = r.Outcome;
            if (b.Outcome == current) continue;
            bool regressed = b.Outcome == "c" && current != "c";
            bool improved = b.Outcome != "c" && current == "c";
            deltas.Add(new TaskDelta(r.Id, b.Outcome, current, regressed, improved));
        }
        return deltas;
    }
}

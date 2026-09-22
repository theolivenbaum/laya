using Laya.JevBench.Scoring;

namespace Laya.JevBench.Reporting;

/// <summary>One task's run outcome — the C# analogue of one line of a JevBench <c>results.jsonl</c>.</summary>
public sealed record TaskResult
{
    public required string Id { get; init; }
    public required string Family { get; init; }
    public required string Tier { get; init; }
    public string? Group { get; init; }
    public required bool Ok { get; init; }
    public string? Error { get; init; }
    public required ScoredOutcome Scored { get; init; }
    public required double LatencyS { get; init; }
    public required int InputTokens { get; init; }

    /// <summary>"c" (correct) / "w" (wrong) / "f" (failed to answer) / "n" (no ground truth) — matches the baseline's per-task letter code.</summary>
    public string Outcome => !Ok || !Scored.Valid ? "f" : Scored.Correct switch
    {
        true => "c",
        false => "w",
        null => "n",
    };
}

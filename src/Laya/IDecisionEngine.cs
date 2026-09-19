using Laya.Runtime;

namespace Laya;

/// <summary>
/// What a <see cref="Router"/> needs from a checkpoint.
///
/// <para><see cref="Agent"/> is the only production implementation. The interface exists so that
/// the router's loading, eviction and attachment bookkeeping can be exercised without 421M
/// parameters of weights on disk.</para>
/// </summary>
public interface IDecisionEngine : IDisposable
{
    DecisionResult SystemOne(object? state, QuestionSet questions);
}

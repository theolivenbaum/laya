namespace StructuredDecisions;

/// <summary>
/// A model that answers typed questions about a state — <c>noul</c> (does it hold?), <c>choice</c> (which
/// option?) and <c>score</c> (which level?) — with a probability for every allowed answer.
///
/// <para>Every model family implements it: Laya's ModernBERT engine (<c>StructuredDecisions.Laya.Agent</c>,
/// one encoder pass for all questions) and Mica's Qwen3.5 decoder (<c>StructuredDecisions.Mica.MicaJudge</c>,
/// one prefill per question over a shared prefix). Code written against this interface — a router, a
/// shortlist, a cascade that asks the small model first and the large one when it is unsure — works with
/// either. Options a model cannot honour (say, <see cref="Question.Multi"/> on a model without multi-answer
/// support) are rejected with <see cref="NotSupportedException"/> rather than silently ignored.</para>
/// </summary>
public interface IDecisionModel : IDisposable
{
    /// <summary>Answers every question in <paramref name="questions"/> about <paramref name="state"/>.</summary>
    /// <param name="state">Text, or anything JSON-serialisable (a <c>JsonElement</c>, a dictionary, a record).</param>
    DecisionResult SystemOne(object? state, QuestionSet questions);

    /// <summary>
    /// The same, with the threads the pass may use. A model that does not schedule threads itself may
    /// leave the default, which ignores the options.
    /// </summary>
    DecisionResult SystemOne(object? state, QuestionSet questions, ParallelOptions? parallel)
        => SystemOne(state, questions);
}

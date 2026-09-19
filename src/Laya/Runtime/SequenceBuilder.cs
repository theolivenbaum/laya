using Laya.Tokenizers;

namespace Laya.Runtime;

/// <summary>A tokenized question: the input ids and where each option's <c>[MASK]</c> marker sits.</summary>
public sealed record BuiltSequence(int[] TokenIds, int[] MarkerPositions);

/// <summary>
/// Builds the input sequence the decision head expects:
/// <c>[CLS] &lt;type&gt; question: &lt;instructions&gt; [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]</c>.
///
/// <para>The budget arithmetic is a literal port of <c>build_sequence</c>: options get at most 48
/// tokens each, the instructions get whatever is left of <c>head_max_len</c> (at least 8 tokens),
/// and if the options alone overrun the head budget every option is trimmed to an equal share.</para>
/// </summary>
public static class SequenceBuilder
{
    private const int MaxOptionTokens = 48;

    public static BuiltSequence Build(HuggingFaceTokenizer tokenizer, object? state, Question question,
        int maxLength = 512, int headMaxLength = 192, IReadOnlyList<int>? optionOrder = null,
        bool truncateLeft = false)
    {
        string maskToken = tokenizer.MaskToken;
        var options = question.RenderOptions();
        var order = optionOrder ?? [.. Enumerable.Range(0, options.Count)];

        string instructions = question.InstructionText().Replace(maskToken, " ", StringComparison.Ordinal);
        var headIds = tokenizer.Encode($"{QuestionTypes.Name(question.Type)} question: {instructions}");

        var optionIds = new List<List<int>>(order.Count);
        foreach (int index in order)
        {
            var ids = new List<int>(MaxOptionTokens + 1) { tokenizer.MaskTokenId };
            var encoded = tokenizer.Encode(" " + options[index].Replace(maskToken, " ", StringComparison.Ordinal));
            ids.AddRange(encoded.Take(MaxOptionTokens));
            optionIds.Add(ids);
        }

        int optionBudget = headMaxLength - optionIds.Sum(o => o.Count);
        if (optionBudget < 16)
        {
            int perOption = Math.Max(4, (headMaxLength - 16) / Math.Max(1, optionIds.Count));
            for (int i = 0; i < optionIds.Count; ++i)
            {
                if (optionIds[i].Count > perOption) optionIds[i] = optionIds[i][..perOption];
            }
            optionBudget = headMaxLength - optionIds.Sum(o => o.Count);
        }

        int headBudget = Math.Max(8, optionBudget);
        if (headIds.Count > headBudget) headIds = headIds[..headBudget];

        var tokens = new List<int>(maxLength) { tokenizer.ClsTokenId };
        tokens.AddRange(headIds);
        tokens.Add(tokenizer.SepTokenId);

        var markers = new List<int>(optionIds.Count);
        foreach (var option in optionIds)
        {
            markers.Add(tokens.Count);
            tokens.AddRange(option);
        }
        tokens.Add(tokenizer.SepTokenId);

        int room = Math.Max(0, maxLength - tokens.Count - 1);
        var stateIds = tokenizer.Encode(SerializeState(state).Replace(maskToken, " ", StringComparison.Ordinal));
        if (truncateLeft)
        {
            // Python's `st[-room:]` keeps the whole list when room is 0, and the trailing
            // `ids[:max_len]` is what actually bounds the sequence. Reproduced rather than
            // "fixed", so the two implementations truncate identically.
            if (room > 0 && stateIds.Count > room) stateIds = stateIds[^room..];
        }
        else if (stateIds.Count > room)
        {
            stateIds = stateIds[..room];
        }
        tokens.AddRange(stateIds);
        tokens.Add(tokenizer.SepTokenId);

        if (tokens.Count > maxLength) tokens.RemoveRange(maxLength, tokens.Count - maxLength);
        return new BuiltSequence([.. tokens], [.. markers.Where(m => m < maxLength)]);
    }

    /// <summary>Strings pass through; dicts and lists become compact JSON. Port of <c>serialize_state</c>.</summary>
    public static string SerializeState(object? state)
        => state as string ?? PythonJson.Dumps(state);
}

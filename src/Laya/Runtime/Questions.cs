using System.Collections;

namespace Laya.Runtime;

/// <summary>The three question types the decision head is trained on.</summary>
public enum QuestionType
{
    /// <summary>Pick one of N named options.</summary>
    Choice = 0,
    /// <summary>Place the state on an ordered scale; the answer is the expected level.</summary>
    Score = 1,
    /// <summary>Does the statement hold? The answer is a calibrated probability.</summary>
    Noul = 2,
}

/// <summary>Names and indices shared with the Python implementation's <c>QTYPES</c>.</summary>
public static class QuestionTypes
{
    public static string Name(QuestionType type) => type switch
    {
        QuestionType.Choice => "choice",
        QuestionType.Score => "score",
        _ => "noul",
    };

    public static QuestionType Parse(string name) => name switch
    {
        "choice" => QuestionType.Choice,
        "score" => QuestionType.Score,
        "noul" => QuestionType.Noul,
        _ => throw new ArgumentException($"unknown question type '{name}'; expected choice, score or noul.", nameof(name)),
    };

    /// <summary>
    /// The calibration bucket a question falls into: the type plus how many options it has.
    /// Mirrors <c>temp_bucket</c>.
    /// </summary>
    public static string TemperatureBucket(QuestionType type, int options)
    {
        string size = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+";
        return Name(type) + ":" + size;
    }
}

/// <summary>
/// One typed question.
///
/// <para>Criteria keep their declaration order, because that order is the label order the model
/// was trained with: option 0 of a <c>choice</c> is whichever key was written first.</para>
/// </summary>
public sealed class Question
{
    public QuestionType Type { get; }

    /// <summary>Usually a string; anything else is JSON-encoded, as <c>_to_internal</c> does.</summary>
    public object Instructions { get; }

    /// <summary>Ordered options for <c>choice</c>, and the <c>true</c>/<c>false</c> texts for <c>noul</c>.</summary>
    public IReadOnlyList<KeyValuePair<string, object?>>? NamedCriteria { get; }

    /// <summary>Ordered levels for <c>score</c>.</summary>
    public IReadOnlyList<object?>? Levels { get; }

    private Question(QuestionType type, object instructions,
        IReadOnlyList<KeyValuePair<string, object?>>? namedCriteria, IReadOnlyList<object?>? levels)
    {
        Type = type;
        Instructions = instructions;
        NamedCriteria = namedCriteria;
        Levels = levels;
    }

    public static Question Choice(object instructions, params string[] options)
        => new(QuestionType.Choice, instructions,
            [.. options.Select(o => new KeyValuePair<string, object?>(o, null))], null);

    public static Question Choice(object instructions, params (string Option, object? Description)[] options)
        => new(QuestionType.Choice, instructions,
            [.. options.Select(o => new KeyValuePair<string, object?>(o.Option, o.Description))], null);

    public static Question Choice(object instructions, IEnumerable<KeyValuePair<string, object?>> options)
        => new(QuestionType.Choice, instructions, [.. options], null);

    public static Question Score(object instructions, params object?[] levels)
        => new(QuestionType.Score, instructions, null, [.. levels]);

    public static Question Noul(object instructions, object? trueCriterion = null, object? falseCriterion = null)
    {
        if (trueCriterion is null && falseCriterion is null)
        {
            return new Question(QuestionType.Noul, instructions, null, null);
        }
        return new Question(QuestionType.Noul, instructions,
        [
            new KeyValuePair<string, object?>("false", falseCriterion),
            new KeyValuePair<string, object?>("true", trueCriterion),
        ], null);
    }

    /// <summary>The option keys of a <c>choice</c> question, in label order.</summary>
    public IReadOnlyList<string> OptionKeys => NamedCriteria is null
        ? []
        : [.. NamedCriteria.Select(c => c.Key)];

    /// <summary>
    /// Renders the option texts in label order — the port of <c>render_options</c>.
    /// </summary>
    public IReadOnlyList<string> RenderOptions()
    {
        switch (Type)
        {
            case QuestionType.Choice:
            {
                if (NamedCriteria is null || NamedCriteria.Count == 0)
                {
                    throw new InvalidOperationException("a choice question needs at least one option.");
                }
                var rendered = new List<string>(NamedCriteria.Count);
                foreach (var (key, value) in NamedCriteria)
                {
                    // Only None/"" mean "no description"; 0 and false are legitimate criterion values.
                    rendered.Add(IsEmpty(value) ? key : key + ": " + RenderCriterion(value));
                }
                return rendered;
            }
            case QuestionType.Score:
            {
                if (Levels is null || Levels.Count == 0)
                {
                    throw new InvalidOperationException("a score question needs at least one level.");
                }
                var rendered = new List<string>(Levels.Count);
                for (int i = 0; i < Levels.Count; ++i)
                {
                    rendered.Add($"level {i}: {RenderCriterion(Levels[i])}");
                }
                return rendered;
            }
            default:
            {
                object? falseCriterion = Lookup("false");
                object? trueCriterion = Lookup("true");
                return
                [
                    "false: " + (IsEmpty(falseCriterion) ? "no, the statement does not hold" : RenderCriterion(falseCriterion)),
                    "true: " + (IsEmpty(trueCriterion) ? "yes, the statement holds" : RenderCriterion(trueCriterion)),
                ];
            }
        }
    }

    private object? Lookup(string key)
    {
        if (NamedCriteria is null) return null;
        foreach (var (name, value) in NamedCriteria)
        {
            if (name == key) return value;
        }
        return null;
    }

    private static bool IsEmpty(object? value) => value is null || (value is string s && s.Length == 0);

    /// <summary>
    /// One criterion as text: strings pass through, anything structured becomes compact JSON so a
    /// rubric reads as JSON rather than a language runtime's repr.
    /// </summary>
    public static string RenderCriterion(object? value)
        => value is string text ? text : PythonJson.Dumps(value);

    /// <summary>The instructions as a string, JSON-encoding anything that is not one.</summary>
    public string InstructionText()
        => Instructions as string ?? PythonJson.Dumps(Instructions, ensureAscii: true);
}

/// <summary>
/// An ordered set of question ids. Insertion order is preserved because it is the order answers
/// come back in — and, for the typed-decisions router, part of the workflow signature.
/// </summary>
public sealed class QuestionSet : IEnumerable<KeyValuePair<string, Question>>
{
    private readonly List<string> _order = [];
    private readonly Dictionary<string, Question> _questions = new(StringComparer.Ordinal);

    public QuestionSet() { }

    public QuestionSet(IEnumerable<KeyValuePair<string, Question>> questions)
    {
        foreach (var (id, question) in questions) Add(id, question);
    }

    public int Count => _order.Count;

    public IReadOnlyList<string> Ids => _order;

    public Question this[string id] => _questions[id];

    public QuestionSet Add(string id, Question question)
    {
        if (_questions.TryAdd(id, question)) _order.Add(id);
        else _questions[id] = question;
        return this;
    }

    public bool Contains(string id) => _questions.ContainsKey(id);

    public IEnumerator<KeyValuePair<string, Question>> GetEnumerator()
    {
        foreach (string id in _order) yield return new KeyValuePair<string, Question>(id, _questions[id]);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using Laya.Runtime;
using Laya.Tokenizers;

namespace Laya.D1;

/// <summary>
/// The d1 prompt and readout — a port of the checkpoint's own <c>prompt.py</c>.
///
/// <para>A System One decision is one forward pass that stops at the answer slot. Everything the
/// model sees is rendered here: the state as the user turn, the question with its options as
/// single-token codes, and the assistant header; the answer is a softmax over the option codes'
/// tokens at the last position and nothing else. The exact bytes matter — a different string
/// tokenizes differently — so every format string mirrors the Python verbatim.</para>
///
/// <code>
/// &lt;|startoftext|&gt;&lt;|im_start|&gt;user
/// {state}\n\n\nQUESTION:\n{instructions}\n\nOptions:\nA {desc}\nB {desc}\n\nReply with the option code only.&lt;|im_end|&gt;
/// &lt;|im_start|&gt;assistant
/// </code>
/// </summary>
public sealed class D1Prompt
{
    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";

    private static readonly string[] YesForms = ["yes", "Yes", "YES"];
    private static readonly string[] NoForms = ["no", "No", "NO"];

    /// <summary>
    /// The codes tried, in order, when a label's own code is not a single token or is taken:
    /// capitals, two digits, lower case, <c>#i</c>, then capital pairs.
    /// </summary>
    private static readonly string[] FallbackPool = BuildFallbackPool();

    private readonly HuggingFaceTokenizer _tokenizer;
    private readonly ConcurrentDictionary<string, (string Code, int Id)[]> _aliases = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, int[]> _ids = new(StringComparer.Ordinal);

    public D1Prompt(HuggingFaceTokenizer tokenizer, string bos)
    {
        _tokenizer = tokenizer;
        Bos = bos;
    }

    /// <summary>The tokenizer's BOS text, written at the head of every prompt.</summary>
    public string Bos { get; }

    private static string[] BuildFallbackPool()
    {
        var pool = new List<string>();
        for (char c = 'A'; c <= 'Z'; ++c) pool.Add(c.ToString());
        for (int i = 0; i < 100; ++i) pool.Add(i.ToString("00", System.Globalization.CultureInfo.InvariantCulture));
        for (char c = 'a'; c <= 'z'; ++c) pool.Add(c.ToString());
        for (int i = 0; i < 200; ++i) pool.Add("#" + i.ToString(System.Globalization.CultureInfo.InvariantCulture));
        for (char a = 'A'; a <= 'Z'; ++a)
        {
            for (char b = 'A'; b <= 'Z'; ++b) pool.Add(new string([a, b]));
        }
        return [.. pool];
    }

    /// <summary>
    /// <c>option_codes</c>: the labels themselves when every one is a single letter, else A..Z,
    /// else 00, 01, …
    /// </summary>
    public static IReadOnlyList<string> OptionCodes(IReadOnlyList<string> labels)
    {
        var stripped = labels.Select(l => l.Trim()).ToArray();
        if (stripped.Length > 0 && stripped.All(l => l.Length == 1 && char.IsLetter(l[0]))) return stripped;
        if (stripped.Length <= 26) return [.. Enumerable.Range(0, stripped.Length).Select(i => ((char)('A' + i)).ToString())];
        return [.. Enumerable.Range(0, stripped.Length).Select(i => i.ToString("00", System.Globalization.CultureInfo.InvariantCulture))];
    }

    /// <summary>
    /// <c>aliases</c>: every label gets a distinct single-token code, its own when that encodes to one
    /// unused token, else the first free entry of the fallback pool.
    /// </summary>
    public IReadOnlyList<(string Code, int Id)> Aliases(IReadOnlyList<string> labels)
    {
        var codes = OptionCodes(labels);
        return _aliases.GetOrAdd(string.Join('\u0001', codes), _ =>
        {
            var used = new HashSet<int>();
            var result = new List<(string, int)>(codes.Count);
            bool Take(string raw)
            {
                var encoded = _tokenizer.Encode(raw);
                if (encoded.Count != 1 || used.Contains(encoded[0])) return false;
                result.Add((raw, encoded[0]));
                used.Add(encoded[0]);
                return true;
            }

            foreach (string code in codes)
            {
                if (Take(code)) continue;
                if (!FallbackPool.Any(Take))
                {
                    throw new InvalidOperationException($"no single-token alias left for {codes.Count} options");
                }
            }
            return [.. result];
        });
    }

    /// <summary><c>_ids</c>: the single-token encodings among <paramref name="texts"/>, distinct, in order.</summary>
    public IReadOnlyList<int> SingleTokenIds(params string[] texts)
    {
        return _ids.GetOrAdd(string.Join('\u0001', texts), _ =>
        {
            var result = new List<int>();
            foreach (string text in texts)
            {
                var encoded = _tokenizer.Encode(text);
                if (encoded.Count == 1 && !result.Contains(encoded[0])) result.Add(encoded[0]);
            }
            return [.. result];
        });
    }

    /// <summary>
    /// <c>readout_ids</c>: the token ids each option is scored by, max-pooled. A noul scores
    /// <c>[yes forms, no forms]</c>, a score the digits <c>0..n-1</c>, a choice its code and the code
    /// with a leading space.
    /// </summary>
    public IReadOnlyList<IReadOnlyList<int>> ReadoutIds(Question question)
    {
        switch (question.Type)
        {
            case QuestionType.Noul:
            {
                var yes = SingleTokenIds(YesForms);
                var no = SingleTokenIds(NoForms);
                if (yes.Count == 0 || no.Count == 0) throw new InvalidOperationException("tokenizer has no single-token yes/no");
                return [yes, no];
            }
            case QuestionType.Score:
            {
                int levels = question.Levels?.Count ?? 0;
                var groups = new List<IReadOnlyList<int>>(levels);
                for (int i = 0; i < levels; ++i)
                {
                    var group = SingleTokenIds(i.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    if (group.Count == 0)
                    {
                        throw new ArgumentException(
                            $"score with {levels} levels needs single-token digits; the primitive is defined for 2 to 10");
                    }
                    groups.Add(group);
                }
                return groups;
            }
            default:
            {
                var groups = new List<IReadOnlyList<int>>();
                foreach (var (code, id) in Aliases(question.OptionKeys))
                {
                    var group = new List<int> { id };
                    group.AddRange(SingleTokenIds(" " + code).Where(i => i != id));
                    groups.Add(group);
                }
                if (groups.Count == 0) throw new ArgumentException("choice with no options");
                return groups;
            }
        }
    }

    /// <summary>
    /// <c>state_block</c> in the default <c>json_only</c> style: a string as it is, anything else as
    /// <c>json.dumps(state, ensure_ascii=False, indent=2)</c>; then a blank line.
    /// </summary>
    public static string StateBlock(object state)
    {
        string body = state switch
        {
            string text => text,
            JsonElement { ValueKind: JsonValueKind.String } element => element.GetString()!,
            _ => PythonJson.Dumps(state, ensureAscii: false, indent: 2),
        };
        return body + "\n\n";
    }

    /// <summary><c>question_block</c> with the default <c>desc</c> option style.</summary>
    public string QuestionBlock(Question question)
    {
        string instructions = Instructions(question);
        switch (question.Type)
        {
            case QuestionType.Choice:
            {
                var labels = question.OptionKeys;
                var codes = Aliases(labels);
                var lines = new StringBuilder();
                var criteria = question.NamedCriteria!;
                for (int i = 0; i < labels.Count; ++i)
                {
                    if (i > 0) lines.Append('\n');
                    // `desc or label.replace("_", " ")`
                    object? description = criteria[i].Value;
                    string text = IsFalsy(description) ? labels[i].Replace('_', ' ') : Text(description);
                    lines.Append(codes[i].Code).Append(' ').Append(text);
                }
                return $"{instructions}\n\nOptions:\n{lines}\n\nReply with the option code only.";
            }
            case QuestionType.Noul:
            {
                string extra = string.Empty;
                if (question.NamedCriteria is { Count: > 0 } criteria)
                {
                    object? Lookup(string key) => criteria.FirstOrDefault(c => c.Key == key).Value;
                    extra = $"\nYes: {PythonStr(Lookup("true"))}\nNo: {PythonStr(Lookup("false"))}";
                }
                return $"{instructions}{extra}\n\nReply with yes or no only.";
            }
            default:
            {
                var levels = question.Levels ?? [];
                string legend = string.Join('\n', levels.Select((level, i) => $"{i} {PythonStr(level)}"));
                return $"{instructions}\n\n{legend}\n\nReply with a single digit 0-{levels.Count - 1} only.";
            }
        }
    }

    /// <summary>
    /// <c>prefix_text</c>: everything before the question, shared by every question on one state —
    /// BOS, the user turn header, the pictures' markup, then the state and <c>QUESTION:</c>. With no
    /// state the question follows the pictures directly.
    /// </summary>
    public string Prefix(object? state, string images = "")
    {
        string body = state is null ? string.Empty : StateBlock(state) + "\nQUESTION:\n";
        return $"{Bos}{ImStart}user\n{images}{body}";
    }

    /// <summary><c>suffix_text</c>: the question and the assistant header, up to the answer slot.</summary>
    public string Suffix(Question question) => $"{QuestionBlock(question)}{ImEnd}\n{ImStart}assistant\n";

    /// <summary>The whole prompt for one question.</summary>
    public string Render(object? state, Question question) => Prefix(state) + Suffix(question);

    private static string Instructions(Question question) => question.Instructions switch
    {
        string text => text,
        JsonElement { ValueKind: JsonValueKind.String } element => element.GetString()!,
        _ => question.InstructionText(),
    };

    private static bool IsFalsy(object? value) => value switch
    {
        null => true,
        string s => s.Length == 0,
        JsonElement { ValueKind: JsonValueKind.Null or JsonValueKind.Undefined } => true,
        JsonElement { ValueKind: JsonValueKind.String } e => e.GetString()!.Length == 0,
        _ => false,
    };

    /// <summary>A criterion as text: strings as they are, anything structured as JSON.</summary>
    private static string Text(object? value) => value switch
    {
        string s => s,
        JsonElement { ValueKind: JsonValueKind.String } e => e.GetString()!,
        _ => Question.RenderCriterion(value),
    };

    /// <summary><c>str()</c> of a criterion in an f-string: <c>None</c> for a missing one.</summary>
    private static string PythonStr(object? value) => value is null ? "None" : Text(value);
}

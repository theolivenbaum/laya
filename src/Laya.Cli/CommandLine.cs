namespace Laya.Cli;

/// <summary>
/// A very small <c>--flag value</c> parser. Repeated flags accumulate, and a flag with no value is
/// a boolean switch.
/// </summary>
internal sealed class CommandLine
{
    private readonly Dictionary<string, List<string>> _options = new(StringComparer.Ordinal);
    private readonly List<string> _positional = [];

    public static CommandLine Parse(ReadOnlySpan<string> args)
    {
        var result = new CommandLine();
        for (int i = 0; i < args.Length; ++i)
        {
            string argument = args[i];
            if (!argument.StartsWith("--", StringComparison.Ordinal))
            {
                result._positional.Add(argument);
                continue;
            }

            string name = argument[2..];
            string? value = null;
            int equals = name.IndexOf('=', StringComparison.Ordinal);
            if (equals >= 0)
            {
                value = name[(equals + 1)..];
                name = name[..equals];
            }
            else if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                value = args[++i];
            }

            if (!result._options.TryGetValue(name, out var values))
            {
                values = [];
                result._options[name] = values;
            }
            if (value is not null) values.Add(value);
        }
        return result;
    }

    public bool Has(string name) => _options.ContainsKey(name);

    public string? Value(string name)
        => _options.TryGetValue(name, out var values) && values.Count > 0 ? values[^1] : null;

    public IReadOnlyList<string> Values(string name)
        => _options.TryGetValue(name, out var values) ? values : [];

    public IReadOnlyList<string> Positional => _positional;
}

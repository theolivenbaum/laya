using System.Collections;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Laya.Runtime;

/// <summary>
/// A <c>json.dumps</c> work-alike.
///
/// <para>Criterion text and serialized state end up inside the token sequence verbatim, so the
/// exact bytes matter: Python writes <c>{"a": 1, "b": [2, 3]}</c> with a space after every colon
/// and comma, and <c>System.Text.Json</c> writes <c>{"a":1,"b":[2,3]}</c>. Two different strings
/// tokenize differently, which is a parity failure, so this writes the Python spelling.</para>
/// </summary>
public static class PythonJson
{
    /// <summary>Serializes a value the way <c>json.dumps</c> would.</summary>
    public static string Dumps(object? value, bool ensureAscii = false)
    {
        var builder = new StringBuilder();
        Write(builder, value, ensureAscii, depth: 0);
        return builder.ToString();
    }

    /// <summary>
    /// <c>json.dumps(value, ensure_ascii=…, indent=<paramref name="indent"/>)</c>: every member on its
    /// own line, nested <paramref name="indent"/> spaces deeper, <c>","</c> with no trailing space
    /// between members, and empty containers written as <c>{}</c> / <c>[]</c>.
    /// </summary>
    public static string Dumps(object? value, bool ensureAscii, int indent)
    {
        var builder = new StringBuilder();
        Write(builder, value, ensureAscii, depth: 0, indent);
        return builder.ToString();
    }

    private static void Open(StringBuilder builder, char bracket, int? indent, int depth)
    {
        builder.Append(bracket);
        if (indent is int width) builder.Append('\n').Append(' ', width * (depth + 1));
    }

    private static void Separate(StringBuilder builder, int? indent, int depth)
    {
        if (indent is int width) builder.Append(",\n").Append(' ', width * (depth + 1));
        else builder.Append(", ");
    }

    private static void Close(StringBuilder builder, char bracket, int? indent, int depth)
    {
        if (indent is int width) builder.Append('\n').Append(' ', width * depth);
        builder.Append(bracket);
    }

    private static void Write(StringBuilder builder, object? value, bool ensureAscii, int depth, int? indent = null)
    {
        if (depth > 32)
        {
            builder.Append("null");
            return;
        }

        switch (value)
        {
            case null:
                builder.Append("null");
                return;
            case string text:
                WriteString(builder, text, ensureAscii);
                return;
            case bool flag:
                builder.Append(flag ? "true" : "false");
                return;
            case float or double or decimal:
                builder.Append(FormatNumber(Convert.ToDouble(value, CultureInfo.InvariantCulture)));
                return;
            case sbyte or byte or short or ushort or int or uint or long or ulong:
                builder.Append(Convert.ToString(value, CultureInfo.InvariantCulture));
                return;
            case JsonElement element:
                WriteJsonElement(builder, element, ensureAscii, depth, indent);
                return;
        }

        if (value is IDictionary dictionary)
        {
            if (dictionary.Count == 0)
            {
                builder.Append("{}");
                return;
            }
            Open(builder, '{', indent, depth);
            bool first = true;
            foreach (DictionaryEntry entry in dictionary)
            {
                if (!first) Separate(builder, indent, depth);
                first = false;
                WriteString(builder, entry.Key?.ToString() ?? string.Empty, ensureAscii);
                builder.Append(": ");
                Write(builder, entry.Value, ensureAscii, depth + 1, indent);
            }
            Close(builder, '}', indent, depth);
            return;
        }

        if (value is IEnumerable<KeyValuePair<string, object?>> pairs)
        {
            var members = pairs.ToList();
            if (members.Count == 0)
            {
                builder.Append("{}");
                return;
            }
            Open(builder, '{', indent, depth);
            bool first = true;
            foreach (var pair in members)
            {
                if (!first) Separate(builder, indent, depth);
                first = false;
                WriteString(builder, pair.Key, ensureAscii);
                builder.Append(": ");
                Write(builder, pair.Value, ensureAscii, depth + 1, indent);
            }
            Close(builder, '}', indent, depth);
            return;
        }

        if (value is IEnumerable sequence)
        {
            var items = sequence.Cast<object?>().ToList();
            if (items.Count == 0)
            {
                builder.Append("[]");
                return;
            }
            Open(builder, '[', indent, depth);
            bool first = true;
            foreach (object? item in items)
            {
                if (!first) Separate(builder, indent, depth);
                first = false;
                Write(builder, item, ensureAscii, depth + 1, indent);
            }
            Close(builder, ']', indent, depth);
            return;
        }

        // json.dumps(..., default=str) is what render_criterion asks for.
        WriteString(builder, value.ToString() ?? string.Empty, ensureAscii);
    }

    private static void WriteJsonElement(StringBuilder builder, JsonElement element, bool ensureAscii, int depth,
        int? indent = null)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                if (!element.EnumerateObject().Any())
                {
                    builder.Append("{}");
                    return;
                }
                Open(builder, '{', indent, depth);
                bool firstProperty = true;
                foreach (var property in element.EnumerateObject())
                {
                    if (!firstProperty) Separate(builder, indent, depth);
                    firstProperty = false;
                    WriteString(builder, property.Name, ensureAscii);
                    builder.Append(": ");
                    WriteJsonElement(builder, property.Value, ensureAscii, depth + 1, indent);
                }
                Close(builder, '}', indent, depth);
                return;
            case JsonValueKind.Array:
                if (element.GetArrayLength() == 0)
                {
                    builder.Append("[]");
                    return;
                }
                Open(builder, '[', indent, depth);
                bool firstItem = true;
                foreach (var item in element.EnumerateArray())
                {
                    if (!firstItem) Separate(builder, indent, depth);
                    firstItem = false;
                    WriteJsonElement(builder, item, ensureAscii, depth + 1, indent);
                }
                Close(builder, ']', indent, depth);
                return;
            case JsonValueKind.String:
                WriteString(builder, element.GetString() ?? string.Empty, ensureAscii);
                return;
            case JsonValueKind.Number:
                builder.Append(element.TryGetInt64(out long integer)
                    ? integer.ToString(CultureInfo.InvariantCulture)
                    : FormatNumber(element.GetDouble()));
                return;
            case JsonValueKind.True:
                builder.Append("true");
                return;
            case JsonValueKind.False:
                builder.Append("false");
                return;
            default:
                builder.Append("null");
                return;
        }
    }

    /// <summary>Python's <c>repr</c> for floats: shortest round-tripping form, always with a point.</summary>
    private static string FormatNumber(double value)
    {
        if (double.IsNaN(value)) return "NaN";
        if (double.IsPositiveInfinity(value)) return "Infinity";
        if (double.IsNegativeInfinity(value)) return "-Infinity";

        string text = value.ToString("R", CultureInfo.InvariantCulture);
        if (text.Contains('.', StringComparison.Ordinal) || text.Contains('e', StringComparison.OrdinalIgnoreCase))
        {
            return text;
        }
        return text + ".0";
    }

    private static void WriteString(StringBuilder builder, string value, bool ensureAscii)
    {
        builder.Append('"');
        foreach (char c in value)
        {
            switch (c)
            {
                case '"': builder.Append("\\\""); break;
                case '\\': builder.Append("\\\\"); break;
                case '\n': builder.Append("\\n"); break;
                case '\r': builder.Append("\\r"); break;
                case '\t': builder.Append("\\t"); break;
                case '\b': builder.Append("\\b"); break;
                case '\f': builder.Append("\\f"); break;
                default:
                    if (c < 0x20 || (ensureAscii && c > 0x7E))
                    {
                        builder.Append(CultureInfo.InvariantCulture, $"\\u{(int)c:x4}");
                    }
                    else
                    {
                        builder.Append(c);
                    }
                    break;
            }
        }
        builder.Append('"');
    }
}

using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace StructuredDecisions.Mica;

/// <summary>
/// <c>json.dumps</c> over a parsed request, as Mica's Python renders state and object-valued criteria.
///
/// <para>The rendered prompt has to match the text the model was trained on byte for byte, so this follows
/// what <c>json.loads</c> then <c>json.dumps</c> produce, not what the request said: integers print as
/// integers of any size, other numbers as Python's float <c>repr</c> (<c>1.50</c> → <c>1.5</c>,
/// <c>1e5</c> → <c>100000.0</c>, <c>1e16</c> → <c>1e+16</c>), and only quotes, backslashes and control
/// characters are escaped (<c>ensure_ascii=False</c>).</para>
/// </summary>
public static class PythonJsonNode
{
    /// <summary>
    /// <paramref name="compact"/> is <c>separators=(",", ":")</c>, <paramref name="sortKeys"/> sorts by code
    /// point; non-finite numbers throw, as <c>allow_nan=False</c> does.
    /// </summary>
    public static string Dumps(JsonNode? node, bool sortKeys = false, bool compact = false)
    {
        var builder = new StringBuilder();
        Write(builder, node, sortKeys, compact ? "," : ", ", compact ? ":" : ": ");
        return builder.ToString();
    }

    private static void Write(StringBuilder builder, JsonNode? node, bool sortKeys, string itemSeparator, string keySeparator)
    {
        switch (node)
        {
            case null:
                builder.Append("null");
                return;
            case JsonObject obj:
            {
                builder.Append('{');
                IEnumerable<KeyValuePair<string, JsonNode?>> properties = obj;
                if (sortKeys) properties = obj.OrderBy(p => p.Key, CodePointComparer.Instance);
                bool first = true;
                foreach (var (key, value) in properties)
                {
                    if (!first) builder.Append(itemSeparator);
                    first = false;
                    WriteString(builder, key);
                    builder.Append(keySeparator);
                    Write(builder, value, sortKeys, itemSeparator, keySeparator);
                }
                builder.Append('}');
                return;
            }
            case JsonArray array:
            {
                builder.Append('[');
                for (int i = 0; i < array.Count; ++i)
                {
                    if (i > 0) builder.Append(itemSeparator);
                    Write(builder, array[i], sortKeys, itemSeparator, keySeparator);
                }
                builder.Append(']');
                return;
            }
        }

        var element = Element(node.AsValue());
        switch (element.ValueKind)
        {
            case JsonValueKind.String: WriteString(builder, element.GetString()!); return;
            case JsonValueKind.True: builder.Append("true"); return;
            case JsonValueKind.False: builder.Append("false"); return;
            case JsonValueKind.Null: builder.Append("null"); return;
            case JsonValueKind.Number: builder.Append(Number(element.GetRawText())); return;
            default: throw new ArgumentException($"cannot serialize a JSON {element.ValueKind}");
        }
    }

    /// <summary>
    /// A value as a <see cref="JsonElement"/>, whether it was parsed (element-backed) or built in code
    /// (<c>JsonValue.Create(0.75)</c>), which the serializer round-trips to the same JSON token.
    /// </summary>
    public static JsonElement Element(JsonValue value)
        => value.TryGetValue<JsonElement>(out var element) ? element : JsonSerializer.SerializeToElement(value);

    /// <summary>A JSON number as Python prints it after <c>json.loads</c>.</summary>
    public static string Number(string raw)
    {
        bool integral = raw.AsSpan().IndexOfAny('.', 'e', 'E') < 0;
        if (integral) return BigInteger.Parse(raw, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
        double value = double.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture);
        if (!double.IsFinite(value)) throw new ArgumentException("Out of range float values are not JSON compliant");
        return FloatRepr(value);
    }

    /// <summary>Python's <c>repr(float)</c>: the shortest round-tripping digits, fixed notation for exponents −4..15.</summary>
    public static string FloatRepr(double value)
    {
        if (value == 0) return double.IsNegative(value) ? "-0.0" : "0.0";
        string shortest = Math.Abs(value).ToString("R", CultureInfo.InvariantCulture);

        // Digits and decimal point position of the shortest form: value = 0.d1d2… × 10^decpt.
        int e = shortest.IndexOfAny(['E', 'e']);
        string mantissa = e < 0 ? shortest : shortest[..e];
        int exponent = e < 0 ? 0 : int.Parse(shortest[(e + 1)..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
        int point = mantissa.IndexOf('.');
        string digits = point < 0 ? mantissa : mantissa.Remove(point, 1);
        int decpt = (point < 0 ? mantissa.Length : point) + exponent;
        int lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0') { lead++; decpt--; }
        digits = digits[lead..].TrimEnd('0');
        if (digits.Length == 0) digits = "0";

        var builder = new StringBuilder();
        if (value < 0) builder.Append('-');
        if (decpt > -4 && decpt <= 16)
        {
            if (decpt <= 0) builder.Append("0.").Append('0', -decpt).Append(digits);
            else if (decpt >= digits.Length) builder.Append(digits).Append('0', decpt - digits.Length).Append(".0");
            else builder.Append(digits, 0, decpt).Append('.').Append(digits, decpt, digits.Length - decpt);
        }
        else
        {
            builder.Append(digits[0]);
            if (digits.Length > 1) builder.Append('.').Append(digits, 1, digits.Length - 1);
            int shown = decpt - 1;
            builder.Append('e').Append(shown < 0 ? '-' : '+').Append(Math.Abs(shown).ToString("00", CultureInfo.InvariantCulture));
        }
        return builder.ToString();
    }

    private static void WriteString(StringBuilder builder, string value)
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
                    if (c < 0x20) builder.Append(CultureInfo.InvariantCulture, $"\\u{(int)c:x4}");
                    else builder.Append(c);
                    break;
            }
        }
        builder.Append('"');
    }

    /// <summary>Python's <c>str</c> ordering: by code point, which UTF-16 ordinal order is not past U+FFFF.</summary>
    internal sealed class CodePointComparer : IComparer<string>
    {
        public static readonly CodePointComparer Instance = new();

        public int Compare(string? x, string? y)
        {
            var a = (x ?? string.Empty).EnumerateRunes().GetEnumerator();
            var b = (y ?? string.Empty).EnumerateRunes().GetEnumerator();
            while (true)
            {
                bool hasA = a.MoveNext(), hasB = b.MoveNext();
                if (!hasA || !hasB) return hasA.CompareTo(hasB);
                int c = a.Current.Value.CompareTo(b.Current.Value);
                if (c != 0) return c;
            }
        }
    }
}

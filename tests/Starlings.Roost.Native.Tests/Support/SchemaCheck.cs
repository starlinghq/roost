using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Starlings.Roost.Native.Tests.Support;

/// <summary>
/// A JSON-schema checker for the keywords the test schemas use (type, properties, required, additionalProperties false, enum,
/// const, minimum, maximum, maxLength, pattern, items, minItems, maxItems, anyOf, $ref into $defs). Anything else throws, so a
/// schema cannot silently use a keyword this checker ignores.
/// </summary>
internal static class SchemaCheck
{
    private static readonly HashSet<string> s_known =
    [
        "type", "properties", "required", "additionalProperties", "enum", "const", "minimum", "maximum", "maxLength", "pattern",
        "items", "minItems", "maxItems", "anyOf", "$ref", "$defs",
    ];

    /// <summary>The first violation of <paramref name="schema"/> by <paramref name="instance"/>, or null.</summary>
    public static string? Violation(JsonElement instance, JsonElement schema) => Check(instance, schema, schema, "$");

    private static string? Check(JsonElement value, JsonElement schema, JsonElement root, string path)
    {
        foreach (var keyword in schema.EnumerateObject())
        {
            if (!s_known.Contains(keyword.Name))
            {
                throw new NotSupportedException($"SchemaCheck does not implement '{keyword.Name}'.");
            }
        }

        if (schema.TryGetProperty("$ref", out var reference))
        {
            var name = reference.GetString()!["#/$defs/".Length..];
            return Check(value, root.GetProperty("$defs").GetProperty(name), root, path);
        }

        if (schema.TryGetProperty("anyOf", out var anyOf))
        {
            return anyOf.EnumerateArray().Any(s => Check(value, s, root, path) is null) ? null : $"{path}: matches no anyOf branch";
        }

        if (schema.TryGetProperty("const", out var constant) && !JsonElement.DeepEquals(value, constant))
        {
            return $"{path}: is not the const {constant.GetRawText()}";
        }

        if (schema.TryGetProperty("enum", out var options) && !options.EnumerateArray().Any(o => JsonElement.DeepEquals(value, o)))
        {
            return $"{path}: {value.GetRawText()} is not in the enum";
        }

        if (schema.TryGetProperty("type", out var type))
        {
            var types = type.ValueKind == JsonValueKind.Array ? type.EnumerateArray().Select(t => t.GetString()!).ToArray() : [type.GetString()!];
            if (!types.Any(t => Is(value, t)))
            {
                return $"{path}: {value.ValueKind} is not {string.Join(" or ", types)}";
            }
        }

        switch (value.ValueKind)
        {
            case JsonValueKind.Number:
                var number = value.GetDouble();
                if (schema.TryGetProperty("minimum", out var min) && number < min.GetDouble())
                {
                    return $"{path}: {number} < minimum";
                }

                if (schema.TryGetProperty("maximum", out var max) && number > max.GetDouble())
                {
                    return $"{path}: {number} > maximum";
                }

                break;
            case JsonValueKind.String:
                var text = value.GetString()!;
                if (schema.TryGetProperty("maxLength", out var maxLength) && text.EnumerateRunes().Count() > maxLength.GetInt32())
                {
                    return $"{path}: longer than maxLength";
                }

                if (schema.TryGetProperty("pattern", out var pattern) && !Regex.IsMatch(text, pattern.GetString()!, RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1)))
                {
                    return $"{path}: does not match the pattern";
                }

                break;
            case JsonValueKind.Array:
                var items = value.EnumerateArray().ToArray();
                if (schema.TryGetProperty("minItems", out var minItems) && items.Length < minItems.GetInt32())
                {
                    return $"{path}: fewer than minItems";
                }

                if (schema.TryGetProperty("maxItems", out var maxItems) && items.Length > maxItems.GetInt32())
                {
                    return $"{path}: more than maxItems";
                }

                if (schema.TryGetProperty("items", out var itemSchema))
                {
                    for (var i = 0; i < items.Length; i++)
                    {
                        if (Check(items[i], itemSchema, root, path + "[" + i.ToString(CultureInfo.InvariantCulture) + "]") is { } bad)
                        {
                            return bad;
                        }
                    }
                }

                break;
            case JsonValueKind.Object:
                var properties = schema.TryGetProperty("properties", out var p) ? p : default;
                if (schema.TryGetProperty("required", out var required))
                {
                    foreach (var name in required.EnumerateArray())
                    {
                        if (!value.TryGetProperty(name.GetString()!, out _))
                        {
                            return $"{path}: missing required '{name.GetString()}'";
                        }
                    }
                }

                foreach (var member in value.EnumerateObject())
                {
                    if (properties.ValueKind == JsonValueKind.Object && properties.TryGetProperty(member.Name, out var memberSchema))
                    {
                        if (Check(member.Value, memberSchema, root, path + "." + member.Name) is { } bad)
                        {
                            return bad;
                        }
                    }
                    else if (schema.TryGetProperty("additionalProperties", out var extra) && extra.ValueKind == JsonValueKind.False)
                    {
                        return $"{path}: unexpected property '{member.Name}'";
                    }
                }

                break;
            default:
                break;
        }

        return null;
    }

    private static bool Is(JsonElement value, string type) => type switch
    {
        "object" => value.ValueKind == JsonValueKind.Object,
        "array" => value.ValueKind == JsonValueKind.Array,
        "string" => value.ValueKind == JsonValueKind.String,
        "boolean" => value.ValueKind is JsonValueKind.True or JsonValueKind.False,
        "null" => value.ValueKind == JsonValueKind.Null,
        "integer" => value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out _),
        "number" => value.ValueKind == JsonValueKind.Number,
        _ => throw new NotSupportedException(type),
    };
}

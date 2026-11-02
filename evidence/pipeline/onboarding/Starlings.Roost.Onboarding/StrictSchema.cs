using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Starlings.Roost.Onboarding;

/// <summary>
/// A JSON Schema (draft 2020-12) evaluator for the subset the roost-v1 recipe schema uses: <c>type</c>, <c>properties</c>,
/// <c>required</c>, <c>additionalProperties</c>, <c>items</c>, <c>enum</c>, <c>const</c>, <c>pattern</c>, <c>minLength</c>,
/// <c>minimum</c>, <c>maximum</c>, <c>uniqueItems</c> and local <c>$ref</c> into <c>$defs</c>.
/// </summary>
/// <remarks>
/// Loading a schema that uses any other keyword throws, so the schema can never carry a rule this evaluator silently
/// ignores. It exists because the onboarding tool has no third-party dependency; the schema file stays the source of truth.
/// </remarks>
internal sealed class StrictSchema
{
    private static readonly HashSet<string> Annotations = new(StringComparer.Ordinal) { "$schema", "$id", "title", "description", "$defs" };

    private static readonly HashSet<string> Keywords = new(StringComparer.Ordinal)
    {
        "type", "properties", "required", "additionalProperties", "items", "enum", "const", "pattern", "minLength",
        "minimum", "maximum", "uniqueItems", "$ref",
    };

    private readonly JsonElement _root;
    private readonly Dictionary<string, Regex> _patterns = new(StringComparer.Ordinal);

    private StrictSchema(JsonElement root)
    {
        _root = root;
        CheckKeywords(root, "#");
    }

    /// <summary>Loads and checks a schema.</summary>
    /// <param name="schema">The schema root, cloned so it does not depend on a live document.</param>
    /// <returns>The evaluator.</returns>
    /// <exception cref="InvalidOperationException">The schema uses a keyword outside the subset or a bad <c>$ref</c>.</exception>
    public static StrictSchema Load(JsonElement schema) => new(schema.Clone());

    /// <summary>Validates <paramref name="instance"/>, adding one "location: message" entry per violation.</summary>
    /// <param name="instance">The instance.</param>
    /// <param name="errors">Receives errors; locations are JSON pointers ("" for the root).</param>
    public void Validate(JsonElement instance, List<string> errors) => Check(_root, instance, "", errors);

    private void CheckKeywords(JsonElement schema, string at)
    {
        if (schema.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidOperationException($"Schema {at}: a subschema must be an object.");
        }

        foreach (var p in schema.EnumerateObject())
        {
            if (!Annotations.Contains(p.Name) && !Keywords.Contains(p.Name))
            {
                throw new InvalidOperationException($"Schema {at}: keyword '{p.Name}' is outside the subset StrictSchema evaluates.");
            }
        }

        if (schema.TryGetProperty("$defs", out var defs))
        {
            foreach (var d in defs.EnumerateObject())
            {
                CheckKeywords(d.Value, at + "/$defs/" + d.Name);
            }
        }

        if (schema.TryGetProperty("properties", out var props))
        {
            foreach (var p in props.EnumerateObject())
            {
                CheckKeywords(p.Value, at + "/properties/" + p.Name);
            }
        }

        if (schema.TryGetProperty("items", out var items))
        {
            CheckKeywords(items, at + "/items");
        }

        if (schema.TryGetProperty("additionalProperties", out var ap) && ap.ValueKind == JsonValueKind.Object)
        {
            CheckKeywords(ap, at + "/additionalProperties");
        }

        if (schema.TryGetProperty("$ref", out var r))
        {
            Resolve(r.GetString()!);
        }
    }

    private JsonElement Resolve(string reference)
    {
        const string Prefix = "#/$defs/";
        if (!reference.StartsWith(Prefix, StringComparison.Ordinal)
            || !_root.TryGetProperty("$defs", out var defs)
            || !defs.TryGetProperty(reference[Prefix.Length..], out var target))
        {
            throw new InvalidOperationException($"Schema: $ref '{reference}' does not name an entry of $defs.");
        }

        return target;
    }

    private void Check(JsonElement schema, JsonElement value, string at, List<string> errors)
    {
        if (schema.TryGetProperty("$ref", out var r))
        {
            Check(Resolve(r.GetString()!), value, at, errors);
        }

        if (schema.TryGetProperty("const", out var c) && !JsonElement.DeepEquals(c, value))
        {
            errors.Add($"{at}: must be {c.GetRawText()}");
        }

        if (schema.TryGetProperty("enum", out var e) && !e.EnumerateArray().Any(x => JsonElement.DeepEquals(x, value)))
        {
            errors.Add($"{at}: {Show(value)} is not one of {e.GetRawText()}");
        }

        if (schema.TryGetProperty("type", out var t) && !IsType(value, t.GetString()!))
        {
            errors.Add($"{at}: {Show(value)} is not of type {t.GetString()}");
            return;
        }

        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                CheckObject(schema, value, at, errors);
                break;
            case JsonValueKind.Array:
                CheckArray(schema, value, at, errors);
                break;
            case JsonValueKind.String:
                CheckString(schema, value.GetString()!, at, errors);
                break;
            case JsonValueKind.Number:
                CheckNumber(schema, value.GetDouble(), at, errors);
                break;
            default:
                break;
        }
    }

    private void CheckObject(JsonElement schema, JsonElement value, string at, List<string> errors)
    {
        if (schema.TryGetProperty("required", out var req))
        {
            foreach (var name in req.EnumerateArray())
            {
                if (!value.TryGetProperty(name.GetString()!, out _))
                {
                    errors.Add($"{at}: '{name.GetString()}' is required");
                }
            }
        }

        bool hasProps = schema.TryGetProperty("properties", out var props);
        bool hasAdditional = schema.TryGetProperty("additionalProperties", out var additional);
        foreach (var p in value.EnumerateObject())
        {
            string child = at + "/" + p.Name.Replace("~", "~0", StringComparison.Ordinal).Replace("/", "~1", StringComparison.Ordinal);
            if (hasProps && props.TryGetProperty(p.Name, out var sub))
            {
                Check(sub, p.Value, child, errors);
            }
            else if (hasAdditional)
            {
                if (additional.ValueKind == JsonValueKind.False)
                {
                    errors.Add($"{at}: unknown property '{p.Name}'");
                }
                else if (additional.ValueKind == JsonValueKind.Object)
                {
                    Check(additional, p.Value, child, errors);
                }
            }
        }
    }

    private void CheckArray(JsonElement schema, JsonElement value, string at, List<string> errors)
    {
        var elements = value.EnumerateArray().ToArray();
        if (schema.TryGetProperty("items", out var items))
        {
            for (int i = 0; i < elements.Length; i++)
            {
                Check(items, elements[i], at + "/" + i.ToString(CultureInfo.InvariantCulture), errors);
            }
        }

        if (schema.TryGetProperty("uniqueItems", out var u) && u.ValueKind == JsonValueKind.True)
        {
            for (int i = 1; i < elements.Length; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    if (JsonElement.DeepEquals(elements[i], elements[j]))
                    {
                        errors.Add($"{at}/{i.ToString(CultureInfo.InvariantCulture)}: duplicates item {j.ToString(CultureInfo.InvariantCulture)}");
                        break;
                    }
                }
            }
        }
    }

    private void CheckString(JsonElement schema, string s, string at, List<string> errors)
    {
        if (schema.TryGetProperty("minLength", out var ml) && s.Length < ml.GetInt32())
        {
            errors.Add($"{at}: shorter than {ml.GetInt32().ToString(CultureInfo.InvariantCulture)} characters");
        }

        if (schema.TryGetProperty("pattern", out var p))
        {
            string pattern = p.GetString()!;
            if (!_patterns.TryGetValue(pattern, out var regex))
            {
                regex = new Regex(pattern, RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1));
                _patterns.Add(pattern, regex);
            }

            if (!regex.IsMatch(s))
            {
                errors.Add($"{at}: \"{s}\" does not match {pattern}");
            }
        }
    }

    private static void CheckNumber(JsonElement schema, double d, string at, List<string> errors)
    {
        if (schema.TryGetProperty("minimum", out var min) && d < min.GetDouble())
        {
            errors.Add($"{at}: {HfConfig.Number(d)} is less than {min.GetRawText()}");
        }

        if (schema.TryGetProperty("maximum", out var max) && d > max.GetDouble())
        {
            errors.Add($"{at}: {HfConfig.Number(d)} is greater than {max.GetRawText()}");
        }
    }

    private static bool IsType(JsonElement v, string type) => type switch
    {
        "object" => v.ValueKind == JsonValueKind.Object,
        "array" => v.ValueKind == JsonValueKind.Array,
        "string" => v.ValueKind == JsonValueKind.String,
        "boolean" => v.ValueKind is JsonValueKind.True or JsonValueKind.False,
        "null" => v.ValueKind == JsonValueKind.Null,
        "number" => v.ValueKind == JsonValueKind.Number,
        "integer" => v.ValueKind == JsonValueKind.Number && v.GetDouble() == Math.Floor(v.GetDouble()),
        _ => throw new InvalidOperationException($"Schema: type '{type}' is not a JSON Schema type."),
    };

    private static string Show(JsonElement v)
    {
        string raw = v.GetRawText();
        return raw.Length > 60 ? raw[..57] + "..." : raw;
    }
}

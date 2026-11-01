using System.Text.Json;
using System.Text.RegularExpressions;

namespace Starlings.Roost.EngineGates.Tests.Support;

/// <summary>
/// A JSON Schema (draft 2020-12) validator for the keywords evidence/schemas/stage-results.schema.json uses, so the gate rows
/// are checked against the contract file itself without a schema package. It fails closed: a keyword it does not implement
/// throws <see cref="NotSupportedException"/>, so a schema change that adds one cannot pass silently. The evaluator's own
/// check (Python <c>jsonschema</c>, evidence/pipeline) remains the reference; the pytest in evidence/pipeline/tests runs it
/// on the sample run this harness wrote.
/// </summary>
internal sealed class SchemaValidator
{
    private static readonly HashSet<string> s_annotations = ["$schema", "$id", "title", "description", "$defs"];

    private readonly JsonElement _root;

    public SchemaValidator(JsonElement root) => _root = root;

    /// <summary>Validates <paramref name="instance"/>; returns one message per violation (empty when valid).</summary>
    /// <param name="instance">The document.</param>
    /// <returns>The violations.</returns>
    public List<string> Validate(JsonElement instance)
    {
        var errors = new List<string>();
        Check(_root, instance, "$", errors);
        return errors;
    }

    private static bool IsType(JsonElement value, string type) => type switch
    {
        "object" => value.ValueKind == JsonValueKind.Object,
        "array" => value.ValueKind == JsonValueKind.Array,
        "string" => value.ValueKind == JsonValueKind.String,
        "boolean" => value.ValueKind is JsonValueKind.True or JsonValueKind.False,
        "null" => value.ValueKind == JsonValueKind.Null,
        "number" => value.ValueKind == JsonValueKind.Number,
        "integer" => value.ValueKind == JsonValueKind.Number && Math.Floor(value.GetDouble()) == value.GetDouble(),
        _ => throw new NotSupportedException($"type {type}"),
    };

    private static bool JsonEquals(JsonElement a, JsonElement b)
    {
        if (a.ValueKind == JsonValueKind.Number && b.ValueKind == JsonValueKind.Number)
        {
            return a.GetDouble() == b.GetDouble();
        }

        if (a.ValueKind != b.ValueKind)
        {
            return false;
        }

        return a.ValueKind switch
        {
            JsonValueKind.Object => a.EnumerateObject().Count() == b.EnumerateObject().Count()
                && a.EnumerateObject().All(p => b.TryGetProperty(p.Name, out var q) && JsonEquals(p.Value, q)),
            JsonValueKind.Array => a.GetArrayLength() == b.GetArrayLength()
                && a.EnumerateArray().Zip(b.EnumerateArray()).All(x => JsonEquals(x.First, x.Second)),
            JsonValueKind.String => a.GetString() == b.GetString(),
            _ => true,
        };
    }

    private bool Valid(JsonElement schema, JsonElement value)
    {
        var errors = new List<string>();
        Check(schema, value, "$", errors);
        return errors.Count == 0;
    }

    private JsonElement Resolve(string reference)
    {
        if (!reference.StartsWith("#/", StringComparison.Ordinal))
        {
            throw new NotSupportedException($"$ref {reference}");
        }

        var node = _root;
        foreach (var part in reference[2..].Split('/'))
        {
            node = node.GetProperty(part);
        }

        return node;
    }

    private void Check(JsonElement schema, JsonElement value, string at, List<string> errors)
    {
        if (schema.ValueKind == JsonValueKind.True)
        {
            return;
        }

        if (schema.ValueKind == JsonValueKind.False)
        {
            errors.Add($"{at}: not allowed");
            return;
        }

        foreach (var keyword in schema.EnumerateObject())
        {
            var k = keyword.Value;
            switch (keyword.Name)
            {
                case var a when s_annotations.Contains(a):
                    break;
                case "$ref":
                    Check(Resolve(k.GetString()!), value, at, errors);
                    break;
                case "type":
                    var types = k.ValueKind == JsonValueKind.Array ? k.EnumerateArray().Select(t => t.GetString()!).ToList() : [k.GetString()!];
                    if (!types.Any(t => IsType(value, t)))
                    {
                        errors.Add($"{at}: expected {string.Join(" or ", types)}, got {value.ValueKind}");
                    }

                    break;
                case "const":
                    if (!JsonEquals(k, value))
                    {
                        errors.Add($"{at}: expected the constant {k.GetRawText()}");
                    }

                    break;
                case "enum":
                    if (!k.EnumerateArray().Any(e => JsonEquals(e, value)))
                    {
                        errors.Add($"{at}: {value.GetRawText()} is not one of {k.GetRawText()}");
                    }

                    break;
                case "pattern":
                    if (value.ValueKind == JsonValueKind.String && !Regex.IsMatch(value.GetString()!, k.GetString()!, RegexOptions.CultureInvariant))
                    {
                        errors.Add($"{at}: does not match {k.GetString()}");
                    }

                    break;
                case "minLength":
                    if (value.ValueKind == JsonValueKind.String && value.GetString()!.EnumerateRunes().Count() < k.GetInt32())
                    {
                        errors.Add($"{at}: shorter than {k.GetInt32()}");
                    }

                    break;
                case "minimum":
                    if (value.ValueKind == JsonValueKind.Number && value.GetDouble() < k.GetDouble())
                    {
                        errors.Add($"{at}: below {k.GetDouble()}");
                    }

                    break;
                case "required":
                    if (value.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var name in k.EnumerateArray().Select(n => n.GetString()!).Where(n => !value.TryGetProperty(n, out _)))
                        {
                            errors.Add($"{at}: lacks {name}");
                        }
                    }

                    break;
                case "properties":
                    if (value.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var p in k.EnumerateObject())
                        {
                            if (value.TryGetProperty(p.Name, out var v))
                            {
                                Check(p.Value, v, $"{at}.{p.Name}", errors);
                            }
                        }
                    }

                    break;
                case "additionalProperties":
                    if (value.ValueKind == JsonValueKind.Object)
                    {
                        var known = schema.TryGetProperty("properties", out var props) ? props.EnumerateObject().Select(p => p.Name).ToHashSet() : [];
                        foreach (var p in value.EnumerateObject().Where(p => !known.Contains(p.Name)))
                        {
                            if (k.ValueKind == JsonValueKind.False)
                            {
                                errors.Add($"{at}: unexpected property {p.Name}");
                            }
                            else
                            {
                                Check(k, p.Value, $"{at}.{p.Name}", errors);
                            }
                        }
                    }

                    break;
                case "propertyNames":
                    if (value.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var p in value.EnumerateObject())
                        {
                            Check(k, JsonDocument.Parse(JsonSerializer.Serialize(p.Name)).RootElement, $"{at}.{p.Name} (name)", errors);
                        }
                    }

                    break;
                case "minProperties":
                    if (value.ValueKind == JsonValueKind.Object && value.EnumerateObject().Count() < k.GetInt32())
                    {
                        errors.Add($"{at}: fewer than {k.GetInt32()} properties");
                    }

                    break;
                case "prefixItems":
                    if (value.ValueKind == JsonValueKind.Array)
                    {
                        int i = 0;
                        foreach (var (item, sub) in value.EnumerateArray().Zip(k.EnumerateArray()))
                        {
                            Check(sub, item, $"{at}[{i++}]", errors);
                        }
                    }

                    break;
                case "items":
                    if (value.ValueKind == JsonValueKind.Array)
                    {
                        int skip = schema.TryGetProperty("prefixItems", out var prefix) ? prefix.GetArrayLength() : 0;
                        int i = 0;
                        foreach (var item in value.EnumerateArray())
                        {
                            if (i >= skip)
                            {
                                Check(k, item, $"{at}[{i}]", errors);
                            }

                            i++;
                        }
                    }

                    break;
                case "minItems":
                    if (value.ValueKind == JsonValueKind.Array && value.GetArrayLength() < k.GetInt32())
                    {
                        errors.Add($"{at}: fewer than {k.GetInt32()} items");
                    }

                    break;
                case "maxItems":
                    if (value.ValueKind == JsonValueKind.Array && value.GetArrayLength() > k.GetInt32())
                    {
                        errors.Add($"{at}: more than {k.GetInt32()} items");
                    }

                    break;
                case "allOf":
                    foreach (var sub in k.EnumerateArray())
                    {
                        Check(sub, value, at, errors);
                    }

                    break;
                case "anyOf":
                    if (!k.EnumerateArray().Any(sub => Valid(sub, value)))
                    {
                        errors.Add($"{at}: matches none of anyOf");
                    }

                    break;
                case "oneOf":
                    if (k.EnumerateArray().Count(sub => Valid(sub, value)) != 1)
                    {
                        errors.Add($"{at}: does not match exactly one of oneOf");
                    }

                    break;
                case "not":
                    if (Valid(k, value))
                    {
                        errors.Add($"{at}: matches a schema under not");
                    }

                    break;
                case "if":
                    if (Valid(k, value))
                    {
                        if (schema.TryGetProperty("then", out var then))
                        {
                            Check(then, value, at, errors);
                        }
                    }
                    else if (schema.TryGetProperty("else", out var otherwise))
                    {
                        Check(otherwise, value, at, errors);
                    }

                    break;
                case "then" or "else":
                    break;
                default:
                    throw new NotSupportedException($"The gate schema validator does not implement '{keyword.Name}' ({at}).");
            }
        }
    }
}

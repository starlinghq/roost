using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using static Starlings.Roost.Server.Routing.RequestFields;

namespace Starlings.Roost.Server.Routing;

/// <summary>
/// Reads a response format (R-051): OpenAI chat completions' <c>response_format</c> and the Responses API's <c>text.format</c>.
/// <c>text</c>, <c>json_object</c> and <c>json_schema</c> become <see cref="ChatOptions.ResponseFormat"/>, which
/// <see cref="Starlings.Generation.Roost.RoostChatClient"/> compiles with the grammar engine (an invalid or unsupported schema is a 400
/// naming the format). Every other type, regular expressions and grammars included, is refused here until it is supported.
/// </summary>
internal static partial class ResponseFormats
{
    /// <summary>Reads a format object.</summary>
    /// <param name="v">The object.</param>
    /// <param name="param">Its parameter name (<c>response_format</c> or <c>text.format</c>).</param>
    /// <param name="flat">True for the Responses API, where the schema's fields sit beside <c>type</c>; false for chat completions,
    /// where they sit in a <c>json_schema</c> object.</param>
    /// <returns>The format.</returns>
    public static ChatResponseFormat Read(JsonElement v, string param, bool flat)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse(param, $"{param} must be an object.");
        }

        var type = v.TryGetProperty("type", out var t) ? StringValue(t, param + ".type") : throw Refuse(param + ".type", $"{param}.type is required.");
        switch (type)
        {
            case "text":
                Only(v, param, "type");
                return ChatResponseFormat.Text;
            case "json_object":
                Only(v, param, "type");
                return ChatResponseFormat.Json;
            case "json_schema":
                if (flat)
                {
                    return Schema(v, param, skipType: true);
                }

                Only(v, param, "type", "json_schema");
                return v.TryGetProperty("json_schema", out var inner) && inner.ValueKind != JsonValueKind.Null
                    ? Schema(inner, param + ".json_schema", skipType: false)
                    : throw Refuse(param + ".json_schema", $"{param}.json_schema is required for type json_schema.");
            default:
                throw Refuse(param + ".type", $"Response format type '{type}' is unsupported; Roost accepts text, json_object and json_schema. Regular expression and grammar formats are refused until they are supported (R-051).");
        }
    }

    private static ChatResponseFormatJson Schema(JsonElement v, string param, bool skipType)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse(param, $"{param} must be an object.");
        }

        string? name = null;
        string? description = null;
        JsonElement? schema = null;
        foreach (var field in Fields(v, param))
        {
            var at = param + "." + field.Name;
            switch (field.Name)
            {
                case "type" when skipType:
                    break;
                case "name":
                    name = StringValue(field.Value, at);
                    if (!SchemaName().IsMatch(name))
                    {
                        throw Refuse(at, $"{at} must be 1 to 64 letters, digits, underscores or dashes.");
                    }

                    break;
                case "description":
                    description = StringValue(field.Value, at);
                    break;
                case "schema":
                    schema = field.Value.ValueKind == JsonValueKind.Object ? field.Value.Clone() : throw Refuse(at, $"{at} must be a JSON object.");
                    break;
                case "strict":
                    _ = Bool(field.Value, at); // the grammar always constrains decoding to the schema, strict or not
                    break;
                default:
                    throw Refuse(at, $"Option '{field.Name}' of {param} is unsupported.");
            }
        }

        if (name is null)
        {
            throw Refuse(param + ".name", $"{param}.name is required.");
        }

        return schema is { } s
            ? ChatResponseFormat.ForJsonSchema(s, name, description)
            : throw Refuse(param + ".schema", $"{param}.schema is required: Roost enforces the schema it is given.");
    }

    private static void Only(JsonElement v, string param, params string[] allowed)
    {
        foreach (var field in Fields(v, param))
        {
            if (Array.IndexOf(allowed, field.Name) < 0)
            {
                throw Refuse(param + "." + field.Name, $"Option '{field.Name}' of {param} is unsupported for this type.");
            }
        }
    }

    [GeneratedRegex("^[A-Za-z0-9_-]{1,64}$")]
    private static partial Regex SchemaName();
}

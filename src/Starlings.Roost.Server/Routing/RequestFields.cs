using System.Text.Json;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Server.Routing;

/// <summary>
/// Field readers shared by every route's request reader (moved out of the R-037 OpenAI reader for R-074). A field that is not
/// what it must be throws a <see cref="RoostChatException"/> with <see cref="RoostErrorCode.InvalidOption"/> whose
/// <see cref="RoostError.Param"/> names it; every route turns that into its own error body, a 400.
/// </summary>
internal static class RequestFields
{
    /// <summary>The set fields of a request body, which must be a JSON object.</summary>
    internal static IEnumerable<JsonProperty> Fields(JsonElement body)
    {
        if (body.ValueKind != JsonValueKind.Object)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The request body must be a JSON object."));
        }

        return Fields(body, null);
    }

    /// <summary>The set fields of an object (null means unset), refusing a field that appears twice.</summary>
    internal static IEnumerable<JsonProperty> Fields(JsonElement v, string? at)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var field in v.EnumerateObject())
        {
            string param = at is null ? field.Name : at + "." + field.Name;
            if (!seen.Add(field.Name))
            {
                throw Refuse(param, $"'{field.Name}' appears more than once.");
            }

            if (field.Value.ValueKind != JsonValueKind.Null)
            {
                yield return field;
            }
        }
    }

    internal static string StringValue(JsonElement v, string param) =>
        v.ValueKind == JsonValueKind.String ? v.GetString()! : throw Refuse(param, $"{param} must be a string.");

    internal static bool Bool(JsonElement v, string param) => v.ValueKind switch
    {
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        _ => throw Refuse(param, $"{param} must be a boolean."),
    };

    internal static float Number(JsonElement v, string param) =>
        v.ValueKind == JsonValueKind.Number ? (float)v.GetDouble() : throw Refuse(param, $"{param} must be a number.");

    internal static int Int(JsonElement v, string param) =>
        v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out var i) ? i : throw Refuse(param, $"{param} must be an integer.");

    internal static int PositiveInt(JsonElement v, string param)
    {
        var i = Int(v, param);
        return i > 0 ? i : throw Refuse(param, Inv($"{param} must be positive; got {i}."));
    }

    internal static string Inv(FormattableString text) => FormattableString.Invariant(text);

    internal static RoostChatException Unknown(string name) =>
        Refuse(name, $"Unknown or unsupported option '{name}'.");

    internal static RoostChatException Refuse(string param, string message) =>
        new(new RoostError(RoostErrorCode.InvalidOption, message, param));
}

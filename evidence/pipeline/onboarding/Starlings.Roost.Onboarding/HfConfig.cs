using System.Globalization;
using System.Text.Json;

namespace Starlings.Roost.Onboarding;

/// <summary>A config.json that inspect cannot read: not an object, a required field absent, or a field of the wrong type.</summary>
public sealed class OnboardingConfigException : Exception
{
    /// <summary>Creates the exception.</summary>
    public OnboardingConfigException()
    {
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="message">What is wrong, naming the field.</param>
    public OnboardingConfigException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="message">What is wrong.</param>
    /// <param name="inner">The cause.</param>
    public OnboardingConfigException(string message, Exception inner)
        : base(message, inner)
    {
    }
}

/// <summary>
/// Typed reads of one JSON object of a Hugging Face config. A JSON null reads as absent, as transformers treats it. Reads
/// fall back to a second object (the top level of a multimodal wrapper) so a field the wrapper keeps outside
/// <c>text_config</c>, such as <c>tie_word_embeddings</c>, is still found.
/// </summary>
internal readonly struct HfConfig
{
    private readonly JsonElement _obj;
    private readonly JsonElement? _fallback;

    public HfConfig(JsonElement obj, JsonElement? fallback = null)
    {
        _obj = obj;
        _fallback = fallback;
    }

    public JsonElement Element => _obj;

    public bool Has(string name) => TryGet(name, out _);

    public bool TryGet(string name, out JsonElement value)
    {
        if (_obj.TryGetProperty(name, out value) && value.ValueKind != JsonValueKind.Null)
        {
            return true;
        }

        if (_fallback is { } f && f.TryGetProperty(name, out value) && value.ValueKind != JsonValueKind.Null)
        {
            return true;
        }

        value = default;
        return false;
    }

    public int RequiredInt(string name) =>
        OptInt(name) ?? throw new OnboardingConfigException($"config.json: required field '{name}' is absent.");

    public int? OptInt(string name)
    {
        if (!TryGet(name, out var v))
        {
            return null;
        }

        if (v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int i))
        {
            return i;
        }

        throw new OnboardingConfigException($"config.json: field '{name}' is {Describe(v)}, not an integer.");
    }

    public double? OptDouble(string name)
    {
        if (!TryGet(name, out var v))
        {
            return null;
        }

        return v.ValueKind == JsonValueKind.Number
            ? v.GetDouble()
            : throw new OnboardingConfigException($"config.json: field '{name}' is {Describe(v)}, not a number.");
    }

    public bool? OptBool(string name)
    {
        if (!TryGet(name, out var v))
        {
            return null;
        }

        return v.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => throw new OnboardingConfigException($"config.json: field '{name}' is {Describe(v)}, not a boolean."),
        };
    }

    public string? OptString(string name)
    {
        if (!TryGet(name, out var v))
        {
            return null;
        }

        return v.ValueKind == JsonValueKind.String
            ? v.GetString()
            : throw new OnboardingConfigException($"config.json: field '{name}' is {Describe(v)}, not a string.");
    }

    public HfConfig? OptObject(string name)
    {
        if (!TryGet(name, out var v))
        {
            return null;
        }

        return v.ValueKind == JsonValueKind.Object
            ? new HfConfig(v)
            : throw new OnboardingConfigException($"config.json: field '{name}' is {Describe(v)}, not an object.");
    }

    public IReadOnlyList<string>? OptStrings(string name)
    {
        if (!TryGet(name, out var v))
        {
            return null;
        }

        if (v.ValueKind != JsonValueKind.Array)
        {
            throw new OnboardingConfigException($"config.json: field '{name}' is {Describe(v)}, not an array.");
        }

        var list = new List<string>(v.GetArrayLength());
        foreach (var e in v.EnumerateArray())
        {
            list.Add(e.ValueKind == JsonValueKind.String
                ? e.GetString()!
                : throw new OnboardingConfigException($"config.json: '{name}' holds {Describe(e)}, not a string."));
        }

        return list;
    }

    public IReadOnlyList<int>? OptInts(string name)
    {
        if (!TryGet(name, out var v))
        {
            return null;
        }

        if (v.ValueKind != JsonValueKind.Array)
        {
            throw new OnboardingConfigException($"config.json: field '{name}' is {Describe(v)}, not an array.");
        }

        var list = new List<int>(v.GetArrayLength());
        foreach (var e in v.EnumerateArray())
        {
            list.Add(e.ValueKind == JsonValueKind.Number && e.TryGetInt32(out int i)
                ? i
                : throw new OnboardingConfigException($"config.json: '{name}' holds {Describe(e)}, not an integer."));
        }

        return list;
    }

    public static string Describe(JsonElement v) => v.ValueKind switch
    {
        JsonValueKind.Number => "the number " + v.GetRawText(),
        JsonValueKind.String => "the string \"" + v.GetString() + "\"",
        JsonValueKind.True or JsonValueKind.False => "a boolean",
        JsonValueKind.Array => "an array",
        JsonValueKind.Object => "an object",
        _ => v.ValueKind.ToString().ToLowerInvariant(),
    };

    public static string Number(double d) => d.ToString("R", CultureInfo.InvariantCulture);
}

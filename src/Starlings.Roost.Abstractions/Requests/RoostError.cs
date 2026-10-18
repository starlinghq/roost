using System.Buffers;
using System.Text.Json;

namespace Starlings.Roost.Requests;

/// <summary>Typed error codes (contract C6). The numeric values and <see cref="RoostError.CodeName"/> strings are stable.</summary>
public enum RoostErrorCode
{
    /// <summary>Not enough device memory for the request or the profile.</summary>
    InsufficientDeviceMemory = 1,

    /// <summary>The prompt exceeds the profile's context length.</summary>
    PromptTooLong = 2,

    /// <summary>Admission refused: the queue is full.</summary>
    QueueFull = 3,

    /// <summary>An option is invalid or unsupported by the profile (including grammar keywords the matcher lacks, S10).</summary>
    InvalidOption = 4,

    /// <summary>The GPU faulted and the request could not be resumed.</summary>
    DeviceFault = 5,

    /// <summary>The worker restarted and the request could not be resumed.</summary>
    WorkerRestarted = 6,

    /// <summary>A kernel, weight or manifest digest does not match.</summary>
    AssetMismatch = 7,

    /// <summary>The requested profile or capability is not supported.</summary>
    ProfileUnsupported = 8,

    /// <summary>Anything else.</summary>
    Internal = 9,
}

/// <summary>An error with its stable code, HTTP status and body.</summary>
/// <param name="Code">The code.</param>
/// <param name="Message">Human-readable message; never contains prompt text.</param>
/// <param name="Param">The request field at fault, or null.</param>
public sealed record RoostError(RoostErrorCode Code, string Message, string? Param = null)
{
    /// <summary>The stable snake_case name of <see cref="Code"/> used in HTTP bodies.</summary>
    public string CodeName => NameOf(Code);

    /// <summary>The HTTP status for <see cref="Code"/>.</summary>
    public int HttpStatus => StatusOf(Code);

    /// <summary>The stable snake_case name of a code.</summary>
    /// <param name="code">The code.</param>
    /// <returns>The name.</returns>
    public static string NameOf(RoostErrorCode code) => code switch
    {
        RoostErrorCode.InsufficientDeviceMemory => "insufficient_device_memory",
        RoostErrorCode.PromptTooLong => "prompt_too_long",
        RoostErrorCode.QueueFull => "queue_full",
        RoostErrorCode.InvalidOption => "invalid_option",
        RoostErrorCode.DeviceFault => "device_fault",
        RoostErrorCode.WorkerRestarted => "worker_restarted",
        RoostErrorCode.AssetMismatch => "asset_mismatch",
        RoostErrorCode.ProfileUnsupported => "profile_unsupported",
        _ => "internal",
    };

    /// <summary>The HTTP status of a code.</summary>
    /// <param name="code">The code.</param>
    /// <returns>The status.</returns>
    public static int StatusOf(RoostErrorCode code) => code switch
    {
        RoostErrorCode.InsufficientDeviceMemory => 503,
        RoostErrorCode.PromptTooLong => 400,
        RoostErrorCode.QueueFull => 429,
        RoostErrorCode.InvalidOption => 400,
        RoostErrorCode.DeviceFault => 503,
        RoostErrorCode.WorkerRestarted => 503,
        RoostErrorCode.AssetMismatch => 500,
        RoostErrorCode.ProfileUnsupported => 400,
        _ => 500,
    };

    /// <summary>
    /// Writes the stable HTTP body: <c>{"error":{"message":…,"type":"roost_error","param":…,"code":…}}</c>, the
    /// OpenAI error envelope with Roost's code names.
    /// </summary>
    /// <param name="output">Destination.</param>
    public void WriteHttpBody(IBufferWriter<byte> output)
    {
        using var writer = new Utf8JsonWriter(output);
        writer.WriteStartObject();
        writer.WriteStartObject("error");
        writer.WriteString("message", Message);
        writer.WriteString("type", "roost_error");
        if (Param is null)
        {
            writer.WriteNull("param");
        }
        else
        {
            writer.WriteString("param", Param);
        }

        writer.WriteString("code", CodeName);
        writer.WriteEndObject();
        writer.WriteEndObject();
    }
}

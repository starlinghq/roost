using System.Buffers;
using System.Text.Json;
using Microsoft.AspNetCore.Diagnostics;
using Microsoft.AspNetCore.WebUtilities;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Server.OpenAi;

/// <summary>
/// The one error body of every failure: <c>{"error":{"message","type","param","code"}}</c>, OpenAI's envelope with
/// <c>type</c> <c>roost_error</c>. Request and generation errors are written by <see cref="RoostError.WriteHttpBody"/>
/// (contract C6, with C6's code names and HTTP statuses); routing failures (no such route, wrong method) get the same shape
/// with <c>code</c> <c>not_found</c> or <c>method_not_allowed</c>.
/// </summary>
internal static class ErrorBodies
{
    /// <summary>The <c>type</c> every error body carries, as <see cref="RoostError.WriteHttpBody"/> writes it.</summary>
    public const string ErrorType = "roost_error";

    /// <summary>The body of an error, as bytes.</summary>
    public static byte[] Body(RoostError error)
    {
        var buffer = new ArrayBufferWriter<byte>();
        error.WriteHttpBody(buffer);
        return buffer.WrittenSpan.ToArray();
    }

    /// <summary>Writes an error that has no C6 code (a refusal by the HTTP layer) in the same body shape.</summary>
    /// <param name="response">The response.</param>
    /// <param name="status">The HTTP status.</param>
    /// <param name="code">The <c>code</c> field, snake_case.</param>
    /// <param name="message">The message.</param>
    public static async Task WriteAsync(HttpResponse response, int status, string code, string message)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteStartObject("error");
            writer.WriteString("message", message);
            writer.WriteString("type", ErrorType);
            writer.WriteNull("param");
            writer.WriteString("code", code);
            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        response.StatusCode = status;
        response.ContentType = "application/json";
        await response.Body.WriteAsync(buffer.WrittenMemory).ConfigureAwait(false);
    }

    /// <summary>Writes an error as the whole response, with its HTTP status.</summary>
    public static async Task WriteAsync(HttpResponse response, RoostError error)
    {
        response.StatusCode = error.HttpStatus;
        response.ContentType = "application/json";
        await response.Body.WriteAsync(Body(error)).ConfigureAwait(false);
    }

    /// <summary>Gives an empty error response from routing (404, 405 and the like) the error body.</summary>
    public static async Task StatusCodePageAsync(StatusCodeContext context)
    {
        var http = context.HttpContext;
        int status = http.Response.StatusCode;
        var (code, message) = status switch
        {
            StatusCodes.Status404NotFound => ("not_found", $"No route matches {http.Request.Method} {http.Request.Path}."),
            StatusCodes.Status405MethodNotAllowed => ("method_not_allowed", $"{http.Request.Method} is not allowed on {http.Request.Path}."),
            _ => ("http_" + status.ToString(System.Globalization.CultureInfo.InvariantCulture), ReasonPhrases.GetReasonPhrase(status)),
        };

        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteStartObject("error");
            writer.WriteString("message", message);
            writer.WriteString("type", ErrorType);
            writer.WriteNull("param");
            writer.WriteString("code", code);
            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        http.Response.ContentType = "application/json";
        await http.Response.Body.WriteAsync(buffer.WrittenMemory).ConfigureAwait(false);
    }
}

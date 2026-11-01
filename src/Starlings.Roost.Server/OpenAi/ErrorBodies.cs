using Microsoft.AspNetCore.Diagnostics;
using Microsoft.AspNetCore.WebUtilities;
using Starlings.Roost.Requests;
using Starlings.Roost.Server.Routing;

namespace Starlings.Roost.Server.OpenAi;

/// <summary>
/// The one error body of every failure outside the Anthropic routes: <c>{"error":{"message","type","param","code"}}</c>,
/// OpenAI's envelope with <c>type</c> <c>roost_error</c>, with the fields in the order <see cref="RoostError.WriteHttpBody"/>
/// writes them. Request and generation errors carry contract C6's code names and HTTP statuses; routing failures (no such
/// route, wrong method) and route failures (<see cref="RouteException"/>, such as an unknown stored response) get the same shape
/// with their own <c>code</c> (<c>not_found</c>, <c>method_not_allowed</c>).
/// </summary>
internal sealed class ErrorBodies : IErrorShape
{
    /// <summary>The <c>type</c> every error body carries, as <see cref="RoostError.WriteHttpBody"/> writes it.</summary>
    public const string ErrorType = "roost_error";

    /// <summary>The shape, for <see cref="RouteRunner"/>.</summary>
    public static readonly ErrorBodies Shape = new();

    private ErrorBodies()
    {
    }

    /// <summary>The body of a C6 error, as bytes.</summary>
    public static byte[] Body(RoostError error) => Body(error.CodeName, error.Message, error.Param);

    /// <summary>The body of an error with any code name, as bytes.</summary>
    public static byte[] Body(string code, string message, string? param) => RouteRunner.Json(writer =>
    {
        writer.WriteStartObject();
        writer.WriteStartObject("error");
        writer.WriteString("message", message);
        writer.WriteString("type", ErrorType);
        if (param is null)
        {
            writer.WriteNull("param");
        }
        else
        {
            writer.WriteString("param", param);
        }

        writer.WriteString("code", code);
        writer.WriteEndObject();
        writer.WriteEndObject();
    });

    /// <inheritdoc />
    byte[] IErrorShape.Body(int status, string code, string message, string? param) => Body(code, message, param);

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

        http.Response.ContentType = "application/json";
        await http.Response.Body.WriteAsync(Body(code, message, null)).ConfigureAwait(false);
    }
}

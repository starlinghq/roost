using Starlings.Roost.Server.Routing;

namespace Starlings.Roost.Server.Anthropic;

/// <summary>
/// Anthropic's error body for the <c>/v1/messages</c> routes: <c>{"type":"error","error":{"type","message"}}</c>, with the error
/// type Anthropic gives the HTTP status (400 <c>invalid_request_error</c>, 404 <c>not_found_error</c>, 429
/// <c>rate_limit_error</c>, 500 <c>api_error</c>, 503 <c>overloaded_error</c>, ...). Anthropic's body has no parameter field, so
/// a refused field is named at the start of the message (<c>"tools: Tools are unsupported ..."</c>).
/// </summary>
internal sealed class AnthropicErrors : IErrorShape
{
    /// <summary>The shape, for <see cref="RouteRunner"/>.</summary>
    public static readonly AnthropicErrors Shape = new();

    private AnthropicErrors()
    {
    }

    /// <summary>Anthropic's error type for an HTTP status.</summary>
    public static string TypeOf(int status) => status switch
    {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 or 529 => "overloaded_error",
        _ => "api_error",
    };

    /// <inheritdoc />
    public byte[] Body(int status, string code, string message, string? param) => RouteRunner.Json(w =>
    {
        w.WriteStartObject();
        w.WriteString("type", "error");
        w.WriteStartObject("error");
        w.WriteString("type", TypeOf(status));
        w.WriteString("message", param is null ? message : param + ": " + message);
        w.WriteEndObject();
        w.WriteEndObject();
    });
}

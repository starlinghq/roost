using System.Net;
using Starlings.Roost.Server.OpenAi;

namespace Starlings.Roost.Server.Routing;

/// <summary>
/// Refuses requests a web page made. Listening on loopback keeps other machines out, not a browser on this one: any page
/// can send a POST to <c>127.0.0.1</c> without a preflight, and DNS rebinding makes such a page same-origin, so it can read
/// the answers. The server has no authentication, so it checks two headers. A browser sends <c>Origin</c> on every
/// cross-origin POST and a page cannot remove it; non-browser clients do not send it, so they are unaffected.
/// </summary>
/// <remarks>
/// <c>Host</c> must name this machine when the listener is loopback (this stops DNS rebinding). <c>Origin</c>, when
/// present, must be this machine or the host the request was sent to. A listener opened with
/// <see cref="RoostServerOptions.AllowNonLoopback"/> keeps the <c>Origin</c> check and accepts any <c>Host</c>, because
/// the operator chose the names it answers to.
/// </remarks>
internal static class RequestOriginGuard
{
    /// <summary>Why a request is refused, or null to let it through.</summary>
    /// <param name="host">The request's <c>Host</c> header.</param>
    /// <param name="origin">The request's <c>Origin</c> header; null or empty when absent.</param>
    /// <param name="loopbackListener">Whether the server listens on loopback.</param>
    /// <returns>The reason, without any header value.</returns>
    public static string? Refusal(HostString host, string? origin, bool loopbackListener)
    {
        if (loopbackListener && !IsThisMachine(host.Host))
        {
            return "The Host header does not name this machine.";
        }

        if (string.IsNullOrEmpty(origin))
        {
            return null;
        }

        if (!Uri.TryCreate(origin, UriKind.Absolute, out var uri))
        {
            return "Requests from a web page are refused.";
        }

        return IsThisMachine(uri.Host) || string.Equals(uri.Host, host.Host, StringComparison.OrdinalIgnoreCase)
            ? null
            : "Requests from a web page are refused.";
    }

    /// <summary>The middleware.</summary>
    /// <param name="loopbackListener">Whether the server listens on loopback.</param>
    /// <returns>The middleware delegate.</returns>
    public static Func<HttpContext, RequestDelegate, Task> Middleware(bool loopbackListener) => (context, next) =>
    {
        var refusal = Refusal(context.Request.Host, context.Request.Headers.Origin.ToString(), loopbackListener);
        return refusal is null
            ? next(context)
            : ErrorBodies.WriteAsync(context.Response, StatusCodes.Status403Forbidden, "forbidden_origin", refusal);
    };

    private static bool IsThisMachine(string host)
    {
        if (string.Equals(host, "localhost", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        var bare = host.StartsWith('[') && host.EndsWith(']') ? host[1..^1] : host;
        return IPAddress.TryParse(bare, out var address) && IPAddress.IsLoopback(address);
    }
}

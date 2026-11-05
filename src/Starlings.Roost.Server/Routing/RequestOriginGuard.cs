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
/// <c>Host</c> must name this machine or a name in <see cref="RoostServerOptions.AllowedHosts"/>, which stops DNS rebinding.
/// A request without one can't come from a browser, so it passes. <c>Origin</c>, when present, must be an allowed name or
/// the request's own host; on a loopback listener any name of this machine also passes, for a local UI on another port.
/// A listener opened with <see cref="RoostServerOptions.AllowNonLoopback"/> and no allowed names accepts any <c>Host</c>:
/// the server can't know the names it answers to, so rebinding is not stopped there.
/// </remarks>
internal static class RequestOriginGuard
{
    private const string PageRefused = "Requests from a web page are refused.";

    /// <summary>Why a request is refused, or null to let it through.</summary>
    /// <param name="host">The request's <c>Host</c> header.</param>
    /// <param name="origin">The request's <c>Origin</c> header; null or empty when absent.</param>
    /// <param name="loopbackListener">Whether the server listens on loopback.</param>
    /// <param name="allowedHosts">Names allowed besides this machine.</param>
    /// <returns>The reason, without any header value.</returns>
    public static string? Refusal(HostString host, string? origin, bool loopbackListener, IReadOnlyList<string> allowedHosts)
    {
        bool checkHost = loopbackListener || allowedHosts.Count > 0;
        if (checkHost && !NamesThisMachine(host) && !IsAllowed(host.Host, allowedHosts))
        {
            return "The Host header does not name this server.";
        }

        if (string.IsNullOrEmpty(origin))
        {
            return null;
        }

        if (!Uri.TryCreate(origin, UriKind.Absolute, out var uri))
        {
            return PageRefused;
        }

        bool accepted = string.Equals(uri.Host, host.Host, StringComparison.OrdinalIgnoreCase)
            || IsAllowed(uri.Host, allowedHosts)
            || (loopbackListener && IsThisMachine(uri.Host));
        return accepted ? null : PageRefused;
    }

    /// <summary>Whether a <c>Host</c> header is absent or names this machine.</summary>
    /// <param name="host">The header.</param>
    /// <returns>True when it does.</returns>
    public static bool NamesThisMachine(HostString host) => !host.HasValue || IsThisMachine(host.Host);

    /// <summary>The middleware.</summary>
    /// <param name="loopbackListener">Whether the server listens on loopback.</param>
    /// <param name="allowedHosts">Names allowed besides this machine.</param>
    /// <returns>The middleware delegate.</returns>
    public static Func<HttpContext, RequestDelegate, Task> Middleware(bool loopbackListener, IReadOnlyList<string> allowedHosts) => (context, next) =>
    {
        var refusal = Refusal(context.Request.Host, context.Request.Headers.Origin.ToString(), loopbackListener, allowedHosts);
        return refusal is null
            ? next(context)
            : ErrorBodies.WriteAsync(context.Response, StatusCodes.Status403Forbidden, "forbidden_origin", refusal);
    };

    private static bool IsAllowed(string host, IReadOnlyList<string> allowedHosts) =>
        allowedHosts.Contains(host, StringComparer.OrdinalIgnoreCase);

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

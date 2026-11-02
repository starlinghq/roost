using System.Net;

namespace Starlings.Roost.Server;

/// <summary>Where a <see cref="RoostServer"/> listens.</summary>
public sealed record RoostServerOptions
{
    /// <summary>The default address: IPv4 loopback.</summary>
    public const string DefaultHost = "127.0.0.1";

    /// <summary>The default port, vLLM's.</summary>
    public const int DefaultPort = 8000;

    /// <summary>
    /// The IP address to listen on, or <c>localhost</c> (IPv4 loopback). A non-loopback address is refused unless
    /// <see cref="AllowNonLoopback"/> is set: the server has no authentication.
    /// </summary>
    public string Host { get; init; } = DefaultHost;

    /// <summary>The TCP port; 0 lets the operating system pick one.</summary>
    public int Port { get; init; } = DefaultPort;

    /// <summary>Allows a <see cref="Host"/> that is not a loopback address. Off by default.</summary>
    public bool AllowNonLoopback { get; init; }

    /// <summary>
    /// Maps the sleep-mode routes (R-100): <c>POST /sleep</c>, <c>POST /wake_up</c>, <c>GET /is_sleeping</c> and
    /// <c>POST /update_weights</c>, as vLLM's <c>--enable-sleep-mode</c> does. Off by default; even when on, they answer only
    /// requests from a loopback address.
    /// </summary>
    public bool EnableSleepMode { get; init; }

    /// <summary>Resolves <see cref="Host"/> and checks it and <see cref="Port"/>.</summary>
    /// <returns>The address to listen on.</returns>
    /// <exception cref="ArgumentException">The host is not an IP address or <c>localhost</c>, the port is out of range, or the
    /// address is not loopback and <see cref="AllowNonLoopback"/> is not set.</exception>
    public IPAddress ResolveAddress()
    {
        if (Port is < 0 or > 65535)
        {
            throw new ArgumentException($"Port {Port} is out of range (0 to 65535).", nameof(Port));
        }

        IPAddress address;
        if (string.Equals(Host, "localhost", StringComparison.OrdinalIgnoreCase))
        {
            address = IPAddress.Loopback;
        }
        else if (!IPAddress.TryParse(Host, out address!))
        {
            throw new ArgumentException($"Host '{Host}' is not an IP address or 'localhost'.", nameof(Host));
        }

        if (!IPAddress.IsLoopback(address) && !AllowNonLoopback)
        {
            throw new ArgumentException(
                $"Host '{Host}' is not a loopback address. The server has no authentication, so it listens on loopback only unless non-loopback listening is allowed explicitly (--allow-non-loopback).",
                nameof(Host));
        }

        return address;
    }
}

using System.Net;
using Starlings.Roost.Server.Pooling;

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
    /// Names the server answers to besides this machine and its listen address, for <c>Host</c> and <c>Origin</c>
    /// (<c>--allowed-host</c>): a same-host
    /// reverse proxy's public name, for example. On a listener opened with <see cref="AllowNonLoopback"/>, leaving this empty
    /// accepts any <c>Host</c>, so DNS rebinding can reach the server; set it there unless the network is trusted.
    /// </summary>
    public IReadOnlyList<string> AllowedHosts { get; init; } = [];

    /// <summary>The default <see cref="ResponseStoreCapacity"/>.</summary>
    public const int DefaultResponseStoreCapacity = 1024;

    /// <summary>
    /// The most Responses API responses kept in memory for <c>GET /v1/responses/{id}</c> (R-074). When the store is full the
    /// oldest finished response makes room; when every stored response is still running, a new stored one is refused (429).
    /// </summary>
    public int ResponseStoreCapacity { get; init; } = DefaultResponseStoreCapacity;

    /// <summary>How long a stored response is kept after it was created; a running one is kept until it finishes. One hour by default.</summary>
    public TimeSpan ResponseStoreTtl { get; init; } = TimeSpan.FromHours(1);

    /// <summary>
    /// The pooling models behind <c>/v1/embeddings</c> and <c>/v1/extract</c> (R-052); none by default. The caller owns them and
    /// disposes them after the server stops.
    /// </summary>
    public RoostPoolingModels Pooling { get; init; } = RoostPoolingModels.None;

    /// <summary>The clock of the response store's expiry (tests replace it).</summary>
    internal TimeProvider TimeProvider { get; init; } = TimeProvider.System;

    /// <summary>The cold-start stages <c>GET /metrics</c> exports (R-046), or null when the host was not started by <c>roost serve</c>.</summary>
    internal ColdStart? ColdStart { get; init; }


    /// <summary>
    /// Maps the sleep-mode routes (R-100): <c>POST /sleep</c>, <c>POST /wake_up</c>, <c>GET /is_sleeping</c> and
    /// <c>POST /update_weights</c>, as vLLM's <c>--enable-sleep-mode</c> does. Off by default; even when on, they answer only
    /// requests from this machine, which a proxy or SSH forward that sends a loopback <c>Host</c> turns every client into.
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

using Starlings.Roost.Generation;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost;

/// <summary>How the worker is started and reached (contract C8).</summary>
public sealed record RoostWorkerOptions
{
    /// <summary>
    /// The transport. <see cref="TransportKind.Process"/> (the default, decision 21) runs the worker in its own process, so a
    /// sticky CUDA fault cannot take the application down. <see cref="TransportKind.InProcess"/> is for tests and development
    /// only.
    /// </summary>
    public TransportKind Transport { get; init; } = WorkerTransports.Default;

    /// <summary>How to start the worker process; required for <see cref="TransportKind.Process"/>.</summary>
    public WorkerCommand? Command { get; init; }

    /// <summary>Makes the step executor for <see cref="TransportKind.InProcess"/>; required for it.</summary>
    public Func<IStepExecutor>? InProcessExecutor { get; init; }

    /// <summary>CUDA device ordinal the worker opens.</summary>
    public int DeviceOrdinal { get; init; }

    /// <summary>Per-step deadline; a step that takes longer is a hang or a stall (C8).</summary>
    public TimeSpan StepDeadline { get; init; } = TimeSpan.FromSeconds(60);

    /// <summary>Heartbeat period of the worker.</summary>
    public TimeSpan HeartbeatPeriod { get; init; } = TimeSpan.FromMilliseconds(1);

    /// <summary>Host-side transport settings (startup and shutdown timeouts, segment directory); defaults when null.</summary>
    public WorkerTransportOptions? TransportOptions { get; init; }
}

/// <summary>Settings of a <see cref="RoostChatClient"/>.</summary>
public sealed record RoostChatClientOptions
{
    /// <summary>The model the worker serves.</summary>
    public required RoostModelProfile Model { get; init; }

    /// <summary>How the worker is started.</summary>
    public RoostWorkerOptions Worker { get; init; } = new();

    /// <summary>
    /// Makes the matcher for a request's stop strings. The default refuses any stop string (a typed
    /// <see cref="Starlings.Roost.Requests.RoostErrorCode.InvalidOption"/> naming <c>stop_sequences</c>) until Starling publishes
    /// a stop-matching API (starling#124); pass a Starling-backed factory to enable them.
    /// </summary>
    public IStopMatcherFactory StopMatcherFactory { get; init; } = PendingStopMatcherFactory.Instance;

    /// <summary>Output tokens when <c>ChatOptions.MaxOutputTokens</c> is not set (always capped by the context left).</summary>
    public int DefaultMaxOutputTokens { get; init; } = 512;
}

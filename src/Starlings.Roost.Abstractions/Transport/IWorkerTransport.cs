using System.Collections.Immutable;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Transport;

/// <summary>Which transport carries plans and results (C8).</summary>
public enum TransportKind
{
    /// <summary>One worker process per GPU over a shared-memory segment; the default (decision 21, S8).</summary>
    Process = 0,

    /// <summary>The worker on a thread of the host process; tests and development only.</summary>
    InProcess,
}

/// <summary>What the host offers in the handshake.</summary>
public sealed record WorkerLaunch
{
    /// <summary>CUDA device ordinal.</summary>
    public required int DeviceOrdinal { get; init; }

    /// <summary>Mesh rank and size (C9).</summary>
    public required (int Rank, int Size) Mesh { get; init; }

    /// <summary>SHA-256 of the kernel manifest (C5), 32 bytes.</summary>
    public required ImmutableArray<byte> ManifestDigest { get; init; }

    /// <summary>
    /// SHA-256 of the weights as recorded in the manifest and profile lock, 32 bytes. Compared, never recomputed at
    /// spawn (S8: hashing 3.44 GB took 9.4 s).
    /// </summary>
    public required ImmutableArray<byte> WeightsDigest { get; init; }

    /// <summary>Per-step deadline; a missed deadline is a hang or a stall.</summary>
    public required TimeSpan StepDeadline { get; init; }

    /// <summary>Heartbeat period.</summary>
    public TimeSpan HeartbeatPeriod { get; init; } = TimeSpan.FromMilliseconds(1);

    /// <summary>Plan buffers; at least 2 so two steps can be in flight (S8).</summary>
    public int PlanBufferCount { get; init; } = 2;

    /// <summary>Result buffers; at least 2.</summary>
    public int ResultBufferCount { get; init; } = 2;

    /// <summary>Bytes per plan buffer.</summary>
    public required long PlanBufferBytes { get; init; }

    /// <summary>Bytes per result buffer.</summary>
    public required long ResultBufferBytes { get; init; }

    /// <summary>Bytes of the input region for embeddings and encoder inputs.</summary>
    public long InputRegionBytes { get; init; }
}

/// <summary>Outcome of the handshake.</summary>
/// <param name="State">Ready or rejected.</param>
/// <param name="Reject">Why, when rejected.</param>
/// <param name="WorkerProcessId">Worker process id; the host's own for in-process.</param>
/// <param name="DeviceUuid">The device's 16-byte UUID.</param>
public sealed record HandshakeResult(HandshakeState State, HandshakeReject Reject, long WorkerProcessId, ImmutableArray<byte> DeviceUuid);

/// <summary>What the host should do about a fault (C8 kill, respawn and resume rules).</summary>
public enum RecoveryAction
{
    /// <summary>Nothing is wrong.</summary>
    None = 0,

    /// <summary>Fail or retry the affected sequences; keep the worker.</summary>
    RetryStep,

    /// <summary>Kill the worker if alive, respawn it, re-handshake, reload weights and resume requests with <c>Recovered</c> set.</summary>
    Respawn,
}

/// <summary>The C8 fault rules (S8).</summary>
public static class FaultRules
{
    /// <summary>The action for a fault class.</summary>
    /// <param name="fault">The fault.</param>
    /// <returns>Recoverable faults keep the worker; sticky faults, hangs, stalls and exits respawn it.</returns>
    public static RecoveryAction ActionFor(FaultClass fault) => fault switch
    {
        FaultClass.None => RecoveryAction.None,
        FaultClass.Recoverable => RecoveryAction.RetryStep,
        _ => RecoveryAction.Respawn,
    };

    /// <summary>Classifies a missed step deadline from the heartbeat.</summary>
    /// <param name="heartbeatAdvanced">Whether the heartbeat counter moved since the step was published.</param>
    /// <returns><see cref="FaultClass.Hang"/> when it moved, <see cref="FaultClass.Stalled"/> when not.</returns>
    public static FaultClass ClassifyMissedDeadline(bool heartbeatAdvanced) =>
        heartbeatAdvanced ? FaultClass.Hang : FaultClass.Stalled;
}

/// <summary>
/// The worker transport (contract C8): plan and result rings over double buffers in a shared-memory segment that the
/// worker registers with <c>cudaHostRegister</c>; spawn and handshake; heartbeat; per-step deadline; fault classes.
/// The process and in-process transports implement the same interface. Single host thread; never allocates per step.
/// </summary>
public interface IWorkerTransport : IAsyncDisposable
{
    /// <summary>The transport kind.</summary>
    TransportKind Kind { get; }

    /// <summary>Spawns (or starts) the worker and runs the handshake.</summary>
    /// <param name="launch">The offer.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The outcome; a rejected worker has exited.</returns>
    ValueTask<HandshakeResult> StartAsync(WorkerLaunch launch, CancellationToken cancellationToken);

    /// <summary>Plan buffers in the segment.</summary>
    int PlanBufferCount { get; }

    /// <summary>A plan buffer to write a step plan into (header, then tables at their offsets).</summary>
    /// <param name="index">Buffer index.</param>
    /// <returns>The buffer.</returns>
    Span<byte> PlanBuffer(int index);

    /// <summary>Publishes a plan slot on the plan ring.</summary>
    /// <param name="slot">The slot; <c>BufferIndex</c> names a plan buffer the worker is not reading.</param>
    /// <returns>False when the ring is full.</returns>
    bool TryPublish(in RingSlot slot);

    /// <summary>Takes the next result slot from the result ring.</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>False when none is ready.</returns>
    bool TryTakeResult(out RingSlot slot);

    /// <summary>A result buffer named by a result slot.</summary>
    /// <param name="index">Buffer index.</param>
    /// <returns>The buffer, valid until <see cref="ReleaseResult"/>.</returns>
    ReadOnlySpan<byte> ResultBuffer(int index);

    /// <summary>Returns a result buffer to the worker.</summary>
    /// <param name="index">Buffer index.</param>
    void ReleaseResult(int index);

    /// <summary>The worker's latest heartbeat.</summary>
    /// <returns>A copy of the record.</returns>
    Heartbeat ReadHeartbeat();

    /// <summary>Checks the in-flight step's deadline, the heartbeat and the process.</summary>
    /// <returns>The fault seen, or <see cref="FaultClass.None"/>.</returns>
    FaultClass Poll();

    /// <summary>Kills the worker. In-process, a hung step cannot be killed and the host has to exit (S8).</summary>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>A task that completes when the worker is reaped.</returns>
    ValueTask KillAsync(CancellationToken cancellationToken);
}

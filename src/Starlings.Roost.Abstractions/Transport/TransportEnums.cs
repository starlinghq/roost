namespace Starlings.Roost.Transport;

/// <summary>Ring slot kinds (<c>ROOST_RING_SLOT_*</c>).</summary>
public enum RingSlotKind : uint
{
    /// <summary>Empty.</summary>
    None = 0,

    /// <summary>A step plan or result.</summary>
    Step = 1,

    /// <summary>Exit after the in-flight steps.</summary>
    Shutdown = 2,

    /// <summary>Finish the in-flight steps, then wait.</summary>
    Drain = 3,

    /// <summary>Layout 3 (R-100): release the KV and recurrent state pools, and the weights at <see cref="SleepLevel.Weights"/>. Carries a <see cref="ControlMessage"/>.</summary>
    Sleep = 4,

    /// <summary>Layout 3 (R-100): restore what the sleep released, checking the weights digest. Carries a <see cref="ControlMessage"/>.</summary>
    Wake = 5,

    /// <summary>Layout 3 (R-100): load weights from a locked file, verified as read, and swap them in. Carries a <see cref="ControlMessage"/>.</summary>
    ReplaceWeights = 6,
}

/// <summary>What a sleep releases (<c>ROOST_SLEEP_LEVEL_*</c>, layout 3, R-100).</summary>
public enum SleepLevel : uint
{
    /// <summary>Not a sleep message.</summary>
    None = 0,

    /// <summary>The KV and recurrent state pools; the weights stay resident.</summary>
    Kv = 1,

    /// <summary>The state pools and the weights; a wake reloads the weights from their locked file.</summary>
    Weights = 2,
}

/// <summary>Outcome of a control message (<c>ROOST_CONTROL_*</c>, layout 3, R-100).</summary>
public enum ControlStatus : uint
{
    /// <summary>Done.</summary>
    Ok = 0,

    /// <summary>Not valid in the worker's state: a sleep while asleep, a wake while awake.</summary>
    RefusedState = 1,

    /// <summary>The weights did not match the digest (wake) or their lock (replace); the worker kept what it had.</summary>
    WeightsDigest = 2,

    /// <summary>The worker cannot do this; nothing was changed.</summary>
    Unsupported = 3,

    /// <summary>Anything else; the error code and text say why.</summary>
    Failed = 4,

    /// <summary>The message did not parse.</summary>
    Malformed = 5,
}

/// <summary>Handshake states (<c>ROOST_HANDSHAKE_*</c>).</summary>
public enum HandshakeState : uint
{
    /// <summary>Nothing written.</summary>
    Empty = 0,

    /// <summary>The host wrote its side.</summary>
    Offered = 1,

    /// <summary>The worker accepted: context, segment registration and weights are done.</summary>
    Ready = 2,

    /// <summary>The worker rejected the offer; see <see cref="HandshakeReject"/>.</summary>
    Rejected = 3,
}

/// <summary>Handshake rejection reasons (<c>ROOST_REJECT_*</c>).</summary>
public enum HandshakeReject : uint
{
    /// <summary>Not rejected.</summary>
    None = 0,

    /// <summary>Contract version differs.</summary>
    Contract = 1,

    /// <summary>Kernel library ABI incompatible.</summary>
    Abi = 2,

    /// <summary>Kernel manifest digest differs.</summary>
    ManifestDigest = 3,

    /// <summary>Weights digest differs from the manifest-recorded one.</summary>
    WeightsDigest = 4,

    /// <summary>Device missing or different.</summary>
    Device = 5,

    /// <summary>Mesh rank or size invalid.</summary>
    Mesh = 6,

    /// <summary>Shared-memory layout version differs.</summary>
    Layout = 7,
}

/// <summary>Worker states published in the heartbeat (<c>ROOST_WORKER_*</c>).</summary>
public enum WorkerState : uint
{
    /// <summary>Spawned, handshake not done.</summary>
    Starting = 0,

    /// <summary>Waiting for plans.</summary>
    Ready = 1,

    /// <summary>Running a step.</summary>
    Stepping = 2,

    /// <summary>Sticky fault seen; exiting.</summary>
    Faulted = 3,

    /// <summary>Shutting down.</summary>
    Exiting = 4,

    /// <summary>Layout 3 (R-100): asleep; step plans are refused until a wake.</summary>
    Sleeping = 5,
}

/// <summary>Helpers over <see cref="RingSlotKind"/>.</summary>
public static class RingSlotKinds
{
    /// <summary>
    /// Whether slots of a kind name a buffer: a step names the plan or result buffer of its slot number, and so does a control
    /// message (layout 3); shutdown and drain name none.
    /// </summary>
    /// <param name="kind">The kind.</param>
    /// <returns>True for <see cref="RingSlotKind.Step"/>, <see cref="RingSlotKind.Sleep"/>, <see cref="RingSlotKind.Wake"/> and <see cref="RingSlotKind.ReplaceWeights"/>.</returns>
    public static bool NamesBuffer(this RingSlotKind kind) => kind is RingSlotKind.Step or RingSlotKind.Sleep or RingSlotKind.Wake or RingSlotKind.ReplaceWeights;

    /// <summary>Whether a kind is a control message (layout 3, R-100).</summary>
    /// <param name="kind">The kind.</param>
    /// <returns>True for sleep, wake and replace weights.</returns>
    public static bool IsControl(this RingSlotKind kind) => kind is RingSlotKind.Sleep or RingSlotKind.Wake or RingSlotKind.ReplaceWeights;
}

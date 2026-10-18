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
}

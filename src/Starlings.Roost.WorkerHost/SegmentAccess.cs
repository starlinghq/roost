using System.Runtime.CompilerServices;
using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost;

/// <summary>
/// Typed access to the handshake and heartbeat records of a mapped <see cref="RingSegment"/>. The engine package keeps
/// the segment base private, so the base is taken from <see cref="RingSegment.Header"/>, which is a reference to the
/// header at offset 0 of the mapping; the view is a memory-mapped file that does not move, so the pointers stay valid
/// for as long as the segment is not disposed. The other records are reached by the byte offsets in that header, as
/// the C8 contract says.
/// </summary>
internal sealed unsafe class SegmentAccess
{
    public SegmentAccess(RingSegment segment)
    {
        Segment = segment;
        Base = (byte*)Unsafe.AsPointer(ref Unsafe.AsRef(in segment.Header));
        Handshake* handshake = (Handshake*)(Base + (long)segment.Header.HandshakeOffset);
        Heartbeat* heartbeat = (Heartbeat*)(Base + (long)segment.Header.HeartbeatOffset);
        HandshakeRecord = handshake;
        HeartbeatRecord = heartbeat;
        var resultRing = Unsafe.Read<RingHeader>(Base + (long)segment.Header.ResultRingOffset);
        ResultProducerCursor = (ulong*)(Base + (long)resultRing.ProducerCursorOffset);
    }

    public RingSegment Segment { get; }

    public byte* Base { get; }

    public Handshake* HandshakeRecord { get; }

    public Heartbeat* HeartbeatRecord { get; }

    /// <summary>The result ring's producer cursor, read by the host to see whether a result is waiting without taking it.</summary>
    public ulong* ResultProducerCursor { get; }

    public HandshakeState ReadState() => (HandshakeState)Volatile.Read(ref Unsafe.As<HandshakeState, uint>(ref HandshakeRecord->State));

    /// <summary>Publishes the handshake state with release semantics: every field written before it is visible to a reader that sees the state.</summary>
    public void WriteState(HandshakeState state) => Volatile.Write(ref Unsafe.As<HandshakeState, uint>(ref HandshakeRecord->State), (uint)state);

    public WorkerState ReadWorkerState() => (WorkerState)Volatile.Read(ref Unsafe.As<WorkerState, uint>(ref HeartbeatRecord->State));

    public void WriteWorkerState(WorkerState state) => Volatile.Write(ref Unsafe.As<WorkerState, uint>(ref HeartbeatRecord->State), (uint)state);

    public void WriteLastStep(ulong stepId) => Volatile.Write(ref HeartbeatRecord->LastStepId, stepId);

    /// <summary>One beat: the time first, the counter last with release, so a host that reads a new counter reads a current time.</summary>
    public void Beat()
    {
        Volatile.Write(ref HeartbeatRecord->WorkerNs, MonotonicClock.NowNs());
        Volatile.Write(ref HeartbeatRecord->Counter, Volatile.Read(ref HeartbeatRecord->Counter) + 1);
    }

    /// <summary>A copy of the heartbeat; the counter is read first with acquire.</summary>
    public Heartbeat ReadHeartbeat()
    {
        var counter = Volatile.Read(ref HeartbeatRecord->Counter);
        return new Heartbeat
        {
            Counter = counter,
            WorkerNs = Volatile.Read(ref HeartbeatRecord->WorkerNs),
            LastStepId = Volatile.Read(ref HeartbeatRecord->LastStepId),
            State = ReadWorkerState(),
        };
    }

    public Handshake ReadHandshake() => *HandshakeRecord;

    public ulong HeartbeatCounter() => Volatile.Read(ref HeartbeatRecord->Counter);

    /// <summary>Writes the host's side of the handshake, then publishes <see cref="HandshakeState.Offered"/> last with release.</summary>
    public void WriteOffer(in Handshake offer)
    {
        var copy = offer;
        copy.State = HandshakeState.Empty;
        *HandshakeRecord = copy;
        WriteState(HandshakeState.Offered);
    }

    public ulong ResultsPublished() => Volatile.Read(ref *ResultProducerCursor);
}

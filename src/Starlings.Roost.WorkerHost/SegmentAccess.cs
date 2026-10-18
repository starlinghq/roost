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
/// <remarks>
/// <para><b>Memory-ordering argument for the WorkerHost layer.</b> The ring cursors and slots are R-027's and are argued in
/// <c>RingProducer</c> and <c>RingConsumer</c>; this note covers the other shared words. <c>Volatile.Read</c> is an acquire
/// load and <c>Volatile.Write</c> a release store; every shared word below is a naturally aligned 4 or 8 byte field, so
/// each access is atomic and a reader never sees a torn value. Every record has exactly one writer thread at any time.</para>
/// <list type="number">
/// <item><b>Offer, <see cref="WriteOffer"/> then <see cref="WriteState"/>(Offered).</b> The host fills the whole
/// <see cref="Handshake"/> with plain stores (state Empty), then publishes Offered with a release store in
/// <see cref="WriteState"/>. The worker polls <see cref="ReadState"/> (acquire); when it sees Offered, every offer field
/// stored before the release is visible, so it reads them plainly. The worker never reads the offer fields before it has
/// seen Offered.</item>
/// <item><b>Answer, worker's <see cref="WriteState"/>(Ready or Rejected).</b> The worker writes its answer fields (pid,
/// device UUID, reject reason) plainly, then releases the state. The host reads the state with acquire in
/// <c>WaitForAnswerAsync</c> and only then calls <see cref="ReadHandshake"/> (plain copy). The host never writes the
/// record again after the offer, so the two writers do not overlap (the host writes the offer fields before Offered, the
/// worker writes the answer fields after seeing it).</item>
/// <item><b>Beat, <see cref="Beat"/>.</b> One writer: the worker's heartbeat thread (the one call from <c>Start</c> runs
/// before that thread starts, and <c>Thread.Start</c> orders it). It stores <c>WorkerNs</c>, then stores <c>Counter</c> with
/// release. The counter increment reads its own earlier value, which no other thread changes, so it needs no
/// interlocked add.</item>
/// <item><b>Read, <see cref="ReadHeartbeat"/>.</b> The host loads <c>Counter</c> first with acquire, then <c>WorkerNs</c>.
/// If the counter shows beat n, the acquire makes the <c>WorkerNs</c> stored for beat n visible, so the time read is at
/// least as new as the counter. It may be newer (the worker can beat between the two loads), and it may be newer than
/// the host's own clock read taken just before; <c>HeartbeatAdvanced</c> clamps that age to zero. The other fields
/// (<c>LastStepId</c>, <c>State</c>) are independent words, each read atomically, with no ordering promised among them;
/// the host uses only the counter and the time to classify.</item>
/// <item><b>Worker state and last step, <see cref="WriteWorkerState"/>, <see cref="WriteLastStep"/>.</b> Diagnostic
/// words, written with release by the worker thread and read with acquire by
/// the host; nothing else is published through them. The final state (Exiting, Faulted) is stored before the worker's
/// exit flag is set, so a host that sees the exit sees the state.</item>
/// <item><b>Exit flag.</b> Process transport: <c>Process.Exited</c> runs on a thread-pool thread after the kernel has
/// reaped the child, and stores <c>_exited</c> with release; <c>Poll</c> reads it with acquire. A dead process cannot
/// store again, and everything it stored to the shared mapping before dying is in the page cache that the host's mapping
/// shares, so it is visible to any load the host makes after it has seen the exit. In-process transport: the worker thread
/// stores <c>_ended</c> with release in a <c>finally</c> after its last ring and heartbeat store, so the same holds by
/// release and acquire.</item>
/// <item><b>Why <c>Poll</c> reads the exit before <c>ResultsPublished</c>.</b> If it read the result cursor first and the
/// exit second, a worker could publish its last result and exit between the two reads: the host would see "no result" and
/// "exited" and report Exited for a worker that left a result. Reading the exit first means that, whenever the exit is
/// seen, the later acquire load of the producer cursor sees every result published before the exit; so "exited and no
/// result" is exact, and a waiting result always wins. The same order is used in <c>WaitForAnswerAsync</c>: exit first,
/// then the handshake state, so a worker that answered and exited is seen answering.</item>
/// <item><b>Stop flags.</b> <c>_stopHeartbeat</c>, <c>_exit</c> (runtime), <c>_exited</c>, <c>_ended</c>, <c>_failure</c> and
/// the transport <c>_state</c> are process-local words with single logical writers; the Volatile pairs there make a
/// value, and the stores before it, visible to the polling thread. None is shared across processes.</item>
/// </list>
/// <para>No compute-sanitizer racecheck run is claimed: racecheck sees device code, and none of this is device code. The
/// substitute is this argument and the two-process fault stress test (<c>FaultStressTests</c>), which is a test of the
/// behaviour, not a proof; accepting the substitution is the maintainer's decision.</para>
/// </remarks>
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

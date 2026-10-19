using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Rings;

/// <summary>
/// The worker's side of the rings: takes plan slots, reads the plan buffer, releases it, and publishes results into
/// the result buffers. It lives in the engine package so the in-process transport and the stress test can use the same
/// ring code as the host; the worker host (R-069) wraps it. Single worker thread; never allocates. Ordering and
/// buffer ownership are described on <see cref="HostRingEndpoint"/>, <see cref="RingProducer"/> and
/// <see cref="RingConsumer"/>.
/// </summary>
public sealed unsafe class WorkerRingEndpoint
{
    private readonly RingSegment _segment;
    private readonly RingConsumer _plan;
    private readonly RingProducer _result;
    private readonly byte* _planBuffers;
    private readonly byte* _resultBuffers;
    private readonly int _planBufferBytes;
    private readonly int _resultBufferBytes;
    private readonly int _planBufferCount;
    private readonly int _resultBufferCount;

    internal WorkerRingEndpoint(RingSegment segment)
    {
        _segment = segment;
        var h = segment.Header;
        var b = segment.Base;
        var planRing = System.Runtime.CompilerServices.Unsafe.Read<RingHeader>(b + (long)h.PlanRingOffset);
        var resultRing = System.Runtime.CompilerServices.Unsafe.Read<RingHeader>(b + (long)h.ResultRingOffset);
        _planBufferCount = (int)h.PlanBufferCount;
        _resultBufferCount = (int)h.ResultBufferCount;
        _planBufferBytes = (int)h.PlanBufferBytes;
        _resultBufferBytes = (int)h.ResultBufferBytes;
        _planBuffers = b + (long)h.PlanBuffersOffset;
        _resultBuffers = b + (long)h.ResultBuffersOffset;
        _plan = new RingConsumer(b, planRing);
        _result = new RingProducer(b, resultRing, Math.Min(resultRing.SlotCount, (uint)_resultBufferCount));
    }

    /// <summary>Plans taken so far.</summary>
    public ulong PlansTaken => _plan.Taken;

    /// <summary>Plans taken and not yet released.</summary>
    public int PlansOutstanding => _plan.Outstanding;

    /// <summary>Results published so far.</summary>
    public ulong ResultsPublished => _result.Published;

    /// <summary>The result buffer the next published result must use.</summary>
    public int NextResultBufferIndex => _result.NextIndex(_resultBufferCount);

    /// <summary>Takes the next plan slot; it and its buffer stay owned by the worker until <see cref="ReleasePlan"/>.</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>False when the host has published nothing new.</returns>
    public bool TryTakePlan(out RingSlot slot)
    {
        var sequence = _plan.Taken;
        if (!_plan.TryTake(out slot))
        {
            return false;
        }

        if (slot.Kind == RingSlotKind.Step && slot.BufferIndex != (uint)(sequence % (ulong)_planBufferCount))
        {
            throw new InvalidDataException("A plan slot names a buffer other than slot number modulo buffer count.");
        }

        return true;
    }

    /// <summary>A plan buffer named by a plan slot.</summary>
    /// <param name="index">Buffer index.</param>
    /// <returns>The buffer, valid until <see cref="ReleasePlan"/>.</returns>
    public ReadOnlySpan<byte> PlanBuffer(int index)
    {
        if ((uint)index >= (uint)_planBufferCount)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return new ReadOnlySpan<byte>(_planBuffers + ((long)index * _planBufferBytes), _planBufferBytes);
    }

    /// <summary>Returns the oldest taken plan slot, and its buffer, to the host. Call it only after the buffer has been fully read, including any asynchronous device copy.</summary>
    public void ReleasePlan() => _plan.Release();

    /// <summary>Whether a result slot and buffer are free.</summary>
    /// <returns>False when the host still holds every result buffer.</returns>
    public bool CanPublishResult() => _result.HasSpace();

    /// <summary>A result buffer to write into. Write only the buffer <see cref="NextResultBufferIndex"/> names, after <see cref="CanPublishResult"/> returned true.</summary>
    /// <param name="index">Buffer index.</param>
    /// <returns>The buffer.</returns>
    public Span<byte> ResultBuffer(int index)
    {
        if ((uint)index >= (uint)_resultBufferCount)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return new Span<byte>(_resultBuffers + ((long)index * _resultBufferBytes), _resultBufferBytes);
    }

    /// <summary>Publishes a result slot.</summary>
    /// <param name="slot">The slot; for a step, <c>BufferIndex</c> must be <see cref="NextResultBufferIndex"/>.</param>
    /// <returns>False when the ring is full.</returns>
    public bool TryPublishResult(in RingSlot slot)
    {
        if (slot.Kind == RingSlotKind.Step && slot.BufferIndex != (uint)NextResultBufferIndex)
        {
            throw new InvalidDataException("A result slot names a buffer other than slot number modulo buffer count.");
        }

        return _result.TryPublish(slot);
    }

    /// <summary>Keeps the segment reachable for as long as the endpoint is.</summary>
    internal RingSegment Segment => _segment;
}

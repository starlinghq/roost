using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Rings;

/// <summary>
/// The host's side of the plan and result rings: writes plans into the double-buffered plan area and publishes them,
/// takes results and returns their buffers. Its members match the ring part of <see cref="IWorkerTransport"/> so the
/// process transport (R-069) can forward to it. Single host thread; never allocates.
/// </summary>
/// <remarks>
/// <para><b>Buffer ownership.</b> The C8 header gives each ring slot a <c>buffer_index</c> but no free list, so the
/// engine fixes the mapping: slot <c>n</c> of a ring uses buffer <c>n % buffer_count</c>, and a ring holds at most
/// <c>min(slot_count, buffer_count)</c> slots in flight. A plan buffer goes back to the host when the worker
/// releases the plan slot (after it has finished reading the buffer); a result buffer goes back to the worker when
/// the host calls <see cref="ReleaseResult"/>. Releases are in order. All ordering is the ring cursors' release and
/// acquire pairs described on <see cref="RingProducer"/> and <see cref="RingConsumer"/>; the buffers themselves are
/// plain memory.</para>
/// </remarks>
public sealed unsafe class HostRingEndpoint
{
    private readonly RingSegment _segment;
    private readonly RingProducer _plan;
    private readonly RingConsumer _result;
    private readonly byte* _planBuffers;
    private readonly byte* _resultBuffers;
    private readonly int _planBufferBytes;
    private readonly int _resultBufferBytes;
    private readonly int _planBufferCount;
    private readonly int _resultBufferCount;

    internal HostRingEndpoint(RingSegment segment)
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
        _plan = new RingProducer(b, planRing, Math.Min(planRing.SlotCount, (uint)_planBufferCount));
        _result = new RingConsumer(b, resultRing);
    }

    /// <summary>Plan buffers in the segment.</summary>
    public int PlanBufferCount => _planBufferCount;

    /// <summary>Plans published so far.</summary>
    public ulong PlansPublished => _plan.Published;

    /// <summary>Results taken so far.</summary>
    public ulong ResultsTaken => _result.Taken;

    /// <summary>Results taken and not yet released.</summary>
    public int ResultsOutstanding => _result.Outstanding;

    /// <summary>The plan buffer the next published plan must use.</summary>
    public int NextPlanBufferIndex => _plan.NextIndex(_planBufferCount);

    /// <summary>Whether the worker has released enough plan slots for another plan to be published.</summary>
    /// <returns>False when every plan buffer is still in flight.</returns>
    public bool CanPublish() => _plan.HasSpace();

    /// <summary>A plan buffer to write a step plan into. Write only the buffer <see cref="NextPlanBufferIndex"/> names, and only after <see cref="CanPublish"/> returned true.</summary>
    /// <param name="index">Buffer index.</param>
    /// <returns>The buffer.</returns>
    public Span<byte> PlanBuffer(int index)
    {
        if ((uint)index >= (uint)_planBufferCount)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return new Span<byte>(_planBuffers + ((long)index * _planBufferBytes), _planBufferBytes);
    }

    /// <summary>Publishes a plan slot.</summary>
    /// <param name="slot">The slot; for a step, <c>BufferIndex</c> must be <see cref="NextPlanBufferIndex"/>.</param>
    /// <returns>False when the ring is full.</returns>
    public bool TryPublish(in RingSlot slot)
    {
        if (slot.Kind == RingSlotKind.Step && slot.BufferIndex != (uint)NextPlanBufferIndex)
        {
            ThrowWrongBuffer();
        }

        return _plan.TryPublish(slot);
    }

    /// <summary>Takes the next result slot; it stays owned by the host until <see cref="ReleaseResult"/>.</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>False when none is ready.</returns>
    public bool TryTakeResult(out RingSlot slot)
    {
        var sequence = _result.Taken;
        if (!_result.TryTake(out slot))
        {
            return false;
        }

        if (slot.Kind == RingSlotKind.Step && slot.BufferIndex != (uint)(sequence % (ulong)_resultBufferCount))
        {
            ThrowWrongBuffer();
        }

        return true;
    }

    /// <summary>A result buffer named by a result slot.</summary>
    /// <param name="index">Buffer index.</param>
    /// <returns>The buffer, valid until <see cref="ReleaseResult"/>.</returns>
    public ReadOnlySpan<byte> ResultBuffer(int index)
    {
        if ((uint)index >= (uint)_resultBufferCount)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return new ReadOnlySpan<byte>(_resultBuffers + ((long)index * _resultBufferBytes), _resultBufferBytes);
    }

    /// <summary>Returns the oldest taken result, and its buffer, to the worker.</summary>
    /// <param name="index">The buffer index of that result; checked.</param>
    public void ReleaseResult(int index)
    {
        if ((ulong)index != _result.Released % (ulong)_resultBufferCount)
        {
            throw new ArgumentException("Results are released in order; this is not the oldest outstanding buffer.", nameof(index));
        }

        _result.Release();
    }

    /// <summary>Keeps the segment reachable for as long as the endpoint is.</summary>
    internal RingSegment Segment => _segment;

    private static void ThrowWrongBuffer() => throw new InvalidDataException("A ring slot names a buffer other than slot number modulo buffer count.");
}

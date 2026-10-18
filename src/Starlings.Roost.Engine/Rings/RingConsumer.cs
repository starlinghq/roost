using System.Runtime.CompilerServices;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Rings;

/// <summary>
/// The consumer half of one single-producer single-consumer ring. Taking a slot and releasing it are separate: a
/// slot (and the buffer it names) stays owned by the consumer until <see cref="Release"/>, which is what lets a
/// worker keep reading a plan buffer asynchronously after it has seen the slot.
/// </summary>
/// <remarks>
/// <para>Protocol and cursors are described on <see cref="RingProducer"/>. The consumer-side operations and their
/// ordering:</para>
/// <list type="number">
/// <item><description><c>Volatile.Read(ref *producerCursor)</c> (acquire), in <see cref="TryTake"/> when the cached
/// value shows nothing new. Pairs with the producer's release of <c>P</c>: if it returns <c>P &gt; n</c>, the slot
/// stores and buffer stores the producer made before publishing slot <c>n</c> are visible, so the plain read of the
/// slot (item 2) and of the buffer by the caller see complete data.</description></item>
/// <item><description>The plain read (struct copy) of the slot. The copy is complete before <see cref="Release"/>
/// publishes the slot back, because the release store cannot move before it.</description></item>
/// <item><description><c>Volatile.Write(ref *consumerCursor, C + 1)</c> (release), in <see cref="Release"/>:
/// returns the slot and its buffer to the producer. All the consumer's reads of the slot and the buffer precede it
/// and cannot be delayed past it. The caller must have finished reading the buffer before calling
/// <see cref="Release"/>. For a worker that uploads a plan to the device that means the host-to-device copy has
/// completed on the stream (event or stream synchronised), because a DMA read of host memory is not ordered by the
/// CPU memory model. Pairs with the producer's acquire of <c>C</c>.</description></item>
/// </list>
/// <para><c>_taken</c>, <c>_released</c> and <c>_cachedHead</c> are private fields of the one consumer thread.
/// <c>_cachedHead</c> may be stale low, which only makes the ring look empty when it is not; item 1 refreshes it
/// before <see cref="TryTake"/> reports empty. Nothing here allocates.</para>
/// </remarks>
public sealed unsafe class RingConsumer
{
    private readonly byte* _slots;
    private readonly uint _slotCount;
    private readonly uint _slotBytes;
    private readonly ulong* _producerCursor;
    private readonly ulong* _consumerCursor;
    private ulong _taken;
    private ulong _released;
    private ulong _cachedHead;

    internal RingConsumer(byte* segment, in RingHeader header)
    {
        if (!Environment.Is64BitProcess)
        {
            throw new PlatformNotSupportedException("The rings need atomic 64-bit loads and stores.");
        }

        _slots = segment + (long)header.SlotsOffset;
        _slotCount = header.SlotCount;
        _slotBytes = header.SlotBytes;
        _producerCursor = (ulong*)(segment + (long)header.ProducerCursorOffset);
        _consumerCursor = (ulong*)(segment + (long)header.ConsumerCursorOffset);
        _released = Volatile.Read(ref *_consumerCursor);
        _taken = _released;
        _cachedHead = Volatile.Read(ref *_producerCursor);
    }

    /// <summary>Slots taken so far; the sequence number the next taken slot has.</summary>
    public ulong Taken => _taken;

    /// <summary>Slots released so far (the consumer cursor).</summary>
    public ulong Released => _released;

    /// <summary>Slots taken and not yet released.</summary>
    public int Outstanding => (int)(_taken - _released);

    /// <summary>Takes the next slot without releasing it.</summary>
    /// <param name="slot">A copy of the slot.</param>
    /// <returns>False when the producer has published nothing new.</returns>
    public bool TryTake(out RingSlot slot)
    {
        if (_taken == _cachedHead)
        {
            _cachedHead = Volatile.Read(ref *_producerCursor); // item 1 (acquire)
            if (_cachedHead - _released > _slotCount)
            {
                ThrowCorrupt();
            }

            if (_taken == _cachedHead)
            {
                slot = default;
                return false;
            }
        }

        slot = Unsafe.ReadUnaligned<RingSlot>(_slots + ((_taken % _slotCount) * _slotBytes)); // item 2
        _taken++;
        return true;
    }

    /// <summary>Returns the oldest taken slot, and the buffer it names, to the producer.</summary>
    /// <exception cref="InvalidOperationException">No slot is outstanding.</exception>
    public void Release()
    {
        if (_released == _taken)
        {
            ThrowNothingOutstanding();
        }

        _released++;
        Volatile.Write(ref *_consumerCursor, _released); // item 3 (release)
    }

    private static void ThrowCorrupt() => throw new InvalidDataException("Ring cursors are inconsistent: the producer cursor is ahead of the consumer by more than the ring holds.");

    private static void ThrowNothingOutstanding() => throw new InvalidOperationException("No taken slot to release.");
}

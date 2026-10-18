using System.Runtime.CompilerServices;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Rings;

/// <summary>
/// The producer half of one single-producer single-consumer ring in the C8 segment. One thread in one process owns
/// it; the matching <see cref="RingConsumer"/> lives in the other process (or thread).
/// </summary>
/// <remarks>
/// <para><b>Protocol.</b> The ring has two cursors, monotonically increasing slot counts, each alone on a cache line:
/// the producer cursor <c>P</c> (slots published) and the consumer cursor <c>C</c> (slots released). Slot <c>n</c>
/// lives at index <c>n % slot_count</c>. The ring is full when <c>P - C &gt;= capacity</c>, where capacity is
/// <c>min(slot_count, buffer_count)</c> so that buffer <c>n % buffer_count</c> is never reused while still in flight.
/// <c>P</c> is written only by the producer and <c>C</c> only by the consumer, so no atomic read-modify-write is
/// needed anywhere on a ring; the only operations are the Volatile ones below. (Both cursors are naturally aligned
/// 64-bit values on a 64-bit process, so each load and store is a single atomic access; <see cref="RingSegment"/>
/// validates the alignment and this class requires a 64-bit process.)</para>
/// <para><b>Memory-ordering argument.</b> In .NET, <c>Volatile.Write</c> has release semantics and <c>Volatile.Read</c>
/// has acquire semantics, for the compiler and for the CPU (on x86-64 they are plain moves, because stores are not
/// reordered with earlier stores or loads and loads are not reordered with later loads or stores; on ARM64 they are
/// <c>stlr</c> and <c>ldar</c>). The operations on the producer side are:</para>
/// <list type="number">
/// <item><description><c>Volatile.Read(ref *consumerCursor)</c> (acquire), in <see cref="TryPublish"/> when the cached
/// value says the ring is full. It pairs with the consumer's <c>Volatile.Write</c> of <c>C</c> (<see cref="RingConsumer.Release"/>).
/// Everything the consumer did to slot <c>n</c> and to the buffer named by it before that release (reading the slot,
/// copying or uploading the buffer) happens before anything the producer does after this acquire (reusing the slot,
/// rewriting the buffer). This is what makes it safe to overwrite a slot and a buffer.</description></item>
/// <item><description>The plain stores of the slot (a 32-byte struct copy) and, by the caller, of the plan or result
/// buffer. They are not atomic and may be torn from another thread's point of view; that is harmless because the
/// consumer does not read them until it has seen the cursor move (item 3), and the producer does not write them
/// again until the consumer has released them (item 1).</description></item>
/// <item><description><c>Volatile.Write(ref *producerCursor, P + 1)</c> (release): publishes the slot. The slot
/// stores and the buffer stores that precede it in program order cannot become visible after it. It pairs with the
/// consumer's <c>Volatile.Read</c> of <c>P</c> in <see cref="RingConsumer.TryTake"/>.</description></item>
/// </list>
/// <para>The producer's own copy of <c>P</c> (<c>_head</c>) is a private field. The producer's cached copy of
/// <c>C</c> (<c>_cachedTail</c>) may be stale low (older than the real <c>C</c>), which only makes the ring look
/// fuller than it is, so the check is conservative; it is refreshed with item 1 before reporting full. No fence
/// stronger than acquire/release is needed because each cursor has exactly one writer and the data flows one way
/// per cursor: slot contents and buffers from producer to consumer under <c>P</c>, and slot and buffer ownership
/// from consumer back to producer under <c>C</c>. There is no store-then-load (Dekker-style) pattern, which is the
/// case acquire/release alone would not cover.</para>
/// <para>Nothing here allocates.</para>
/// </remarks>
public sealed unsafe class RingProducer
{
    private readonly byte* _slots;
    private readonly uint _slotCount;
    private readonly uint _slotBytes;
    private readonly uint _capacity;
    private readonly ulong* _producerCursor;
    private readonly ulong* _consumerCursor;
    private ulong _head;
    private ulong _cachedTail;

    internal RingProducer(byte* segment, in RingHeader header, uint capacity)
    {
        if (!Environment.Is64BitProcess)
        {
            throw new PlatformNotSupportedException("The rings need atomic 64-bit loads and stores.");
        }

        if (capacity == 0 || capacity > header.SlotCount)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity));
        }

        _slots = segment + (long)header.SlotsOffset;
        _slotCount = header.SlotCount;
        _slotBytes = header.SlotBytes;
        _capacity = capacity;
        _producerCursor = (ulong*)(segment + (long)header.ProducerCursorOffset);
        _consumerCursor = (ulong*)(segment + (long)header.ConsumerCursorOffset);
        _head = Volatile.Read(ref *_producerCursor);
        _cachedTail = Volatile.Read(ref *_consumerCursor);
    }

    /// <summary>Slots published so far (the producer cursor).</summary>
    public ulong Published => _head;

    /// <summary>Slots that may be in flight at once.</summary>
    public int Capacity => (int)_capacity;

    /// <summary>The cursor this producer would publish next, taken modulo <paramref name="modulus"/>.</summary>
    /// <param name="modulus">Slot or buffer count.</param>
    /// <returns>The index.</returns>
    public int NextIndex(int modulus) => (int)(_head % (ulong)modulus);

    /// <summary>Whether a slot can be published now.</summary>
    /// <returns>False when <c>capacity</c> slots are unreleased.</returns>
    public bool HasSpace()
    {
        if (_head - _cachedTail < _capacity)
        {
            return true;
        }

        _cachedTail = Volatile.Read(ref *_consumerCursor); // item 1 (acquire)
        if (_head - _cachedTail > _slotCount)
        {
            ThrowCorrupt();
        }

        return _head - _cachedTail < _capacity;
    }

    /// <summary>Writes a slot and publishes it.</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>False when the ring is full; nothing is written then.</returns>
    public bool TryPublish(in RingSlot slot)
    {
        if (!HasSpace())
        {
            return false;
        }

        Unsafe.WriteUnaligned(_slots + ((_head % _slotCount) * _slotBytes), slot); // item 2: plain store
        Volatile.Write(ref *_producerCursor, _head + 1); // item 3 (release)
        _head++;
        return true;
    }

    private static void ThrowCorrupt() => throw new InvalidDataException("Ring cursors are inconsistent: the consumer cursor is behind the producer by more than the ring holds.");
}

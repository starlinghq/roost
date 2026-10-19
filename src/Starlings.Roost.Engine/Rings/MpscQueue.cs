using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

namespace Starlings.Roost.Engine.Rings;

/// <summary>
/// A bounded multi-producer single-consumer queue in managed memory: the engine's request queue. Any number of
/// frontend threads enqueue; the one engine thread dequeues. It is not in the C8 segment because its items are
/// managed objects (a <c>Request</c>); only plans and results cross the process boundary.
/// </summary>
/// <remarks>
/// <para>Bounded array queue after D. Vyukov's bounded MPMC queue, with the consumer side specialised to one thread.
/// Storage is allocated once in the constructor. <see cref="TryEnqueue"/> and <see cref="TryDequeue"/> allocate
/// nothing, take no lock and never block; a full queue returns false (admission turns that into backpressure).</para>
/// <para><b>State.</b> <c>_tail</c> is the next position to claim (producers). <c>_head</c> is the next position to
/// read (consumer only, a plain field). Each cell has a <c>Sequence</c> that says whose turn the cell is: for the
/// cell at position <c>p</c>, <c>Sequence == p</c> means empty and writable for the producer that claims <c>p</c>;
/// <c>Sequence == p + 1</c> means full and readable for the consumer at <c>p</c>; after the read the consumer sets
/// it to <c>p + capacity</c>, which is "empty" for the next lap.</para>
/// <para><b>Memory-ordering argument</b>, per operation:</para>
/// <list type="number">
/// <item><description><c>Volatile.Read(ref _tail)</c> (producer): an initial hint only. A stale value costs a
/// retry, never correctness, because the claim below is validated by the compare-exchange.</description></item>
/// <item><description><c>Volatile.Read(ref cell.Sequence)</c> (producer, acquire): pairs with the consumer's release
/// store of <c>p + capacity</c> (item 7). Seeing <c>Sequence == pos</c> therefore also shows the consumer's earlier
/// <c>cell.Value = default</c>, so the producer's write of the new value is ordered after the consumer is done
/// with the cell. Without the acquire a producer could overwrite a value the consumer is still reading.</description></item>
/// <item><description><c>Interlocked.CompareExchange(ref _tail, pos + 1, pos)</c> (producer): claims position
/// <c>pos</c> for exactly one producer. It is a full fence in .NET, which is stronger than needed; the property it
/// must give is atomicity of the claim. Producers that lose retry with the observed value.</description></item>
/// <item><description><c>cell.Value = item</c> (producer, plain store): safe because the claim gave this producer
/// exclusive ownership of the cell for this lap (the consumer will not touch it until the sequence says full) and
/// no other producer can claim the same position.</description></item>
/// <item><description><c>Volatile.Write(ref cell.Sequence, pos + 1)</c> (producer, release): publishes the value.
/// The plain store of <c>Value</c> cannot move after it. Pairs with item 6.</description></item>
/// <item><description><c>Volatile.Read(ref cell.Sequence)</c> (consumer, acquire): sees <c>head + 1</c> only after
/// item 5, and then the read of <c>Value</c> sees the producer's store. If the sequence is not yet <c>head + 1</c>
/// the queue is empty from the consumer's point of view, even if a producer has claimed the position (it has not
/// published yet); the consumer returns false and retries later. It does not spin on that cell, but it is not
/// independent of the producer either: later cells that are already published stay undelivered until this one
/// is published (head-of-line blocking).</description></item>
/// <item><description><c>Volatile.Write(ref cell.Sequence, head + capacity)</c> (consumer, release): returns the cell
/// to the producers after the consumer has read and cleared <c>Value</c>. Pairs with item 2.</description></item>
/// <item><description><c>Volatile.Write(ref _head.Value, head + 1)</c> (consumer): see below; nothing is ordered through it.</description></item>
/// </list>
/// <para>Item 8, and the <c>Volatile.Read</c> of both cursors in
/// <see cref="Count"/>: the head is written by the consumer thread only and read by others only for
/// <see cref="Count"/>, an estimate for metrics; nothing is synchronised through it. The consumer's own plain read of
/// <c>_head</c> is safe because no other thread writes it.</para>
/// <para>Producers are lock-free but not wait-free: a producer that is descheduled between items 3 and 5 holds its
/// cell, and the consumer sees an empty queue at that position until it resumes (later cells are not delivered
/// early, so order is the claim order). That is the usual trade for this queue. Whether the planner tolerates the
/// resulting delay (bounded by one producer's scheduling latency between its claim and its publish, with no
/// upper bound if that thread is suspended) has not been decided here; it is an open question for the planner
/// owner.</para>
/// </remarks>
/// <typeparam name="T">The item type. A reference type is cleared on dequeue so the queue does not keep it alive.</typeparam>
[SuppressMessage("Naming", "CA1711:Identifiers should not have incorrect suffix", Justification = "It is a queue; the plan and the backlog call it the MPSC request queue.")]
public sealed class MpscQueue<T>
{
    private readonly Cell[] _cells;
    private readonly long _mask;
    private PaddedInt64 _tail;
    private PaddedInt64 _head;

    /// <summary>Creates a queue.</summary>
    /// <param name="capacity">Slots; a power of two, at least 2.</param>
    public MpscQueue(int capacity)
    {
        if (capacity < 2 || (capacity & (capacity - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be a power of two, at least 2.");
        }

        _cells = new Cell[capacity];
        for (var i = 0; i < _cells.Length; i++)
        {
            _cells[i].Sequence = i;
        }

        _mask = capacity - 1;
    }

    /// <summary>Slots in the queue.</summary>
    public int Capacity => _cells.Length;

    /// <summary>An estimate of the items queued; exact only when no other thread is running.</summary>
    public int Count
    {
        get
        {
            var head = Volatile.Read(ref _head.Value);
            var tail = Volatile.Read(ref _tail.Value);
            return (int)Math.Clamp(tail - head, 0, _cells.Length);
        }
    }

    /// <summary>Adds an item. Safe from any number of threads.</summary>
    /// <param name="item">The item.</param>
    /// <returns>False when the queue is full.</returns>
    public bool TryEnqueue(T item)
    {
        var pos = Volatile.Read(ref _tail.Value); // item 1
        while (true)
        {
            ref var cell = ref _cells[(int)(pos & _mask)];
            var seq = Volatile.Read(ref cell.Sequence); // item 2
            var diff = seq - pos;
            if (diff == 0)
            {
                var seen = Interlocked.CompareExchange(ref _tail.Value, pos + 1, pos); // item 3
                if (seen == pos)
                {
                    cell.Value = item; // item 4
                    Volatile.Write(ref cell.Sequence, pos + 1); // item 5
                    return true;
                }

                pos = seen;
            }
            else if (diff < 0)
            {
                return false; // the cell still holds last lap's item: full
            }
            else
            {
                pos = Volatile.Read(ref _tail.Value); // another producer already took pos; item 1
            }
        }
    }

    /// <summary>Removes the oldest item. Only the one consumer thread may call this.</summary>
    /// <param name="item">The item.</param>
    /// <returns>False when nothing is ready.</returns>
    public bool TryDequeue(out T item)
    {
        var head = _head.Value; // consumer-owned
        ref var cell = ref _cells[(int)(head & _mask)];
        if (Volatile.Read(ref cell.Sequence) != head + 1) // item 6
        {
            item = default!;
            return false;
        }

        item = cell.Value;
        cell.Value = default!;
        Volatile.Write(ref cell.Sequence, head + _cells.Length); // item 7 (release, pairs with item 2)
        Volatile.Write(ref _head.Value, head + 1); // item 8: publishes the head for Count only
        return true;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct Cell
    {
        public long Sequence;
        public T Value;
    }
}

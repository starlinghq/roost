using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Blocks;

/// <summary>What a <see cref="BlockPool"/> holds for its group.</summary>
public enum BlockPoolRole
{
    /// <summary>Live blocks, or live recurrent slots: what the step plan's block tables name.</summary>
    Live = 0,

    /// <summary>Recurrent snapshots (C4 <c>SnapshotSlots</c>): a pool of its own, shared only by reference count.</summary>
    Snapshot,
}

/// <summary>The outcome of <see cref="BlockPool.TryMakeExclusive"/>.</summary>
public enum ExclusiveResult
{
    /// <summary>The block had one reference; the caller may write it in place.</summary>
    AlreadyExclusive = 0,

    /// <summary>The block was shared; a fresh block was allocated and a copy op was produced.</summary>
    Copied,

    /// <summary>The block was shared and no block was free. Nothing changed.</summary>
    OutOfBlocks,
}

/// <summary>
/// The typed block pool of one state group and role (contract C4): ids <c>0..Capacity-1</c>, a reference count each,
/// and a LIFO stack of free ids. Alloc, free, fork and copy-on-write are O(1) per block and allocate no managed memory.
/// </summary>
/// <remarks>
/// <para><b>Threading.</b> A pool is owned by the engine thread (R-027): it takes no lock and uses no atomics. Every
/// caller (planner, prefix cache, transfer service) runs on that thread; the worker never sees a pool, only the ids
/// and cache ops in a step plan. Cross-thread use is a bug, not a supported mode.</para>
/// <para><b>State.</b> <c>_refs[b]</c> is the number of holders of block <c>b</c> (a sequence's block table, a
/// snapshot owner, an export pin). <c>_free[0.._freeTop)</c> is the stack of free ids.</para>
/// <para><b>Invariants</b>, true between any two public calls (checked by <see cref="CheckInvariants"/>; a
/// call that throws leaves them holding: <see cref="Release"/>, <see cref="Retain"/> and <see cref="Fork"/> are all or
/// none, while <see cref="ReleaseOutOfWindow"/> keeps the releases it made before the bad entry, each already nulled in
/// the table):</para>
/// <list type="number">
/// <item><description><c>_refs[b] &gt;= 0</c> for every <c>b</c>.</description></item>
/// <item><description><c>_refs[b] == 0</c> if and only if <c>b</c> is on the free stack, exactly once. A referenced
/// block is never on the stack, so it is never handed out twice, and a freed block is on the stack once, so a double
/// free is detected (it throws) instead of corrupting the stack.</description></item>
/// <item><description><c>FreeCount == _freeTop</c> and <c>FreeCount + #{b : _refs[b] &gt; 0} == Capacity</c>.</description></item>
/// <item><description>Paged <see cref="Fork"/> keeps the block id and raises the count by one per fork; a block that is
/// shared (count at least two) is never written in place: the writer first calls <see cref="TryMakeExclusive"/>,
/// which gives it a private copy. An export pin is a reference, so a pinned block is shared and gets the same
/// treatment.</description></item>
/// <item><description>Recurrent live <see cref="Fork"/> allocates fresh slots and never shares one, because a recurrent
/// slot is written in place every step; the count of a live slot is therefore 1 while a sequence holds it. A pin on a
/// live slot does not freeze its contents.</description></item>
/// <item><description>Allocation order is deterministic: ids are handed out in the order a LIFO stack gives, starting
/// at 0, so a recorded request stream replays to the same ids (R-029 replay tests).</description></item>
/// </list>
/// <para>Id <see cref="NullBlock"/> (-1) marks a released entry in a block table (sliding window). It is accepted by
/// <see cref="Release"/> and <see cref="Fork"/> and ignored; any other id outside the pool throws.</para>
/// <para>Ops produced here (<see cref="Fork"/>, <see cref="TryMakeExclusive"/>) run on the worker before the next
/// forward. A block named by an op in a plan that has not completed must not be released and then reallocated by the
/// caller before that plan's result: the pool cannot see plans.</para>
/// </remarks>
public sealed class BlockPool : IBlockPool
{
    /// <summary>The id that marks an entry of a block table with no block (released out of a sliding window).</summary>
    public const int NullBlock = -1;

    private readonly int[] _refs;
    private readonly int[] _free;
    private int _freeTop;

    /// <summary>Creates a pool with every block free.</summary>
    /// <param name="group">The group descriptor.</param>
    /// <param name="capacity">Blocks (slots for a recurrent group).</param>
    /// <param name="role">What the pool holds.</param>
    public BlockPool(StateGroupDescriptor group, int capacity, BlockPoolRole role = BlockPoolRole.Live)
    {
        ArgumentNullException.ThrowIfNull(group);
        ArgumentOutOfRangeException.ThrowIfNegative(capacity);
        if (group.BlockSize <= 0 || group.BytesPerBlock <= 0)
        {
            throw new ArgumentException("The group needs a positive block size and bytes per block.", nameof(group));
        }

        if (role == BlockPoolRole.Snapshot && group.Kind != StateGroupKind.Recurrent)
        {
            throw new ArgumentException("Only a recurrent group has a snapshot pool.", nameof(group));
        }

        Group = group;
        Role = role;
        Capacity = capacity;
        _refs = new int[capacity];
        _free = new int[capacity];
        for (var i = 0; i < capacity; i++)
        {
            _free[i] = capacity - 1 - i; // the stack pops 0 first
        }

        _freeTop = capacity;
    }

    /// <inheritdoc/>
    public StateGroupDescriptor Group { get; }

    /// <summary>What the pool holds.</summary>
    public BlockPoolRole Role { get; }

    /// <inheritdoc/>
    public BlockTier Tier => Group.Tier;

    /// <inheritdoc/>
    public int Capacity { get; }

    /// <inheritdoc/>
    public int FreeCount => _freeTop;

    /// <summary>The number of holders of a block; 0 when it is free.</summary>
    /// <param name="block">Block id.</param>
    /// <returns>The reference count.</returns>
    public int RefCount(int block)
    {
        CheckId(block);
        return _refs[block];
    }

    /// <inheritdoc/>
    public bool TryAllocate(Span<int> blocks)
    {
        if (blocks.Length > _freeTop)
        {
            return false;
        }

        for (var i = 0; i < blocks.Length; i++)
        {
            var b = _free[--_freeTop];
            _refs[b] = 1;
            blocks[i] = b;
        }

        return true;
    }

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">A block was already free. Nothing was released.</exception>
    public void Release(ReadOnlySpan<int> blocks)
    {
        for (var i = 0; i < blocks.Length; i++)
        {
            var b = blocks[i];
            if (b == NullBlock)
            {
                continue;
            }

            if ((uint)b >= (uint)Capacity || _refs[b] <= 0)
            {
                // Undo in reverse: a block freed above is on top of the stack again, in the order it was pushed.
                for (var j = i - 1; j >= 0; j--)
                {
                    Unrelease(blocks[j]);
                }

                CheckId(b);
                throw new InvalidOperationException($"Block {b} of group {Group.Index} is already free (double release).");
            }

            ReleaseValid(b);
        }
    }

    /// <summary>Adds one reference to each block (an export pin, a snapshot owner). All or none.</summary>
    /// <param name="blocks">Live block ids; <see cref="NullBlock"/> entries are ignored.</param>
    public void Retain(ReadOnlySpan<int> blocks)
    {
        RequireLive(blocks);
        for (var i = 0; i < blocks.Length; i++)
        {
            if (blocks[i] != NullBlock)
            {
                _refs[blocks[i]]++;
            }
        }
    }

    /// <inheritdoc/>
    public int Fork(ReadOnlySpan<int> source, Span<int> forked, Span<PlanCacheOp> copies)
    {
        if (forked.Length != source.Length)
        {
            throw new ArgumentException("The forked span must be as long as the source.", nameof(forked));
        }

        RequireLive(source);
        if (Role == BlockPoolRole.Live && Group.Kind == StateGroupKind.Recurrent)
        {
            return ForkByCopy(source, forked, copies);
        }

        // Copy-on-write: share the blocks. Nothing is copied until a holder writes (TryMakeExclusive).
        for (var i = 0; i < source.Length; i++)
        {
            forked[i] = source[i];
            if (source[i] != NullBlock)
            {
                _refs[source[i]]++;
            }
        }

        return 0;
    }

    /// <summary>
    /// The write side of copy-on-write: returns <paramref name="block"/> when its holder is alone, otherwise a private
    /// copy and the op that fills it. The caller calls this before it writes into a block that a fork may share.
    /// </summary>
    /// <param name="block">The block the caller holds and wants to write.</param>
    /// <param name="writable">The block to write: <paramref name="block"/> itself, or the new one that replaces it in the caller's table.</param>
    /// <param name="copy">The <see cref="CacheOpKind.Copy"/> op for the next plan when the result is <see cref="ExclusiveResult.Copied"/>; otherwise default.</param>
    /// <returns>What happened.</returns>
    public ExclusiveResult TryMakeExclusive(int block, out int writable, out PlanCacheOp copy)
    {
        if (block == NullBlock)
        {
            throw new ArgumentOutOfRangeException(nameof(block), block, "A null block entry has nothing to write; skip it.");
        }

        RequireLive([block]);
        copy = default;
        writable = block;
        if (_refs[block] == 1)
        {
            return ExclusiveResult.AlreadyExclusive;
        }

        if (_freeTop == 0)
        {
            return ExclusiveResult.OutOfBlocks;
        }

        var fresh = _free[--_freeTop];
        _refs[fresh] = 1;
        _refs[block]--; // it was shared, so it stays referenced and is not freed
        writable = fresh;
        copy = MakeOp(CacheOpKind.Copy, block, fresh);
        return ExclusiveResult.Copied;
    }

    /// <summary>The number of leading blocks of a sliding sequence that are no longer needed.</summary>
    /// <remarks>
    /// <paramref name="tokens"/> tokens are in the cache, so the next query is at position <paramref name="tokens"/>.
    /// The rule keeps the keys from position <c>tokens - window</c> on, one more than a window of <c>window</c> keys
    /// including the query needs, so it never releases a block a query could still read. A block is released when
    /// every one of its positions is below that: <c>(tokens - window) / blockSize</c> blocks, rounded down, when
    /// <c>tokens &gt; window</c>.
    /// </remarks>
    /// <param name="tokens">Tokens already in the cache for the sequence.</param>
    /// <param name="window">Window in tokens.</param>
    /// <param name="blockSize">Tokens per block.</param>
    /// <returns>Blocks that may be released.</returns>
    public static int LeadingReleasableBlocks(int tokens, int window, int blockSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(window);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        return tokens > window ? (tokens - window) / blockSize : 0;
    }

    /// <summary>
    /// Sliding-window release: drops the reference to each leading block of <paramref name="table"/> that left the
    /// window and sets its entry to <see cref="NullBlock"/>, so positions keep their table index.
    /// </summary>
    /// <param name="table">The sequence's block table for this group, indexed by position / block size.</param>
    /// <param name="tokens">Tokens already in the cache.</param>
    /// <returns>Blocks released by this call (entries that were already null are not counted).</returns>
    public int ReleaseOutOfWindow(Span<int> table, int tokens)
    {
        if (Group.Kind != StateGroupKind.Sliding || Group.Window <= 0)
        {
            throw new InvalidOperationException($"Group {Group.Index} is not a sliding-window group.");
        }

        var n = Math.Min(LeadingReleasableBlocks(tokens, Group.Window, Group.BlockSize), table.Length);
        var released = 0;
        for (var i = 0; i < n; i++)
        {
            var b = table[i];
            if (b == NullBlock)
            {
                continue;
            }

            CheckId(b);
            if (_refs[b] <= 0)
            {
                throw new InvalidOperationException($"Block {b} of group {Group.Index} is already free (double release).");
            }

            ReleaseValid(b);
            table[i] = NullBlock; // set per entry, so a throw on a later entry leaves the table consistent
            released++;
        }

        return released;
    }

    /// <summary>Checks the invariants in the class remarks in O(capacity). For tests and diagnostics.</summary>
    /// <exception cref="InvalidOperationException">An invariant does not hold.</exception>
    public void CheckInvariants()
    {
        var onStack = new bool[Capacity];
        for (var i = 0; i < _freeTop; i++)
        {
            var b = _free[i];
            if ((uint)b >= (uint)Capacity || onStack[b])
            {
                throw new InvalidOperationException($"Free stack holds block {b} out of range or twice.");
            }

            onStack[b] = true;
        }

        var used = 0;
        for (var b = 0; b < Capacity; b++)
        {
            if (_refs[b] < 0)
            {
                throw new InvalidOperationException($"Block {b} has count {_refs[b]}.");
            }

            if ((_refs[b] == 0) != onStack[b])
            {
                throw new InvalidOperationException($"Block {b}: count {_refs[b]} but on free stack = {onStack[b]}.");
            }

            if (_refs[b] > 0)
            {
                used++;
            }
        }

        if (used + _freeTop != Capacity)
        {
            throw new InvalidOperationException("Free count and referenced blocks do not add up to the capacity.");
        }
    }

    internal PlanCacheOp MakeOp(CacheOpKind kind, int src, int dst) => new()
    {
        Kind = kind,
        Group = (uint)Group.Index,
        Src = (uint)src,
        Dst = (uint)dst,
        Count = 1,
        SrcTier = Tier,
        DstTier = Tier,
    };

    /// <summary>Appends a one-block op, merging it into the previous op when src and dst both continue its run.</summary>
    internal static void Append(Span<PlanCacheOp> ops, ref int count, in PlanCacheOp op)
    {
        if (count > 0)
        {
            ref var last = ref ops[count - 1];
            if (last.Kind == op.Kind && last.Group == op.Group && last.SrcTier == op.SrcTier && last.DstTier == op.DstTier
                && last.Src + last.Count == op.Src && last.Dst + last.Count == op.Dst)
            {
                last.Count += op.Count;
                return;
            }
        }

        ops[count++] = op;
    }

    private int ForkByCopy(ReadOnlySpan<int> source, Span<int> forked, Span<PlanCacheOp> copies)
    {
        var needed = 0;
        for (var i = 0; i < source.Length; i++)
        {
            if (source[i] != NullBlock)
            {
                needed++;
            }
        }

        if (copies.Length < needed)
        {
            throw new ArgumentException("The copies span must hold one op per source slot.", nameof(copies));
        }

        if (needed > _freeTop)
        {
            return -1;
        }

        var ops = 0;
        for (var i = 0; i < source.Length; i++)
        {
            if (source[i] == NullBlock)
            {
                forked[i] = NullBlock;
                continue;
            }

            var dst = _free[--_freeTop];
            _refs[dst] = 1;
            forked[i] = dst;
            Append(copies, ref ops, MakeOp(CacheOpKind.Fork, source[i], dst));
        }

        return ops;
    }

    private void ReleaseValid(int b)
    {
        if (--_refs[b] == 0)
        {
            _free[_freeTop++] = b;
        }
    }

    private void Unrelease(int b)
    {
        if (b == NullBlock)
        {
            return;
        }

        if (_refs[b] == 0)
        {
            _freeTop--; // b is on top: undo runs in reverse order of the pushes
        }

        _refs[b]++;
    }

    private void RequireLive(ReadOnlySpan<int> blocks)
    {
        for (var i = 0; i < blocks.Length; i++)
        {
            var b = blocks[i];
            if (b == NullBlock)
            {
                continue;
            }

            CheckId(b);
            if (_refs[b] <= 0)
            {
                throw new InvalidOperationException($"Block {b} of group {Group.Index} is not allocated.");
            }
        }
    }

    private void CheckId(int block)
    {
        if ((uint)block >= (uint)Capacity)
        {
            throw new ArgumentOutOfRangeException(nameof(block), block, $"Group {Group.Index} has {Capacity} blocks.");
        }
    }
}

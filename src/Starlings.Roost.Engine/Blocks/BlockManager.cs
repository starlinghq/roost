using System.Collections.Immutable;
using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Blocks;

/// <summary>
/// The typed block manager (contract C4, stream G): one <see cref="BlockPool"/> per state group, sized by
/// <see cref="BlockSizing"/>, and for each recurrent group a second pool for its snapshots.
/// </summary>
/// <remarks>
/// <para>Threading and invariants are those of <see cref="BlockPool"/>: engine thread only, no locks. The manager adds
/// one invariant of its own: the snapshot pool of a recurrent group and its live pool never share an id space, so a
/// snapshot id is never a valid state slot. The ops say which space an id is in: <see cref="CacheOpKind.SnapshotWrite"/>
/// has a live slot as <c>Src</c> and a snapshot id as <c>Dst</c>; <see cref="CacheOpKind.SnapshotRestore"/> the reverse.</para>
/// <para>Snapshot ids are held by reference count, like blocks, so the prefix cache (R-030) can share one snapshot
/// between several cached prefixes and release it when the last one is evicted.</para>
/// </remarks>
public sealed class BlockManager
{
    private readonly BlockPool[] _pools;
    private readonly BlockPool?[] _snapshots;

    /// <summary>Creates the pools with explicit capacities.</summary>
    /// <param name="groups">Groups in index order; <c>groups[i].Index == i</c>.</param>
    /// <param name="capacities">One entry per group, as <see cref="BlockSizing.Compute"/> returns.</param>
    public BlockManager(ImmutableArray<StateGroupDescriptor> groups, ImmutableArray<GroupCapacity> capacities)
    {
        if (groups.IsDefault || capacities.IsDefault || groups.Length != capacities.Length)
        {
            throw new ArgumentException("One capacity per group is required.", nameof(capacities));
        }

        Groups = groups;
        _pools = new BlockPool[groups.Length];
        _snapshots = new BlockPool?[groups.Length];
        for (var i = 0; i < groups.Length; i++)
        {
            if (groups[i].Index != i || capacities[i].Group != i)
            {
                throw new ArgumentException($"Group at position {i} is out of order.", nameof(groups));
            }

            _pools[i] = new BlockPool(groups[i], capacities[i].Blocks);
            if (groups[i].Kind == StateGroupKind.Recurrent)
            {
                _snapshots[i] = new BlockPool(groups[i], capacities[i].SnapshotSlots, BlockPoolRole.Snapshot);
            }
            else if (capacities[i].SnapshotSlots != 0)
            {
                throw new ArgumentException($"Group {groups[i].Name} is not recurrent.", nameof(capacities));
            }
        }
    }

    /// <summary>The groups.</summary>
    public ImmutableArray<StateGroupDescriptor> Groups { get; }

    /// <summary>Sizes the pools from a byte budget and creates the manager.</summary>
    /// <param name="groups">Groups in index order.</param>
    /// <param name="budgetBytes">Bytes for all pools.</param>
    /// <param name="maxSequences">Live sequences.</param>
    /// <returns>The manager.</returns>
    public static BlockManager Create(ImmutableArray<StateGroupDescriptor> groups, long budgetBytes, int maxSequences) =>
        new(groups, BlockSizing.Compute(groups, budgetBytes, maxSequences));

    /// <summary>The live pool of a group.</summary>
    /// <param name="group">Group index.</param>
    /// <returns>The pool.</returns>
    public BlockPool GetPool(int group) => _pools[group];

    /// <summary>The snapshot pool of a recurrent group.</summary>
    /// <param name="group">Group index.</param>
    /// <returns>The pool.</returns>
    /// <exception cref="InvalidOperationException">The group is not recurrent.</exception>
    public BlockPool GetSnapshotPool(int group) =>
        _snapshots[group] ?? throw new InvalidOperationException($"Group {group} has no snapshot pool.");

    /// <summary>Copies a live recurrent slot into a new snapshot slot, as an op for the next plan.</summary>
    /// <param name="group">Recurrent group index.</param>
    /// <param name="slot">The live slot to snapshot.</param>
    /// <param name="snapshot">The new snapshot id, held once by the caller.</param>
    /// <param name="write">The <see cref="CacheOpKind.SnapshotWrite"/> op.</param>
    /// <returns>False, changing nothing, when no snapshot slot is free.</returns>
    public bool TrySnapshot(int group, int slot, out int snapshot, out PlanCacheOp write)
    {
        var live = _pools[group];
        var snaps = GetSnapshotPool(group);
        if (live.RefCount(slot) <= 0)
        {
            throw new InvalidOperationException($"Slot {slot} of group {group} is not allocated.");
        }

        Span<int> one = stackalloc int[1];
        if (!snaps.TryAllocate(one))
        {
            snapshot = BlockPool.NullBlock;
            write = default;
            return false;
        }

        snapshot = one[0];
        write = live.MakeOp(CacheOpKind.SnapshotWrite, slot, snapshot);
        return true;
    }

    /// <summary>Restores a snapshot into a live slot, as an op for the next plan. The snapshot stays held.</summary>
    /// <param name="group">Recurrent group index.</param>
    /// <param name="snapshot">A held snapshot id.</param>
    /// <param name="slot">The live slot to overwrite.</param>
    /// <returns>The <see cref="CacheOpKind.SnapshotRestore"/> op.</returns>
    public PlanCacheOp Restore(int group, int snapshot, int slot)
    {
        var live = _pools[group];
        if (GetSnapshotPool(group).RefCount(snapshot) <= 0 || live.RefCount(slot) <= 0)
        {
            throw new InvalidOperationException("Both the snapshot and the slot must be allocated.");
        }

        return live.MakeOp(CacheOpKind.SnapshotRestore, snapshot, slot);
    }
}

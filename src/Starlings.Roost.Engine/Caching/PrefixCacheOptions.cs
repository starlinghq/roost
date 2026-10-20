namespace Starlings.Roost.Engine.Caching;

/// <summary>Settings of the prefix cache (R-030). A planner has a cache when <c>PlannerOptions.PrefixCache</c> is set.</summary>
public sealed record PrefixCacheOptions
{
    /// <summary>
    /// For a model with a recurrent group, a snapshot is taken at every this-many attention blocks, and a prefill chunk
    /// never crosses such a boundary, because a snapshot can only copy the state a finished plan left. 1 snapshots every
    /// attention block boundary (the R-030 criterion) and caps a prefill chunk at one block; a larger value trades hits for
    /// fewer, larger chunks. Ignored without a recurrent group.
    /// </summary>
    public int SnapshotStrideBlocks { get; init; } = 1;

    /// <summary>Mixed into every hash chain: set it to something that names the weights (a digest), so caches never mix models.</summary>
    public ulong Salt { get; init; }

    internal void Validate() => ArgumentOutOfRangeException.ThrowIfNegativeOrZero(SnapshotStrideBlocks);
}

/// <summary>Counters of the prefix cache since creation.</summary>
/// <param name="Lookups">Prefix lookups made for sequences gaining residency.</param>
/// <param name="Hits">Lookups that reused at least one block.</param>
/// <param name="HitTokens">Tokens whose computation a hit skipped.</param>
/// <param name="NodesInserted">Block nodes added.</param>
/// <param name="NodesEvicted">Block nodes evicted (LRU).</param>
/// <param name="SnapshotsTaken">Recurrent snapshots written for the cache.</param>
/// <param name="SnapshotsDropped">Snapshots dropped to make room for newer ones.</param>
public readonly record struct PrefixCacheStats(
    long Lookups, long Hits, long HitTokens, long NodesInserted, long NodesEvicted, long SnapshotsTaken, long SnapshotsDropped);

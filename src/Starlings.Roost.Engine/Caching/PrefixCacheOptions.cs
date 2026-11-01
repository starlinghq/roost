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

    /// <summary>
    /// Bytes of pinned host memory for the host KV tier (R-094); 0 turns the tier off. With a budget, device eviction
    /// demotes a cached block to the host tier with a <c>TierMove</c> op instead of dropping it, a lookup that reaches
    /// host blocks restores them to the device one plan ahead of use, and host blocks are dropped only when the host tier
    /// is full. Sized by <see cref="Blocks.HostTier.BlocksFor"/>. The CUDA worker does not execute host-tier ops yet
    /// (see <c>STATUS.md</c>), so this is for the CPU worker and planner tests until it does.
    /// </summary>
    public long HostTierBytes { get; init; }

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(SnapshotStrideBlocks);
        ArgumentOutOfRangeException.ThrowIfNegative(HostTierBytes);
    }
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

/// <summary>Counters of the host KV tier (R-094) since creation. All zero when the tier is off.</summary>
/// <param name="Demoted">Cached blocks moved from the device to the host tier instead of being dropped.</param>
/// <param name="Restored">Cached blocks moved back from the host tier to the device for a lookup.</param>
/// <param name="HostHits">Lookups whose reused prefix included at least one host-tier block.</param>
/// <param name="HostHitTokens">Tokens of reused prefixes that were on the host tier at lookup time.</param>
/// <param name="HostEvicted">Host-tier blocks dropped because the host tier was full.</param>
/// <param name="HostBlocks">Host-tier nodes now.</param>
public readonly record struct HostTierStats(long Demoted, long Restored, long HostHits, long HostHitTokens, long HostEvicted, int HostBlocks);

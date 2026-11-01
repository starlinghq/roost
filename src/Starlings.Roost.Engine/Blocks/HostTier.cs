using System.Collections.Immutable;
using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Blocks;

/// <summary>
/// The host-memory KV tier (R-094, stream G): a second <see cref="BlockManager"/> whose paged attention pools live in
/// pinned host memory (<see cref="BlockTier.Host"/>, contract C4), sized from a byte budget.
/// </summary>
/// <remarks>
/// <para><b>Shape.</b> The host manager has the same group indices as the device manager, so a
/// <see cref="CacheOpKind.TierMove"/> op names one group and an id in each tier. Only the attention kinds the prefix cache
/// serves (<see cref="StateGroupKind.Full"/> and <see cref="StateGroupKind.MlaLatent"/>) get host blocks; every other group
/// has capacity 0 in this tier. Recurrent snapshots stay in the device snapshot pool.</para>
/// <para><b>Sizing.</b> A cached prefix block names one block in every served group, so the budget buys
/// <c>budgetBytes / sum(BytesPerBlock of the served groups)</c> blocks in each served group, rounded down.</para>
/// <para><b>Status.</b> The planner side (pools, ops, pins) is tested on CPU. Execution of host-tier
/// <see cref="CacheOpKind.TierMove"/> ops in the CUDA worker (pinned host allocation and the copies) is not built:
/// the capability is <i>untested</i> on GPU, see <c>src/Starlings.Roost.Engine/STATUS.md</c>.</para>
/// </remarks>
public static class HostTier
{
    /// <summary>Whether the host tier holds blocks of a group of this kind.</summary>
    /// <param name="kind">The group kind.</param>
    /// <returns>True for the paged attention kinds the prefix cache serves.</returns>
    public static bool Serves(StateGroupKind kind) => kind is StateGroupKind.Full or StateGroupKind.MlaLatent;

    /// <summary>Host bytes one cached prefix block takes: one block of every served group.</summary>
    /// <param name="groups">The device groups.</param>
    /// <returns>The bytes, or 0 when no group is served.</returns>
    public static long BytesPerPrefixBlock(ImmutableArray<StateGroupDescriptor> groups)
    {
        if (groups.IsDefault)
        {
            throw new ArgumentException("Groups are required.", nameof(groups));
        }

        long bytes = 0;
        foreach (var g in groups)
        {
            if (Serves(g.Kind))
            {
                bytes = checked(bytes + g.BytesPerBlock);
            }
        }

        return bytes;
    }

    /// <summary>The blocks per served group a budget buys.</summary>
    /// <param name="groups">The device groups.</param>
    /// <param name="budgetBytes">Host bytes for the tier.</param>
    /// <returns>The block count (0 when the budget is below one prefix block or no group is served).</returns>
    public static int BlocksFor(ImmutableArray<StateGroupDescriptor> groups, long budgetBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(budgetBytes);
        var per = BytesPerPrefixBlock(groups);
        return per == 0 ? 0 : (int)Math.Min(int.MaxValue, budgetBytes / per);
    }

    /// <summary>Creates the host-tier manager for a set of device groups.</summary>
    /// <param name="groups">The device groups, in index order.</param>
    /// <param name="budgetBytes">Host bytes for the tier.</param>
    /// <returns>A manager whose groups are the device groups with <see cref="BlockTier.Host"/>; unserved groups have no blocks.</returns>
    public static BlockManager Create(ImmutableArray<StateGroupDescriptor> groups, long budgetBytes)
    {
        var blocks = BlocksFor(groups, budgetBytes);
        var host = ImmutableArray.CreateBuilder<StateGroupDescriptor>(groups.Length);
        var capacities = ImmutableArray.CreateBuilder<GroupCapacity>(groups.Length);
        foreach (var g in groups)
        {
            host.Add(g with { Tier = BlockTier.Host });
            capacities.Add(new GroupCapacity(g.Index, Serves(g.Kind) ? blocks : 0, 0));
        }

        return new BlockManager(host.MoveToImmutable(), capacities.MoveToImmutable());
    }
}

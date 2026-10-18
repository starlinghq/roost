using System.Collections.Immutable;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Blocks;

/// <summary>Block counts for one group.</summary>
/// <param name="Group">Group index.</param>
/// <param name="Blocks">Live blocks (slots for a recurrent group).</param>
/// <param name="SnapshotSlots">Slots of the separate snapshot pool; 0 for paged groups.</param>
public readonly record struct GroupCapacity(int Group, int Blocks, int SnapshotSlots);

/// <summary>
/// The sizing policy: how a byte budget becomes per-group pool capacities. It is a pure function, apart from the pools,
/// so admission control (R-031) can replace the policy without touching them.
/// </summary>
/// <remarks>
/// <para>Rule: recurrent groups first, at <c>maxSequences</c> live slots plus the descriptor's snapshot slots, each at
/// <c>BytesPerBlock</c>. The rest of the budget goes to the paged groups so that every one of them can hold the same
/// number of tokens <c>T</c>: with <c>bytes per token = BytesPerBlock / BlockSize</c>, <c>T</c> is a value found by
/// an estimate and a downward correction, with <c>sum(floor(T / BlockSize) * BytesPerBlock) &lt;= rest</c>, and a group holds <c>floor(T / BlockSize)</c>
/// blocks. A sliding group never needs more than <c>maxSequences * (ceil(window / BlockSize) + 1)</c> blocks (a window
/// spans one more block than its length when it is not aligned); a group capped there frees its share for the others.
/// The chunk being prefilled and a speculation tree are not in that cap: reserving them is admission control's job.</para>
/// <para>Because recurrent state sits in its own pool, the attention block size is whatever the model compiler chose;
/// nothing here pads it to the recurrent state size (bet X2).</para>
/// </remarks>
public static class BlockSizing
{
    /// <summary>Computes the capacities.</summary>
    /// <param name="groups">Groups in index order; <c>groups[i].Index == i</c>.</param>
    /// <param name="budgetBytes">Bytes available for all pools of this tier.</param>
    /// <param name="maxSequences">Live sequences, which is the number of live recurrent slots.</param>
    /// <returns>One entry per group, in order.</returns>
    /// <exception cref="ArgumentException">The groups are out of order, or the budget cannot hold the recurrent slots.</exception>
    public static ImmutableArray<GroupCapacity> Compute(ImmutableArray<StateGroupDescriptor> groups, long budgetBytes, int maxSequences)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(budgetBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(maxSequences);
        var blocks = new long[groups.Length];
        var snapshots = new int[groups.Length];
        var active = new List<int>();
        var rest = budgetBytes;
        for (var i = 0; i < groups.Length; i++)
        {
            var g = groups[i];
            if (g.Index != i)
            {
                throw new ArgumentException($"Group at position {i} has index {g.Index}.", nameof(groups));
            }

            if (g.BlockSize <= 0 || g.BytesPerBlock <= 0)
            {
                throw new ArgumentException($"Group {g.Name} needs a positive block size and bytes per block.", nameof(groups));
            }

            if (g.Kind == StateGroupKind.Recurrent)
            {
                blocks[i] = maxSequences;
                snapshots[i] = g.SnapshotSlots;
                rest -= ((long)maxSequences + g.SnapshotSlots) * g.BytesPerBlock;
            }
            else if (g.SnapshotSlots != 0)
            {
                throw new ArgumentException($"Group {g.Name} is not recurrent but has snapshot slots.", nameof(groups));
            }
            else
            {
                active.Add(i);
            }
        }

        if (rest < 0)
        {
            throw new ArgumentException("The budget cannot hold the recurrent slots and snapshots.", nameof(budgetBytes));
        }

        // Fix sliding groups at their cap while the cap is below an even share; repeat, as fixing one raises the share.
        long tokens;
        while (true)
        {
            tokens = MaxTokens(groups, active, rest);
            var changed = false;
            for (var k = active.Count - 1; k >= 0; k--)
            {
                var g = groups[active[k]];
                if (g.Kind != StateGroupKind.Sliding || g.Window <= 0)
                {
                    continue;
                }

                var cap = (long)maxSequences * (((g.Window + g.BlockSize - 1) / g.BlockSize) + 1);
                if (cap <= tokens / g.BlockSize)
                {
                    blocks[active[k]] = cap;
                    rest -= cap * g.BytesPerBlock;
                    active.RemoveAt(k);
                    changed = true;
                }
            }

            if (!changed)
            {
                break;
            }
        }

        foreach (var i in active)
        {
            blocks[i] = tokens / groups[i].BlockSize;
        }

        var result = ImmutableArray.CreateBuilder<GroupCapacity>(groups.Length);
        for (var i = 0; i < groups.Length; i++)
        {
            result.Add(new GroupCapacity(i, checked((int)blocks[i]), snapshots[i]));
        }

        return result.MoveToImmutable();
    }

    private static long MaxTokens(ImmutableArray<StateGroupDescriptor> groups, List<int> active, long rest)
    {
        if (active.Count == 0)
        {
            return 0;
        }

        decimal perToken = 0;
        foreach (var i in active)
        {
            perToken += (decimal)groups[i].BytesPerBlock / groups[i].BlockSize;
        }

        var t = (long)Math.Floor(rest / perToken);
        while (t > 0 && Cost(groups, active, t) > rest)
        {
            t--; // the estimate may be a token too high through rounding
        }

        return t;
    }

    private static long Cost(ImmutableArray<StateGroupDescriptor> groups, List<int> active, long tokens)
    {
        long cost = 0;
        foreach (var i in active)
        {
            cost += tokens / groups[i].BlockSize * groups[i].BytesPerBlock;
        }

        return cost;
    }
}

using System.Collections.Immutable;
using Starlings.Roost.Requests;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Admission;

/// <summary>
/// What a request can hold in one group at its worst, in the group's own units (blocks, or slots for a recurrent group).
/// The request holds <c>Shared + rows * PerRow</c>.
/// </summary>
/// <param name="Shared">Blocks every row shares: the full blocks of the prompt, which forked rows share copy-on-write.</param>
/// <param name="PerRow">Blocks one row owns: its generated tokens, the partial prompt block it copies on first write, or its recurrent slot.</param>
public readonly record struct GroupReservation(long Shared, long PerRow)
{
    /// <summary>Blocks for a number of rows.</summary>
    /// <param name="rows">Rows (sequences).</param>
    /// <returns>The total.</returns>
    public long For(int rows) => Shared + (rows * PerRow);
}

/// <summary>The worst-case reservation of one request, one entry per state group.</summary>
public static class WorstCaseReservation
{
    /// <summary>Rows a request needs: <c>n</c>, <c>best_of</c> and the beam width each ask for that many sequences.</summary>
    /// <param name="sampling">The sampling parameters.</param>
    /// <returns>At least 1.</returns>
    public static int Rows(SamplingParameters sampling)
    {
        ArgumentNullException.ThrowIfNull(sampling);
        return Math.Max(1, Math.Max(sampling.N, Math.Max(sampling.BestOf ?? 1, sampling.BeamWidth)));
    }

    /// <summary>
    /// Tokens a sequence can reach: the prompt plus <c>max_tokens</c> (the context length when it is unset), never
    /// beyond the context length.
    /// </summary>
    /// <param name="promptLength">Prompt tokens.</param>
    /// <param name="maxTokens">The request's <c>max_tokens</c>.</param>
    /// <param name="contextLength">The model's context length.</param>
    /// <returns>The worst-case length.</returns>
    public static long WorstTokens(int promptLength, int? maxTokens, int contextLength) =>
        Math.Min((long)contextLength, promptLength + (long)(maxTokens ?? contextLength));

    /// <summary>Computes the reservation per group (adapter groups are reserved separately, by handle).</summary>
    /// <param name="groups">Groups in index order.</param>
    /// <param name="promptLength">Prompt tokens.</param>
    /// <param name="maxTokens">The request's <c>max_tokens</c>, or null.</param>
    /// <param name="contextLength">The model's context length.</param>
    /// <returns>One entry per group.</returns>
    /// <remarks>
    /// A paged group needs one block for every token up to the worst-case length plus one, because the planner reserves the
    /// block of the token it is about to write (the same <c>+ 1</c> as <c>StepPlanner.Add</c>). A sliding group is capped
    /// at a window plus one block, as <see cref="Blocks.BlockSizing"/> caps it, per row, without credit for sharing. A
    /// recurrent group needs one live slot per row; its snapshot pool is the prefix cache's and is evictable, so it is
    /// not reserved. The reservation is deliberately conservative: it refuses a request the planner could have run by
    /// preempting others.
    /// </remarks>
    public static ImmutableArray<GroupReservation> Compute(ImmutableArray<StateGroupDescriptor> groups, int promptLength, int? maxTokens, int contextLength)
    {
        var worst = WorstTokens(promptLength, maxTokens, contextLength) + 1;
        var result = ImmutableArray.CreateBuilder<GroupReservation>(groups.Length);
        foreach (var g in groups)
        {
            switch (g.Kind)
            {
                case StateGroupKind.Recurrent:
                    result.Add(new GroupReservation(0, 1));
                    break;
                case StateGroupKind.Sliding when g.Window > 0:
                    result.Add(new GroupReservation(0, CeilDiv(Math.Min(worst, (long)g.Window + g.BlockSize + 1), g.BlockSize)));
                    break;
                case StateGroupKind.Full or StateGroupKind.Sliding or StateGroupKind.MlaLatent:
                    var total = CeilDiv(worst, g.BlockSize);
                    var shared = Math.Min(total, promptLength / g.BlockSize);
                    result.Add(new GroupReservation(shared, total - shared));
                    break;
                case StateGroupKind.Adapter:
                    result.Add(default);
                    break;
                default:
                    throw new NotSupportedException($"State group kind {g.Kind} (group {g.Name}) is not supported by admission control.");
            }
        }

        return result.MoveToImmutable();
    }

    private static long CeilDiv(long a, long b) => (a + b - 1) / b;
}

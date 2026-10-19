using System.Collections.Immutable;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Engine.Scheduling;

/// <summary>
/// One step plan in managed form: the tables of C2 (<c>RoostStepPlanHeader</c> and its records) as arrays, in the
/// order the shared-memory writer lays them out. Rows are in scheduling order (priority, then deadline, then arrival).
/// </summary>
/// <remarks>
/// <see cref="GroupBlocks"/> has <c>GroupCount</c> entries per row, row-major. A recurrent group has a zero block
/// count there: its slot is <see cref="PlanSequence.RecurrentSlot"/>. A sliding group's table keeps
/// <c>-1</c> entries for blocks that left the window, so a position keeps its table index.
/// </remarks>
public sealed class PlannedStep
{
    internal PlannedStep(ulong stepId, int groupCount, PlanOptions flags, MathMode mathMode, ImmutableArray<PlanSequence> sequences,
        ImmutableArray<int> tokenIds, ImmutableArray<int> positions, ImmutableArray<PlanGroupBlocks> groupBlocks,
        ImmutableArray<int> blockIds, ImmutableArray<PlanSampling> sampling, ImmutableArray<PlanCacheOp> cacheOps,
        ImmutableArray<int> stopTokenIds, string digest)
    {
        StepId = stepId;
        GroupCount = groupCount;
        Flags = flags;
        MathMode = mathMode;
        Sequences = sequences;
        TokenIds = tokenIds;
        Positions = positions;
        GroupBlocks = groupBlocks;
        BlockIds = blockIds;
        Sampling = sampling;
        CacheOps = cacheOps;
        StopTokenIds = stopTokenIds;
        Digest = digest;
    }

    /// <summary>The step id, 1 for the first plan of a planner.</summary>
    public ulong StepId { get; }

    /// <summary>State groups per row in <see cref="GroupBlocks"/>.</summary>
    public int GroupCount { get; }

    /// <summary>Plan flags.</summary>
    public PlanOptions Flags { get; }

    /// <summary>Math mode.</summary>
    public MathMode MathMode { get; }

    /// <summary>The sequence table.</summary>
    public ImmutableArray<PlanSequence> Sequences { get; }

    /// <summary>Flat token ids.</summary>
    public ImmutableArray<int> TokenIds { get; }

    /// <summary>Flat positions, parallel to <see cref="TokenIds"/>.</summary>
    public ImmutableArray<int> Positions { get; }

    /// <summary>Per row and group, the span of <see cref="BlockIds"/>.</summary>
    public ImmutableArray<PlanGroupBlocks> GroupBlocks { get; }

    /// <summary>Block ids.</summary>
    public ImmutableArray<int> BlockIds { get; }

    /// <summary>One sampling record per row.</summary>
    public ImmutableArray<PlanSampling> Sampling { get; }

    /// <summary>Cache ops the worker runs before the forward: pending forks first, then copy-on-write copies.</summary>
    public ImmutableArray<PlanCacheOp> CacheOps { get; }

    /// <summary>Stop ids, per row.</summary>
    public ImmutableArray<int> StopTokenIds { get; }

    /// <summary>SHA-256 over every field above, as hex. Equal inputs give equal digests, in any process.</summary>
    public string Digest { get; }

    /// <summary>The block ids of one row and group.</summary>
    /// <param name="row">Row index.</param>
    /// <param name="group">Group index.</param>
    /// <returns>The ids, including <c>-1</c> entries.</returns>
    public ReadOnlySpan<int> BlocksOf(int row, int group)
    {
        var gb = GroupBlocks[(row * GroupCount) + group];
        return BlockIds.AsSpan((int)gb.BlockStart, (int)gb.BlockCount);
    }
}

/// <summary>What the planner reports besides plans.</summary>
public enum PlannerNoticeKind
{
    /// <summary>A resident sequence lost its state and will recompute from token 0.</summary>
    Preempted = 0,

    /// <summary>A sequence can never fit in the pools, alone; it was dropped.</summary>
    Aborted,
}

/// <summary>A preemption or abort, drained with <see cref="StepPlanner.DrainNotices"/>.</summary>
/// <param name="Kind">What happened.</param>
/// <param name="Handle">The sequence handle (the request id value).</param>
public readonly record struct PlannerNotice(PlannerNoticeKind Kind, ulong Handle);

/// <summary>A sequence that finished when a result was applied.</summary>
/// <param name="Handle">The sequence handle.</param>
/// <param name="Reason">Why it finished.</param>
public readonly record struct SequenceFinished(ulong Handle, FinishReason Reason);

/// <summary>Read-only view of one sequence, for tests and metrics.</summary>
/// <param name="Handle">The handle.</param>
/// <param name="TokenCount">Prompt plus generated tokens known so far.</param>
/// <param name="ComputedTokens">Tokens whose state is (or will be, once planned steps complete) in the cache.</param>
/// <param name="Resident">True when the sequence holds blocks.</param>
/// <param name="InFlightPlans">Uncompleted plans that carry it.</param>
/// <param name="Preemptions">Times it was preempted.</param>
public readonly record struct SequenceInfo(ulong Handle, int TokenCount, int ComputedTokens, bool Resident, int InFlightPlans, int Preemptions);

using System.Collections.Immutable;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Engine.Scheduling;

/// <summary>One sequence's part of a step result (C2 <c>RoostResultSequence</c> with its tokens).</summary>
/// <param name="Handle">The sequence handle echoed by the worker.</param>
/// <param name="Tokens">Tokens sampled this step: empty for a prefill chunk that did not end the prompt, one otherwise (speculation is not supported yet).</param>
/// <param name="Finish">The worker's finish flag.</param>
public sealed record SequenceResult(ulong Handle, ImmutableArray<int> Tokens, FinishReason Finish);

/// <summary>A completed step, as the planner consumes it. A recorded run is a list of these.</summary>
/// <param name="StepId">The plan it answers.</param>
/// <param name="Sequences">One entry per row that sampled or prefilled.</param>
public sealed record StepOutcome(ulong StepId, ImmutableArray<SequenceResult> Sequences)
{
    /// <summary>Reads the result tables of a C2 step result.</summary>
    /// <param name="stepId">The result header's step id.</param>
    /// <param name="sequences">The <c>Sequences</c> table.</param>
    /// <param name="tokenIds">The <c>TokenIds</c> table.</param>
    /// <returns>The outcome.</returns>
    public static StepOutcome FromTables(ulong stepId, ReadOnlySpan<ResultSequence> sequences, ReadOnlySpan<int> tokenIds)
    {
        var rows = ImmutableArray.CreateBuilder<SequenceResult>(sequences.Length);
        foreach (var s in sequences)
        {
            rows.Add(new SequenceResult(s.SequenceHandle, [.. tokenIds.Slice((int)s.TokenStart, (int)s.TokenCount)], s.Finish));
        }

        return new StepOutcome(stepId, rows.MoveToImmutable());
    }
}

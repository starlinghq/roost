using System.Collections.Immutable;
using Starlings.Roost.Engine.Caching;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Engine.Scheduling;

/// <summary>Limits and fixed fields of the planner (R-029).</summary>
/// <remarks>
/// There is no switch for chunked prefill: it is always on, so a prompt longer than the token budget is split over
/// several steps and never refused for its length alone.
/// </remarks>
public sealed record PlannerOptions
{
    /// <summary>Tokens one plan may carry (decode rows count one, prefill chunks as many as they hold).</summary>
    public int MaxTokensPerStep { get; init; } = 2048;

    /// <summary>Rows one plan may carry. Forked branch rows (n &gt; 1, beam search) count like any other sequence.</summary>
    public int MaxSequencesPerStep { get; init; } = 64;

    /// <summary>Upper bound on a prefill chunk, applied before the token budget. Defaults to no extra bound.</summary>
    public int MaxChunkTokens { get; init; } = int.MaxValue;

    /// <summary>The model's own stop ids, appended to every row's request stop ids (C2 <c>stop_token_ids</c>).</summary>
    public ImmutableArray<int> ModelStopTokenIds { get; init; } = [];

    /// <summary>Math mode stamped on every plan.</summary>
    public MathMode MathMode { get; init; } = MathMode.Default;

    /// <summary>
    /// The prefix cache (R-030). Null, the default, means no cache and plans exactly as before R-030. Setting it makes
    /// the planner reuse cached blocks and snapshots when a sequence gains residency, and for a hybrid model cap every
    /// prefill chunk at the next snapshot boundary (see <see cref="PrefixCacheOptions.SnapshotStrideBlocks"/>).
    /// </summary>
    public PrefixCacheOptions? PrefixCache { get; init; }

    /// <summary>
    /// Async scheduling (R-032). False, the default, plans exactly as before: a sequence waits for its sampled token. True lets
    /// a decode row ride in the plan after the one sampling its input, marked <see cref="SequenceOptions.SampledInput"/> for the
    /// worker to patch (see <see cref="StepPlanner"/>). Only for a worker that patches inputs; the caller keeps at most the
    /// plans in flight its transport double-buffers and applies results in order.
    /// </summary>
    public bool AsyncScheduling { get; init; }

    internal void Validate()
    {
        PrefixCache?.Validate();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxTokensPerStep);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxSequencesPerStep);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxChunkTokens);
    }
}

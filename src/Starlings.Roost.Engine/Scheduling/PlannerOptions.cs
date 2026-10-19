using System.Collections.Immutable;
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

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxTokensPerStep);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxSequencesPerStep);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxChunkTokens);
    }
}

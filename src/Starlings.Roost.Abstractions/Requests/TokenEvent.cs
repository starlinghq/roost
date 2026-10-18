using System.Threading.Channels;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Requests;

/// <summary>Kinds of request event.</summary>
public enum TokenEventKind
{
    /// <summary>New tokens for one sequence of the request.</summary>
    Tokens = 0,

    /// <summary>A pooling or decision head output.</summary>
    HeadOutput,

    /// <summary>The sequence finished; <see cref="TokenEvent.Finish"/> says why.</summary>
    Finished,

    /// <summary>The request failed; <see cref="TokenEvent.Error"/> says why.</summary>
    Failed,
}

/// <summary>A logprob row entry.</summary>
/// <param name="TokenId">Token.</param>
/// <param name="Logprob">Natural-log probability.</param>
public readonly record struct TokenLogprob(int TokenId, float Logprob);

/// <summary>
/// One event on a request's channel (contract C6). Token ids and logprobs are views into engine-owned buffers, valid
/// until the next read from the channel.
/// </summary>
public readonly record struct TokenEvent
{
    /// <summary>The request.</summary>
    public required RequestId RequestId { get; init; }

    /// <summary>The kind.</summary>
    public required TokenEventKind Kind { get; init; }

    /// <summary>Sequence index within the request (n, best_of, beams).</summary>
    public int Sequence { get; init; }

    /// <summary>Tokens emitted this step for the sequence.</summary>
    public ReadOnlyMemory<int> TokenIds { get; init; }

    /// <summary>Logprob rows: (1 + top_logprobs) entries per token, chosen token first.</summary>
    public ReadOnlyMemory<TokenLogprob> Logprobs { get; init; }

    /// <summary>Head output (pooling vector or decision scores).</summary>
    public ReadOnlyMemory<float> HeadOutput { get; init; }

    /// <summary>Why the sequence finished, for <see cref="TokenEventKind.Finished"/>.</summary>
    public FinishReason Finish { get; init; }

    /// <summary>The error, for <see cref="TokenEventKind.Failed"/>.</summary>
    public RoostError? Error { get; init; }

    /// <summary>
    /// True for every event after the worker was restarted and the request resumed (R-070): tokens before it were
    /// produced by the previous worker.
    /// </summary>
    public bool Recovered { get; init; }

    /// <summary>Prompt tokens, completion tokens so far.</summary>
    public (int Prompt, int Completion) Usage { get; init; }
}

/// <summary>A submitted request's event channel.</summary>
public interface IRequestChannel
{
    /// <summary>The request.</summary>
    RequestId Id { get; }

    /// <summary>Events in order; completes after the last <see cref="TokenEventKind.Finished"/> or a failure.</summary>
    ChannelReader<TokenEvent> Events { get; }

    /// <summary>Cancels the request; the channel then completes.</summary>
    void Cancel();
}

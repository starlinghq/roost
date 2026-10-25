using System.Threading.Channels;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost.Driver;

/// <summary>
/// What the chat client runs requests on, in contract C6 terms: a <see cref="Request"/> in, a channel of
/// <see cref="TokenEvent"/>s out. M0's implementation is <see cref="WorkerDriver"/> (the eager worker, one step at a time);
/// from M1 the engine implements it, and the public API does not change.
/// </summary>
internal interface IGenerationBackend : IAsyncDisposable
{
    /// <summary>Queues a request. Refusals arrive as a <see cref="TokenEventKind.Failed"/> event on the channel.</summary>
    /// <param name="request">The request.</param>
    /// <returns>Its event channel.</returns>
    IRequestChannel Submit(Request request);
}

/// <summary>An <see cref="IRequestChannel"/> over a channel reader and a cancel callback.</summary>
/// <param name="id">The request.</param>
/// <param name="events">The events.</param>
/// <param name="cancel">Asks the backend to drop the request.</param>
internal sealed class RequestChannel(RequestId id, ChannelReader<TokenEvent> events, Action<RequestId> cancel) : IRequestChannel
{
    private int _cancelled;

    public RequestId Id { get; } = id;

    public ChannelReader<TokenEvent> Events { get; } = events;

    public void Cancel()
    {
        if (Interlocked.Exchange(ref _cancelled, 1) == 0)
        {
            cancel(Id);
        }
    }
}

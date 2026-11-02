using System.Globalization;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Server.Responses;

/// <summary>
/// The in-memory store behind <c>GET /v1/responses/{id}</c> and <c>POST /v1/responses/{id}/cancel</c>: at most
/// <c>capacity</c> responses, each kept for <c>ttl</c> after it was created. A finished response past its expiry is dropped on
/// the next access; a running one is never dropped. When the store is full the oldest finished response makes room, and when
/// every stored response is still running a new one is refused with <see cref="RoostErrorCode.QueueFull"/> (429). Nothing
/// survives a restart.
/// </summary>
internal sealed class ResponseStore
{
    private readonly Lock _gate = new();
    private readonly Dictionary<string, StoredResponse> _byId = new(StringComparer.Ordinal);
    private readonly LinkedList<StoredResponse> _byAge = new();
    private readonly int _capacity;
    private readonly TimeSpan _ttl;
    private readonly TimeProvider _time;

    public ResponseStore(int capacity, TimeSpan ttl, TimeProvider time)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(ttl, TimeSpan.Zero);
        _capacity = capacity;
        _ttl = ttl;
        _time = time;
    }

    public TimeProvider Time => _time;

    public int Count
    {
        get
        {
            lock (_gate)
            {
                Purge();
                return _byId.Count;
            }
        }
    }

    /// <summary>Stores a response.</summary>
    /// <exception cref="RoostChatException">Every stored response is still running and the store is full.</exception>
    public void Add(StoredResponse response)
    {
        lock (_gate)
        {
            Purge();
            if (_byId.Count >= _capacity)
            {
                var oldest = _byAge.First;
                while (oldest is not null && !oldest.Value.IsTerminal)
                {
                    oldest = oldest.Next;
                }

                if (oldest is null)
                {
                    throw new RoostChatException(new RoostError(
                        RoostErrorCode.QueueFull,
                        string.Create(CultureInfo.InvariantCulture, $"The response store holds {_capacity} responses that are all still running; try again when one finishes.")));
                }

                Remove(oldest);
            }

            _byId.Add(response.Id, response);
            _byAge.AddLast(response);
        }
    }

    public bool TryGet(string id, out StoredResponse response)
    {
        lock (_gate)
        {
            Purge();
            return _byId.TryGetValue(id, out response!);
        }
    }

    private void Purge()
    {
        var now = _time.GetUtcNow();
        var node = _byAge.First;
        while (node is not null)
        {
            var next = node.Next;
            if (now - node.Value.CreatedAt <= _ttl)
            {
                break; // the list is in creation order: everything after is younger
            }

            if (node.Value.IsTerminal)
            {
                Remove(node);
            }

            node = next;
        }
    }

    private void Remove(LinkedListNode<StoredResponse> node)
    {
        _byId.Remove(node.Value.Id);
        _byAge.Remove(node);
    }
}

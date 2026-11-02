using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;
using Starlings.Roost.Server.OpenAi;
using Starlings.Roost.Server.Routing;

namespace Starlings.Roost.Server.Responses;

/// <summary>
/// The OpenAI Responses API routes (R-074): <c>POST /v1/responses</c> (whole, streamed with the Responses API's named events, or
/// in the background), <c>GET /v1/responses/{id}</c> and <c>POST /v1/responses/{id}/cancel</c>, over
/// <see cref="RoostChatClient"/> and an in-memory <see cref="ResponseStore"/>. Failures use the OpenAI error body
/// (<see cref="ErrorBodies"/>).
/// </summary>
/// <remarks>
/// <para>A response with <c>store</c> true (OpenAI's default) is kept for retrieval: a whole response once it is complete, a
/// streamed one from its <c>response.created</c> event, a background one from its creation. A stream ends with
/// <c>response.completed</c>, <c>response.incomplete</c> (output limit or end of context, <c>incomplete_details.reason</c>
/// <c>max_output_tokens</c>) or <c>response.failed</c>; there is no <c>[DONE]</c>, as in OpenAI's API.</para>
/// <para>As in OpenAI's API, only a background response can be cancelled. A client that disconnects from a stream cancels the
/// generation, and the stored response reads <c>cancelled</c>.</para>
/// </remarks>
internal sealed class ResponsesRoutes
{
    private readonly RoostChatClient _client;
    private readonly ResponseStore _store;
    private readonly CancellationToken _stopping;
    private readonly RouteRunner _runner;

    public ResponsesRoutes(RoostChatClient client, ResponseStore store, ILogger logger, CancellationToken stopping)
    {
        _client = client;
        _store = store;
        _stopping = stopping;
        _runner = new RouteRunner(logger, ErrorBodies.Shape);
    }

    /// <summary><c>POST /v1/responses</c>.</summary>
    public Task CreateAsync(HttpContext context) => _runner.RunAsync(context, async ct =>
    {
        ResponsesRequest request;
        using (var body = await RouteRunner.ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            request = ResponsesRequestReader.Read(body.RootElement);
        }

        var stored = new StoredResponse(request, _client.Model.ModelId, _store.Time.GetUtcNow());
        if (request.Background)
        {
            await StartBackgroundAsync(context, stored).ConfigureAwait(false);
            return;
        }

        // Refused options throw here, before anything is written.
        var updates = Start(request, ct);
        var enumerator = updates.GetAsyncEnumerator(ct);
        await using (enumerator.ConfigureAwait(false))
        {
            // The first update is awaited before anything is stored or written, so a request refused on its first step (a
            // prompt longer than the context) is an HTTP error and leaves nothing behind.
            bool more = await enumerator.MoveNextAsync().ConfigureAwait(false);
            if (request.Stream)
            {
                await StreamAsync(context, stored, enumerator, more, ct).ConfigureAwait(false);
                return;
            }

            await ConsumeAsync(stored, enumerator, more, null).ConfigureAwait(false);
        }

        if (request.Store)
        {
            _store.Add(stored);
        }

        await RouteRunner.WriteJsonAsync(context, RouteRunner.Json(stored.Write)).ConfigureAwait(false);
    });

    /// <summary><c>GET /v1/responses/{id}</c>: the stored response as it stands.</summary>
    public Task RetrieveAsync(HttpContext context) => _runner.RunAsync(context, async ct =>
    {
        if (context.Request.Query.Keys.FirstOrDefault() is { } key)
        {
            throw RequestFields.Refuse(key, $"Query parameter '{key}' is unsupported on retrieve (streaming a stored response is not built).");
        }

        var stored = Find(context);
        await RouteRunner.WriteJsonAsync(context, RouteRunner.Json(stored.Write)).ConfigureAwait(false);
    });

    /// <summary><c>POST /v1/responses/{id}/cancel</c>: cancels a background response and answers it, now <c>cancelled</c>.</summary>
    public Task CancelAsync(HttpContext context) => _runner.RunAsync(context, async ct =>
    {
        var stored = Find(context);
        if (!stored.Request.Background)
        {
            throw new RouteException(StatusCodes.Status400BadRequest, "invalid_request", "Only a response created with background true can be cancelled.", "response_id");
        }

        if (stored.Cancellation is { } cancellation && !stored.IsTerminal)
        {
            try
            {
                await cancellation.CancelAsync().ConfigureAwait(false);
            }
            catch (ObjectDisposedException)
            {
                // The generation finished between the check and the cancel.
            }

            if (stored.Run is { } run)
            {
                await run.WaitAsync(ct).ConfigureAwait(false);
            }
        }

        var status = stored.Status;
        if (status != StoredResponse.Cancelled)
        {
            throw new RouteException(StatusCodes.Status400BadRequest, "invalid_request", $"The response cannot be cancelled: its status is '{status}'.", "response_id");
        }

        await RouteRunner.WriteJsonAsync(context, RouteRunner.Json(stored.Write)).ConfigureAwait(false);
    });

    private StoredResponse Find(HttpContext context)
    {
        var id = context.Request.RouteValues["id"] as string ?? string.Empty;
        return _store.TryGet(id, out var stored)
            ? stored
            : throw new RouteException(StatusCodes.Status404NotFound, "not_found", "No stored response has this id: it was created with store false, it expired, or the server restarted.", "response_id");
    }

    /// <summary>Starts the request on the client; a refused response format is named <c>text.format</c>, this API's field.</summary>
    private IAsyncEnumerable<ChatResponseUpdate> Start(ResponsesRequest request, CancellationToken ct)
    {
        try
        {
            return _client.GetStreamingResponseAsync(request.Messages, request.Options, ct);
        }
        catch (RoostChatException e) when (e.Error.Param == "response_format")
        {
            throw new RoostChatException(e.Error with { Param = "text.format" }, e);
        }
    }

    private async Task StartBackgroundAsync(HttpContext context, StoredResponse stored)
    {
        var request = stored.Request;
        var cancellation = CancellationTokenSource.CreateLinkedTokenSource(_stopping);
        IAsyncEnumerable<ChatResponseUpdate> updates;
        try
        {
            // Refused options throw here, before anything is stored.
            updates = Start(request, cancellation.Token);
            stored.Cancellation = cancellation;
            _store.Add(stored);
        }
        catch
        {
            cancellation.Dispose();
            throw;
        }

        stored.Run = Task.Run(() => RunBackgroundAsync(stored, updates, cancellation), CancellationToken.None);
        await RouteRunner.WriteJsonAsync(context, RouteRunner.Json(stored.Write)).ConfigureAwait(false);
    }

    private async Task RunBackgroundAsync(StoredResponse stored, IAsyncEnumerable<ChatResponseUpdate> updates, CancellationTokenSource cancellation)
    {
        var ct = cancellation.Token;
        try
        {
            stored.Start();
            var enumerator = updates.GetAsyncEnumerator(ct);
            await using (enumerator.ConfigureAwait(false))
            {
                bool more = await enumerator.MoveNextAsync().ConfigureAwait(false);
                await ConsumeAsync(stored, enumerator, more, null).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            stored.MarkCancelled();
        }
        catch (RoostChatException e)
        {
            stored.Fail(e.Error);
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            ServerLog.Unhandled(_runner.Logger, e);
            stored.Fail(new RoostError(RoostErrorCode.Internal, "Internal error."));
        }
        finally
        {
            cancellation.Dispose();
        }
    }

    /// <summary>Reads the rest of the updates into <paramref name="stored"/>, passing each piece of text to <paramref name="onText"/>.</summary>
    private static async Task ConsumeAsync(StoredResponse stored, IAsyncEnumerator<ChatResponseUpdate> enumerator, bool more, Func<string, Task>? onText)
    {
        ChatFinishReason? finish = null;
        UsageDetails? usage = null;
        while (more)
        {
            var update = enumerator.Current;
            var text = update.Text;
            if (text.Length > 0)
            {
                stored.Append(text);
                if (onText is not null)
                {
                    await onText(text).ConfigureAwait(false);
                }
            }

            finish ??= update.FinishReason;
            usage = update.Contents.OfType<UsageContent>().LastOrDefault()?.Details ?? usage;
            more = await enumerator.MoveNextAsync().ConfigureAwait(false);
        }

        stored.Finish(finish, usage);
    }

    /// <summary>
    /// Streams the Responses API's events: <c>response.created</c>, <c>response.in_progress</c>,
    /// <c>response.output_item.added</c>, <c>response.content_part.added</c>, <c>response.output_text.delta</c>s,
    /// <c>response.output_text.done</c>, <c>response.content_part.done</c>, <c>response.output_item.done</c>, then
    /// <c>response.completed</c> or <c>response.incomplete</c>; a failure after the headers is <c>response.failed</c>.
    /// </summary>
    private async Task StreamAsync(HttpContext context, StoredResponse stored, IAsyncEnumerator<ChatResponseUpdate> enumerator, bool more, CancellationToken ct)
    {
        if (stored.Request.Store)
        {
            _store.Add(stored);
        }

        var response = context.Response;
        RouteRunner.BeginEventStream(context);
        var sequence = 0;
        Task EmitAsync(string type, Action<Utf8JsonWriter> write) => RouteRunner.WriteEventAsync(response, type, RouteRunner.Json(w =>
        {
            w.WriteStartObject();
            w.WriteString("type", type);
            w.WriteNumber("sequence_number", sequence++);
            write(w);
            w.WriteEndObject();
        }), ct);

        void Position(Utf8JsonWriter w, bool part)
        {
            if (part)
            {
                w.WriteString("item_id", stored.MessageId);
            }

            w.WriteNumber("output_index", 0);
            if (part)
            {
                w.WriteNumber("content_index", 0);
            }
        }

        try
        {
            await EmitAsync("response.created", w => Response(w, stored)).ConfigureAwait(false);
            await EmitAsync("response.in_progress", w => Response(w, stored)).ConfigureAwait(false);
            await EmitAsync("response.output_item.added", w =>
            {
                Position(w, part: false);
                w.WritePropertyName("item");
                stored.WriteItem(w);
            }).ConfigureAwait(false);
            await EmitAsync("response.content_part.added", w =>
            {
                Position(w, part: true);
                w.WritePropertyName("part");
                StoredResponse.WritePart(w, string.Empty);
            }).ConfigureAwait(false);

            await ConsumeAsync(stored, enumerator, more, text => EmitAsync("response.output_text.delta", w =>
            {
                Position(w, part: true);
                w.WriteString("delta", text);
                w.WriteStartArray("logprobs");
                w.WriteEndArray();
            })).ConfigureAwait(false);

            await EmitAsync("response.output_text.done", w =>
            {
                Position(w, part: true);
                w.WriteString("text", stored.Text);
                w.WriteStartArray("logprobs");
                w.WriteEndArray();
            }).ConfigureAwait(false);
            await EmitAsync("response.content_part.done", w =>
            {
                Position(w, part: true);
                w.WritePropertyName("part");
                stored.WritePart(w);
            }).ConfigureAwait(false);
            await EmitAsync("response.output_item.done", w =>
            {
                Position(w, part: false);
                w.WritePropertyName("item");
                stored.WriteItem(w);
            }).ConfigureAwait(false);
            var final = stored.Status == StoredResponse.Incomplete ? "response.incomplete" : "response.completed";
            await EmitAsync(final, w => Response(w, stored)).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            stored.MarkCancelled();
            throw;
        }
        catch (RoostChatException e)
        {
            stored.Fail(e.Error);
            await EmitAsync("response.failed", w => Response(w, stored)).ConfigureAwait(false);
        }
        catch (Exception e) when (!ct.IsCancellationRequested && e is not OutOfMemoryException)
        {
            ServerLog.Unhandled(_runner.Logger, e);
            stored.Fail(new RoostError(RoostErrorCode.Internal, "Internal error."));
            await EmitAsync("response.failed", w => Response(w, stored)).ConfigureAwait(false);
        }
    }

    private static void Response(Utf8JsonWriter w, StoredResponse stored)
    {
        w.WritePropertyName("response");
        stored.Write(w);
    }
}

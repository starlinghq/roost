using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;
using Starlings.Roost.Server.Routing;

namespace Starlings.Roost.Server.Anthropic;

/// <summary>
/// The Anthropic Messages API routes (R-074): <c>POST /v1/messages</c>, whole or streamed with Anthropic's named events
/// (<c>message_start</c>, <c>content_block_start</c>, <c>ping</c>, <c>content_block_delta</c>, <c>content_block_stop</c>,
/// <c>message_delta</c>, <c>message_stop</c>), and <c>POST /v1/messages/count_tokens</c>. Both map onto
/// <see cref="RoostChatClient"/>; failures use Anthropic's error body (<see cref="AnthropicErrors"/>), and a failure inside a
/// stream is an <c>error</c> event. The <c>anthropic-version</c> and <c>x-api-key</c> headers are not checked (the server has no
/// authentication).
/// </summary>
/// <remarks>
/// <see cref="RoostChatClient"/> reports a stop token and a stop string alike as <see cref="ChatFinishReason.Stop"/>, so
/// <c>stop_reason</c> is <c>end_turn</c> for both and <c>stop_sequence</c> is always null; <c>max_tokens</c> is reported for the
/// output limit and for the end of the context. Usage arrives with the last update, so a stream's <c>message_start</c> carries 0
/// input tokens and <c>message_delta</c> carries both counts.
/// </remarks>
internal sealed class MessagesRoutes
{
    private static readonly byte[] Ping = """{"type":"ping"}"""u8.ToArray();
    private static readonly byte[] BlockStart = """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"""u8.ToArray();
    private static readonly byte[] BlockStop = """{"type":"content_block_stop","index":0}"""u8.ToArray();
    private static readonly byte[] MessageStop = """{"type":"message_stop"}"""u8.ToArray();

    private readonly RoostChatClient _client;
    private readonly RouteRunner _runner;

    public MessagesRoutes(RoostChatClient client, ILogger logger)
    {
        _client = client;
        _runner = new RouteRunner(logger, AnthropicErrors.Shape);
    }

    /// <summary><c>POST /v1/messages</c>.</summary>
    public Task MessagesAsync(HttpContext context) => _runner.RunAsync(context, async ct =>
    {
        MessagesRequest request;
        using (var body = await RouteRunner.ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            request = MessagesRequestReader.Read(body.RootElement);
        }

        // Refused options throw here, before anything is written.
        var updates = _client.GetStreamingResponseAsync(request.Messages, request.Options, ct);
        if (request.Stream)
        {
            await StreamAsync(context, updates, ct).ConfigureAwait(false);
            return;
        }

        var response = await updates.ToChatResponseAsync(ct).ConfigureAwait(false);
        var json = RouteRunner.Json(w =>
        {
            w.WriteStartObject();
            WriteMessageHead(w, response.ResponseId, response.ModelId);
            w.WriteStartArray("content");
            w.WriteStartObject();
            w.WriteString("type", "text");
            w.WriteString("text", response.Text);
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteString("stop_reason", StopReason(response.FinishReason));
            w.WriteNull("stop_sequence");
            WriteUsage(w, response.Usage);
            w.WriteEndObject();
        });
        await RouteRunner.WriteJsonAsync(context, json).ConfigureAwait(false);
    });

    /// <summary><c>POST /v1/messages/count_tokens</c>: <c>{"input_tokens"}</c>, the prompt tokens the conversation renders to.</summary>
    public Task CountTokensAsync(HttpContext context) => _runner.RunAsync(context, async ct =>
    {
        MessagesRequest request;
        using (var body = await RouteRunner.ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            request = MessagesRequestReader.Read(body.RootElement, countTokens: true);
        }

        var model = _client.Model;
        if (!string.Equals(request.Options.ModelId, model.ModelId, StringComparison.Ordinal))
        {
            throw RequestFields.Refuse("model", $"Model '{request.Options.ModelId}' is not served here; this server serves '{model.ModelId}'.");
        }

        var count = ChatPrompts.Encode(model, ChatPrompts.Render(model, request.Messages, request.Options.Instructions)).Length;
        await RouteRunner.WriteJsonAsync(context, RouteRunner.Json(w =>
        {
            w.WriteStartObject();
            w.WriteNumber("input_tokens", count);
            w.WriteEndObject();
        })).ConfigureAwait(false);
    });

    /// <summary>
    /// Streams Anthropic's events. The first update is awaited before the headers are sent, so a request refused on its first
    /// step still gets its HTTP status; a failure after that is an <c>error</c> event and ends the stream.
    /// </summary>
    private async Task StreamAsync(HttpContext context, IAsyncEnumerable<ChatResponseUpdate> updates, CancellationToken ct)
    {
        var enumerator = updates.GetAsyncEnumerator(ct);
        await using (enumerator.ConfigureAwait(false))
        {
            bool more = await enumerator.MoveNextAsync().ConfigureAwait(false);
            var first = more ? enumerator.Current : throw new RoostChatException(new RoostError(RoostErrorCode.Internal, "The generation ended without an update."));
            var response = context.Response;
            RouteRunner.BeginEventStream(context);
            try
            {
                await RouteRunner.WriteEventAsync(response, "message_start", RouteRunner.Json(w =>
                {
                    w.WriteStartObject();
                    w.WriteString("type", "message_start");
                    w.WriteStartObject("message");
                    WriteMessageHead(w, first.ResponseId, first.ModelId);
                    w.WriteStartArray("content");
                    w.WriteEndArray();
                    w.WriteNull("stop_reason");
                    w.WriteNull("stop_sequence");
                    WriteUsage(w, null);
                    w.WriteEndObject();
                    w.WriteEndObject();
                }), ct).ConfigureAwait(false);
                await RouteRunner.WriteEventAsync(response, "content_block_start", BlockStart, ct).ConfigureAwait(false);
                await RouteRunner.WriteEventAsync(response, "ping", Ping, ct).ConfigureAwait(false);

                ChatFinishReason? finish = null;
                UsageDetails? usage = null;
                while (more)
                {
                    var update = enumerator.Current;
                    var text = update.Text;
                    if (text.Length > 0)
                    {
                        await RouteRunner.WriteEventAsync(response, "content_block_delta", RouteRunner.Json(w =>
                        {
                            w.WriteStartObject();
                            w.WriteString("type", "content_block_delta");
                            w.WriteNumber("index", 0);
                            w.WriteStartObject("delta");
                            w.WriteString("type", "text_delta");
                            w.WriteString("text", text);
                            w.WriteEndObject();
                            w.WriteEndObject();
                        }), ct).ConfigureAwait(false);
                    }

                    finish ??= update.FinishReason;
                    usage = update.Contents.OfType<UsageContent>().LastOrDefault()?.Details ?? usage;
                    more = await enumerator.MoveNextAsync().ConfigureAwait(false);
                }

                await RouteRunner.WriteEventAsync(response, "content_block_stop", BlockStop, ct).ConfigureAwait(false);
                await RouteRunner.WriteEventAsync(response, "message_delta", RouteRunner.Json(w =>
                {
                    w.WriteStartObject();
                    w.WriteString("type", "message_delta");
                    w.WriteStartObject("delta");
                    w.WriteString("stop_reason", StopReason(finish));
                    w.WriteNull("stop_sequence");
                    w.WriteEndObject();
                    WriteUsage(w, usage);
                    w.WriteEndObject();
                }), ct).ConfigureAwait(false);
                await RouteRunner.WriteEventAsync(response, "message_stop", MessageStop, ct).ConfigureAwait(false);
            }
            catch (RoostChatException e)
            {
                await RouteRunner.WriteEventAsync(response, "error", _runner.ErrorBody(e.Error), ct).ConfigureAwait(false);
            }
            catch (Exception e) when (!ct.IsCancellationRequested && e is not OutOfMemoryException)
            {
                ServerLog.Unhandled(_runner.Logger, e);
                await RouteRunner.WriteEventAsync(response, "error", _runner.ErrorBody(new RoostError(RoostErrorCode.Internal, "Internal error.")), ct).ConfigureAwait(false);
            }
        }
    }

    private void WriteMessageHead(Utf8JsonWriter w, string? responseId, string? modelId)
    {
        w.WriteString("id", "msg_" + responseId);
        w.WriteString("type", "message");
        w.WriteString("role", "assistant");
        w.WriteString("model", modelId ?? _client.Model.ModelId);
    }

    private static void WriteUsage(Utf8JsonWriter w, UsageDetails? usage)
    {
        w.WriteStartObject("usage");
        w.WriteNumber("input_tokens", usage?.InputTokenCount ?? 0);
        w.WriteNumber("output_tokens", usage?.OutputTokenCount ?? 0);
        w.WriteEndObject();
    }

    private static string StopReason(ChatFinishReason? reason) => reason == ChatFinishReason.Length ? "max_tokens" : "end_turn";
}

using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Server.OpenAi;

/// <summary>
/// The OpenAI routes of the PRD's V1 set (R-037): <c>POST /v1/chat/completions</c> and <c>POST /v1/completions</c>, whole or
/// streamed as server-sent events ending in <c>data: [DONE]</c>, and <c>GET /v1/models</c>. Every failure has the one error
/// body (<see cref="ErrorBodies"/>). A request is cancelled when its client disconnects: the request's abort token is the
/// generation's cancellation token, so the worker drops the sequence from its next step.
/// </summary>
internal sealed class OpenAiRoutes
{
    private static readonly byte[] DataPrefix = "data: "u8.ToArray();
    private static readonly byte[] EventEnd = "\n\n"u8.ToArray();
    private static readonly byte[] Done = "data: [DONE]\n\n"u8.ToArray();

    private readonly RoostChatClient _client;
    private readonly long _created;
    private readonly ILogger _logger;

    public OpenAiRoutes(RoostChatClient client, DateTimeOffset started, ILogger logger)
    {
        _client = client;
        _created = started.ToUnixTimeSeconds();
        _logger = logger;
    }

    private string ModelId => _client.Model.ModelId;

    /// <summary><c>GET /v1/models</c>: the one model this server serves.</summary>
    public Task ModelsAsync(HttpContext context) =>
        WriteJsonAsync(context, new ModelListBody("list", [new ModelBody(ModelId, "model", _created, "roost")]), OpenAiJsonContext.Default.ModelListBody);

    /// <summary><c>POST /v1/chat/completions</c>.</summary>
    public Task ChatCompletionsAsync(HttpContext context) => RunAsync(context, async ct =>
    {
        ChatCompletionRequest request;
        using (var body = await ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            request = OpenAiRequestReader.ReadChat(body.RootElement);
        }

        // Refused options throw here, before anything is written.
        var updates = _client.GetStreamingResponseAsync(request.Messages, request.Options, ct);
        if (request.Stream)
        {
            await StreamAsync(context, updates, request.IncludeUsage, ChatChunks.Instance, ct).ConfigureAwait(false);
            return;
        }

        var response = await updates.ToChatResponseAsync(ct).ConfigureAwait(false);
        var completion = new ChatCompletionBody(
            "chatcmpl-" + response.ResponseId,
            "chat.completion",
            Created(response.CreatedAt),
            response.ModelId ?? ModelId,
            [new ChatChoiceBody(0, new ChatMessageBody("assistant", response.Text), null, Finish(response.FinishReason))],
            Usage(response.Usage));
        await WriteJsonAsync(context, completion, OpenAiJsonContext.Default.ChatCompletionBody).ConfigureAwait(false);
    });

    /// <summary><c>POST /v1/completions</c>: a raw prompt, without the chat template.</summary>
    public Task CompletionsAsync(HttpContext context) => RunAsync(context, async ct =>
    {
        CompletionRequest request;
        using (var body = await ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            request = OpenAiRequestReader.ReadCompletion(body.RootElement);
        }

        var updates = _client.GetStreamingCompletionAsync(request.Prompt, request.Options, ct);
        if (request.Stream)
        {
            await StreamAsync(context, updates, request.IncludeUsage, TextChunks.Instance, ct).ConfigureAwait(false);
            return;
        }

        var response = await updates.ToChatResponseAsync(ct).ConfigureAwait(false);
        var completion = new TextCompletionBody(
            "cmpl-" + response.ResponseId,
            "text_completion",
            Created(response.CreatedAt),
            response.ModelId ?? ModelId,
            [new TextChoiceBody(0, response.Text, null, Finish(response.FinishReason))],
            Usage(response.Usage));
        await WriteJsonAsync(context, completion, OpenAiJsonContext.Default.TextCompletionBody).ConfigureAwait(false);
    });

    private async Task RunAsync(HttpContext context, Func<CancellationToken, Task> handle)
    {
        var ct = context.RequestAborted;
        try
        {
            await handle(ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // The client disconnected; the generation was cancelled with it and there is no one to answer.
        }
        catch (RoostChatException e) when (!context.Response.HasStarted)
        {
            await ErrorBodies.WriteAsync(context.Response, e.Error).ConfigureAwait(false);
        }
        catch (Exception e) when (!context.Response.HasStarted && e is not OutOfMemoryException)
        {
            // Exception messages in Roost never carry prompt text (C6); the body still says only "Internal error."
            ServerLog.Unhandled(_logger, e);
            await ErrorBodies.WriteAsync(context.Response, new RoostError(RoostErrorCode.Internal, "Internal error.")).ConfigureAwait(false);
        }
    }

    private static async Task<JsonDocument> ReadBodyAsync(HttpContext context, CancellationToken ct)
    {
        try
        {
            return await JsonDocument.ParseAsync(context.Request.Body, default, ct).ConfigureAwait(false);
        }
        catch (JsonException)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The request body is not valid JSON."));
        }
    }

    /// <summary>
    /// Streams updates as server-sent events. The first update is awaited before the headers are sent, so a request refused
    /// on its first step (a prompt longer than the context, say) still gets its HTTP status and error body. A failure after
    /// that is sent as a <c>data:</c> event carrying the error body, followed by <c>[DONE]</c>.
    /// </summary>
    private async Task StreamAsync<TChunk>(HttpContext context, IAsyncEnumerable<ChatResponseUpdate> updates, bool includeUsage, IChunkWriter<TChunk> chunks, CancellationToken ct)
    {
        var enumerator = updates.GetAsyncEnumerator(ct);
        await using (enumerator.ConfigureAwait(false))
        {
            bool more = await enumerator.MoveNextAsync().ConfigureAwait(false);
            var first = more ? enumerator.Current : throw new RoostChatException(new RoostError(RoostErrorCode.Internal, "The generation ended without an update."));
            string id = chunks.IdPrefix + first.ResponseId;
            long created = Created(first.CreatedAt);
            string model = first.ModelId ?? ModelId;

            var response = context.Response;
            response.StatusCode = StatusCodes.Status200OK;
            response.ContentType = "text/event-stream";
            response.Headers.CacheControl = "no-cache";
            response.Headers["X-Accel-Buffering"] = "no";
            context.Features.Get<IHttpResponseBodyFeature>()?.DisableBuffering();

            ChatFinishReason? finish = null;
            UsageDetails? usage = null;
            try
            {
                if (chunks.WritesRoleChunk)
                {
                    await WriteEventAsync(response, chunks.Chunk(id, created, model, null, null, null, role: true), chunks.TypeInfo, ct).ConfigureAwait(false);
                }

                while (more)
                {
                    var update = enumerator.Current;
                    var text = update.Text;
                    if (text.Length > 0)
                    {
                        await WriteEventAsync(response, chunks.Chunk(id, created, model, text, null, null, role: false), chunks.TypeInfo, ct).ConfigureAwait(false);
                    }

                    finish ??= update.FinishReason;
                    usage = update.Contents.OfType<UsageContent>().LastOrDefault()?.Details ?? usage;
                    more = await enumerator.MoveNextAsync().ConfigureAwait(false);
                }

                await WriteEventAsync(response, chunks.Chunk(id, created, model, null, Finish(finish) ?? "stop", null, role: false), chunks.TypeInfo, ct).ConfigureAwait(false);
                if (includeUsage)
                {
                    await WriteEventAsync(response, chunks.UsageChunk(id, created, model, Usage(usage)), chunks.TypeInfo, ct).ConfigureAwait(false);
                }
            }
            catch (RoostChatException e)
            {
                await WriteErrorEventAsync(response, e.Error, ct).ConfigureAwait(false);
            }
            catch (Exception e) when (!ct.IsCancellationRequested && e is not OutOfMemoryException)
            {
                ServerLog.Unhandled(_logger, e);
                await WriteErrorEventAsync(response, new RoostError(RoostErrorCode.Internal, "Internal error."), ct).ConfigureAwait(false);
            }

            await response.Body.WriteAsync(Done, ct).ConfigureAwait(false);
            await response.Body.FlushAsync(ct).ConfigureAwait(false);
        }
    }

    private static async Task WriteEventAsync<T>(HttpResponse response, T value, JsonTypeInfo<T> typeInfo, CancellationToken ct)
    {
        var json = JsonSerializer.SerializeToUtf8Bytes(value, typeInfo);
        await WriteDataAsync(response, json, ct).ConfigureAwait(false);
    }

    private static Task WriteErrorEventAsync(HttpResponse response, RoostError error, CancellationToken ct) =>
        WriteDataAsync(response, ErrorBodies.Body(error), ct);

    private static async Task WriteDataAsync(HttpResponse response, byte[] json, CancellationToken ct)
    {
        await response.Body.WriteAsync(DataPrefix, ct).ConfigureAwait(false);
        await response.Body.WriteAsync(json, ct).ConfigureAwait(false);
        await response.Body.WriteAsync(EventEnd, ct).ConfigureAwait(false);
        await response.Body.FlushAsync(ct).ConfigureAwait(false);
    }

    private static async Task WriteJsonAsync<T>(HttpContext context, T value, JsonTypeInfo<T> typeInfo)
    {
        context.Response.StatusCode = StatusCodes.Status200OK;
        context.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(context.Response.Body, value, typeInfo, context.RequestAborted).ConfigureAwait(false);
    }

    private static long Created(DateTimeOffset? at) => at?.ToUnixTimeSeconds() ?? DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    private static string? Finish(ChatFinishReason? reason)
    {
        if (reason is not { } r)
        {
            return null;
        }

        if (r == ChatFinishReason.Stop)
        {
            return "stop";
        }

        if (r == ChatFinishReason.Length)
        {
            return "length";
        }

        if (r == ChatFinishReason.ToolCalls)
        {
            return "tool_calls";
        }

        return r == ChatFinishReason.ContentFilter ? "content_filter" : r.Value;
    }

    private static UsageBody Usage(UsageDetails? usage)
    {
        int prompt = (int)(usage?.InputTokenCount ?? 0);
        int completion = (int)(usage?.OutputTokenCount ?? 0);
        return new UsageBody(prompt, completion, prompt + completion);
    }

    /// <summary>Makes the stream chunks of one route.</summary>
    private interface IChunkWriter<TChunk>
    {
        string IdPrefix { get; }

        bool WritesRoleChunk { get; }

        JsonTypeInfo<TChunk> TypeInfo { get; }

        TChunk Chunk(string id, long created, string model, string? text, string? finish, UsageBody? usage, bool role);

        TChunk UsageChunk(string id, long created, string model, UsageBody usage);
    }

    /// <summary><c>chat.completion.chunk</c>s: a first chunk with the role, content deltas, a finish chunk, and the usage chunk.</summary>
    private sealed class ChatChunks : IChunkWriter<ChatCompletionChunkBody>
    {
        public static readonly ChatChunks Instance = new();

        public string IdPrefix => "chatcmpl-";

        public bool WritesRoleChunk => true;

        public JsonTypeInfo<ChatCompletionChunkBody> TypeInfo => OpenAiJsonContext.Default.ChatCompletionChunkBody;

        public ChatCompletionChunkBody Chunk(string id, long created, string model, string? text, string? finish, UsageBody? usage, bool role) =>
            new(id, "chat.completion.chunk", created, model, [new ChatChunkChoiceBody(0, new ChatDeltaBody(role ? "assistant" : null, role ? string.Empty : text), null, finish)], usage);

        public ChatCompletionChunkBody UsageChunk(string id, long created, string model, UsageBody usage) =>
            new(id, "chat.completion.chunk", created, model, [], usage);
    }

    /// <summary>Streamed <c>text_completion</c>s: text chunks, a finish chunk, and the usage chunk.</summary>
    private sealed class TextChunks : IChunkWriter<TextCompletionBody>
    {
        public static readonly TextChunks Instance = new();

        public string IdPrefix => "cmpl-";

        public bool WritesRoleChunk => false;

        public JsonTypeInfo<TextCompletionBody> TypeInfo => OpenAiJsonContext.Default.TextCompletionBody;

        public TextCompletionBody Chunk(string id, long created, string model, string? text, string? finish, UsageBody? usage, bool role) =>
            new(id, "text_completion", created, model, [new TextChoiceBody(0, text ?? string.Empty, null, finish)], usage);

        public TextCompletionBody UsageChunk(string id, long created, string model, UsageBody usage) =>
            new(id, "text_completion", created, model, [], usage);
    }
}

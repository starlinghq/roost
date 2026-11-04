using System.Buffers;
using System.Text.Json;
using Microsoft.AspNetCore.Http.Features;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Server.Routing;

/// <summary>
/// A failure a route answers with an HTTP status and code of its own, outside contract C6's codes: a stored response that does
/// not exist (404 <c>not_found</c>) or cannot be cancelled (400 <c>invalid_request</c>).
/// </summary>
internal sealed class RouteException : Exception
{
    public RouteException(int status, string code, string message, string? param = null)
        : base(message)
    {
        Status = status;
        Code = code;
        Param = param;
    }

    public RouteException()
        : this(StatusCodes.Status500InternalServerError, "internal", "Internal error.")
    {
    }

    public RouteException(string message)
        : this(StatusCodes.Status500InternalServerError, "internal", message)
    {
    }

    public RouteException(string message, Exception innerException)
        : base(message, innerException)
    {
        Status = StatusCodes.Status500InternalServerError;
        Code = "internal";
    }

    /// <summary>The HTTP status.</summary>
    public int Status { get; }

    /// <summary>The error code name.</summary>
    public string Code { get; }

    /// <summary>The request field the error is about, if any.</summary>
    public string? Param { get; }
}

/// <summary>The error body of one API family (OpenAI's envelope, Anthropic's).</summary>
internal interface IErrorShape
{
    /// <summary>The body of an error with this HTTP status, code name, message and parameter.</summary>
    byte[] Body(int status, string code, string message, string? param);
}

/// <summary>
/// Runs a route handler: maps <see cref="RoostChatException"/>, <see cref="RouteException"/> and unexpected exceptions to an error
/// body in the route's <see cref="IErrorShape"/> while nothing has been written, and treats a client disconnect as the end of the
/// request. Also holds the request and response helpers the routes share.
/// </summary>
internal sealed class RouteRunner
{
    private static readonly byte[] EventPrefix = "event: "u8.ToArray();
    private static readonly byte[] DataPrefix = "data: "u8.ToArray();
    private static readonly byte[] LineEnd = "\n"u8.ToArray();
    private static readonly byte[] EventEnd = "\n\n"u8.ToArray();

    private readonly ILogger _logger;
    private readonly IErrorShape _shape;

    public RouteRunner(ILogger logger, IErrorShape shape)
    {
        _logger = logger;
        _shape = shape;
    }

    public ILogger Logger => _logger;

    public IErrorShape Shape => _shape;

    public async Task RunAsync(HttpContext context, Func<CancellationToken, Task> handle)
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
            await WriteErrorAsync(context.Response, e.Error.HttpStatus, e.Error.CodeName, e.Error.Message, e.Error.Param).ConfigureAwait(false);
        }
        catch (RouteException e) when (!context.Response.HasStarted)
        {
            await WriteErrorAsync(context.Response, e.Status, e.Code, e.Message, e.Param).ConfigureAwait(false);
        }
        catch (BadHttpRequestException e) when (!context.Response.HasStarted)
        {
            // Kestrel's refusal of the body: too large, or the client stopped sending it.
            await WriteErrorAsync(context.Response, e.StatusCode, "bad_request", "The request body could not be read.", null).ConfigureAwait(false);
        }
        catch (Exception e) when (!context.Response.HasStarted && e is not OutOfMemoryException)
        {
            // Exception messages in Roost never carry prompt text (C6); the body still says only "Internal error."
            ServerLog.Unhandled(_logger, e);
            var internalError = new RoostError(RoostErrorCode.Internal, "Internal error.");
            await WriteErrorAsync(context.Response, internalError.HttpStatus, internalError.CodeName, internalError.Message, null).ConfigureAwait(false);
        }
    }

    /// <summary>The body of a C6 error in this route's shape (for an error event inside a stream).</summary>
    public byte[] ErrorBody(RoostError error) => _shape.Body(error.HttpStatus, error.CodeName, error.Message, error.Param);

    private async Task WriteErrorAsync(HttpResponse response, int status, string code, string message, string? param)
    {
        response.StatusCode = status;
        response.ContentType = "application/json";
        await response.Body.WriteAsync(_shape.Body(status, code, message, param)).ConfigureAwait(false);
    }

    /// <summary>Parses the request body; a body that is not JSON is an <see cref="RoostErrorCode.InvalidOption"/>.</summary>
    public static async Task<JsonDocument> ReadBodyAsync(HttpContext context, CancellationToken ct)
    {
        JsonDocument document;
        try
        {
            document = await JsonDocument.ParseAsync(context.Request.Body, default, ct).ConfigureAwait(false);
        }
        catch (JsonException)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The request body is not valid JSON."));
        }

        try
        {
            CheckText(document.RootElement);
        }
        catch (InvalidOperationException)
        {
            document.Dispose();
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The request body holds text that is not valid Unicode."));
        }

        return document;
    }

    /// <summary>
    /// Reads every string and property name, so a lone surrogate or invalid UTF-8 fails here and not wherever a route's
    /// reader first touches it: <see cref="JsonDocument"/> accepts both, and reading them throws.
    /// </summary>
    private static void CheckText(JsonElement element)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                foreach (var field in element.EnumerateObject())
                {
                    _ = field.Name;
                    CheckText(field.Value);
                }

                break;
            case JsonValueKind.Array:
                foreach (var item in element.EnumerateArray())
                {
                    CheckText(item);
                }

                break;
            case JsonValueKind.String:
                _ = element.GetString();
                break;
        }
    }

    /// <summary>Writes a JSON value with <see cref="Utf8JsonWriter"/> (no reflection; Native AOT) and returns its bytes.</summary>
    public static byte[] Json(Action<Utf8JsonWriter> write)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            write(writer);
        }

        return buffer.WrittenSpan.ToArray();
    }

    /// <summary>Answers 200 with a JSON body.</summary>
    public static async Task WriteJsonAsync(HttpContext context, byte[] json)
    {
        context.Response.StatusCode = StatusCodes.Status200OK;
        context.Response.ContentType = "application/json";
        await context.Response.Body.WriteAsync(json, context.RequestAborted).ConfigureAwait(false);
    }

    /// <summary>Sends the headers of a server-sent event stream, unbuffered.</summary>
    public static void BeginEventStream(HttpContext context)
    {
        var response = context.Response;
        response.StatusCode = StatusCodes.Status200OK;
        response.ContentType = "text/event-stream";
        response.Headers.CacheControl = "no-cache";
        response.Headers["X-Accel-Buffering"] = "no";
        context.Features.Get<IHttpResponseBodyFeature>()?.DisableBuffering();
    }

    /// <summary>Writes one server-sent event (<c>event:</c> line when <paramref name="eventName"/> is set, then <c>data:</c>) and flushes.</summary>
    public static async Task WriteEventAsync(HttpResponse response, string? eventName, ReadOnlyMemory<byte> data, CancellationToken ct)
    {
        if (eventName is not null)
        {
            await response.Body.WriteAsync(EventPrefix, ct).ConfigureAwait(false);
            await response.Body.WriteAsync(System.Text.Encoding.UTF8.GetBytes(eventName), ct).ConfigureAwait(false);
            await response.Body.WriteAsync(LineEnd, ct).ConfigureAwait(false);
        }

        await response.Body.WriteAsync(DataPrefix, ct).ConfigureAwait(false);
        await response.Body.WriteAsync(data, ct).ConfigureAwait(false);
        await response.Body.WriteAsync(EventEnd, ct).ConfigureAwait(false);
        await response.Body.FlushAsync(ct).ConfigureAwait(false);
    }
}

using System.Buffers;
using System.Collections.Immutable;
using System.Net;
using System.Text;
using System.Text.Json;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;
using Starlings.Roost.Server.OpenAi;
using Starlings.Roost.Server.Routing;
using Starlings.Roost.WorkerHost;

namespace Starlings.Roost.Server;

/// <summary>
/// The sleep-mode routes (R-100), named as vLLM names them: <c>POST /sleep?level=1|2</c>, <c>POST /wake_up</c>,
/// <c>GET /is_sleeping</c> and <c>POST /update_weights</c>. Mapped only with <see cref="RoostServerOptions.EnableSleepMode"/>
/// (<c>--enable-sleep-mode</c>), and they answer only requests from this machine (a loopback peer and a <c>Host</c> that names it), whatever the listen address: they
/// release and replace the model, and the server has no authentication. Failures use the one error body.
/// </summary>
/// <remarks>
/// The check can't tell who is behind a proxy or an SSH forward on this machine that sends a loopback <c>Host</c>: every
/// client of such a forwarder is local. Don't put one in front of a server with these routes enabled.
/// </remarks>
/// <remarks>
/// Level 1 releases the KV cache and keeps the weights; level 2 also releases the weights, which the wake streams again from
/// their locked file. (vLLM's level 1 offloads weights to host memory, which Roost does not do.) <c>/update_weights</c> takes
/// <c>{"path": absolute path, "size": bytes, "sha256": hex}</c>: the worker verifies the file against that lock as it reads it
/// and keeps the old weights when it does not match.
/// </remarks>
internal sealed class AdminRoutes
{
    private readonly RoostChatClient _client;
    private readonly ILogger _logger;

    public AdminRoutes(RoostChatClient client, ILogger logger)
    {
        _client = client;
        _logger = logger;
    }

    /// <summary>Maps the routes.</summary>
    public void Map(WebApplication app)
    {
        app.MapPost("/sleep", (RequestDelegate)SleepAsync);
        app.MapPost("/wake_up", (RequestDelegate)WakeUpAsync);
        app.MapGet("/is_sleeping", (RequestDelegate)IsSleepingAsync);
        app.MapPost("/update_weights", (RequestDelegate)UpdateWeightsAsync);
    }

    /// <summary>
    /// Whether a request comes from this machine: a loopback remote address, and a <c>Host</c> that names this machine, so a
    /// page rebound to loopback is refused on any listener. A request with no remote address is not local.
    /// </summary>
    internal static bool IsLoopback(HttpContext context) =>
        context.Connection.RemoteIpAddress is { } address && IPAddress.IsLoopback(address) && RequestOriginGuard.NamesThisMachine(context.Request.Host);

    private Task SleepAsync(HttpContext context) => RunAsync(context, async ct =>
    {
        var level = RoostSleepLevel.KvCache;
        if (context.Request.Query.TryGetValue("level", out var values))
        {
            level = values.ToString() switch
            {
                "1" => RoostSleepLevel.KvCache,
                "2" => RoostSleepLevel.KvCacheAndWeights,
                _ => throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "level must be 1 (KV cache) or 2 (KV cache and weights).", "level")),
            };
        }

        return await _client.SleepAsync(level, ct).ConfigureAwait(false);
    });

    private Task WakeUpAsync(HttpContext context) => RunAsync(context, async ct =>
    {
        var result = await _client.WakeUpAsync(ct).ConfigureAwait(false);
        ServerLog.Woke(_logger, result.Duration.TotalMilliseconds, result.WorkerDuration.TotalMilliseconds);
        return result;
    });

    private Task UpdateWeightsAsync(HttpContext context) => RunAsync(context, async ct =>
    {
        WeightSource source;
        using (var body = await RouteRunner.ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            source = ReadSource(body.RootElement);
        }

        return await _client.UpdateWeightsAsync(source, ct).ConfigureAwait(false);
    });

    private async Task IsSleepingAsync(HttpContext context)
    {
        if (!IsLoopback(context))
        {
            await RefuseAsync(context).ConfigureAwait(false);
            return;
        }

        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteBoolean("is_sleeping", _client.IsSleeping);
            writer.WriteEndObject();
        }

        await WriteAsync(context, buffer).ConfigureAwait(false);
    }

    private static WeightSource ReadSource(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw Invalid("The body must be a JSON object with path, size and sha256.", null);
        }

        string? path = null;
        long? size = null;
        string? sha = null;
        foreach (var property in root.EnumerateObject())
        {
            switch (property.Name)
            {
                case "path" when property.Value.ValueKind == JsonValueKind.String:
                    path = property.Value.GetString();
                    break;
                case "size" when property.Value.ValueKind == JsonValueKind.Number && property.Value.TryGetInt64(out var s) && s >= 0:
                    size = s;
                    break;
                case "sha256" when property.Value.ValueKind == JsonValueKind.String:
                    sha = property.Value.GetString();
                    break;
                case "path" or "size" or "sha256":
                    throw Invalid($"{property.Name} has the wrong type or value.", property.Name);
                default:
                    throw Invalid($"Unknown field {property.Name}.", property.Name);
            }
        }

        if (string.IsNullOrEmpty(path) || !Path.IsPathFullyQualified(path))
        {
            throw Invalid("path must be an absolute path to the safetensors file, as the worker sees it.", "path");
        }

        if (Encoding.UTF8.GetByteCount(path) > ControlMessages.MaxPathBytes)
        {
            throw Invalid(FormattableString.Invariant($"path is limited to {ControlMessages.MaxPathBytes} UTF-8 bytes."), "path");
        }

        if (size is null)
        {
            throw Invalid("size (the locked size in bytes) is required.", "size");
        }

        if (sha is null || sha.Length != 64 || !sha.All(Uri.IsHexDigit))
        {
            throw Invalid("sha256 must be the locked SHA-256 as 64 hex digits.", "sha256");
        }

        return new WeightSource(path, size.Value, ImmutableArray.Create(Convert.FromHexString(sha)));
    }

    private static RoostChatException Invalid(string message, string? param) => new(new RoostError(RoostErrorCode.InvalidOption, message, param));

    private static Task RefuseAsync(HttpContext context) => ErrorBodies.WriteAsync(
        context.Response,
        StatusCodes.Status403Forbidden,
        "forbidden",
        $"{context.Request.Path} answers clients on this machine only: it releases or replaces the model, and the server has no authentication.");

    private static async Task WriteAsync(HttpContext context, ArrayBufferWriter<byte> buffer)
    {
        context.Response.StatusCode = StatusCodes.Status200OK;
        context.Response.ContentType = "application/json";
        await context.Response.Body.WriteAsync(buffer.WrittenMemory, context.RequestAborted).ConfigureAwait(false);
    }

    private async Task RunAsync(HttpContext context, Func<CancellationToken, Task<RoostWorkerControlResult>> handle)
    {
        if (!IsLoopback(context))
        {
            await RefuseAsync(context).ConfigureAwait(false);
            return;
        }

        var ct = context.RequestAborted;
        RoostWorkerControlResult result;
        try
        {
            result = await handle(ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            return; // the client disconnected; an operation already sent to the worker completes there
        }
        catch (IOException) when (ct.IsCancellationRequested)
        {
            return; // the client reset the connection while sending the body
        }
        catch (BadHttpRequestException e) when (!context.Response.HasStarted)
        {
            await ErrorBodies.WriteAsync(context.Response, e.StatusCode, "bad_request", "The request body could not be read.").ConfigureAwait(false);
            return;
        }
        catch (RoostChatException e) when (!context.Response.HasStarted)
        {
            await ErrorBodies.WriteAsync(context.Response, e.Error).ConfigureAwait(false);
            return;
        }
        catch (Exception e) when (!context.Response.HasStarted && e is not OutOfMemoryException)
        {
            ServerLog.Unhandled(_logger, e);
            await ErrorBodies.WriteAsync(context.Response, new RoostError(RoostErrorCode.Internal, "Internal error.")).ConfigureAwait(false);
            return;
        }

        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteString("operation", result.Operation switch
            {
                RoostWorkerOperation.Sleep => "sleep",
                RoostWorkerOperation.WakeUp => "wake_up",
                _ => "update_weights",
            });
            writer.WriteNumber("duration_ms", Math.Round(result.Duration.TotalMilliseconds, 3));
            writer.WriteNumber("worker_duration_ms", Math.Round(result.WorkerDuration.TotalMilliseconds, 3));
            writer.WriteNumber("bytes", result.Bytes);
            writer.WriteBoolean("is_sleeping", result.IsSleeping);
            if (result.WeightsDigest.IsDefaultOrEmpty)
            {
                writer.WriteNull("weights_sha256");
            }
            else
            {
                writer.WriteString("weights_sha256", Convert.ToHexStringLower(result.WeightsDigest.AsSpan()));
            }

            writer.WriteEndObject();
        }

        await WriteAsync(context, buffer).ConfigureAwait(false);
    }
}

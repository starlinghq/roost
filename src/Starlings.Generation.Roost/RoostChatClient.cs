using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Driver;
using Starlings.Roost.Generation;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost;

/// <summary>
/// Roost behind <see cref="IChatClient"/> (R-036): a conversation is rendered by the model's chat template, tokenised,
/// decoded on a GPU worker, detokenised incrementally and streamed back. The worker runs in its own process by default
/// (decision 21); <see cref="Starlings.Roost.Transport.TransportKind.InProcess"/> is for tests and development.
/// </summary>
/// <remarks>
/// <para><b>M0 scope.</b> Greedy decoding only. Every <see cref="ChatOptions"/> member that is set and not supported is
/// refused with a <see cref="RoostChatException"/> whose <see cref="RoostError.Code"/> is
/// <see cref="RoostErrorCode.InvalidOption"/> and whose <see cref="RoostError.Param"/> names the option; see
/// <see cref="Capabilities"/>. Requests may be issued concurrently; M0 runs one sequence per step, so they take turns.</para>
/// <para><b>Streaming.</b> Each update carries text that can no longer change: incomplete UTF-8 is held back by the
/// detokenizer and a possible stop string by the stop matcher. The last update carries the finish reason
/// (<see cref="ChatFinishReason.Stop"/> for a stop token or stop string, <see cref="ChatFinishReason.Length"/> for the
/// output limit or the end of the context) and a <see cref="UsageContent"/> with the prompt and generated token counts
/// (generated tokens include a stop token).</para>
/// <para><b>Cancellation.</b> Cancelling the token, or ending the enumeration early, drops the request from the worker's
/// next step and frees its blocks.</para>
/// <para><b>Telemetry.</b> Every request submitted to the worker records metrics on <see cref="Telemetry"/> and one activity
/// (<see cref="RoostTelemetry.RequestActivityName"/>) tagged with its id, token counts and finish reason, never its text.</para>
/// </remarks>
public sealed class RoostChatClient : IChatClient, IAsyncDisposable
{
    private readonly RoostChatClientOptions _options;
    private readonly IGenerationBackend _backend;
    private readonly ChatClientMetadata _metadata;
    private long _nextId;
    private int _disposed;

    private RoostChatClient(RoostChatClientOptions options, IGenerationBackend backend)
    {
        _options = options;
        _backend = backend;
        _metadata = new ChatClientMetadata("roost", null, options.Model.ModelId);
        Capabilities = new RoostCapabilities();
        Telemetry = new RoostTelemetry(options.Model.ModelId, () => backend.KvCacheUsage);
    }

    /// <summary>This client's metrics and per-request activities (R-038).</summary>
    public RoostTelemetry Telemetry { get; }

    /// <summary>What this client supports; anything not listed is unsupported.</summary>
    public RoostCapabilities Capabilities { get; }

    /// <summary>The model profile.</summary>
    public RoostModelProfile Model => _options.Model;

    internal IGenerationBackend Backend => _backend;

    /// <summary>Starts the worker (spawn, handshake and weight load) and returns a client over it.</summary>
    /// <param name="options">The settings.</param>
    /// <param name="cancellationToken">Cancels the start.</param>
    /// <returns>The client; dispose it to stop the worker.</returns>
    /// <exception cref="ArgumentException">The options are incomplete or inconsistent, or the packaged worker has no kernel library to load.</exception>
    /// <exception cref="FileNotFoundException">The default process transport found no <c>roost-worker</c> executable (<see cref="RoostWorkerRuntime"/>).</exception>
    /// <exception cref="UnauthorizedAccessException">The <c>roost-worker</c> executable lacks the execute permission and its mode cannot be changed.</exception>
    /// <exception cref="RoostChatException">The worker rejected the handshake (<see cref="RoostErrorCode.AssetMismatch"/>, <see cref="RoostErrorCode.ProfileUnsupported"/>) or did not start.</exception>
    public static Task<RoostChatClient> StartAsync(RoostChatClientOptions options, CancellationToken cancellationToken = default) =>
        StartAsync(options, null, cancellationToken);

    /// <summary>Starts a client whose transports are wrapped by <paramref name="decorate"/> (tests inject transport failures).</summary>
    internal static async Task<RoostChatClient> StartAsync(RoostChatClientOptions options, Func<IDriverTransport, IDriverTransport>? decorate, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(options.Model);
        ArgumentNullException.ThrowIfNull(options.Worker);
        ArgumentNullException.ThrowIfNull(options.StopMatcherFactory);
        options.Model.Validate();
        if (options.DefaultMaxOutputTokens <= 0)
        {
            throw new ArgumentException("DefaultMaxOutputTokens must be positive.", nameof(options));
        }

        var driver = await WorkerDriver.StartAsync(options.Model, options.Worker, cancellationToken, decorate).ConfigureAwait(false);
        return new RoostChatClient(options, driver);
    }

    /// <inheritdoc />
    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
        await GetStreamingResponseAsync(messages, options, cancellationToken).ToChatResponseAsync(cancellationToken).ConfigureAwait(false);

    /// <inheritdoc />
    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        // Validate before anything is enumerated, so a refused option throws at the call.
        var prepared = ChatRequestBuilder.Build(messages, options, _options.Model.ModelId, _options.StopMatcherFactory, _options.DefaultMaxOutputTokens);
        return StreamChatAsync(prepared, cancellationToken);
    }

    /// <summary>
    /// Completes a raw text prompt (the OpenAI completions route, R-037): the prompt is tokenised as given, with no chat
    /// template and no special tokens added, and decoded as a chat request is. The options are checked as for a chat, and
    /// <c>Instructions</c> is refused. Updates, finish reasons, usage and cancellation are those of
    /// <see cref="GetStreamingResponseAsync"/>.
    /// </summary>
    /// <param name="prompt">The prompt text.</param>
    /// <param name="options">The request options.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The streamed completion.</returns>
    /// <exception cref="RoostChatException">An option is unsupported or invalid (thrown at the call), or the request failed.</exception>
    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingCompletionAsync(string prompt, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var prepared = ChatRequestBuilder.BuildRaw(prompt, options, _options.Model.ModelId, _options.StopMatcherFactory, _options.DefaultMaxOutputTokens);
        return StreamAsync(prepared.Prompt, "prompt", prepared.StopSequences, prepared.StopMatcher, prepared.MaxOutputTokens, cancellationToken);
    }

    /// <summary>Completes a raw text prompt; see <see cref="GetStreamingCompletionAsync"/>.</summary>
    /// <param name="prompt">The prompt text.</param>
    /// <param name="options">The request options.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The completion.</returns>
    /// <exception cref="RoostChatException">An option is unsupported or invalid, or the request failed.</exception>
    public async Task<ChatResponse> GetCompletionAsync(string prompt, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
        await GetStreamingCompletionAsync(prompt, options, cancellationToken).ToChatResponseAsync(cancellationToken).ConfigureAwait(false);

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(ChatClientMetadata))
        {
            return _metadata;
        }

        if (serviceType == typeof(RoostCapabilities))
        {
            return Capabilities;
        }

        if (serviceType == typeof(RoostModelProfile))
        {
            return _options.Model;
        }

        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    /// <summary>Stops the worker. Requests still running fail with <see cref="RoostErrorCode.Internal"/>.</summary>
    /// <returns>A task that completes when the worker has exited.</returns>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            try
            {
                await _backend.DisposeAsync().ConfigureAwait(false);
            }
            finally
            {
                Telemetry.Dispose();
            }
        }
    }

    /// <summary>Stops the worker; blocks until it has exited. Prefer <see cref="DisposeAsync"/>.</summary>
    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    private async IAsyncEnumerable<ChatResponseUpdate> StreamChatAsync(PreparedChat prepared, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var prompt = Render(_options.Model.ChatTemplate, prepared.Messages);
        await foreach (var update in StreamAsync(prompt, "messages", prepared.StopSequences, prepared.StopMatcher, prepared.MaxOutputTokens, cancellationToken).ConfigureAwait(false))
        {
            yield return update;
        }
    }

    private async IAsyncEnumerable<ChatResponseUpdate> StreamAsync(string prompt, string promptParam, IReadOnlyList<string> stopSequences, IStopMatcher matcher, int maxOutputTokens, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        long started = Stopwatch.GetTimestamp();
        var model = _options.Model;
        int[] ids = [.. model.Tokenizer.Encode(prompt, false).Ids];
        if (ids.Length == 0)
        {
            throw RoostChatException.InvalidOption(promptParam, "The prompt has no tokens.");
        }

        int room = model.ContextTokens - ids.Length;
        if (room < 1)
        {
            throw new RoostChatException(new RoostError(
                RoostErrorCode.PromptTooLong,
                string.Create(CultureInfo.InvariantCulture, $"The prompt has {ids.Length} tokens; the worker's context holds {model.ContextTokens}, including at least one output token."),
                promptParam));
        }

        var id = new RequestId((ulong)Interlocked.Increment(ref _nextId));
        var request = new Request
        {
            Id = id,
            PromptTokenIds = [.. ids],
            Sampling = new SamplingParameters { Temperature = 0f, MaxTokens = Math.Min(maxOutputTokens, room) },
            StopSequences = [.. stopSequences],
        };

        string responseId = "roost-" + id.Value.ToString(CultureInfo.InvariantCulture);
        var created = DateTimeOffset.UtcNow;
        var detokenizer = new RoostIncrementalDetokenizer(model.Tokenizer, skipSpecialTokens: true);
        var telemetry = Telemetry.Start(responseId, started, ids.Length);
        var channel = _backend.Submit(request);
        var done = false;
        var ended = false; // the backend finished or failed the request itself
        ChatFinishReason? finish = null;
        RoostError? failure = null;
        (int Prompt, int Completion) usage = (ids.Length, 0);
        try
        {
            while (!done && await channel.Events.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
            {
                while (!done && channel.Events.TryRead(out var ev))
                {
                    usage = ev.Usage;
                    switch (ev.Kind)
                    {
                        case TokenEventKind.Tokens:
                            var tokens = ev.TokenIds;
                            telemetry.Tokens(tokens.Length);
                            for (var i = 0; i < tokens.Length; i++)
                            {
                                var match = matcher.Push(detokenizer.Push(tokens.Span[i]));
                                if (match.Emit.Length > 0)
                                {
                                    yield return Update(responseId, created, match.Emit);
                                }

                                if (match.Matched is not null)
                                {
                                    finish = ChatFinishReason.Stop;
                                    done = true;
                                    break;
                                }
                            }

                            break;
                        case TokenEventKind.Finished:
                            var tail = matcher.Push(detokenizer.Flush());
                            var text = tail.Matched is null ? tail.Emit + matcher.Flush() : tail.Emit;
                            if (text.Length > 0)
                            {
                                yield return Update(responseId, created, text);
                            }

                            ended = true;
                            finish = tail.Matched is not null ? ChatFinishReason.Stop : MapFinish(ev.Finish);
                            if (finish is null)
                            {
                                failure = new RoostError(RoostErrorCode.Internal, $"The worker finished the sequence with {ev.Finish}.");
                                throw new RoostChatException(failure);
                            }

                            done = true;
                            break;
                        case TokenEventKind.Failed:
                            ended = true;
                            done = true;
                            failure = ev.Error ?? new RoostError(RoostErrorCode.Internal, "The request failed.");
                            throw new RoostChatException(failure);
                        default:
                            break;
                    }
                }
            }

            if (!done)
            {
                failure = new RoostError(RoostErrorCode.Internal, "The request ended without a finish event.");
                throw new RoostChatException(failure);
            }
        }
        finally
        {
            if (!ended)
            {
                channel.Cancel(); // cancelled, abandoned or stopped on a string: drop it from the next step
            }

            string reason = failure is not null ? "error" : finish == ChatFinishReason.Stop ? "stop" : finish == ChatFinishReason.Length ? "length" : "abort";
            telemetry.End(reason, usage.Prompt, usage.Completion, failure);
        }

        yield return new ChatResponseUpdate(ChatRole.Assistant, [new UsageContent(new UsageDetails
        {
            InputTokenCount = usage.Prompt,
            OutputTokenCount = usage.Completion,
            TotalTokenCount = usage.Prompt + usage.Completion,
        })])
        {
            ResponseId = responseId,
            MessageId = responseId,
            ModelId = model.ModelId,
            CreatedAt = created,
            FinishReason = finish,
        };
    }

    private ChatResponseUpdate Update(string responseId, DateTimeOffset created, string text) => new(ChatRole.Assistant, text)
    {
        ResponseId = responseId,
        MessageId = responseId,
        ModelId = _options.Model.ModelId,
        CreatedAt = created,
    };

    private static ChatFinishReason? MapFinish(FinishReason reason) => reason switch
    {
        FinishReason.Eos or FinishReason.Grammar => ChatFinishReason.Stop,
        FinishReason.Length => ChatFinishReason.Length,
        _ => null,
    };

    private static string Render(IRoostChatTemplate template, IReadOnlyList<RoostChatMessage> messages)
    {
        try
        {
            return template.Render(messages);
        }
        catch (Exception e) when (e is not OutOfMemoryException and not OperationCanceledException)
        {
            // The template's own exception (Qwen35TemplateException for Qwen3.5) means the conversation is invalid for it.
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The chat template rejected the conversation: " + e.Message, "messages"), e);
        }
    }
}

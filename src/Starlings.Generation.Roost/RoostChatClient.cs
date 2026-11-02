using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Driver;
using Starlings.Generation.Roost.Parsing;
using Starlings.Roost.Generation;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost;

/// <summary>
/// Roost behind <see cref="IChatClient"/> (R-036): a conversation is rendered by the model's chat template, tokenised,
/// decoded on a GPU worker, detokenised incrementally and streamed back. The worker runs in its own process by default
/// (decision 21); <see cref="Starlings.Roost.Transport.TransportKind.InProcess"/> is for tests and development.
/// </summary>
/// <remarks>
/// <para><b>Scope.</b> Greedy decoding on the CUDA worker; with <see cref="RoostSampler.Cpu"/> (a worker that runs the CPU
/// sampler) also temperature, top-p, top-k, min-p, seeds, penalties and logprobs (R-042, CPU half). Every
/// <see cref="ChatOptions"/> member that is set and not supported is refused with a <see cref="RoostChatException"/> whose
/// <see cref="RoostError.Code"/> is <see cref="RoostErrorCode.InvalidOption"/> and whose <see cref="RoostError.Param"/> names the
/// option; see <see cref="Capabilities"/>. Requests may be issued concurrently; M0 runs one sequence per step, so they take
/// turns.</para>
/// <para><b>Seeds.</b> A request without <c>Seed</c> gets a random one, so two unseeded requests do not draw the same tokens.
/// With a seed, the draws depend only on the seed and the token position (Philox per sequence), never on other requests.</para>
/// <para><b>Logprobs.</b> With <see cref="LogprobsKey"/> set, each update's <c>AdditionalProperties</c> carries, under
/// <see cref="LogprobsKey"/>, an <c>IReadOnlyList&lt;</c><see cref="RoostTokenLogprob"/><c>&gt;</c> for the tokens decoded since
/// the previous update (the final update carries any left), each with up to <see cref="TopLogprobsKey"/> alternatives.
/// Logprobs are of the model's raw distribution, before penalties and temperature.</para>
/// <para><b>Streaming.</b> Each update carries text that can no longer change: incomplete UTF-8 is held back by the
/// detokenizer and a possible stop string by the stop matcher. The last update carries the finish reason
/// (<see cref="ChatFinishReason.Stop"/> for a stop token or stop string, <see cref="ChatFinishReason.Length"/> for the
/// output limit or the end of the context) and a <see cref="UsageContent"/> with the prompt and generated token counts
/// (generated tokens include a stop token).</para>
/// <para><b>Tools and reasoning (R-053).</b> <see cref="ChatOptions.Tools"/> (function declarations) are rendered by the chat
/// template and the model's calls come back as <see cref="FunctionCallContent"/>, with <see cref="ChatFinishReason.ToolCalls"/>;
/// <see cref="EnableThinkingProperty"/> enables thinking mode and the reasoning comes back as <see cref="TextReasoningContent"/>.
/// Both are parsed incrementally in the formats of <see cref="RoostModelProfile.EffectiveToolCallFormat"/> and
/// <see cref="RoostModelProfile.EffectiveReasoningFormat"/>; nothing constrains decoding, so the model may answer in text, call an
/// undeclared tool or omit arguments. A malformed call is returned as text.</para>
/// <para><b>Response formats (R-051).</b> <see cref="ChatOptions.ResponseFormat"/> JSON and JSON-schema formats are compiled with
/// <see cref="RoostChatClientOptions.Grammar"/>, so a schema the grammar engine cannot enforce is refused with its reason, and
/// are then refused: the worker does not apply token masks yet. Both refusals are <see cref="RoostErrorCode.InvalidOption"/>
/// naming <c>response_format</c>.</para>
/// <para><b>Cancellation.</b> Cancelling the token, or ending the enumeration early, drops the request from the worker's
/// next step and frees its blocks.</para>
/// <para><b>Telemetry.</b> Every request submitted to the worker records metrics on <see cref="Telemetry"/> and one activity
/// (<see cref="RoostTelemetry.RequestActivityName"/>) tagged with its id, token counts and finish reason, never its text.</para>
/// <para><b>Sleep, wake and weight update (R-100).</b> <see cref="SleepAsync"/> has the worker release its KV cache (and, at
/// <see cref="RoostSleepLevel.KvCacheAndWeights"/>, its weights) while the client is idle, for example while a training step
/// uses the GPU; <see cref="WakeUpAsync"/> restores them, checking the weights' digest, and records the wake time
/// (<see cref="LastWakeDuration"/>); <see cref="UpdateWeightsAsync"/> replaces the weights in place from a locked file, verified as
/// it is read, for reinforcement-learning loops. Each needs a worker with no request running, and requests made while the
/// worker is asleep are refused with <see cref="RoostErrorCode.InsufficientDeviceMemory"/>. A worker whose executor cannot do
/// them refuses with <see cref="RoostErrorCode.ProfileUnsupported"/>: the packaged CUDA <c>roost-worker</c> does today (see
/// <see cref="Capabilities"/>).</para>
/// </remarks>
public sealed class RoostChatClient : IChatClient, IAsyncDisposable
{
    /// <summary>
    /// The <see cref="ChatOptions.AdditionalProperties"/> key that enables the model's thinking mode (a Boolean; the chat
    /// template's <c>enable_thinking</c>). The completion is then split into <see cref="TextReasoningContent"/> and the answer.
    /// Refused for a model with no reasoning format (<see cref="RoostModelProfile.EffectiveReasoningFormat"/>) and by the
    /// Qwen3 template. The other accepted additional properties are the CPU sampler's (<see cref="MinPKey"/> and the keys after it).
    /// </summary>
    public const string EnableThinkingProperty = "enable_thinking";

    /// <summary>The <c>ChatOptions.AdditionalProperties</c> key for <c>min_p</c> (a number in [0, 1]); CPU sampler only.</summary>
    public const string MinPKey = "min_p";

    /// <summary>The <c>ChatOptions.AdditionalProperties</c> key for <c>repetition_penalty</c> (a positive number, 1 disables); CPU sampler only.</summary>
    public const string RepetitionPenaltyKey = "repetition_penalty";

    /// <summary>
    /// The <c>ChatOptions.AdditionalProperties</c> key for <c>logprobs</c> (a boolean; CPU sampler only), and the
    /// <c>ChatResponseUpdate.AdditionalProperties</c> key under which the logprobs come back.
    /// </summary>
    public const string LogprobsKey = "logprobs";

    /// <summary>The <c>ChatOptions.AdditionalProperties</c> key for <c>top_logprobs</c> (an integer in [0, 20], needs <see cref="LogprobsKey"/>); CPU sampler only.</summary>
    public const string TopLogprobsKey = "top_logprobs";

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
        var packagedCudaWorker = options.Worker.Transport == TransportKind.Process && options.Worker.Command is null;
        Capabilities = new RoostCapabilities(options.Worker.Sampler, options.Worker.AsyncScheduling, weightLifecycle: !packagedCudaWorker);
        Telemetry = new RoostTelemetry(options.Model.ModelId, () => backend.KvCacheUsage);
    }

    /// <summary>This client's metrics and per-request activities (R-038).</summary>
    public RoostTelemetry Telemetry { get; }

    /// <summary>What this client supports; anything not listed is unsupported.</summary>
    public RoostCapabilities Capabilities { get; }

    /// <summary>The model profile.</summary>
    public RoostModelProfile Model => _options.Model;

    /// <summary>Whether the worker is asleep (<see cref="SleepAsync"/>).</summary>
    public bool IsSleeping => _backend.IsSleeping;

    /// <summary>
    /// The time the last successful <see cref="WakeUpAsync"/> took, from the call reaching the driver to the worker's answer
    /// (state pools reallocated and, after a level 2 sleep, the weights streamed and verified again); null before the first wake.
    /// </summary>
    public TimeSpan? LastWakeDuration => _backend.LastWakeDuration;

    /// <summary>
    /// Puts the worker to sleep (R-100): it releases its KV cache and recurrent state pools, and at
    /// <see cref="RoostSleepLevel.KvCacheAndWeights"/> also its weights. Their contents are lost, so the client must be idle.
    /// </summary>
    /// <param name="level">What to release.</param>
    /// <param name="cancellationToken">Cancels the wait; a sleep already sent to the worker still completes there.</param>
    /// <returns>What the worker did, with the bytes it released.</returns>
    /// <exception cref="RoostChatException">Requests are running or the worker is already asleep (<see cref="RoostErrorCode.InvalidOption"/>), the worker cannot sleep (<see cref="RoostErrorCode.ProfileUnsupported"/>), or it was lost (<see cref="RoostErrorCode.WorkerRestarted"/>).</exception>
    public Task<RoostWorkerControlResult> SleepAsync(RoostSleepLevel level = RoostSleepLevel.KvCache, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _backend.ControlAsync(RoostWorkerOperation.Sleep, level, null, cancellationToken);
    }

    /// <summary>
    /// Wakes the worker (R-100): it reallocates what the sleep released and, after a level 2 sleep, streams the weights again
    /// from their locked file, verifying them as read. The weights must have the digest of the weights the client last
    /// verified (the profile's, or the last <see cref="UpdateWeightsAsync"/>'s); a file substituted while the worker slept fails
    /// with <see cref="RoostErrorCode.AssetMismatch"/> and the worker stays asleep. The duration is recorded in
    /// <see cref="LastWakeDuration"/>.
    /// </summary>
    /// <param name="cancellationToken">Cancels the wait; a wake already sent to the worker still completes there.</param>
    /// <returns>What the worker did, with the bytes it allocated and the wake time.</returns>
    /// <exception cref="RoostChatException">The worker is not asleep (<see cref="RoostErrorCode.InvalidOption"/>), the weights do not match (<see cref="RoostErrorCode.AssetMismatch"/>), or the worker was lost (<see cref="RoostErrorCode.WorkerRestarted"/>).</exception>
    public Task<RoostWorkerControlResult> WakeUpAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _backend.ControlAsync(RoostWorkerOperation.WakeUp, default, null, cancellationToken);
    }

    /// <summary>
    /// Replaces the worker's weights in place from a locked file (R-100): the worker streams it through the verified loader
    /// (R-024) and swaps it in only once its SHA-256 matched <see cref="WeightSource.Sha256"/>, so a substituted or truncated file
    /// fails with <see cref="RoostErrorCode.AssetMismatch"/> and the old weights stay. Allowed while asleep, where the new file
    /// becomes the one the wake restores. The file must have the model's tensors, dtypes and shapes. A worker the client starts
    /// after a fault loads the replacement too.
    /// </summary>
    /// <param name="source">The new weights: an absolute path as the worker sees it, the locked size and SHA-256.</param>
    /// <param name="cancellationToken">Cancels the wait; an update already sent to the worker still completes there.</param>
    /// <returns>What the worker did, with the digest it now serves.</returns>
    /// <exception cref="ArgumentException"><paramref name="source"/> is invalid.</exception>
    /// <exception cref="RoostChatException">Requests are running (<see cref="RoostErrorCode.InvalidOption"/>), the file does not match its lock (<see cref="RoostErrorCode.AssetMismatch"/>), the worker cannot replace weights (<see cref="RoostErrorCode.ProfileUnsupported"/>), or it was lost (<see cref="RoostErrorCode.WorkerRestarted"/>).</exception>
    public Task<RoostWorkerControlResult> UpdateWeightsAsync(WeightSource source, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _backend.ControlAsync(RoostWorkerOperation.UpdateWeights, default, source, cancellationToken);
    }

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
        var prepared = ChatRequestBuilder.Build(messages, options, _options.Model, _options.StopMatcherFactory, _options.DefaultMaxOutputTokens, _options.Worker.Sampler, _options.Grammar);
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
        var prepared = ChatRequestBuilder.BuildRaw(prompt, options, _options.Model.ModelId, _options.StopMatcherFactory, _options.DefaultMaxOutputTokens, _options.Worker.Sampler, _options.Grammar);
        return StreamAsync(prepared.Prompt, "prompt", prepared.StopSequences, prepared.StopMatcher, prepared.MaxOutputTokens, prepared.Sampling, null, cancellationToken);
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
        var model = _options.Model;
        var prompt = Render(model.ChatTemplate, prepared);
        var parser = RoostOutputParser.Create(prepared.EnableThinking, model.EffectiveReasoningFormat, prepared.Tools, model.EffectiveToolCallFormat);
        await foreach (var update in StreamAsync(prompt, "messages", prepared.StopSequences, prepared.StopMatcher, prepared.MaxOutputTokens, prepared.Sampling, parser, cancellationToken).ConfigureAwait(false))
        {
            yield return update;
        }
    }

    private async IAsyncEnumerable<ChatResponseUpdate> StreamAsync(string prompt, string promptParam, IReadOnlyList<string> stopSequences, IStopMatcher matcher, int maxOutputTokens, SamplingParameters sampling, RoostOutputParser? parser, [EnumeratorCancellation] CancellationToken cancellationToken)
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
            Sampling = sampling with
            {
                MaxTokens = Math.Min(maxOutputTokens, room),
                Seed = sampling.Seed ?? BitConverter.ToUInt64(RandomNumberGenerator.GetBytes(sizeof(ulong))),
            },
            StopSequences = [.. stopSequences],
        };

        string responseId = "roost-" + id.Value.ToString(CultureInfo.InvariantCulture);
        var created = DateTimeOffset.UtcNow;
        var detokenizer = new RoostIncrementalDetokenizer(model.Tokenizer, skipSpecialTokens: true);
        var telemetry = Telemetry.Start(responseId, started, ids.Length);
        var channel = _backend.Submit(request);
        var calls = new StrongBox<int>(0); // tool calls returned so far, for their ids
        var done = false;
        var ended = false; // the backend finished or failed the request itself
        ChatFinishReason? finish = null;
        RoostError? failure = null;
        (int Prompt, int Completion) usage = (ids.Length, 0);
        var pending = sampling.Logprobs ? new List<RoostTokenLogprob>() : null;
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
                            int perToken = pending is null || tokens.Length == 0 ? 0 : ev.Logprobs.Length / tokens.Length;
                            for (var i = 0; i < tokens.Length; i++)
                            {
                                if (perToken > 0)
                                {
                                    pending!.Add(ToLogprob(ev.Logprobs.Span.Slice(i * perToken, perToken), sampling.TopLogprobs));
                                }

                                var match = matcher.Push(detokenizer.Push(tokens.Span[i]));
                                if (match.Emit.Length > 0 && Shape(match.Emit, parser, calls, responseId, created, pending) is { } update)
                                {
                                    yield return update;
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
                            if (text.Length > 0 && Shape(text, parser, calls, responseId, created, pending) is { } tailUpdate)
                            {
                                yield return tailUpdate;
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

        if (parser is not null && Shape(null, parser, calls, responseId, created, pending) is { } last)
        {
            yield return last;
        }

        if (calls.Value > 0 && finish == ChatFinishReason.Stop)
        {
            finish = ChatFinishReason.ToolCalls; // the model stopped after calling tools
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
            AdditionalProperties = TakeLogprobs(pending),
        };
    }

    /// <summary>
    /// The update for a text delta: the text itself without a parser; otherwise the parts the parser has made final, as
    /// <see cref="TextContent"/>, <see cref="TextReasoningContent"/> and <see cref="FunctionCallContent"/> (null when there are
    /// none yet). A null <paramref name="text"/> completes the parser. The logprobs gathered since the last update go with it.
    /// </summary>
    private ChatResponseUpdate? Shape(string? text, RoostOutputParser? parser, StrongBox<int> calls, string responseId, DateTimeOffset created, List<RoostTokenLogprob>? pending)
    {
        List<AIContent> contents;
        if (parser is null)
        {
            contents = [new TextContent(text)];
        }
        else
        {
            var parts = text is null ? parser.Complete() : parser.Append(text);
            if (parts.Count == 0)
            {
                return null;
            }

            contents = new List<AIContent>(parts.Count);
            foreach (var part in parts)
            {
                contents.Add(part.Kind switch
                {
                    OutputPartKind.Reasoning => new TextReasoningContent(part.Text),
                    OutputPartKind.ToolCall => ToolCall(part.ToolCall!, responseId, calls.Value++),
                    _ => new TextContent(part.Text),
                });
            }
        }

        return new ChatResponseUpdate(ChatRole.Assistant, contents)
        {
            ResponseId = responseId,
            MessageId = responseId,
            ModelId = _options.Model.ModelId,
            CreatedAt = created,
            AdditionalProperties = TakeLogprobs(pending),
        };
    }

    /// <summary>
    /// A parsed call as <see cref="FunctionCallContent"/>: the id is <c>call_</c>, the response id and the call's index; the
    /// arguments are the JSON object's members as <see cref="JsonElement"/> values; <see cref="AIContent.RawRepresentation"/> is
    /// the <see cref="RoostToolCall"/>, whose JSON keeps the model's member order (after schema casting for Qwen3.5).
    /// </summary>
    private static FunctionCallContent ToolCall(RoostToolCall call, string responseId, int index)
    {
        var arguments = new Dictionary<string, object?>(StringComparer.Ordinal);
        using (var document = JsonDocument.Parse(call.ArgumentsJson))
        {
            foreach (var member in document.RootElement.EnumerateObject())
            {
                arguments[member.Name] = member.Value.Clone();
            }
        }

        return new FunctionCallContent("call_" + responseId + "_" + index.ToString(CultureInfo.InvariantCulture), call.Name, arguments)
        {
            RawRepresentation = call,
        };
    }

    // The logprobs gathered since the last update, moved into the update's properties; null when there are none.
    private static AdditionalPropertiesDictionary? TakeLogprobs(List<RoostTokenLogprob>? pending)
    {
        if (pending is not { Count: > 0 })
        {
            return null;
        }

        var taken = pending.ToArray();
        pending.Clear();
        return new AdditionalPropertiesDictionary { [LogprobsKey] = (IReadOnlyList<RoostTokenLogprob>)taken };
    }

    // One token's rows: the chosen token first, then the worker's alternatives, of which the request asked for `top`.
    private static RoostTokenLogprob ToLogprob(ReadOnlySpan<TokenLogprob> rows, int top) =>
        new(rows[0].TokenId, rows[0].Logprob, rows.Slice(1, Math.Min(top, rows.Length - 1)).ToArray());

    private static ChatFinishReason? MapFinish(FinishReason reason) => reason switch
    {
        FinishReason.Eos or FinishReason.Grammar => ChatFinishReason.Stop,
        FinishReason.Length => ChatFinishReason.Length,
        _ => null,
    };

    private static string Render(IRoostChatTemplate template, PreparedChat prepared)
    {
        try
        {
            return template.Render(prepared.Messages, prepared.Tools, prepared.EnableThinking);
        }
        catch (NotSupportedException e) when (!prepared.Tools.IsDefaultOrEmpty || prepared.EnableThinking)
        {
            // The Qwen3 template refuses tools and thinking mode, which Starling's published template does not support.
            var param = prepared.Tools.IsDefaultOrEmpty ? EnableThinkingProperty : "tools";
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The chat template does not support this request: " + e.Message, param), e);
        }
        catch (Exception e) when (e is not OutOfMemoryException and not OperationCanceledException)
        {
            // The template's own exception (Qwen35TemplateException for Qwen3.5) means the conversation is invalid for it.
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The chat template rejected the conversation: " + e.Message, "messages"), e);
        }
    }
}

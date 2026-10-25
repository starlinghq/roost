using System.Collections.Frozen;

namespace Starlings.Generation.Roost;

/// <summary>A capability's status. Missing never means supported: a name not in the table is <see cref="Unsupported"/>.</summary>
public enum RoostCapabilityStatus
{
    /// <summary>Not built, or refused with a typed error.</summary>
    Unsupported = 0,

    /// <summary>Built and tested on the CPU test path; no hardware evidence yet.</summary>
    Experimental,

    /// <summary>Built, but not exercised by any test or evidence run.</summary>
    Untested,

    /// <summary>Built, tested and backed by committed evidence.</summary>
    Supported,
}

/// <summary>
/// The capability table of a <see cref="RoostChatClient"/>. M0 (R-036): greedy decoding over the eager worker, streaming,
/// cancellation, usage and stop token ids are built and CPU-tested (<see cref="RoostCapabilityStatus.Experimental"/>; the
/// Preview 1 drill, R-062, attaches the GPU evidence). Everything not built is <see cref="RoostCapabilityStatus.Unsupported"/>
/// and is refused with a typed error naming the option.
/// </summary>
public sealed class RoostCapabilities
{
    /// <summary>Chat completion through <c>IChatClient.GetResponseAsync</c>.</summary>
    public const string Chat = "chat";

    /// <summary>Token streaming through <c>IChatClient.GetStreamingResponseAsync</c>.</summary>
    public const string Streaming = "streaming";

    /// <summary>Cancelling a request through its <see cref="CancellationToken"/> or by ending the enumeration.</summary>
    public const string Cancellation = "cancellation";

    /// <summary>Input and output token counts on every response.</summary>
    public const string Usage = "usage";

    /// <summary>Greedy (argmax) decoding: <c>Temperature</c> null or 0.</summary>
    public const string GreedyDecoding = "greedy_decoding";

    /// <summary>The model's stop token ids end generation with <c>ChatFinishReason.Stop</c>.</summary>
    public const string StopTokenIds = "stop_token_ids";

    /// <summary><c>MaxOutputTokens</c>, ending with <c>ChatFinishReason.Length</c>.</summary>
    public const string MaxOutputTokens = "max_output_tokens";

    /// <summary><c>ChatOptions.Instructions</c> as a leading system message.</summary>
    public const string Instructions = "instructions";

    /// <summary><c>ChatOptions.StopSequences</c>, matched on detokenized text in the host process.</summary>
    public const string StopSequences = "stop_sequences";

    /// <summary>Temperature, top-p, top-k and seeded sampling.</summary>
    public const string Sampling = "sampling";

    /// <summary>Frequency and presence penalties.</summary>
    public const string Penalties = "penalties";

    /// <summary>Tool definitions and tool calls.</summary>
    public const string Tools = "tools";

    /// <summary>JSON and JSON-schema response formats (structured output).</summary>
    public const string ResponseFormat = "response_format";

    /// <summary>Log probabilities.</summary>
    public const string Logprobs = "logprobs";

    /// <summary>Reasoning (think block) output.</summary>
    public const string Reasoning = "reasoning";

    /// <summary>Image, audio and other non-text input.</summary>
    public const string MultimodalInput = "multimodal_input";

    /// <summary>Server-side conversation state (<c>ConversationId</c>).</summary>
    public const string ConversationState = "conversation_state";

    /// <summary>Several requests batched into one step. M0 runs one sequence per step; concurrent requests wait their turn.</summary>
    public const string ContinuousBatching = "continuous_batching";

    /// <summary>Resuming requests on a respawned worker (R-070). In M0 a worker fault fails the requests it carried.</summary>
    public const string WorkerRecovery = "worker_recovery";

    private readonly FrozenDictionary<string, RoostCapabilityStatus> _statuses;

    internal RoostCapabilities(bool stopSequences)
    {
        var experimental = RoostCapabilityStatus.Experimental;
        var unsupported = RoostCapabilityStatus.Unsupported;
        _statuses = new Dictionary<string, RoostCapabilityStatus>
        {
            [Chat] = experimental,
            [Streaming] = experimental,
            [Cancellation] = experimental,
            [Usage] = experimental,
            [GreedyDecoding] = experimental,
            [StopTokenIds] = experimental,
            [MaxOutputTokens] = experimental,
            [Instructions] = experimental,
            [StopSequences] = stopSequences ? experimental : unsupported,
            [Sampling] = unsupported,
            [Penalties] = unsupported,
            [Tools] = unsupported,
            [ResponseFormat] = unsupported,
            [Logprobs] = unsupported,
            [Reasoning] = unsupported,
            [MultimodalInput] = unsupported,
            [ConversationState] = unsupported,
            [ContinuousBatching] = unsupported,
            [WorkerRecovery] = unsupported,
        }.ToFrozenDictionary(StringComparer.Ordinal);
    }

    /// <summary>Every listed capability and its status.</summary>
    public IReadOnlyDictionary<string, RoostCapabilityStatus> Statuses => _statuses;

    /// <summary>The status of a capability; <see cref="RoostCapabilityStatus.Unsupported"/> for a name not in the table.</summary>
    /// <param name="name">The capability name (the constants of this class).</param>
    /// <returns>The status.</returns>
    public RoostCapabilityStatus StatusOf(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return _statuses.TryGetValue(name, out var status) ? status : RoostCapabilityStatus.Unsupported;
    }
}

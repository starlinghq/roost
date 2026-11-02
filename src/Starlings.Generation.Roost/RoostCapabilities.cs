using System.Collections.Frozen;

namespace Starlings.Generation.Roost;

/// <summary>A capability's status. Missing never means supported: a name not in the table is <see cref="Unsupported"/>.</summary>
public enum RoostCapabilityStatus
{
    /// <summary>Not built, or refused with a typed error.</summary>
    Unsupported = 0,

    /// <summary>Built, with hardware evidence, and with known caveats.</summary>
    Experimental,

    /// <summary>Built and tested on the CPU path, without hardware evidence (AGENTS.md: no hardware evidence means untested).</summary>
    Untested,

    /// <summary>Built, tested and backed by committed evidence.</summary>
    Supported,
}

/// <summary>
/// The capability table of a <see cref="RoostChatClient"/>. M0 (R-036): greedy decoding over the eager worker, streaming,
/// cancellation, usage, stop token ids and stop strings are built and CPU-tested, with no GPU evidence yet
/// (<see cref="RoostCapabilityStatus.Untested"/>; what passes the Preview 1 drill, R-062, moves to
/// <see cref="RoostCapabilityStatus.Experimental"/>). Sampling, penalties and logprobs (R-042, CPU half) are
/// <see cref="RoostCapabilityStatus.Untested"/> for a client whose worker runs the CPU sampler (<see cref="RoostSampler.Cpu"/>)
/// and <see cref="RoostCapabilityStatus.Unsupported"/> on the CUDA worker until the GPU sampler exists. Everything not built is
/// <see cref="RoostCapabilityStatus.Unsupported"/> and is refused with a typed error naming the option.
/// </summary>
public sealed class RoostCapabilities
{
    /// <summary>Chat completion through <c>IChatClient.GetResponseAsync</c>.</summary>
    public const string Chat = "chat";

    /// <summary>Raw-text completion through <see cref="RoostChatClient.GetStreamingCompletionAsync"/> (no chat template).</summary>
    public const string RawCompletion = "raw_completion";

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

    /// <summary>
    /// Temperature, top-p, top-k and min-p sampling with a per-request seed (Philox per sequence: the same seed gives the same
    /// output whatever else is batched), on the CPU sampler only. (A seed is accepted under greedy decoding, where it changes
    /// nothing.)
    /// </summary>
    public const string Sampling = "sampling";

    /// <summary>Frequency, presence and repetition penalties, on the CPU sampler only.</summary>
    public const string Penalties = "penalties";

    /// <summary>Tool definitions and tool calls.</summary>
    public const string Tools = "tools";

    /// <summary>JSON and JSON-schema response formats (structured output).</summary>
    public const string ResponseFormat = "response_format";

    /// <summary>Log probabilities of the chosen token and up to 20 alternatives per token, on the CPU sampler only.</summary>
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

    /// <summary>
    /// Refusing new requests with <c>InsufficientDeviceMemory</c> while a co-tenant holds the device's free memory below
    /// <see cref="RoostWorkerOptions.MinFreeDeviceBytes"/> (R-044), from <see cref="RoostWorkerOptions.DeviceMemory"/>. CPU-tested
    /// only; the co-tenant GPU test is open and no NVML source is the default yet.
    /// </summary>
    public const string CoTenantAdmission = "co_tenant_admission";

    /// <summary>
    /// Async scheduling (R-032): step N+1 is planned and published while step N runs, with decode inputs patched by the worker
    /// and stop ids checked one step late without emitting the extra token. Only with <see cref="RoostWorkerOptions.AsyncScheduling"/>
    /// and a worker that patches inputs (the CPU path); the CUDA worker refuses such plans until stream F patches on the device.
    /// </summary>
    public const string AsyncScheduling = "async_scheduling";

    /// <summary>
    /// <see cref="RoostChatClient.SleepAsync"/> and <see cref="RoostChatClient.WakeUpAsync"/> (R-100): the worker releases its KV
    /// cache, and optionally its weights, and restores them with a digest check; the wake time is recorded. CPU path only (an
    /// in-process executor or custom worker that implements it); the packaged CUDA worker refuses it.
    /// </summary>
    public const string SleepWake = "sleep_wake";

    /// <summary>
    /// <see cref="RoostChatClient.UpdateWeightsAsync"/> (R-100): weights replaced in place from a locked file, verified as read,
    /// for reinforcement-learning loops. CPU path only; the packaged CUDA worker refuses it.
    /// </summary>
    public const string WeightUpdate = "weight_update";

    private static readonly string[] SamplerRows = [Sampling, Penalties, Logprobs];

    private readonly FrozenDictionary<string, RoostCapabilityStatus> _statuses;

    internal RoostCapabilities(RoostSampler sampler = RoostSampler.GreedyOnly, bool asyncScheduling = false, bool weightLifecycle = false)
    {
        var untested = RoostCapabilityStatus.Untested;
        var unsupported = RoostCapabilityStatus.Unsupported;
        var table = new Dictionary<string, RoostCapabilityStatus>
        {
            [Chat] = untested,
            [RawCompletion] = untested,
            [Streaming] = untested,
            [Cancellation] = untested,
            [Usage] = untested,
            [GreedyDecoding] = untested,
            [StopTokenIds] = untested,
            [MaxOutputTokens] = untested,
            [Instructions] = untested,
            [StopSequences] = untested,
            [CoTenantAdmission] = untested,
            [Sampling] = untested,
            [Penalties] = untested,
            [Tools] = unsupported,
            [ResponseFormat] = unsupported,
            [Logprobs] = untested,
            [Reasoning] = unsupported,
            [MultimodalInput] = unsupported,
            [ConversationState] = unsupported,
            [ContinuousBatching] = unsupported,
            [WorkerRecovery] = unsupported,
            [AsyncScheduling] = untested,
            [SleepWake] = untested,
            [WeightUpdate] = untested,
        };

        // Sleep, wake and weight update are the CPU path's (an executor with IWeightLifecycle); the packaged CUDA worker refuses them.
        if (!weightLifecycle)
        {
            table[SleepWake] = unsupported;
            table[WeightUpdate] = unsupported;
        }

        // Async scheduling is the CPU path's (an executor that patches inputs); a client that does not ask for it has none.
        if (!asyncScheduling)
        {
            table[AsyncScheduling] = unsupported;
        }

        // The rows above are the CPU sampler's (docs/capabilities.md lists them so); the CUDA worker has no sampler yet.
        if (sampler != RoostSampler.Cpu)
        {
            foreach (var name in SamplerRows)
            {
                table[name] = unsupported;
            }
        }

        _statuses = table.ToFrozenDictionary(StringComparer.Ordinal);
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

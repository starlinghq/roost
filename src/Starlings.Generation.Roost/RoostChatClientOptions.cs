using Starlings.Roost.Engine.Admission;
using Starlings.Roost.Generation;
using Starlings.Roost.Native;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost;

/// <summary>Where a worker samples, which decides the sampling options a client accepts (R-042).</summary>
public enum RoostSampler
{
    /// <summary>
    /// Greedy decoding only: the CUDA eager worker, which refuses sampling, penalty and logprob plans until the GPU sampler of
    /// R-042 exists. The default.
    /// </summary>
    GreedyOnly = 0,

    /// <summary>
    /// The worker runs the CPU sampler (<c>HostSampler</c> over <c>CpuOps.Sample</c>) and accepts the whole C2 sampling table:
    /// temperature, top-p, top-k, min-p, seeds, penalties and logprobs. Only a CPU worker or an executor built on that sampler
    /// may claim it; the packaged CUDA <c>roost-worker</c> may not (<see cref="RoostChatClient.StartAsync(RoostChatClientOptions, CancellationToken)"/>
    /// refuses the combination).
    /// </summary>
    Cpu = 1,
}

/// <summary>How the worker is started and reached (contract C8).</summary>
public sealed record RoostWorkerOptions
{
    /// <summary>
    /// The transport. <see cref="TransportKind.Process"/> (the default, decision 21) runs the worker in its own process, so a
    /// sticky CUDA fault cannot take the application down. <see cref="TransportKind.InProcess"/> is for tests and development
    /// only.
    /// </summary>
    public TransportKind Transport { get; init; } = WorkerTransports.Default;

    /// <summary>
    /// Starts the packaged <c>roost-worker</c> (R-062) for <see cref="TransportKind.Process"/>: the weights and, optionally,
    /// where the executable, kernels and manifest are. The command passes the profile's capacities to the worker.
    /// </summary>
    public RoostWorkerLaunch? Launch { get; init; }

    /// <summary>
    /// A custom command that starts the worker process for <see cref="TransportKind.Process"/>, used instead of
    /// <see cref="Launch"/> (for example a test executable re-run as the worker).
    /// </summary>
    public WorkerCommand? Command { get; init; }

    /// <summary>Makes the step executor for <see cref="TransportKind.InProcess"/>; required for it.</summary>
    public Func<IStepExecutor>? InProcessExecutor { get; init; }

    /// <summary>
    /// Where the worker samples. <see cref="RoostSampler.GreedyOnly"/> (the default, and the only value the packaged CUDA
    /// worker takes) refuses every sampling option that is not greedy-equivalent; <see cref="RoostSampler.Cpu"/> accepts them,
    /// for a worker that runs the CPU sampler (an in-process executor or a custom <see cref="Command"/>). The handshake does
    /// not carry this, so the setting must match the worker: a worker without the CPU sampler refuses a sampling plan and the
    /// request fails with <c>DeviceFault</c>.
    /// </summary>
    public RoostSampler Sampler { get; init; } = RoostSampler.GreedyOnly;

    /// <summary>
    /// Async scheduling (R-032): the driver builds and publishes step N+1 while step N runs, and a decode row whose input is the
    /// token step N is sampling carries <c>ROOST_SEQ_FLAG_SAMPLED_INPUT</c> for the worker to patch. Stop ids are checked one
    /// step late and the extra token is never emitted. False, the default, keeps one step in flight. Only for a worker that
    /// patches inputs (an in-process executor or a custom <see cref="Command"/> whose worker reports
    /// <c>PlanLimits.SampledInputs</c>): the packaged CUDA worker refuses such plans until its device-side patching exists
    /// (stream F), and the request fails with <c>DeviceFault</c>.
    /// </summary>
    public bool AsyncScheduling { get; init; }

    /// <summary>CUDA device ordinal the worker opens.</summary>
    public int DeviceOrdinal { get; init; }

    /// <summary>Per-step deadline; a step that takes longer is a hang or a stall (C8).</summary>
    public TimeSpan StepDeadline { get; init; } = TimeSpan.FromSeconds(60);

    /// <summary>Heartbeat period of the worker.</summary>
    public TimeSpan HeartbeatPeriod { get; init; } = TimeSpan.FromMilliseconds(1);

    /// <summary>
    /// How long the driver waits for the worker to answer a sleep, wake or weight update (R-100) before it kills the worker. A
    /// wake from <see cref="RoostSleepLevel.KvCacheAndWeights"/> and an update stream a whole weights file, so the default is
    /// generous: 10 minutes.
    /// </summary>
    public TimeSpan ControlTimeout { get; init; } = TimeSpan.FromMinutes(10);

    /// <summary>Host-side transport settings (startup and shutdown timeouts, segment directory); defaults when null.</summary>
    public WorkerTransportOptions? TransportOptions { get; init; }

    /// <summary>
    /// Where the driver reads the device's free memory before it queues a request (R-044, co-tenant admission), or null to
    /// decide from the worker's pools alone (the default). Read on the driver thread with
    /// <see cref="IDeviceMemorySource.TryRead"/>, so it must not block: a <see cref="PolledDeviceMemorySource"/> over an NVML
    /// reader is the intended source. The client does not dispose it. An unreadable source leaves the decision to the pools.
    /// </summary>
    /// <remarks>
    /// No NVML source is the default yet for <see cref="TransportKind.Process"/>: the NVML reader is worker-side code, which
    /// this host package may not reference (decision 21), so how a free-memory reading reaches the host (an NVML reader the
    /// host may load, or the worker reporting it over C8) is a stream A decision.
    /// </remarks>
    public IDeviceMemorySource? DeviceMemory { get; init; }

    /// <summary>
    /// Free device bytes below which a new request is refused with <c>InsufficientDeviceMemory</c> instead of being queued,
    /// because a co-tenant has taken memory the worker may still need. Zero (the default) disables the check; it needs
    /// <see cref="DeviceMemory"/>. Requests already running are never aborted by it.
    /// </summary>
    public long MinFreeDeviceBytes { get; init; }
}

/// <summary>Settings of a <see cref="RoostChatClient"/>.</summary>
public sealed record RoostChatClientOptions
{
    /// <summary>The model the worker serves.</summary>
    public required RoostModelProfile Model { get; init; }

    /// <summary>How the worker is started.</summary>
    public RoostWorkerOptions Worker { get; init; } = new();

    /// <summary>
    /// Makes the matcher for a request's stop strings, matched on detokenized text in the host process. The default is
    /// <see cref="StarlingStopMatcherFactory.Instance"/>, over Starling's public <c>StopSequenceMatcher</c> (starling#124).
    /// </summary>
    public IStopMatcherFactory StopMatcherFactory { get; init; } = StarlingStopMatcherFactory.Instance;

    /// <summary>Output tokens when <c>ChatOptions.MaxOutputTokens</c> is not set (always capped by the context left).</summary>
    public int DefaultMaxOutputTokens { get; init; } = 512;

    /// <summary>
    /// The grammar engine for <c>ChatOptions.ResponseFormat</c> (R-051): a <see cref="GrammarTokenizer"/> over the model's
    /// <c>tokenizer.json</c> and stop ids, from the <c>roost-grammar</c> library (<see cref="RoostGrammarLibrary"/>). A JSON or
    /// JSON-schema response format is compiled with it, so an invalid or unsupported schema is refused as an
    /// <c>InvalidOption</c> naming <c>response_format</c>. Null (the default) refuses every JSON response format. The client does
    /// not dispose it.
    /// </summary>
    /// <remarks>
    /// A format that compiles is still refused: the worker's sampler takes no token masks until contract C2 carries them to it,
    /// so the output could not be constrained. The masks and the CPU sampler path that applies them are built and tested in
    /// <c>Starlings.Roost.Native</c> (<see cref="GrammarSlots"/>, <see cref="GrammarSampler"/>).
    /// </remarks>
    public GrammarTokenizer? Grammar { get; init; }
}

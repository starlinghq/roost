using System.Runtime.CompilerServices;
using Starlings.Generation.Roost.Driver;
using Starlings.Roost.Generation;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Transport;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.WorkerHost;
using Starlings.Tokenizers;

namespace Starlings.Generation.Roost.Tests.Support;

/// <summary>
/// The test model: the tiny byte-level tokenizer (264 ids; <c>&lt;|im_end|&gt;</c> is 263), Starling's Qwen3.5 template, and
/// roost-tiny-hybrid's two state groups (paged attention, block 16, and one recurrent group), with an eager worker sized
/// like a small M0 worker: 16 attention blocks (a 256-token context), one recurrent slot, 32 tokens and one sequence per step.
/// </summary>
internal static class TestRig
{
    public const int StopId = 263;
    public const int AttentionBlocks = 16;
    public const int MaxTokensPerStep = 32;
    public const string ModelId = "roost-tiny-hybrid-test";
    public const string WorkerFlag = "--roost-chat-test-worker";
    public const string ScriptPrefix = "--script=";
    public const string TinyWeightsPrefix = "--tiny-weights=";

    private static readonly Lazy<Tokenizer> s_tokenizer = new(() =>
        Tokenizer.FromFile(Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny-bytelevel-bpe.tokenizer.json")));

    public static Tokenizer Tokenizer => s_tokenizer.Value;

    public static RoostModelProfile Profile() => new()
    {
        ModelId = ModelId,
        Tokenizer = Tokenizer,
        ChatTemplate = Qwen35RoostChatTemplate.Instance,
        StopTokenIds = [StopId],
        StateGroups = RoostTinyHybrid.CreateStateGroups(),
        AttentionBlocks = AttentionBlocks,
        RecurrentSlots = 1,
        MaxTokensPerStep = MaxTokensPerStep,
        MaxSequencesPerStep = 1,
        ManifestDigest = ScriptedExecutor.ManifestDigest,
        WeightsDigest = ScriptedExecutor.WeightsDigest,
    };

    /// <summary>The eager worker's limits for <see cref="Profile"/> (what <c>EagerWorker.Limits</c> would be for the same options).</summary>
    public static PlanLimits Limits() => new(
        MaxTokens: MaxTokensPerStep,
        MaxSequences: 1,
        MaxKvBlocks: AttentionBlocks,
        MaxSelected: 1,
        Vocab: Tokenizer.GetVocabularySize(true),
        BlockSize: 16,
        AttentionBlocks: AttentionBlocks,
        RecurrentSlots: 1,
        AttentionGroup: 0,
        GroupCount: 2);

    public static int[] Encode(string text) => [.. Tokenizer.Encode(text, false).Ids];

    /// <summary>A client over the in-process transport; <paramref name="executors"/> receives every executor made (one per worker start).</summary>
    public static Task<RoostChatClient> InProcessAsync(
        Script script,
        List<ScriptedExecutor>? executors = null,
        Func<RoostChatClientOptions, RoostChatClientOptions>? configure = null,
        Func<IDriverTransport, IDriverTransport>? decorate = null,
        PlanLimits? limits = null)
    {
        var made = 0;
        var options = new RoostChatClientOptions
        {
            Model = Profile(),
            Worker = new RoostWorkerOptions
            {
                Transport = TransportKind.InProcess,
                InProcessExecutor = () =>
                {
                    // An injected fault happens in the first worker only; a restarted worker runs clean.
                    var first = Interlocked.Increment(ref made) == 1;
                    var e = new ScriptedExecutor(first ? script : script with { FaultAtStep = 0, ThrowAtStep = 0 }, limits ?? Limits(), StopId);
                    if (executors is not null)
                    {
                        lock (executors)
                        {
                            executors.Add(e);
                        }
                    }

                    return e;
                },
                StepDeadline = TimeSpan.FromSeconds(10),
                TransportOptions = new WorkerTransportOptions { StartupTimeout = TimeSpan.FromSeconds(30) },
            },
        };
        return RoostChatClient.StartAsync(configure?.Invoke(options) ?? options, decorate, TestContext.Current.CancellationToken);
    }

    /// <summary>A client over the in-process transport whose executor runs the CPU sampler, with <see cref="RoostSampler.Cpu"/>.</summary>
    public static Task<RoostChatClient> SamplingAsync(Script script, List<ScriptedExecutor>? executors = null) =>
        InProcessAsync(script with { Sample = true }, executors, o => o with { Worker = o.Worker with { Sampler = RoostSampler.Cpu } });

    /// <summary>A client over the default (process) transport: this test executable re-run as the worker.</summary>
    public static Task<RoostChatClient> ProcessAsync(Script script, TimeSpan? stepDeadline = null) => RoostChatClient.StartAsync(new RoostChatClientOptions
    {
        Model = Profile(),
        Worker = new RoostWorkerOptions
        {
            Sampler = script.Sample ? RoostSampler.Cpu : RoostSampler.GreedyOnly,
            Command = WorkerCommand.ForCurrentProcess(WorkerFlag, ScriptPrefix + script.Encode()),
            StepDeadline = stepDeadline ?? TimeSpan.FromSeconds(10),
            TransportOptions = new WorkerTransportOptions { StartupTimeout = TimeSpan.FromSeconds(60) },
        },
    });

    /// <summary>
    /// A client whose worker runs roost-tiny-hybrid on <see cref="Starlings.Roost.Ops.CpuOps"/> with the weights at
    /// <paramref name="weightsPath"/> (<see cref="TinyCpuExecutor"/>), over the in-process transport or this executable re-run as
    /// the worker process. The profile's weights digest is the file's locked one.
    /// </summary>
    public static Task<RoostChatClient> TinyAsync(TransportKind transport, string weightsPath, Starlings.Roost.Worker.Weights.LockedFile locked, List<TinyCpuExecutor>? executors = null)
    {
        var worker = new RoostWorkerOptions
        {
            StepDeadline = TimeSpan.FromSeconds(30),
            TransportOptions = new WorkerTransportOptions { StartupTimeout = TimeSpan.FromSeconds(60) },
        };
        worker = transport == TransportKind.InProcess
            ? worker with
            {
                Transport = TransportKind.InProcess,
                InProcessExecutor = () =>
                {
                    var e = new TinyCpuExecutor(weightsPath, locked, Limits());
                    if (executors is not null)
                    {
                        lock (executors)
                        {
                            executors.Add(e);
                        }
                    }

                    return e;
                },
            }
            : worker with
            {
                Command = WorkerCommand.ForCurrentProcess(WorkerFlag, TinyWeightsPrefix + string.Join('|', Path.GetFullPath(weightsPath), locked.Size.ToString(System.Globalization.CultureInfo.InvariantCulture), locked.Sha256)),
            };
        return RoostChatClient.StartAsync(
            new RoostChatClientOptions
            {
                Model = Profile() with { WeightsDigest = [.. Convert.FromHexString(locked.Sha256)] },
                Worker = worker,
            },
            TestContext.Current.CancellationToken);
    }

    /// <summary>Turns this executable into a scripted worker process when the host starts it with <see cref="WorkerFlag"/>.</summary>
    [ModuleInitializer]
    internal static void EnterWorker()
    {
        var args = Environment.GetCommandLineArgs();
        if (Array.IndexOf(args, WorkerFlag) < 0)
        {
            return;
        }

        Environment.Exit(WorkerEntry.Run(args, a =>
        {
            if (a.FirstOrDefault(x => x.StartsWith(TinyWeightsPrefix, StringComparison.Ordinal)) is { } tiny)
            {
                var p = tiny[TinyWeightsPrefix.Length..].Split('|');
                var locked = new Starlings.Roost.Worker.Weights.LockedFile(Path.GetFileName(p[0]), long.Parse(p[1], System.Globalization.CultureInfo.InvariantCulture), p[2]);
                return new TinyCpuExecutor(p[0], locked, Limits());
            }

            var script = Script.Decode(a.First(x => x.StartsWith(ScriptPrefix, StringComparison.Ordinal))[ScriptPrefix.Length..]);
            return new ScriptedExecutor(script, Limits(), StopId);
        }));
    }
}

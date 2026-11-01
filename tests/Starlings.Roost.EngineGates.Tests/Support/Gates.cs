using System.Collections.Concurrent;
using System.Collections.Immutable;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Engine.Caching;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Generation;
using Starlings.Roost.Requests;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;
using static Starlings.Roost.EngineGates.Tests.Support.EngineRunner;

namespace Starlings.Roost.EngineGates.Tests.Support;

/// <summary>Tokens and the logits that selected them, for one request of one run.</summary>
/// <param name="Tokens">Generated tokens.</param>
/// <param name="Logits">One float32 logits row per token.</param>
internal sealed record CaseOutput(ImmutableArray<int> Tokens, ImmutableArray<float[]> Logits);

/// <summary>A variant run of one case against its baseline.</summary>
/// <param name="TokensEqual">Same generated tokens.</param>
/// <param name="Steps">Steps compared (the shorter run's length).</param>
/// <param name="Step0MaxAbsLogprob">Largest |log-softmax difference| over the vocabulary at the first generated step.</param>
/// <param name="MaxAbsLogprob">The same over every compared step.</param>
/// <param name="BitIdentical">Every compared logits row is the same, bit for bit.</param>
internal sealed record Comparison(bool TokensEqual, int Steps, double Step0MaxAbsLogprob, double MaxAbsLogprob, bool BitIdentical)
{
    public static Comparison Of(CaseOutput baseline, CaseOutput variant)
    {
        int steps = Math.Min(baseline.Logits.Length, variant.Logits.Length);
        double step0 = 0, worst = 0;
        bool bits = true;
        for (int i = 0; i < steps; i++)
        {
            double d = MaxAbs(LogSoftmax(baseline.Logits[i]), LogSoftmax(variant.Logits[i]));
            step0 = i == 0 ? d : step0;
            worst = Math.Max(worst, d);
            bits &= baseline.Logits[i].AsSpan().SequenceEqual(variant.Logits[i]);
        }

        return new Comparison(baseline.Tokens.SequenceEqual(variant.Tokens), steps, step0, worst, bits);
    }

    /// <summary>Log-softmax in double precision.</summary>
    /// <param name="logits">One row.</param>
    /// <returns>The log-probabilities.</returns>
    public static double[] LogSoftmax(float[] logits)
    {
        double top = logits.Max();
        double sum = logits.Sum(v => Math.Exp(v - top));
        double log = Math.Log(sum);
        return [.. logits.Select(v => v - top - log)];
    }

    private static double MaxAbs(double[] a, double[] b)
    {
        double worst = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = Math.Abs(a[i] - b[i]);
            worst = double.IsNaN(d) ? double.PositiveInfinity : Math.Max(worst, d);
        }

        return worst;
    }
}

/// <summary>The cached-vs-uncached measurement of format v3 (<c>sections.cached_vs_uncached</c>).</summary>
/// <param name="Case">The case.</param>
/// <param name="Steps">Decode steps compared.</param>
/// <param name="MaxAbsLogprob">Largest |log-softmax difference| between the cached step and a fresh prefill of the same prefix.</param>
/// <param name="TokensEqual">Every fresh prefill selects the token the cached run generated.</param>
/// <param name="HitTokens">Prompt tokens the prefix cache served.</param>
/// <param name="Restores">Snapshot restores in the cached run (hybrid).</param>
internal sealed record CachedVsUncached(string Case, int Steps, double MaxAbsLogprob, bool TokensEqual, long HitTokens, int Restores);

/// <summary>A run of every case together, against each alone.</summary>
/// <param name="Results">Per case, the comparison with the baseline.</param>
/// <param name="MaxRowsPerStep">The most rows one step carried.</param>
/// <param name="Preemptions">Preemptions in the run.</param>
internal sealed record TogetherRun(ImmutableDictionary<string, Comparison> Results, int MaxRowsPerStep, int Preemptions);

/// <summary>Streaming against non-streaming completions of one case through the chat client.</summary>
/// <param name="StreamTextEqualsResult">Concatenated streamed text equals the non-streaming response text.</param>
/// <param name="StreamIdsEqualResult">The ids the worker emitted for the streamed request equal those for the non-streaming one.</param>
/// <param name="StreamTextEqualsDecode">Concatenated streamed text equals the tokenizer's decode of the emitted ids.</param>
/// <param name="StreamIds">The ids emitted for the streamed request.</param>
/// <param name="PromptTokens">Usage: prompt tokens reported.</param>
/// <param name="CompletionTokens">Usage: completion tokens reported.</param>
/// <param name="FinishReason">The streamed finish reason.</param>
/// <param name="Rejections">Plans the worker's reader refused.</param>
internal sealed record StreamingResult(
    bool StreamTextEqualsResult,
    bool StreamIdsEqualResult,
    bool StreamTextEqualsDecode,
    ImmutableArray<int> StreamIds,
    long PromptTokens,
    long CompletionTokens,
    ChatFinishReason? FinishReason,
    int Rejections);

/// <summary>
/// The engine gates of R-039, computed on CpuOps. Each gate runs a variant of the engine and compares it with the baseline:
/// every case alone, its prompt prefilled in one call, no prefix cache, no preemption (<see cref="EngineSetup.OneCall"/>).
/// </summary>
internal static class Gates
{
    /// <summary>Prefill chunk sizes of the chunked gate (each case runs every size below its prompt length).</summary>
    public static readonly ImmutableArray<int> ChunkSizes = [1, 5, 16];

    private static readonly ConcurrentDictionary<TinyProfile, Lazy<ImmutableDictionary<string, CaseOutput>>> s_baselines = new();

    /// <summary>The baseline output of every case.</summary>
    /// <param name="profile">The model.</param>
    /// <returns>Outputs by case id.</returns>
    public static ImmutableDictionary<string, CaseOutput> Baseline(TinyProfile profile) =>
        s_baselines.GetOrAdd(profile, p => new Lazy<ImmutableDictionary<string, CaseOutput>>(() =>
            GateCases.All.ToImmutableDictionary(c => c.Id, c => Alone(p, EngineSetup.OneCall, c.PromptIds, c.MaxTokens)))).Value;

    /// <summary>Chunked prefill against one call: the same case with prefill chunks capped at <paramref name="chunk"/> tokens.</summary>
    /// <param name="profile">The model.</param>
    /// <param name="gateCase">The case.</param>
    /// <param name="chunk">Most tokens per prefill chunk.</param>
    /// <returns>The comparison and the chunk sizes the planner used.</returns>
    public static (Comparison Result, ImmutableArray<int> Chunks) Chunked(TinyProfile profile, GateCase gateCase, int chunk)
    {
        var setup = EngineSetup.OneCall with { Planner = EngineSetup.OneCall.Planner with { MaxChunkTokens = chunk } };
        var run = EngineRunner.Run(profile, setup, [[Req(1, gateCase.PromptIds, gateCase.MaxTokens)]]);
        return (Comparison.Of(Baseline(profile)[gateCase.Id], Output(run, 1)), [.. run.ChunkSizes]);
    }

    /// <summary>
    /// Cached against uncached, as format v3 defines it: <see cref="GateCases.Cached"/> runs after <see cref="GateCases.Primer"/>
    /// with the prefix cache on, so its prompt prefix comes from cached blocks (and, on the hybrid model, a restored recurrent
    /// snapshot) and every decode step reads the KV cache and the recurrent state. Each step's logits are compared with a fresh,
    /// uncached one-call prefill of the prompt plus the tokens generated before that step.
    /// </summary>
    /// <param name="profile">The model.</param>
    /// <returns>The measurement.</returns>
    public static CachedVsUncached CachedAgainstFreshPrefill(TinyProfile profile)
    {
        var target = GateCases.Cached;
        var run = EngineRunner.Run(profile, CacheOn(), [[Req(1, GateCases.Primer.PromptIds, GateCases.Primer.MaxTokens)], [Req(2, target.PromptIds, target.MaxTokens)]]);
        var cached = Output(run, 2);
        double worst = 0;
        bool equal = true;
        for (int i = 0; i < cached.Tokens.Length; i++)
        {
            var fresh = Alone(profile, EngineSetup.OneCall, target.PromptIds.Concat(cached.Tokens.Take(i)), 1);
            equal &= fresh.Tokens[0] == cached.Tokens[i];
            worst = Math.Max(worst, Comparison.Of(fresh, new CaseOutput([cached.Tokens[i]], [cached.Logits[i]])).MaxAbsLogprob);
        }

        return new CachedVsUncached(target.Id, cached.Tokens.Length, worst, equal, run.Cache!.Value.HitTokens, run.SnapshotRestores);
    }

    /// <summary>Prefix cache on against off: every case after the primer, with the cache on, against its baseline.</summary>
    /// <param name="profile">The model.</param>
    /// <returns>The comparisons and the cache counters.</returns>
    public static (ImmutableDictionary<string, Comparison> Results, PrefixCacheStats Stats) PrefixCacheOnOff(TinyProfile profile)
    {
        var others = GateCases.All.Where(c => c != GateCases.Primer).ToList();
        ulong id = 2;
        var ids = others.ToDictionary(c => c.Id, _ => id++);
        var run = EngineRunner.Run(profile, CacheOn(), [[Req(1, GateCases.Primer.PromptIds, GateCases.Primer.MaxTokens)], [.. others.Select(c => Req(ids[c.Id], c.PromptIds, c.MaxTokens))]]);
        var baseline = Baseline(profile);
        var results = others.ToImmutableDictionary(c => c.Id, c => Comparison.Of(baseline[c.Id], Output(run, ids[c.Id])))
            .Add(GateCases.Primer.Id, Comparison.Of(baseline[GateCases.Primer.Id], Output(run, 1)));
        return (results, run.Cache!.Value);
    }

    /// <summary>
    /// Recurrent snapshot restore against no restore, both with the prefix cache on (so prefill chunks stop at the same snapshot
    /// boundaries): <see cref="GateCases.Cached"/> after <see cref="GateCases.Primer"/> (its state at the shared boundary is a
    /// restored snapshot) against the same case alone (the state is computed).
    /// </summary>
    /// <param name="profile">The model (the hybrid one; the dense model has no recurrent group).</param>
    /// <returns>The comparison and the restores in each run.</returns>
    public static (Comparison Result, int RestoresWith, int RestoresWithout) RecurrentRestore(TinyProfile profile)
    {
        var target = GateCases.Cached;
        var with = EngineRunner.Run(profile, CacheOn(), [[Req(1, GateCases.Primer.PromptIds, GateCases.Primer.MaxTokens)], [Req(2, target.PromptIds, target.MaxTokens)]]);
        var without = EngineRunner.Run(profile, CacheOn(), [[Req(2, target.PromptIds, target.MaxTokens)]]);
        return (Comparison.Of(Output(without, 2), Output(with, 2)), with.SnapshotRestores, without.SnapshotRestores);
    }

    /// <summary>Every case in one wave, batched by the planner, against each alone.</summary>
    /// <param name="profile">The model.</param>
    /// <param name="maxInFlight">Plans issued before the oldest result returns.</param>
    /// <returns>The run.</returns>
    public static TogetherRun Concurrent(TinyProfile profile, int maxInFlight) =>
        Together(profile, new EngineSetup(64, 8, 0, new PlannerOptions { MaxTokensPerStep = 64, MaxSequencesPerStep = 4 }, maxInFlight));

    /// <summary>
    /// Every case in one wave on an attention pool of 6 blocks (96 tokens), which holds the longest case alone (90 tokens
    /// with its output) but not all of them: the first three cases fill it, and when the first one's decode crosses into a
    /// new block the planner preempts the last resident one, which recomputes later. Against each case alone and
    /// uninterrupted.
    /// </summary>
    /// <param name="profile">The model.</param>
    /// <returns>The run.</returns>
    public static TogetherRun Preempted(TinyProfile profile) =>
        Together(profile, new EngineSetup(6, 4, 0, new PlannerOptions { MaxTokensPerStep = 64, MaxSequencesPerStep = 4 }));

    /// <summary>Streaming against non-streaming completions of every case through the chat client on the in-process transport.</summary>
    /// <param name="profile">The model.</param>
    /// <param name="cancellationToken">Cancels the runs.</param>
    /// <returns>Per case, the result.</returns>
    public static async Task<ImmutableDictionary<string, StreamingResult>> Streaming(TinyProfile profile, CancellationToken cancellationToken)
    {
        const int Blocks = 64, MaxTokens = 32;
        CpuTinyExecutor? executor = null;
        var limits = CpuTinyExecutor.LimitsFor(profile, Blocks, 1, MaxTokens, 1);
        var options = new RoostChatClientOptions
        {
            Model = new RoostModelProfile
            {
                ModelId = profile == TinyProfile.Dense ? "roost-tiny" : "roost-tiny-hybrid",
                Tokenizer = GateCases.Tokenizer,
                ChatTemplate = Qwen35RoostChatTemplate.Instance,
                StopTokenIds = [263],
                StateGroups = TinyCpuModel.GroupsOf(profile),
                AttentionBlocks = Blocks,
                RecurrentSlots = 1,
                MaxTokensPerStep = MaxTokens,
                MaxSequencesPerStep = 1,
                ManifestDigest = CpuTinyExecutor.ManifestDigest,
                WeightsDigest = TinyCpuModel.WeightsDigest(profile),
            },
            Worker = new RoostWorkerOptions
            {
                Transport = TransportKind.InProcess,
                InProcessExecutor = () => executor = new CpuTinyExecutor(profile, limits),
                StepDeadline = TimeSpan.FromSeconds(60),
                TransportOptions = new WorkerTransportOptions { StartupTimeout = TimeSpan.FromSeconds(60) },
            },
        };

        await using var client = await RoostChatClient.StartAsync(options, cancellationToken);
        var results = ImmutableDictionary.CreateBuilder<string, StreamingResult>();
        foreach (var c in GateCases.All)
        {
            var chat = new ChatOptions { MaxOutputTokens = c.MaxTokens };
            var before = executor!.Emitted.Keys.ToHashSet();
            var text = new System.Text.StringBuilder();
            UsageDetails? usage = null;
            ChatFinishReason? finish = null;
            await foreach (var u in client.GetStreamingCompletionAsync(c.Prompt, chat, cancellationToken))
            {
                text.Append(u.Text);
                finish = u.FinishReason ?? finish;
                usage = u.Contents.OfType<UsageContent>().FirstOrDefault()?.Details ?? usage;
            }

            var streamIds = NewIds(executor, before);
            before = executor.Emitted.Keys.ToHashSet();
            var response = await client.GetCompletionAsync(c.Prompt, chat, cancellationToken);
            var resultIds = NewIds(executor, before);
            string streamed = text.ToString();
            results[c.Id] = new StreamingResult(
                streamed == response.Text,
                streamIds.SequenceEqual(resultIds),
                streamed == GateCases.Tokenizer.Decode(streamIds.AsSpan(), true),
                streamIds,
                usage?.InputTokenCount ?? -1,
                usage?.OutputTokenCount ?? -1,
                finish,
                executor.Rejections.Count);
        }

        return results.ToImmutable();
    }

    private static ImmutableArray<int> NewIds(CpuTinyExecutor executor, HashSet<ulong> before)
    {
        var handle = Assert.Single(executor.Emitted.Keys, k => !before.Contains(k));
        return [.. executor.Emitted[handle]];
    }

    private static EngineSetup CacheOn() => new(64, 4, 8, new PlannerOptions { MaxTokensPerStep = 256, MaxSequencesPerStep = 4, PrefixCache = new PrefixCacheOptions() });

    private static TogetherRun Together(TinyProfile profile, EngineSetup setup)
    {
        var ids = GateCases.All.Select((c, i) => (c, Id: (ulong)(i + 1))).ToList();
        var run = EngineRunner.Run(profile, setup, [[.. ids.Select(x => Req(x.Id, x.c.PromptIds, x.c.MaxTokens))]]);
        var baseline = Baseline(profile);
        return new TogetherRun(ids.ToImmutableDictionary(x => x.c.Id, x => Comparison.Of(baseline[x.c.Id], Output(run, x.Id))), run.MaxRowsPerStep, run.Preemptions);
    }

    private static CaseOutput Alone(TinyProfile profile, EngineSetup setup, IEnumerable<int> prompt, int maxTokens) =>
        Output(EngineRunner.Run(profile, setup, [[Req(1, prompt, maxTokens)]]), 1);

    private static CaseOutput Output(EngineRun run, ulong id) => new([.. run.Tokens[id]], [.. run.Logits[id]]);
}

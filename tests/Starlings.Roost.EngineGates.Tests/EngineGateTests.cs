using Starlings.Roost.EngineGates.Tests.Support;

namespace Starlings.Roost.EngineGates.Tests;

/// <summary>
/// The engine gates of R-039 on CpuOps, for roost-tiny and roost-tiny-hybrid. Each gate runs the real planner, block manager
/// and prefix cache (or, for streaming, the real chat client and driver) against the generated tiny model and asserts that the
/// variant generates the baseline's tokens. Measured log-probability differences are printed as diagnostics and written to
/// the stage results (<see cref="StageResultsTests"/>); where the computation is the same by construction, the logits must be
/// bit-identical.
/// </summary>
public sealed class EngineGateTests
{
    public static TheoryData<TinyProfile> Profiles => [TinyProfile.Dense, TinyProfile.Hybrid];

    [Theory]
    [MemberData(nameof(Profiles))]
    public void ChunkedPrefillGeneratesTheOneCallTokens(TinyProfile profile)
    {
        int compared = 0;
        foreach (var c in GateCases.All)
        {
            foreach (int chunk in Gates.ChunkSizes.Where(k => k < c.PromptIds.Length))
            {
                var (result, chunks) = Gates.Chunked(profile, c, chunk);
                Report($"{profile} chunked {c.Id}/{chunk}", result);
                Assert.True(chunks.Max() <= chunk, $"{c.Id}: a prefill chunk of {chunks.Max()} tokens is over the cap of {chunk}");
                Assert.True(result.TokensEqual, $"{c.Id}/{chunk}: chunked prefill generated other tokens than one call");
                AssertBitIdenticalWhenDense(profile, result, $"{c.Id}/{chunk}");
                compared++;
            }
        }

        Assert.True(compared >= 10, $"only {compared} chunked runs");
    }

    [Theory]
    [MemberData(nameof(Profiles))]
    public void CachedDecodeMatchesAFreshPrefillOfTheSamePrefix(TinyProfile profile)
    {
        var r = Gates.CachedAgainstFreshPrefill(profile);
        TestContext.Current.SendDiagnosticMessage($"MEASURED {profile} cached_vs_uncached {r.Case}: steps {r.Steps}, max abs logprob {r.MaxAbsLogprob:E3}, hit tokens {r.HitTokens}, restores {r.Restores}");
        Assert.True(r.HitTokens >= GateCases.SharedPrefix.Length, $"the prefix cache served {r.HitTokens} tokens");
        Assert.True(r.TokensEqual, "a fresh prefill selects another token than the cached decode");
        Assert.Equal(GateCases.Cached.MaxTokens, r.Steps);
        Assert.True(double.IsFinite(r.MaxAbsLogprob));
        Assert.True(profile == TinyProfile.Hybrid || r.MaxAbsLogprob == 0, $"float32 cached decode differs from a fresh prefill by {r.MaxAbsLogprob:E3}");
        if (profile == TinyProfile.Hybrid)
        {
            Assert.True(r.Restores > 0, "the hybrid cached run restored no snapshot");
        }
    }

    [Theory]
    [MemberData(nameof(Profiles))]
    public void PrefixCacheOnGeneratesTheUncachedTokens(TinyProfile profile)
    {
        var (results, stats) = Gates.PrefixCacheOnOff(profile);
        Assert.True(stats.HitTokens > 0, "the prefix cache never hit");
        foreach (var (id, result) in results)
        {
            Report($"{profile} prefix-cache {id}", result);
            Assert.True(result.TokensEqual, $"{id}: the cached run generated other tokens");
            AssertBitIdenticalWhenDense(profile, result, id);
        }
    }

    [Fact]
    public void RecurrentSnapshotRestoreIsBitIdenticalToNoRestore()
    {
        var (result, with, without) = Gates.RecurrentRestore(TinyProfile.Hybrid);
        Report("Hybrid recurrent-restore", result);
        Assert.True(with > 0, "the run after the primer restored no snapshot");
        Assert.Equal(0, without);
        Assert.True(result.TokensEqual, "the restored run generated other tokens");
        Assert.True(result.BitIdentical, $"restored and computed state give different logits (max abs logprob {result.MaxAbsLogprob:E3})");
    }

    [Theory]
    [MemberData(nameof(Profiles))]
    public void BatchedRequestsGenerateTheSingleRequestTokens(TinyProfile profile)
    {
        foreach (int inFlight in new[] { 1, 2 })
        {
            var run = Gates.Concurrent(profile, inFlight);
            Assert.True(run.MaxRowsPerStep > 1, "no step carried more than one request");
            Assert.Equal(0, run.Preemptions);
            foreach (var (id, result) in run.Results)
            {
                Report($"{profile} concurrency(in flight {inFlight}) {id}", result);
                Assert.True(result.TokensEqual, $"{id}: the batched run generated other tokens than the single run");

                // Every op of both models computes a row from that row alone, so batching changes no bit (both models).
                Assert.True(result.BitIdentical, $"{id}: batching changed the logits (max abs logprob {result.MaxAbsLogprob:E3})");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Profiles))]
    public void PreemptedRequestsGenerateTheUninterruptedTokens(TinyProfile profile)
    {
        var run = Gates.Preempted(profile);
        TestContext.Current.SendDiagnosticMessage($"MEASURED {profile} preemptions {run.Preemptions}");
        Assert.True(run.Preemptions > 0, "the small pool forced no preemption");
        foreach (var (id, result) in run.Results)
        {
            Report($"{profile} preemption {id}", result);
            Assert.True(result.TokensEqual, $"{id}: the preempted run generated other tokens than the uninterrupted run");
            AssertBitIdenticalWhenDense(profile, result, id);
        }
    }

    [Theory]
    [MemberData(nameof(Profiles))]
    public async Task StreamingReassemblesTheNonStreamingResult(TinyProfile profile)
    {
        var results = await Gates.Streaming(profile, TestContext.Current.CancellationToken);
        var baseline = Gates.Baseline(profile);
        foreach (var c in GateCases.All)
        {
            var r = results[c.Id];
            Assert.Equal(0, r.Rejections);
            Assert.True(r.StreamTextEqualsResult, $"{c.Id}: streamed text differs from the non-streaming text");
            Assert.True(r.StreamIdsEqualResult, $"{c.Id}: streamed ids differ from the non-streaming ids");
            Assert.True(r.StreamTextEqualsDecode, $"{c.Id}: streamed text differs from the decode of the ids");
            Assert.Equal(baseline[c.Id].Tokens, r.StreamIds);
            Assert.Equal(c.PromptIds.Length, r.PromptTokens);
            Assert.Equal(c.MaxTokens, r.CompletionTokens);
            Assert.Equal(Microsoft.Extensions.AI.ChatFinishReason.Length, r.FinishReason);
        }
    }

    /// <summary>
    /// roost-tiny is float32 and every CpuOps op it uses computes a token row from that row and the cache alone, so how a
    /// prompt is chunked, cached, batched or recomputed changes no bit; that is asserted. roost-tiny-hybrid rounds activations
    /// to bfloat16 and its logits do move with chunking (prefill in one-token chunks of the long case measured a step-0
    /// log-probability difference of 0.15), so outside batching only its tokens are asserted; the measured differences are
    /// in the diagnostics and the stage results.
    /// </summary>
    private static void AssertBitIdenticalWhenDense(TinyProfile profile, Comparison r, string what) =>
        Assert.True(profile == TinyProfile.Hybrid || r.BitIdentical, $"{what}: roost-tiny logits changed (max abs logprob {r.MaxAbsLogprob:E3})");

    private static void Report(string what, Comparison r) =>
        TestContext.Current.SendDiagnosticMessage($"MEASURED {what}: tokens equal {r.TokensEqual}, step0 max abs logprob {r.Step0MaxAbsLogprob:E3}, all steps {r.MaxAbsLogprob:E3}, bit-identical {r.BitIdentical}");
}

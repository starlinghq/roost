using Starlings.Roost.Engine.Caching;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Requests;
using static Starlings.Roost.Engine.Tests.Caching.CacheRun;

namespace Starlings.Roost.Engine.Tests.Caching;

/// <summary>Cached and uncached runs of a Qwen3-shaped model on CpuOps: same tokens and the same logits, bit for bit.</summary>
public sealed class CpuOpsPrefixCacheTests
{
    private static readonly PlannerOptions Off = new() { MaxTokensPerStep = 64 };
    private static readonly PlannerOptions On = Off with { PrefixCache = new PrefixCacheOptions() };

    private static CacheRunResult RunOnCpu(int blocks, PlannerOptions options, Request[][] waves, int inFlight, bool corrupt = false) =>
        Run(() => Attention(blocks), options, waves, inFlight, b => new CpuOpsTinyWorker(b) { CorruptFirstCachedBlock = corrupt });

    private static void AssertBitIdentical(CacheRunResult a, CacheRunResult b)
    {
        Assert.Equal(a.Outputs.Keys.Order(), b.Outputs.Keys.Order());
        var x = ((CpuOpsTinyWorker)a.Worker).Logits;
        var y = ((CpuOpsTinyWorker)b.Worker).Logits;
        foreach (var (id, tokens) in a.Outputs)
        {
            Assert.Equal(tokens, b.Outputs[id]);
            Assert.Equal(x[id].Count, y[id].Count);
            for (var i = 0; i < x[id].Count; i++)
            {
                Assert.True(x[id][i].AsSpan().SequenceEqual(y[id][i]), $"request {id} sample {i}: logits differ");
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void SharedPrefixesGiveBitIdenticalLogits(int inFlight)
    {
        Request[][] waves =
        [
            [Req(1, Seq(0, 30))],
            [Req(2, Seq(0, 30).Concat(Seq(500, 5))), Req(3, Seq(0, 17).Concat(Seq(700, 9))), Req(4, Seq(2000, 13))],
            [Req(5, Seq(0, 30).Concat(Seq(500, 5)).Concat(Seq(900, 6)))],
        ];
        var plain = RunOnCpu(200, Off, waves, inFlight);
        var cached = RunOnCpu(200, On, waves, inFlight);
        AssertBitIdentical(plain, cached);
        Assert.True(cached.Stats!.Value.HitTokens >= 4 * 16, $"hit tokens {cached.Stats.Value.HitTokens}");
        Assert.True(cached.ComputedTokens < plain.ComputedTokens);
    }

    [Fact]
    public void PreemptionAndEvictionKeepLogitsBitIdentical()
    {
        long preemptions = 0, evictions = 0;
        for (var seed = 1; seed <= 6; seed++)
        {
            var rng = new Random(seed);
            var prefixes = Enumerable.Range(0, 2).Select(i => Seq(1000 * (i + 1), 8 + (rng.Next(5) * 4))).ToArray();
            var waves = Enumerable.Range(0, 3).Select(w => Enumerable.Range(0, 4).Select(i =>
                Req((ulong)((w * 10) + i + 1), prefixes[rng.Next(2)].Concat(Seq(rng.Next(3000), 1 + rng.Next(12))), maxTokens: 1 + rng.Next(6))).ToArray()).ToArray();
            var plain = RunOnCpu(24, Off, waves, 2);
            var cached = RunOnCpu(24, On, waves, 2);
            AssertBitIdentical(plain, cached);
            preemptions += cached.Preemptions;
            evictions += cached.Stats!.Value.NodesEvicted;
        }

        Assert.True(preemptions > 0 && evictions > 0, $"preemptions {preemptions} evictions {evictions}");
    }

    /// <summary>
    /// The bit-identity check has teeth: with one cached block's contents perturbed, the cached run's logits differ from the
    /// uncached run's (so the attention really reads the cached keys and values), while the same run uncorrupted matches.
    /// </summary>
    [Fact]
    public void CorruptedCachedBlockMakesLogitsDiffer()
    {
        Request[][] waves =
        [
            [Req(1, Seq(0, 30))],
            [Req(2, Seq(0, 30).Concat(Seq(500, 5)))],
        ];
        var plain = RunOnCpu(200, Off, waves, 1);
        AssertBitIdentical(plain, RunOnCpu(200, On, waves, 1));

        var corrupted = RunOnCpu(200, On, waves, 1, corrupt: true);
        Assert.True(corrupted.Stats!.Value.HitTokens > 0, "the cache never hit, so nothing was corrupted");
        Assert.ThrowsAny<Exception>(() => AssertBitIdentical(plain, corrupted));
        var x = ((CpuOpsTinyWorker)plain.Worker).Logits[2];
        var y = ((CpuOpsTinyWorker)corrupted.Worker).Logits[2];
        Assert.False(x[0].AsSpan().SequenceEqual(y[0]), "request 2's logits ignore the cached block");
    }
}

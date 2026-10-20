using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Caching;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using static Starlings.Roost.Engine.Tests.Caching.CacheRun;

namespace Starlings.Roost.Engine.Tests.Caching;

public sealed class PrefixCacheTests
{
    private static readonly PlannerOptions Off = new() { MaxTokensPerStep = 64 };
    private static readonly PlannerOptions On = Off with { PrefixCache = new PrefixCacheOptions() };

    private static void AssertSameOutputs(CacheRunResult a, CacheRunResult b)
    {
        Assert.Equal(a.Outputs.Keys.Order(), b.Outputs.Keys.Order());
        foreach (var (k, v) in a.Outputs)
        {
            Assert.Equal(v, b.Outputs[k]);
        }
    }

    private static Request[][] SharedPrefixWaves() =>
    [
        [Req(1, Seq(0, 30))],
        [Req(2, Seq(0, 30).Concat(Seq(500, 5))), Req(3, Seq(0, 17).Concat(Seq(700, 9))), Req(4, Seq(2000, 13))],
        [Req(5, Seq(0, 30).Concat(Seq(500, 5)).Concat(Seq(900, 6)))],
    ];

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void AttentionOnlyCachedMatchesUncachedAndSkipsWork(int inFlight)
    {
        var plain = Run(() => Attention(200), Off, SharedPrefixWaves(), inFlight);
        var cached = Run(() => Attention(200), On, SharedPrefixWaves(), inFlight);
        AssertSameOutputs(plain, cached);
        Assert.True(cached.Stats!.Value.Hits >= 3, $"hits {cached.Stats.Value.Hits}");
        Assert.True(cached.ComputedTokens < plain.ComputedTokens, $"{cached.ComputedTokens} vs {plain.ComputedTokens}");
        Assert.Equal(0, cached.Restores); // no recurrent group
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(3, 1)]
    [InlineData(1, 2)]
    public void HybridCachedMatchesUncachedWithSnapshotsAtBlockBoundaries(int inFlight, int stride)
    {
        var options = Off with { PrefixCache = new PrefixCacheOptions { SnapshotStrideBlocks = stride } };
        var plain = Run(() => Hybrid(200, 8, 0), Off, SharedPrefixWaves(), inFlight);
        var cached = Run(() => Hybrid(200, 8, 40), options, SharedPrefixWaves(), inFlight);
        AssertSameOutputs(plain, cached);
        Assert.True(cached.Restores >= 3, $"restores {cached.Restores}");
        Assert.True(cached.SnapshotWrites > 0);
        Assert.True(cached.ComputedTokens < plain.ComputedTokens);
    }

    [Fact]
    public void HybridPrefillIsCutAtEveryBlockBoundary()
    {
        // Request 1 has 30 prompt tokens: with block size 4 and stride 1, chunks of 4 up to 28, then the last 2.
        var blocks = Hybrid(100, 4, 20);
        var planner = new StepPlanner(blocks, On);
        planner.Add(Req(1, Seq(0, 30), maxTokens: 1));
        var counts = new List<uint>();
        PlannedStep? plan;
        while ((plan = planner.Plan()) is not null)
        {
            counts.Add(plan.Sequences[0].TokenCount);
            planner.ApplyResult(new StepOutcome(plan.StepId, [new(1, (plan.Sequences[0].Flags & SequenceOptions.Sample) != 0 ? [5] : [], FinishReason.None)]));
        }

        Assert.Equal([4u, 4u, 4u, 4u, 4u, 4u, 4u, 2u], counts);
    }

    [Fact]
    public void PreemptionStormWithEvictionKeepsOutputsExact()
    {
        long hits = 0, evictions = 0, preemptions = 0, dropped = 0, restores = 0;
        for (var seed = 1; seed <= 24; seed++)
        {
            var rng = new Random(seed);
            var prefixes = Enumerable.Range(0, 3).Select(i => Seq(1000 * (i + 1), 8 + (rng.Next(6) * 4))).ToArray();
            var waves = Enumerable.Range(0, 4).Select(w => Enumerable.Range(0, 3 + rng.Next(3)).Select(i =>
                Req((ulong)((w * 10) + i + 1), prefixes[rng.Next(3)].Concat(Seq(rng.Next(3000), 1 + rng.Next(14))), maxTokens: 1 + rng.Next(8))).ToArray()).ToArray();
            foreach (var inFlight in new[] { 1, 2 })
            {
                var hybrid = (seed & 1) == 0;
                var plain = Run(() => hybrid ? Hybrid(26, 3, 0) : Attention(26), Off, waves, inFlight);
                var cached = Run(() => hybrid ? Hybrid(26, 3, 5) : Attention(26), On, waves, inFlight);
                AssertSameOutputs(plain, cached);
                hits += cached.Stats!.Value.Hits;
                evictions += cached.Stats.Value.NodesEvicted;
                dropped += cached.Stats.Value.SnapshotsDropped;
                preemptions += cached.Preemptions;
                restores += cached.Restores;
            }
        }

        // The storm must actually exercise the paths it claims to.
        Assert.True(hits > 20 && evictions > 20 && preemptions > 5 && dropped > 0 && restores > 5, $"hits {hits} evictions {evictions} preemptions {preemptions} dropped {dropped} restores {restores}");
    }

    [Fact]
    public void TheCacheFillsThePoolAndGivesBlocksBackBeforeAnyPreemption()
    {
        // 12 blocks: three finished 16-token prompts would hold 12 blocks in the cache, and a fourth request still fits.
        var waves = new[]
        {
            new[] { Req(1, Seq(0, 16), maxTokens: 1) },
            [Req(2, Seq(100, 16), maxTokens: 1)],
            [Req(3, Seq(200, 16), maxTokens: 1)],
            [Req(4, Seq(300, 16), maxTokens: 1)],
            [Req(5, Seq(0, 16).Concat(Seq(50, 3)), maxTokens: 1)],
        };
        var cached = Run(() => Attention(12), On, waves);
        Assert.Equal(0, cached.Preemptions);
        Assert.True(cached.Stats!.Value.NodesEvicted > 0);
        AssertSameOutputs(Run(() => Attention(12), Off, waves), cached);
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(false, 2)]
    [InlineData(true, 1)]
    [InlineData(true, 2)]
    public void ForkedBranchesWithTheCacheOnMatchTheUncachedRun(bool hybrid, int inFlight)
    {
        var parent = Req(1, Seq(0, 22), maxTokens: 12);
        var child = Req(2, Seq(0, 22), maxTokens: 12);
        Request[] later = [Req(3, Seq(0, 22).Concat(Seq(60, 4)), maxTokens: 3), Req(4, Seq(0, 10), maxTokens: 3)];
        var plain = RunWithFork(() => hybrid ? Hybrid(60, 6, 0) : Attention(60), Off, parent, child, later, inFlight);
        var cached = RunWithFork(() => hybrid ? Hybrid(60, 6, 24) : Attention(60), On, parent, child, later, inFlight);
        AssertSameOutputs(plain, cached);
        Assert.Contains(2UL, cached.Outputs.Keys);
        Assert.True(cached.Stats!.Value.Hits >= 2, $"hits {cached.Stats.Value.Hits}");
    }

    [Fact]
    public void ReplayIsDeterministicWithTheCacheOn()
    {
        var a = Run(() => Hybrid(30, 3, 6), On, SharedPrefixWaves(), 2);
        var b = Run(() => Hybrid(30, 3, 6), On, SharedPrefixWaves(), 2);
        Assert.Equal(a.Transcript, b.Transcript);
        Assert.Equal(a.Stats, b.Stats);
    }

    [Fact]
    public void RestartDropsTheCacheAndTheRunStillCompletes()
    {
        var blocks = Hybrid(60, 4, 10);
        var planner = new StepPlanner(blocks, On);
        var worker = new ContentWorker(blocks);
        planner.Add(Req(1, Seq(0, 24), maxTokens: 2));
        for (var i = 0; i < 40 && planner.Plan() is { } plan; i++)
        {
            planner.ApplyResult(worker.Execute(plan));
            planner.CheckInvariants();
        }

        Assert.True(planner.PrefixCache!.NodeCount > 0);
        planner.OnWorkerRestarted();
        Assert.Equal(0, planner.PrefixCache.NodeCount);
        planner.CheckInvariants();
        Assert.Equal(60, blocks.GetPool(0).FreeCount);
        Assert.Equal(10, blocks.GetSnapshotPool(1).FreeCount);
    }

    [Fact]
    public void DifferentLoraIdsNeverShareNodes()
    {
        // The planner refuses LoRA requests until R-050, so exercise the cache directly.
        var blocks = Attention(40);
        var cache = new PrefixCache(blocks, new PrefixCacheOptions(), _ => false);
        var tokens = Seq(0, 13).ToArray();
        var tables = new List<int>[] { [] };
        var pool = blocks.GetPool(0);
        var ids = new int[3];
        Assert.True(pool.TryAllocate(ids));
        tables[0].AddRange(ids);
        var nodes = new List<PrefixCache.Node>();
        Assert.True(cache.Sync(tokens, 12, lora: 7, tables, nodes));
        Assert.Equal(3, cache.NodeCount);

        List<int>[] other = [[]];
        var found = new List<PrefixCache.Node>();
        Assert.False(cache.TryAttach(tokens, 8, other, found, out var computed, out _));
        Assert.Equal(0, computed);
        Assert.True(cache.TryAttach(tokens, 7, other, found, out computed, out _));
        Assert.Equal(12, computed);
        Assert.Equal(ids, other[0]);
    }

    [Fact]
    public void ARequestWithALoraIsStillRefusedByThePlanner()
    {
        var planner = new StepPlanner(Attention(20), On);
        Assert.Equal(AddResult.Unsupported, planner.Add(Req(1, Seq(0, 8), lora: 3)));
    }

    [Fact]
    public void UnsupportedGroupsAreRefusedWhenTheCacheIsOn()
    {
        var sliding = new BlockManager([Starlings.Roost.Engine.Tests.Blocks.TestGroups.Sliding(0)], [new GroupCapacity(0, 10, 0)]);
        Assert.Throws<NotSupportedException>(() => new StepPlanner(sliding, On));
        var mixed = new BlockManager([Starlings.Roost.Engine.Tests.Blocks.TestGroups.Full(0, 4), Starlings.Roost.Engine.Tests.Blocks.TestGroups.Full(1, 8)], [new GroupCapacity(0, 10, 0), new GroupCapacity(1, 10, 0)]);
        Assert.Throws<NotSupportedException>(() => new StepPlanner(mixed, On));
        var noSnapshots = Hybrid(10, 2, 0);
        Assert.Throws<InvalidOperationException>(() => new StepPlanner(noSnapshots, On));
    }
}

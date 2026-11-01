using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Caching;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Engine.Tests.Blocks;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.State;
using static Starlings.Roost.Engine.Tests.Caching.CacheRun;

namespace Starlings.Roost.Engine.Tests.Caching;

/// <summary>
/// The host-memory KV tier (R-094) on CPU: demotion and restore through the planner against workers that keep real
/// contents, a recorded multi-turn agentic trace, plan determinism, and a demote and restore through
/// <see cref="InProcessBlockTransfer"/> on the CpuOps model. No GPU is involved: these tests say nothing about the CUDA
/// worker's pinned-host copies or about TTFT.
/// </summary>
public sealed class HostTierTests
{
    private const long BlockBytes = 1024; // TestGroups.Full
    private static readonly PlannerOptions Off = new() { MaxTokensPerStep = 32 };
    private static readonly PlannerOptions DeviceOnly = Off with { PrefixCache = new PrefixCacheOptions() };

    private static PlannerOptions HostOn(int hostBlocks) => Off with { PrefixCache = new PrefixCacheOptions { HostTierBytes = hostBlocks * BlockBytes } };

    private static string RecordingDirectory([CallerFilePath] string here = "") => Path.Combine(Path.GetDirectoryName(here)!, "Recordings");

    private static string RecordingOutputDirectory => Path.Combine(AppContext.BaseDirectory, "Caching", "Recordings");

    [Fact]
    public void TheBudgetSizesEveryServedGroupAndNoOther()
    {
        ImmutableArray<StateGroupDescriptor> groups =
        [
            TestGroups.Full(0, 4, 1000),
            TestGroups.Full(1, 4, 500) with { Kind = StateGroupKind.MlaLatent },
            TestGroups.Recurrent(2, snapshotSlots: 3),
        ];
        Assert.Equal(1500, HostTier.BytesPerPrefixBlock(groups));
        Assert.Equal(6, HostTier.BlocksFor(groups, 9999));
        Assert.Equal(0, HostTier.BlocksFor(groups, 1499));

        var host = HostTier.Create(groups, 9999);
        Assert.All(host.Groups, g => Assert.Equal(BlockTier.Host, g.Tier));
        Assert.Equal(6, host.GetPool(0).Capacity);
        Assert.Equal(6, host.GetPool(1).Capacity);
        Assert.Equal(0, host.GetPool(2).Capacity);
        Assert.Equal(0, host.GetSnapshotPool(2).Capacity);
        Assert.Equal(BlockTier.Host, host.GetPool(0).Tier);
        Assert.Throws<ArgumentOutOfRangeException>(() => HostTier.BlocksFor(groups, -1));
    }

    [Fact]
    public void WithoutABudgetThereIsNoHostTier()
    {
        var planner = new StepPlanner(Attention(20), DeviceOnly);
        Assert.Null(planner.HostBlocks);
        Assert.False(planner.PrefixCache!.HasHostTier);
        Assert.Equal(default, planner.PrefixCache.HostStats);
    }

    /// <summary>
    /// One prompt is cached, an unrelated one pushes it off the device (demoted, not dropped), and the first prompt's
    /// continuation restores it from the host and reuses it. Outputs match an uncached run token for token.
    /// </summary>
    [Fact]
    public void ADemotedPrefixIsRestoredAndReused()
    {
        Request[][] waves =
        [
            [Req(1, Seq(0, 16), maxTokens: 1)],
            [Req(2, Seq(5000, 24), maxTokens: 1)],
            [Req(3, Seq(0, 16).Concat(Seq(800, 5)), maxTokens: 3)],
        ];
        var plain = Run(() => Attention(8), Off, waves);
        var device = Run(() => Attention(8), DeviceOnly, waves);
        var host = Run(() => Attention(8), HostOn(16), waves);

        Assert.Equal(plain.Outputs, host.Outputs);
        Assert.Equal(plain.Outputs, device.Outputs);
        var h = host.Host!.Value;
        Assert.True(h.Demoted >= 2, $"demoted {h.Demoted}");
        Assert.True(h.Restored >= 2, $"restored {h.Restored}");
        Assert.Equal(1, h.HostHits);
        Assert.True(h.HostHitTokens >= 8);
        Assert.True(host.Stats!.Value.HitTokens > device.Stats!.Value.HitTokens, $"host {host.Stats.Value.HitTokens} device {device.Stats.Value.HitTokens}");
        Assert.True(host.ComputedTokens < device.ComputedTokens);
        var worker = (ContentWorker)host.Worker;
        Assert.Equal(h.Demoted, worker.Demotions);
        Assert.Equal(h.Restored, worker.Restores);
        Assert.Equal(h.Demoted + h.Restored, host.TierMoves);
    }

    [Fact]
    public void AHybridModelKeepsItsSnapshotsCorrectAcrossTheHostTier()
    {
        Request[][] waves =
        [
            [Req(1, Seq(0, 16), maxTokens: 1)],
            [Req(2, Seq(5000, 24), maxTokens: 1)],
            [Req(3, Seq(0, 16).Concat(Seq(800, 5)), maxTokens: 3)],
            [Req(4, Seq(0, 12).Concat(Seq(900, 7)), maxTokens: 2)],
        ];
        var plain = Run(() => Hybrid(8, 2, 8), Off, waves);
        var host = Run(() => Hybrid(8, 2, 8), HostOn(16), waves);
        Assert.Equal(plain.Outputs, host.Outputs);
        Assert.True(host.Host!.Value.Restored > 0 && host.Host.Value.Demoted > 0, $"{host.Host}");
        Assert.True(host.Restores > 0, "no recurrent snapshot was restored");
    }

    /// <summary>A small host tier fills up: host blocks are dropped (LRU), never device blocks in use, and the outputs still match.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void HostPressureDropsOnlyFromTheHostTier(int inFlight)
    {
        var waves = Enumerable.Range(0, 6).Select(i => new[] { Req((ulong)i + 1, Seq(1000 * (i % 3), 16).Concat(Seq(7000 + (i * 50), 3)), maxTokens: 2) }).ToArray();
        var plain = Run(() => Attention(10), Off, waves, inFlight);
        var host = Run(() => Attention(10), HostOn(3), waves, inFlight);
        Assert.Equal(plain.Outputs, host.Outputs);
        Assert.True(host.Host!.Value.HostEvicted > 0, $"{host.Host}");
        Assert.True(host.Host.Value.HostBlocks <= 3);
    }

    /// <summary>
    /// With another row to run, the restore rides one plan ahead of the row that reads it, and the restored blocks stay
    /// held for that row in between.
    /// </summary>
    [Fact]
    public void RestoresRunOnePlanAheadOfTheirFirstUse()
    {
        var d = new Driver(() => Attention(11), HostOn(16));
        d.Add(Req(1, Seq(0, 16), maxTokens: 1));
        d.Drain(1);
        d.Add(Req(2, Seq(5000, 32), maxTokens: 1));
        d.Drain(1);
        Assert.True(d.Planner.PrefixCache!.HostStats.Demoted > 0);

        d.Add(Req(4, Seq(3500, 2), maxTokens: 12)); // a decoding row to overlap with
        d.Step(); // its prefill
        d.Add(Req(3, Seq(0, 16).Concat(Seq(800, 5)), maxTokens: 2));
        d.Drain(1);

        var restorePlan = d.Plans.FindIndex(p => p.CacheOps.Any(o => o.Kind == CacheOpKind.TierMove && o.SrcTier == BlockTier.Host));
        var firstRow = d.Plans.FindIndex(p => p.Sequences.Any(s => s.SequenceHandle == 3));
        Assert.True(restorePlan >= 0, "nothing was restored");
        Assert.Equal(restorePlan + 1, firstRow);
        Assert.DoesNotContain(d.Plans[restorePlan].Sequences.ToArray(), s => s.SequenceHandle == 3);
        Assert.True(d.Plans[firstRow].Sequences.Single(s => s.SequenceHandle == 3).PositionOffset >= 8);

        var plain = new Driver(() => Attention(11), Off);
        plain.Add(Req(1, Seq(0, 16), maxTokens: 1));
        plain.Drain(1);
        plain.Add(Req(2, Seq(5000, 32), maxTokens: 1));
        plain.Drain(1);
        plain.Add(Req(4, Seq(3500, 2), maxTokens: 12));
        plain.Step();
        plain.Add(Req(3, Seq(0, 16).Concat(Seq(800, 5)), maxTokens: 2));
        plain.Drain(1);
        Assert.Equal(plain.Outputs, d.Outputs);
    }

    [Fact]
    public void TheHostTierIsDroppedOnAWorkerRestart()
    {
        var d = new Driver(() => Attention(8), HostOn(16));
        d.Add(Req(1, Seq(0, 16), maxTokens: 1));
        d.Drain(1);
        d.Add(Req(2, Seq(5000, 24), maxTokens: 1));
        d.Drain(1);
        Assert.True(d.Planner.PrefixCache!.HostStats.HostBlocks > 0);

        d.Add(Req(3, Seq(0, 16).Concat(Seq(800, 5)), maxTokens: 3));
        d.Replay.Apply(new PlannerInput.Restart()); // the restore ops were queued and are now void
        Assert.Equal(0, d.Planner.PrefixCache.NodeCount);
        Assert.Equal(16, d.Planner.HostBlocks!.GetPool(0).FreeCount);
        Assert.Equal(8, d.Blocks.GetPool(0).FreeCount);
    }

    /// <summary>Random shared-prefix workloads with cancels and restarts keep every invariant (checked after every input) and replay identically.</summary>
    [Theory]
    [MemberData(nameof(Seeds))]
    public void RandomWorkloadsKeepEveryInvariantAndReplayIdentically(int seed) => _ = RunRandom(seed);

    [Fact]
    public void TheRandomWorkloadsReachDemotionRestoreAndHostEviction()
    {
        var all = Enumerable.Range(1, 40).Select(RunRandom).ToList();
        Assert.True(all.Sum(h => h.Demoted) > 0 && all.Sum(h => h.Restored) > 0 && all.Sum(h => h.HostHits) > 0 && all.Sum(h => h.HostEvicted) > 0,
            string.Join('\n', all));
    }

    public static TheoryData<int> Seeds() => [.. Enumerable.Range(1, 40)];

    private static HostTierStats RunRandom(int seed)
    {
        var blocks = Rebuild(seed);
        var rng = new Random(seed * 7919);
        var options = HostOn(rng.Next(12)) with { MaxTokensPerStep = 8 + rng.Next(30), MaxSequencesPerStep = 1 + rng.Next(4) };
        var planner = new StepPlanner(blocks, options);
        var replay = new PlannerReplay(planner);
        var log = new List<PlannerInput>();
        var pending = new Queue<StepOutcome>();
        var prefixes = Enumerable.Range(0, 3).Select(i => Seq(1000 * (i + 1), 4 + rng.Next(16)).ToArray()).ToArray();
        ulong next = 1;
        var maxInFlight = 1 + rng.Next(3);

        void Do(PlannerInput i)
        {
            log.Add(i);
            replay.Apply(i);
        }

        for (var step = 0; step < 500; step++)
        {
            var roll = rng.Next(100);
            if (roll < 18 && next < 60)
            {
                var prompt = prefixes[rng.Next(3)].Concat(Seq(rng.Next(4000), 1 + rng.Next(10)));
                Do(new PlannerInput.Add(Req(next++, prompt, maxTokens: 1 + rng.Next(6))));
            }
            else if (roll < 21 && next > 1)
            {
                Do(new PlannerInput.Cancel((ulong)(1 + rng.Next((int)next))));
            }
            else if (roll < 22)
            {
                Do(new PlannerInput.Restart());
                pending.Clear();
            }
            else if (roll < 65 && pending.Count < maxInFlight)
            {
                log.Add(new PlannerInput.Plan());
                if (replay.Apply(log[^1]) is { } plan)
                {
                    pending.Enqueue(Scheduling.PlannerHarness.Simulate(plan));
                }
            }
            else if (pending.Count > 0)
            {
                log.Add(new PlannerInput.Result(pending.Dequeue()));
                replay.Apply(log[^1]);
            }
        }

        Assert.Equal(replay.Transcript, new PlannerReplay(new StepPlanner(Rebuild(seed), options)).Run(log));
        return planner.PrefixCache!.HostStats;
    }

    /// <summary>
    /// A recorded multi-turn agentic trace: four sessions share a tool preamble, and each turn's prompt is the session's
    /// previous prompt plus its reply plus a tool result. The device pool holds about half the working set, so a session's
    /// history is pushed off the device between its turns; with the host tier it comes back from host memory.
    /// </summary>
    [Fact]
    public void TheAgenticTraceHitsTheHostTierAndMatchesUncachedOutputs()
    {
        var plain = RunAgentic(Off);
        var device = RunAgentic(DeviceOnly);
        var host = RunAgentic(AgenticHost);
        Assert.Equal(plain.Outputs, host.Outputs);
        Assert.Equal(plain.Outputs, device.Outputs);

        var h = host.Planner.PrefixCache!.HostStats;
        Assert.True(h.HostHits >= 4, $"{h}");
        Assert.True(h.HostHitTokens >= 64, $"{h}");
        var hostHit = host.Planner.PrefixCache.Stats.HitTokens;
        var deviceHit = device.Planner.PrefixCache!.Stats.HitTokens;
        Assert.True(hostHit > deviceHit, $"host-tier run reused {hostHit} tokens, device-only {deviceHit}");
        Assert.True(host.Worker.ComputedTokens < device.Worker.ComputedTokens);
        Assert.Equal(h.Restored, host.Worker.Restores);
    }

    [Fact]
    public void TheAgenticTraceReplaysToTheSamePlans()
    {
        var a = RunAgentic(AgenticHost);
        var b = RunAgentic(AgenticHost);
        Assert.Equal(a.Replay.Transcript, b.Replay.Transcript);
        Assert.Equal(Scheduling.PlannerHarness.Encode(a.Log), Scheduling.PlannerHarness.Encode(b.Log));
        var fresh = new PlannerReplay(new StepPlanner(Attention(AgenticDeviceBlocks), AgenticHost));
        Assert.Equal(a.Replay.Transcript, fresh.Run(a.Log));
        Assert.Equal(a.Plans.Select(p => p.Digest), b.Plans.Select(p => p.Digest));
    }

    /// <summary>
    /// The committed recording pins the host-tier policy: the trace still produces it, and replaying it on a fresh planner
    /// gives the committed plans (SHA-256 digests). Regenerate with <c>ROOST_UPDATE_RECORDINGS=1</c> and review the diff.
    /// </summary>
    [Fact]
    public void TheCommittedAgenticRecordingStillReplaysToTheCommittedPlans()
    {
        var live = RunAgentic(AgenticHost);
        if (Environment.GetEnvironmentVariable("ROOST_UPDATE_RECORDINGS") == "1")
        {
            Directory.CreateDirectory(RecordingDirectory());
            File.WriteAllText(Path.Combine(RecordingDirectory(), "agentic-host-tier.inputs.txt"), Scheduling.PlannerHarness.Encode(live.Log));
            File.WriteAllText(Path.Combine(RecordingDirectory(), "agentic-host-tier.plans.txt"), string.Join('\n', live.Replay.Transcript) + "\n");
        }

        var recorded = Scheduling.PlannerHarness.Decode(File.ReadAllText(Path.Combine(RecordingOutputDirectory, "agentic-host-tier.inputs.txt")));
        Assert.Equal(Scheduling.PlannerHarness.Encode(live.Log), Scheduling.PlannerHarness.Encode(recorded));

        var planner = new StepPlanner(Attention(AgenticDeviceBlocks), AgenticHost);
        var replayed = new PlannerReplay(planner).Run(recorded);
        Assert.Equal(File.ReadAllText(Path.Combine(RecordingOutputDirectory, "agentic-host-tier.plans.txt")).TrimEnd('\n').Split('\n'), replayed);
        Assert.True(planner.PrefixCache!.HostStats.HostHits > 0, "the recorded trace has no host-tier prefix hit");
    }

    [Fact]
    public void AHybridAgenticTraceMatchesUncachedOutputs()
    {
        var plain = RunAgentic(Off, hybrid: true);
        var host = RunAgentic(AgenticHost, hybrid: true);
        Assert.Equal(plain.Outputs, host.Outputs);
        Assert.True(host.Planner.PrefixCache!.HostStats.HostHits > 0, $"{host.Planner.PrefixCache.HostStats}");
    }

    /// <summary>
    /// The CpuOps model through the planner: the host-tier run gives the same logits as an uncached run, bit for bit, while
    /// demoting and restoring.
    /// </summary>
    [Fact]
    public void CpuOpsLogitsAreBitIdenticalWithTheHostTier()
    {
        Request[][] waves =
        [
            [Req(1, Seq(0, 30), maxTokens: 2)],
            [Req(2, Seq(3300, 40), maxTokens: 2)],
            [Req(3, Seq(0, 30).Concat(Seq(500, 5)), maxTokens: 3)],
            [Req(4, Seq(0, 17).Concat(Seq(700, 9)), maxTokens: 2), Req(5, Seq(2000, 13), maxTokens: 2)],
        ];
        var plain = Run(() => Attention(16), Off with { MaxTokensPerStep = 64 }, waves, 2, (b, h) => new CpuOpsTinyWorker(b, h));
        var host = Run(() => Attention(16), HostOn(24) with { MaxTokensPerStep = 64 }, waves, 2, (b, h) => new CpuOpsTinyWorker(b, h));
        Assert.Equal(plain.Outputs, host.Outputs);
        Assert.True(host.Host!.Value.Restored > 0, $"{host.Host}");
        var x = ((CpuOpsTinyWorker)plain.Worker).Logits;
        var y = ((CpuOpsTinyWorker)host.Worker).Logits;
        foreach (var (id, rows) in x)
        {
            Assert.Equal(rows.Count, y[id].Count);
            for (var i = 0; i < rows.Count; i++)
            {
                Assert.True(rows[i].AsSpan().SequenceEqual(y[id][i]), $"request {id} sample {i}: logits differ");
            }
        }
    }

    /// <summary>
    /// <see cref="InProcessBlockTransfer"/> between the device manager and the host tier on the CpuOps model: a prefilled
    /// prefix is demoted (TierMove ops, device blocks freed and overwritten by another prompt), restored into different
    /// device blocks, and the continuation's logits equal those of a run that never left the device. Skipping the restore
    /// changes them.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task InProcessTransferPreservesLogitsAcrossADemoteAndRestore(bool restore)
    {
        int[] prefix = [.. Seq(0, 12)];
        int[] suffix = [.. Seq(600, 5)];

        // Reference: prefix and continuation on the device.
        var refBlocks = Attention(16);
        using var reference = new CpuOpsTinyWorker(refBlocks);
        var d = Alloc(refBlocks, 3);
        reference.Execute(Plan(1, 7, prefix, 0, d, sample: false));
        var e = Alloc(refBlocks, 2);
        reference.Execute(Plan(2, 7, suffix, 12, [.. d, .. e], sample: true));

        var blocks = Attention(16);
        var host = HostTier.Create(blocks.Groups, 8 * BlockBytes);
        using var worker = new CpuOpsTinyWorker(blocks, host);
        var device = Alloc(blocks, 3);
        worker.Execute(Plan(1, 7, prefix, 0, device, sample: false));

        // Demote.
        var hostIds = new int[3];
        using (var down = new InProcessBlockTransfer(blocks, host))
        {
            var export = await down.ExportBlocksAsync(0, device, CancellationToken.None);
            await down.ImportBlocksAsync(export, hostIds, CancellationToken.None);
            var ops = new PlanCacheOp[3];
            var n = down.Drain(ops, out var ticket);
            Assert.All(ops[..n], o => Assert.Equal((CacheOpKind.TierMove, BlockTier.Device, BlockTier.Host), (o.Kind, o.SrcTier, o.DstTier)));
            worker.ApplyCacheOps(ops.AsSpan(0, n));
            down.Complete(ticket);
        }

        blocks.GetPool(0).Release(device);
        Assert.Equal(16, blocks.GetPool(0).FreeCount);

        // Another prompt reuses the freed device blocks.
        var other = Alloc(blocks, 3);
        Assert.Equal(device.Order(), other.Order());
        worker.Execute(Plan(3, 8, [.. Seq(3000, 12)], 0, other, sample: false));

        // Restore into different device blocks.
        var restored = new int[3];
        using (var up = new InProcessBlockTransfer(host, blocks))
        {
            var export = await up.ExportBlocksAsync(0, hostIds, CancellationToken.None);
            await up.ImportBlocksAsync(export, restored, CancellationToken.None);
            var ops = new PlanCacheOp[3];
            var n = up.Drain(ops, out var ticket);
            Assert.All(ops[..n], o => Assert.Equal((CacheOpKind.TierMove, BlockTier.Host, BlockTier.Device), (o.Kind, o.SrcTier, o.DstTier)));
            if (restore)
            {
                worker.ApplyCacheOps(ops.AsSpan(0, n));
            }

            up.Complete(ticket);
        }

        host.GetPool(0).Release(hostIds);
        Assert.Empty(restored.Intersect(other));
        var tail = Alloc(blocks, 2);
        worker.Execute(Plan(4, 7, suffix, 12, [.. restored, .. tail], sample: true));

        var want = reference.Logits[7].Single();
        var got = worker.Logits[7].Single();
        Assert.Equal(restore, want.AsSpan().SequenceEqual(got));
        Assert.Equal(8, host.GetPool(0).FreeCount);
    }

    private const int AgenticDeviceBlocks = 24;

    private static readonly PlannerOptions AgenticHost = HostOn(64);

    private static BlockManager Rebuild(int seed)
    {
        var rng = new Random(seed);
        return seed % 2 == 0 ? Hybrid(10 + rng.Next(20), 2 + rng.Next(3), 2 + rng.Next(6)) : Attention(10 + rng.Next(20));
    }

    private static Driver RunAgentic(PlannerOptions options, bool hybrid = false)
    {
        const int sessions = 4;
        const int turns = 4;
        var d = new Driver(() => hybrid ? Hybrid(AgenticDeviceBlocks, 2, 24) : Attention(AgenticDeviceBlocks), options);
        var preamble = Seq(0, 16).ToArray(); // tool definitions shared by every session
        var history = Enumerable.Range(0, sessions).Select(s => preamble.Concat(Seq(1000 * (s + 1), 6)).ToList()).ToArray();
        ulong id = 1;
        for (var t = 0; t < turns; t++)
        {
            for (var s = 0; s < sessions; s++)
            {
                var request = new Request
                {
                    Id = new RequestId(id),
                    PromptTokenIds = [.. history[s]],
                    Sampling = new SamplingParameters { MaxTokens = 4, Temperature = 0.7f }, // as the recording codec decodes it
                };
                d.Add(request);
                d.Drain(2);
                history[s].AddRange(d.Outputs[id]);
                history[s].AddRange(Seq(3000 + (s * 97) + (t * 13), 6)); // the tool result
                id++;
            }
        }

        return d;
    }

    private static int[] Alloc(BlockManager blocks, int n)
    {
        var ids = new int[n];
        Assert.True(blocks.GetPool(0).TryAllocate(ids));
        return ids;
    }

    /// <summary>A one-row plan for the CpuOps model: <paramref name="tokens"/> at <paramref name="offset"/> over <paramref name="table"/>.</summary>
    private static PlannedStep Plan(ulong step, ulong handle, int[] tokens, int offset, int[] table, bool sample) =>
        new(
            step,
            1,
            PlanOptions.Prefill,
            default,
            [new PlanSequence
            {
                SequenceHandle = handle,
                TokenCount = (uint)tokens.Length,
                PositionOffset = (uint)offset,
                Flags = SequenceOptions.Prefill | (sample ? SequenceOptions.Sample : SequenceOptions.None),
                RecurrentSlot = -1,
                LoraSlot = -1,
                GrammarSlot = -1,
                EmbeddingSlot = -1,
                HeadRow = -1,
                EncoderOutput = -1,
            }],
            [.. tokens],
            [.. Enumerable.Range(offset, tokens.Length)],
            [new PlanGroupBlocks { BlockStart = 0, BlockCount = (uint)table.Length }],
            [.. table],
            [default(PlanSampling)],
            [],
            [],
            string.Empty);

    /// <summary>Drives a planner and a <see cref="ContentWorker"/>, logging inputs and keeping every plan.</summary>
    private sealed class Driver
    {
        public Driver(Func<BlockManager> makeBlocks, PlannerOptions options)
        {
            Blocks = makeBlocks();
            Planner = new StepPlanner(Blocks, options);
            Replay = new PlannerReplay(Planner);
            Worker = new ContentWorker(Blocks, Planner.HostBlocks);
        }

        public BlockManager Blocks { get; }

        public StepPlanner Planner { get; }

        public PlannerReplay Replay { get; }

        public ContentWorker Worker { get; }

        public List<PlannerInput> Log { get; } = [];

        public List<PlannedStep> Plans { get; } = [];

        public Dictionary<ulong, List<int>> Outputs { get; } = [];

        private Queue<StepOutcome> Pending { get; } = new();

        public void Add(Request r) => Do(new PlannerInput.Add(r));

        /// <summary>Builds and executes one plan, then applies its result.</summary>
        public void Step()
        {
            Assert.True(TryPlan());
            Complete();
        }

        public void Drain(int maxInFlight)
        {
            for (var guard = 0; guard < 100_000; guard++)
            {
                if (Pending.Count < maxInFlight && TryPlan())
                {
                    continue;
                }

                if (Pending.Count == 0)
                {
                    return;
                }

                Complete();
            }

            throw new InvalidOperationException("The planner did not drain.");
        }

        private bool TryPlan()
        {
            Log.Add(new PlannerInput.Plan());
            if (Replay.Apply(Log[^1]) is not { } plan)
            {
                return false;
            }

            Plans.Add(plan);
            Pending.Enqueue(Worker.Execute(plan));
            return true;
        }

        private void Complete()
        {
            var outcome = Pending.Dequeue();
            foreach (var row in outcome.Sequences.Where(x => !x.Tokens.IsDefaultOrEmpty))
            {
                if (!Outputs.TryGetValue(row.Handle, out var list))
                {
                    Outputs[row.Handle] = list = [];
                }

                list.Add(row.Tokens[0]);
            }

            Do(new PlannerInput.Result(outcome));
        }

        private void Do(PlannerInput input)
        {
            Log.Add(input);
            Replay.Apply(input);
        }
    }
}

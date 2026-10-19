using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Engine.Tests.Blocks;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using static Starlings.Roost.Engine.Tests.Scheduling.PlannerHarness;

namespace Starlings.Roost.Engine.Tests.Scheduling;

public sealed class StepPlannerTests
{
    private static StepPlanner Make(BlockManager m, int tokens = 64, int rows = 8, int chunk = int.MaxValue) =>
        new(m, new PlannerOptions { MaxTokensPerStep = tokens, MaxSequencesPerStep = rows, MaxChunkTokens = chunk });

    [Fact]
    public void TokenBudgetAndSequenceCapBoundEveryPlan()
    {
        var p = Make(Hybrid(200, 16), tokens: 10, rows: 3);
        for (ulong i = 1; i <= 6; i++)
        {
            Assert.Equal(AddResult.Added, p.Add(Req(i, 6)));
        }

        var plan = p.Plan()!;
        Assert.Equal(2, plan.Sequences.Length); // budget 10: 6 + 4 (the second prompt is cut)
        Assert.Equal(10, plan.TokenIds.Length);
        Assert.Equal([6u, 4u], plan.Sequences.Select(s => s.TokenCount).ToArray());
        Assert.Equal(PlanOptions.Prefill, plan.Flags);

        var capped = Make(Hybrid(200, 16), tokens: 100, rows: 3);
        for (ulong i = 1; i <= 6; i++)
        {
            capped.Add(Req(i, 6));
        }

        Assert.Equal(3, capped.Plan()!.Sequences.Length); // row cap, 18 tokens of 100
    }

    [Fact]
    public void ChunkedPrefillSplitsALongPromptAndSamplesOnlyAtTheEnd()
    {
        var p = Make(Hybrid(100, 4), tokens: 32);
        p.Add(Req(1, 100, maxTokens: 2));
        var chunks = new List<PlanSequence>();
        PlannedStep? plan;
        while ((plan = p.Plan()) is not null && chunks.Count < 4)
        {
            chunks.Add(plan.Sequences[0]);
            Assert.Equal(plan.Sequences[0].PositionOffset, (uint)plan.Positions[0]);
            if ((plan.Sequences[0].Flags & SequenceOptions.Sample) != 0)
            {
                break;
            }
        }

        Assert.Equal([32u, 32u, 32u, 4u], chunks.Select(c => c.TokenCount).ToArray());
        Assert.Equal([0u, 32u, 64u, 96u], chunks.Select(c => c.PositionOffset).ToArray());
        Assert.Equal([false, false, false, true], chunks.Select(c => (c.Flags & SequenceOptions.Sample) != 0).ToArray());
        Assert.True(chunks.All(c => c.RecurrentSlot == chunks[0].RecurrentSlot));
        p.CheckInvariants();
    }

    [Fact]
    public void ADecodeRowIsOneTokenAndWaitsForItsSampledToken()
    {
        var p = Make(Hybrid(50, 4));
        p.Add(Req(1, 5));
        var first = p.Plan()!;
        Assert.Null(p.Plan()); // sampled token not back yet
        var outcome = Simulate(first);
        p.ApplyResult(outcome);
        var second = p.Plan()!;
        Assert.Equal(1u, second.Sequences[0].TokenCount);
        Assert.Equal(5u, second.Sequences[0].PositionOffset);
        Assert.Equal(outcome.Sequences[0].Tokens[0], second.TokenIds[0]);
        Assert.Equal(SequenceOptions.Sample, second.Sequences[0].Flags);
    }

    [Fact]
    public void PriorityThenDeadlineThenArrivalOrdersTheRows()
    {
        var p = Make(Hybrid(200, 16));
        var t = new DateTimeOffset(2026, 10, 19, 12, 0, 0, TimeSpan.Zero);
        p.Add(Req(1, 4)); // normal, no deadline, first
        p.Add(Req(2, 4, deadline: t.AddSeconds(30)));
        p.Add(Req(3, 4, deadline: t.AddSeconds(10)));
        p.Add(Req(4, 4, priority: RequestPriority.High));
        p.Add(Req(5, 4, priority: RequestPriority.Low));
        p.Add(Req(6, 4));
        Assert.Equal([4ul, 3, 2, 1, 6, 5], p.Plan()!.Sequences.Select(s => s.SequenceHandle).ToArray());
    }

    [Fact]
    public void AHigherPriorityRequestPreemptsALowerOneAndTheVictimRecomputes()
    {
        var m = Hybrid(attentionBlocks: 6, slots: 4); // 24 tokens
        var p = Make(m);
        p.Add(Req(1, 20, priority: RequestPriority.Low));
        var a = p.Plan()!;
        p.ApplyResult(Simulate(a));
        Assert.Equal(5, m.GetPool(0).Capacity - m.GetPool(0).FreeCount); // 20 tokens planned -> 5 blocks

        p.Add(Req(2, 8, priority: RequestPriority.High));
        var b = p.Plan()!;
        Assert.Equal([2ul], b.Sequences.Select(s => s.SequenceHandle).ToArray());
        Assert.Equal([new PlannerNotice(PlannerNoticeKind.Preempted, 1)], p.DrainNotices());
        Assert.True(p.TryGetInfo(1, out var info));
        Assert.False(info.Resident);
        Assert.Equal(0, info.ComputedTokens);
        Assert.Equal(21, info.TokenCount); // generated token kept
        p.CheckInvariants();

        p.ApplyResult(Simulate(b));
        // 2 holds 3 blocks; 1 needs 6 for its 21 tokens: it waits until 2 is done or it can preempt nobody better.
        var outputs = new Dictionary<ulong, List<int>>();
        var log = new List<PlannerInput>();
        Drain(new PlannerReplay(p), log, outputs, 1);
        Assert.False(p.TryGetInfo(1, out _));
        Assert.False(p.TryGetInfo(2, out _));
        Assert.Equal(m.GetPool(0).Capacity, m.GetPool(0).FreeCount);
        Assert.Equal(m.GetPool(1).Capacity, m.GetPool(1).FreeCount);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void PreemptionDoesNotChangeWhatEachSequenceGenerates(int maxInFlight)
    {
        Dictionary<ulong, List<int>> Run(int blocks, int slots)
        {
            var p = Make(Hybrid(blocks, slots), tokens: 24, rows: 6);
            var log = new List<PlannerInput>();
            var replay = new PlannerReplay(p);
            for (ulong i = 1; i <= 7; i++)
            {
                var r = Req(i, 5 + (int)(i * 3), maxTokens: 30, priority: (RequestPriority)((int)(i % 3) - 1));
                log.Add(new PlannerInput.Add(r));
                Assert.Equal(AddResult.Added, p.Add(r));
            }

            var outputs = new Dictionary<ulong, List<int>>();
            Drain(replay, log, outputs, maxInFlight);
            Assert.Empty(replay.Transcript.Where(l => l.Contains("Aborted", StringComparison.Ordinal)));
            return outputs;
        }

        var roomy = Run(blocks: 400, slots: 16);
        var tight = Run(blocks: 28, slots: 3);
        Assert.Equal(7, roomy.Count);
        foreach (var (h, tokens) in roomy)
        {
            Assert.Equal(tokens, tight[h]);
        }
    }

    [Fact]
    public void NobodyIsPreemptedWhenEvenAllVictimsCouldNotSatisfyTheRequest()
    {
        var m = Hybrid(attentionBlocks: 6, slots: 4); // 24 tokens
        var p = Make(m);
        p.Add(Req(1, 8, priority: RequestPriority.High));
        p.Add(Req(2, 8, priority: RequestPriority.Low));
        p.ApplyResult(Simulate(p.Plan()!));
        Assert.Empty(p.DrainNotices());

        // 40 tokens need 10 blocks; the pool has 6 and the only possible victim (2) frees 2 more, which is 4 in all.
        p.Add(Req(3, 40));
        var plan = p.Plan()!;
        Assert.DoesNotContain(plan.Sequences, s => s.SequenceHandle == 3);
        Assert.Empty(p.DrainNotices());
        Assert.True(p.TryGetInfo(2, out var info));
        Assert.True(info.Resident);
        Assert.True(info.ComputedTokens > 0);
        p.CheckInvariants();
    }

    [Fact]
    public void ARecomputedSeededSequenceUsesTheSamePhiloxOffsetAtEveryPosition()
    {
        static (Dictionary<ulong, ulong> Offsets, bool Preempted) Run(int blocks)
        {
            var p = Make(Hybrid(blocks, 4));
            p.Add(Req(1, 8, maxTokens: 12, priority: RequestPriority.Low, seed: 7));
            p.Add(Req(2, 12, maxTokens: 6, priority: RequestPriority.High));
            var offsets = new Dictionary<ulong, ulong>();
            var preempted = false;
            for (var i = 0; i < 1000; i++)
            {
                var plan = p.Plan();
                preempted |= p.DrainNotices().Any(n => n.Kind == PlannerNoticeKind.Preempted && n.Handle == 1);
                if (plan is null)
                {
                    break;
                }

                for (var r = 0; r < plan.Sequences.Length; r++)
                {
                    var row = plan.Sequences[r];
                    if (row.SequenceHandle != 1 || (row.Flags & SequenceOptions.Sample) == 0)
                    {
                        continue;
                    }

                    var at = row.PositionOffset + row.TokenCount;
                    var offset = plan.Sampling[r].PhiloxOffset;
                    if (offsets.TryGetValue(at, out var seen))
                    {
                        Assert.Equal(seen, offset);
                    }

                    offsets[at] = offset;
                }

                p.ApplyResult(Simulate(plan));
            }

            return (offsets, preempted);
        }

        var roomy = Run(40);
        var tight = Run(6);
        Assert.False(roomy.Preempted);
        Assert.True(tight.Preempted);
        Assert.True(tight.Offsets.Count > 1);
        foreach (var (at, offset) in tight.Offsets)
        {
            Assert.Equal(roomy.Offsets[at], offset);
        }
    }

    [Fact]
    public void TightPoolsActuallyPreempt()
    {
        var p = Make(Hybrid(28, 3), tokens: 24, rows: 6);
        var replay = new PlannerReplay(p);
        var log = new List<PlannerInput>();
        for (ulong i = 1; i <= 7; i++)
        {
            p.Add(Req(i, 5 + (int)(i * 3), maxTokens: 30, priority: (RequestPriority)((int)(i % 3) - 1)));
        }

        Drain(replay, log, [], 1);
        Assert.Contains(replay.Transcript, l => l.Contains("Preempted", StringComparison.Ordinal));
    }

    [Fact]
    public void ABlockNamedByAnUncompletedPlanIsNotReleasedOrReallocatedUntilItsResult()
    {
        var m = Hybrid(attentionBlocks: 3, slots: 2);
        var p = Make(m);
        p.Add(Req(1, 9)); // 3 blocks, the whole pool
        var plan = p.Plan()!;
        var named = plan.BlocksOf(0, 0).ToArray();
        Assert.Equal(3, named.Length);

        Assert.True(p.Cancel(1));
        Assert.Equal(0, m.GetPool(0).FreeCount); // parked, not freed
        Assert.True(p.ParkedBlocks >= 3);
        p.Add(Req(2, 4));
        Assert.Null(p.Plan()); // nothing to hand out: the old blocks are still named
        p.CheckInvariants();

        p.ApplyResult(Simulate(plan));
        Assert.Equal(0, p.ParkedBlocks);
        Assert.Equal(3, m.GetPool(0).FreeCount);
        var next = p.Plan()!;
        Assert.Equal(2ul, next.Sequences[0].SequenceHandle);
        p.CheckInvariants();
    }

    [Fact]
    public void ForkedBranchRowsShareBlocksCopyOnWriteAndCountAgainstTheRowCap()
    {
        var m = Hybrid(attentionBlocks: 40, slots: 8);
        var p = Make(m, rows: 2);
        p.Add(Req(1, 6, maxTokens: 20));
        p.ApplyResult(Simulate(p.Plan()!));
        Assert.Equal(ForkResult.Forked, p.Fork(1, Req(2, 1, maxTokens: 20, seed: 1)));
        Assert.Equal(ForkResult.Forked, p.Fork(1, Req(3, 1, maxTokens: 20, seed: 2)));
        p.CheckInvariants();

        var plan = p.Plan()!;
        Assert.Equal(2, plan.Sequences.Length); // 3 branch rows, cap 2
        Assert.Contains(plan.CacheOps, o => o.Kind == CacheOpKind.Fork); // recurrent copies of the pending forks
        // 7 tokens in 2 blocks; the last block (positions 4..7) is shared and the row writes position 6: copy-on-write.
        Assert.Contains(plan.CacheOps, o => o.Kind == CacheOpKind.Copy);
        var rowsBlocks = Enumerable.Range(0, 2).Select(r => plan.BlocksOf(r, 0).ToArray()).ToArray();
        Assert.Equal(rowsBlocks[0][0], rowsBlocks[1][0]); // the full first block stays shared
        Assert.NotEqual(rowsBlocks[0][1], rowsBlocks[1][1]);
        Assert.NotEqual(plan.Sequences[0].RecurrentSlot, plan.Sequences[1].RecurrentSlot);
        p.CheckInvariants();
    }

    [Fact]
    public void AForkOpsSourceIsNotReleasedBeforeThePlanThatRunsIt()
    {
        var m = Hybrid(attentionBlocks: 40, slots: 4);
        var p = Make(m);
        p.Add(Req(1, 6));
        p.ApplyResult(Simulate(p.Plan()!));
        Assert.Equal(ForkResult.Forked, p.Fork(1, Req(2, 1)));
        var slotFree = m.GetPool(1).FreeCount;
        Assert.True(p.Cancel(1)); // the fork op in the pending list still names the parent's slot
        Assert.Equal(slotFree, m.GetPool(1).FreeCount);
        Assert.True(p.ParkedBlocks > 0);
        var plan = p.Plan()!;
        Assert.Equal(2ul, plan.Sequences[0].SequenceHandle);
        p.ApplyResult(Simulate(plan));
        Assert.Equal(0, p.ParkedBlocks);
        p.CheckInvariants();
    }

    [Fact]
    public void WorkerRestartPreemptsEverythingAndReleasesAll()
    {
        var m = Hybrid(60, 6);
        var p = Make(m);
        for (ulong i = 1; i <= 3; i++)
        {
            p.Add(Req(i, 7));
        }

        var plan = p.Plan()!;
        Assert.Equal(1, p.InFlightPlans);
        p.OnWorkerRestarted();
        Assert.Equal(0, p.InFlightPlans);
        Assert.Equal(m.GetPool(0).Capacity, m.GetPool(0).FreeCount);
        Assert.Equal(m.GetPool(1).Capacity, m.GetPool(1).FreeCount);
        Assert.Equal(3, p.DrainNotices().Length);
        Assert.Throws<InvalidOperationException>(() => p.ApplyResult(Simulate(plan)));
        var again = p.Plan()!;
        Assert.Equal(3, again.Sequences.Length);
        Assert.All(again.Sequences, s => Assert.Equal(0u, s.PositionOffset));
        p.CheckInvariants();
    }

    [Fact]
    public void ASequenceThatCanNeverFitIsAbortedInsteadOfStallingTheQueue()
    {
        var m = Hybrid(attentionBlocks: 4, slots: 2); // 16 tokens
        var p = Make(m);
        p.Add(Req(1, 12, maxTokens: 100)); // grows past 16 tokens while generating
        p.Add(Req(2, 3, maxTokens: 3, priority: RequestPriority.Low));
        var replay = new PlannerReplay(p);
        var outputs = new Dictionary<ulong, List<int>>();
        Drain(replay, [], outputs, 1);
        Assert.Equal(0, p.SequenceCount);
        Assert.Equal(m.GetPool(0).Capacity, m.GetPool(0).FreeCount);
        Assert.Equal(3, outputs[2].Count);
        // Sequence 1 either ended on its own (EOS) before outgrowing the pool or was aborted; it must not hold anything.
        Assert.False(p.TryGetInfo(1, out _));
    }

    [Fact]
    public void RequestsThatCannotBePlannedAreRefusedWithATypedResult()
    {
        var p = Make(Hybrid(4, 2));
        Assert.Equal(AddResult.PromptTooLong, p.Add(Req(1, 40)));
        Assert.Equal(AddResult.InvalidOption, p.Add(Req(2, 0)));
        Assert.Equal(AddResult.InvalidOption, p.Add(Req(3, 4, maxTokens: 0)));
        Assert.Equal(AddResult.Unsupported, p.Add(Req(4, 4) with { Lora = new LoraHandle(1) }));
        Assert.Equal(AddResult.Added, p.Add(Req(5, 4)));
        Assert.Equal(AddResult.DuplicateId, p.Add(Req(5, 4)));
    }

    [Fact]
    public void ASlidingGroupDropsBlocksThatLeftTheWindowAndKeepsTableIndexes()
    {
        var groups = System.Collections.Immutable.ImmutableArray.Create(TestGroups.Sliding(0, blockSize: 2, window: 4), TestGroups.Recurrent(1, snapshotSlots: 0));
        var m = new BlockManager(groups, [new GroupCapacity(0, 8, 0), new GroupCapacity(1, 2, 0)]);
        var p = Make(m);
        p.Add(Req(1, 6, maxTokens: 40));
        p.ApplyResult(Simulate(p.Plan()!));
        for (var i = 0; i < 12; i++)
        {
            var plan = p.Plan()!;
            var table = plan.BlocksOf(0, 0).ToArray();
            Assert.Equal(((int)plan.Sequences[0].PositionOffset + 2) / 2, table.Length);
            if (plan.Sequences[0].PositionOffset > 6)
            {
                Assert.Equal(-1, table[0]);
            }

            p.ApplyResult(Simulate(plan));
            p.CheckInvariants();
            if (!p.TryGetInfo(1, out _))
            {
                break;
            }
        }

        Assert.True(m.GetPool(0).FreeCount >= 8 - 4);
    }

    [Fact]
    public void ThePlanDoesNotDependOnHowManyPlansAreInFlightForPrefill()
    {
        // Mid-prefill rows are planned again without waiting for results (nothing sampled).
        var p = Make(Hybrid(100, 4), tokens: 8);
        p.Add(Req(1, 30));
        var a = p.Plan()!;
        var b = p.Plan()!;
        Assert.Equal(0u, a.Sequences[0].PositionOffset);
        Assert.Equal(8u, b.Sequences[0].PositionOffset);
        Assert.Equal(2, p.InFlightPlans);
    }

    [Fact]
    public void AHybridQwen35ShapedManagerPlansWithOneRecurrentSlotPerRow()
    {
        var m = BlockManager.Create(TestGroups.Qwen35Shaped(attentionBlockSize: 16), budgetBytes: 2L << 30, maxSequences: 4);
        var p = Make(m);
        p.Add(Req(1, 50));
        p.Add(Req(2, 20));
        var plan = p.Plan()!;
        Assert.Equal(2, plan.GroupCount);
        Assert.Equal(4, plan.BlocksOf(0, 0).Length); // 50 tokens / 16
        Assert.Equal(0, plan.BlocksOf(0, 1).Length);
        Assert.NotEqual(plan.Sequences[0].RecurrentSlot, plan.Sequences[1].RecurrentSlot);
    }
}

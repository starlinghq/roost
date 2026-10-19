using Starlings.Roost.Engine.Admission;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using static Starlings.Roost.Engine.Tests.Scheduling.PlannerHarness;

namespace Starlings.Roost.Engine.Tests.Admission;

public sealed class AdmittedPlannerTests
{
    private static AdmittedPlanner Make(BlockManager m, IDeviceMemorySource? memory = null) =>
        new(
            new StepPlanner(m, new PlannerOptions { MaxTokensPerStep = 64, MaxSequencesPerStep = 8 }),
            new AdmissionController(m, new AdmissionOptions { ContextLength = 64, MinFreeDeviceBytes = memory is null ? 0 : 1 }, memory));

    [Fact]
    public void ARefusalChangesNothingAndReachesNeitherThePoolsNorThePlanner()
    {
        var m = Hybrid(attentionBlocks: 8, slots: 4);
        var p = Make(m);
        Assert.Null(p.TryAdd(Req(1, 4, 8))); // 13 tokens = 4 blocks
        Assert.Null(p.TryAdd(Req(2, 4, 8)));
        var free = (m.GetPool(0).FreeCount, m.GetPool(1).FreeCount);

        var error = p.TryAdd(Req(3, 4, 8));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, error!.Code);
        Assert.Equal(2, p.Planner.SequenceCount);
        Assert.Equal(free, (m.GetPool(0).FreeCount, m.GetPool(1).FreeCount));
        Assert.False(p.Planner.TryGetInfo(3, out _));
        Assert.Equal(2, p.Admission.LiveRows);
        p.Planner.CheckInvariants();
    }

    [Fact]
    public void ARefusalOnAnIdlePlannerProducesNoWorkAtAll()
    {
        var m = Hybrid(attentionBlocks: 3, slots: 1);
        var p = Make(m);
        Assert.NotNull(p.TryAdd(Req(1, 4, 40)));
        Assert.Null(p.Plan()); // nothing to run: a refused request never reaches a worker
        Assert.Equal(0, p.Planner.SequenceCount);
        Assert.Equal(3, m.GetPool(0).FreeCount);
    }

    [Fact]
    public void PlannerRefusalsAreTypedAndReleaseTheReservation()
    {
        var p = Make(Hybrid(40, 4));
        var grammar = Req(1, 4, 4) with { Grammar = new GrammarHandle(3) };
        Assert.Equal(RoostErrorCode.ProfileUnsupported, p.TryAdd(grammar)!.Code);
        Assert.Equal(0, p.Admission.Reserved(0));
        Assert.Equal(0, p.Admission.LiveRows);
        Assert.Null(p.TryAdd(Req(1, 4, 4))); // the id is free again
    }

    [Fact]
    public void ForkRefusalsAreTypedAndTheForkNeedsReservedRows()
    {
        var m = Hybrid(attentionBlocks: 100, slots: 2);
        var p = Make(m);
        Assert.Null(p.TryAdd(Req(1, 6, 8)));
        Assert.Equal(RoostErrorCode.InvalidOption, p.TryFork(1, Req(2, 1, 8))!.Code); // not resident yet
        Assert.Equal(1, p.Admission.LiveRows); // and the failed fork gave its row back

        p.ApplyResult(Simulate(p.Plan()!));
        Assert.Null(p.TryFork(1, Req(2, 1, 8)));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, p.TryFork(1, Req(3, 1, 8))!.Code); // 2 slots, 2 rows
        Assert.Equal(2, p.Planner.SequenceCount);
        p.Planner.CheckInvariants();
    }

    [Fact]
    public void AdmittedRequestsNeverNeedPreemptionAndEveryReservationComesBack()
    {
        // 40 requests of varied sizes against pools that hold about six at a time: admission refuses the rest and the
        // caller retries them as others finish. Worst-case reservation means the planner never runs out of blocks.
        var m = Hybrid(attentionBlocks: 36, slots: 6);
        var p = Make(m);
        var waiting = new Queue<Request>(Enumerable.Range(1, 40).Select(i => Req((ulong)i, 3 + (i * 7 % 9), 4 + (i * 5 % 9))));
        var refused = 0;
        var finished = 0;
        for (var step = 0; step < 5_000 && (waiting.Count > 0 || p.Planner.SequenceCount > 0); step++)
        {
            while (waiting.Count > 0)
            {
                var error = p.TryAdd(waiting.Peek());
                if (error is not null)
                {
                    Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, error.Code);
                    refused++;
                    break;
                }

                waiting.Dequeue();
            }

            var plan = p.Plan();
            if (plan is not null)
            {
                finished += p.ApplyResult(Simulate(plan)).Length;
            }

            Assert.True(p.Admission.Reserved(0) <= m.GetPool(0).Capacity);
            Assert.True(p.Admission.Reserved(1) <= m.GetPool(1).Capacity);
            p.Planner.CheckInvariants();
        }

        Assert.Equal(40, finished);
        Assert.True(refused > 0);
        Assert.Empty(p.DrainNotices()); // no preemption, no abort
        Assert.All(Enumerable.Range(0, 2), g => Assert.Equal(0, p.Admission.Reserved(g)));
        Assert.Equal(0, p.Admission.LiveRows);
        Assert.Equal(m.GetPool(0).Capacity, m.GetPool(0).FreeCount);
    }

    [Fact]
    public void CancelReleasesTheReservation()
    {
        var p = Make(Hybrid(20, 4));
        Assert.Null(p.TryAdd(Req(1, 4, 8)));
        Assert.True(p.Cancel(1));
        Assert.Equal(0, p.Admission.Reserved(0));
        Assert.False(p.Cancel(1));
    }

    [Fact]
    public void BranchesOfAFinishedParentKeepTheSharedBlocksReserved()
    {
        var m = Hybrid(attentionBlocks: 100, slots: 4);
        var p = Make(m);
        Assert.Null(p.TryAdd(Req(1, 9, 8)));
        p.ApplyResult(Simulate(p.Plan()!));
        Assert.Null(p.TryFork(1, Req(2, 1, 8)));
        var before = p.Admission.Reserved(0);
        Assert.True(p.Cancel(1));
        Assert.Equal(before - 3, p.Admission.Reserved(0)); // the parent's own tail (ceil(18 / 4) = 5 blocks, 2 of them shared), not the shared prompt
        Assert.True(p.Admission.Reserved(0) > 0);
        Assert.True(p.Cancel(2));
        Assert.Equal(0, p.Admission.Reserved(0));
    }
}

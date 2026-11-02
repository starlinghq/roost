using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using static Starlings.Roost.Engine.Tests.Scheduling.PlannerHarness;

namespace Starlings.Roost.Engine.Tests.Scheduling;

/// <summary>Async scheduling in the planner (R-032): plan N+1 built while step N runs, with the input patched by the worker.</summary>
public sealed class AsyncSchedulingTests
{
    private static readonly PlannerOptions Sync = new() { MaxTokensPerStep = 64, MaxSequencesPerStep = 8 };
    private static readonly PlannerOptions Async = Sync with { AsyncScheduling = true };

    private static StepOutcome Token(PlannedStep plan, ulong handle, int token, FinishReason finish = FinishReason.None) =>
        new(plan.StepId, [.. plan.Sequences.Select(s => new SequenceResult(
            s.SequenceHandle,
            s.SequenceHandle == handle && (s.Flags & SequenceOptions.Sample) != 0 ? [token] : [],
            s.SequenceHandle == handle ? finish : FinishReason.None))]);

    [Fact]
    public void PlanNPlusOneDecodesTheTokenStepNIsSampling()
    {
        var p = new StepPlanner(Hybrid(64, 4), Async);
        p.Add(Req(1, 6, maxTokens: 4));
        var prefill = p.Plan()!;
        Assert.Equal(0UL, prefill.InputStepId);

        var next = p.Plan()!; // the prefill's token is still in flight
        var row = Assert.Single(next.Sequences);
        Assert.Equal(SequenceOptions.Sample | SequenceOptions.SampledInput, row.Flags);
        Assert.Equal(prefill.StepId, next.InputStepId);
        Assert.Equal([SharedLayout.PendingTokenId], next.TokenIds.ToArray());
        Assert.Equal([6], next.Positions.ToArray());
        Assert.Equal(6u, row.PositionOffset);
        Assert.Equal(1UL, next.Sampling[0].PhiloxOffset); // one token generated, in flight
        Assert.Equal(3u, row.MaxNewTokensLeft);
        Assert.Equal(PlanOptions.None, next.Flags);
        Assert.True(p.TryGetInfo(1, out var info) && info.InFlightPlans == 2 && info.ComputedTokens == 7 && info.TokenCount == 6);

        // A synchronous planner waits instead.
        var s = new StepPlanner(Hybrid(64, 4), Sync);
        s.Add(Req(1, 6, maxTokens: 4));
        Assert.NotNull(s.Plan());
        Assert.Null(s.Plan());
    }

    [Fact]
    public void TheChainedRowIsTheRowASynchronousPlannerBuildsOnceTheTokenReturns()
    {
        var a = new StepPlanner(Hybrid(64, 4), Async);
        var s = new StepPlanner(Hybrid(64, 4), Sync);
        a.Add(Req(1, 6, maxTokens: 4));
        s.Add(Req(1, 6, maxTokens: 4));
        var a1 = a.Plan()!;
        var a2 = a.Plan()!;
        var s1 = s.Plan()!;
        s.ApplyResult(Token(s1, 1, 77));
        var s2 = s.Plan()!;

        Assert.Equal(s1.Digest, a1.Digest);
        Assert.Equal(s2.Sequences[0] with { Flags = s2.Sequences[0].Flags | SequenceOptions.SampledInput }, a2.Sequences[0]);
        Assert.Equal(s2.Positions, a2.Positions);
        Assert.Equal(s2.Sampling, a2.Sampling);
        Assert.Equal(s2.BlockIds, a2.BlockIds);
        Assert.Equal([77], s2.TokenIds.ToArray());
        Assert.NotEqual(s2.Digest, a2.Digest);
        a.CheckInvariants();
    }

    [Fact]
    public void ChainsOnlyFromThePlanBuiltLastAndNeverPastTheLengthLimit()
    {
        var p = new StepPlanner(Hybrid(64, 4), Async);
        p.Add(Req(1, 6, maxTokens: 3));
        var p1 = p.Plan()!;
        var p2 = p.Plan()!;
        var p3 = p.Plan()!;
        Assert.Equal(p2.StepId, p3.InputStepId);
        Assert.Null(p.Plan()); // three tokens in flight: the limit
        Assert.Empty(p.ApplyResult(Token(p1, 1, 10)));
        Assert.Empty(p.ApplyResult(Token(p2, 1, 11)));
        Assert.Null(p.Plan());
        Assert.Equal([new SequenceFinished(1, FinishReason.Length)], p.ApplyResult(Token(p3, 1, 12)).ToArray());
        Assert.Equal(0, p.SequenceCount);

        // A sequence that was left out of the last plan falls back to the synchronous rule.
        var q = new StepPlanner(Hybrid(64, 4), Async with { MaxSequencesPerStep = 1 });
        q.Add(Req(1, 6, maxTokens: 5, priority: RequestPriority.High));
        q.Add(Req(2, 6, maxTokens: 5));
        var q1 = q.Plan()!; // 1's prefill
        var q2 = q.Plan()!; // 1 chains (it ranks first)
        Assert.Equal(1UL, q2.Sequences[0].SequenceHandle);
        q.ApplyResult(Token(q1, 1, 20));
        q.ApplyResult(Token(q2, 1, 21));
        var q3 = q.Plan()!; // 1 again, a synchronous row with its real token
        Assert.Equal(0UL, q3.InputStepId);
        Assert.Equal([21], q3.TokenIds.ToArray());
        q.CheckInvariants();
    }

    [Fact]
    public void AStopIdSampledInStepNFinishesTheSequenceAndItsRowInNPlusOneIsIgnored()
    {
        var blocks = Hybrid(64, 4);
        var p = new StepPlanner(blocks, Async);
        p.Add(Req(1, 6, maxTokens: 10));
        var p1 = p.Plan()!;
        var p2 = p.Plan()!;
        Assert.Equal([new SequenceFinished(1, FinishReason.Eos)], p.ApplyResult(Token(p1, 1, 30, FinishReason.Eos)).ToArray());
        Assert.Equal(0, p.SequenceCount);
        Assert.True(p.ParkedBlocks > 0); // plan 2 still names its blocks and slot
        Assert.Null(p.Plan());
        Assert.Empty(p.ApplyResult(Token(p2, 1, 31)));
        Assert.Equal(0, p.ParkedBlocks);
        Assert.Equal(blocks.GetPool(0).Capacity, blocks.GetPool(0).FreeCount);
        p.CheckInvariants();
    }

    [Fact]
    public void ARowThatSampledButReturnedNoTokenFailsItsSequence()
    {
        var p = new StepPlanner(Hybrid(64, 4), Async);
        p.Add(Req(1, 6));
        var p1 = p.Plan()!;
        var p2 = p.Plan()!;
        Assert.Equal([new SequenceFinished(1, FinishReason.Error)], p.ApplyResult(new StepOutcome(p1.StepId, [new SequenceResult(1, [], FinishReason.None)])).ToArray());
        Assert.Empty(p.ApplyResult(Token(p2, 1, 5)));
        Assert.Throws<InvalidOperationException>(() =>
        {
            var q = new StepPlanner(Hybrid(64, 4), Sync with { MaxTokensPerStep = 4 });
            q.Add(Req(2, 6));
            var chunk = q.Plan()!; // a prefill chunk that does not sample
            q.ApplyResult(new StepOutcome(chunk.StepId, [new SequenceResult(2, [9], FinishReason.None)]));
        });
    }

    [Fact]
    public void AWorkerRestartDropsTheTokensInFlightAndTheSequenceRecomputes()
    {
        var p = new StepPlanner(Hybrid(64, 4), Async);
        p.Add(Req(1, 6, maxTokens: 6));
        var p1 = p.Plan()!;
        p.ApplyResult(Token(p1, 1, 40));
        _ = p.Plan();
        _ = p.Plan();
        p.OnWorkerRestarted();
        p.CheckInvariants();
        var again = p.Plan()!;
        Assert.Equal(0UL, again.InputStepId);
        Assert.Equal(7u, again.Sequences[0].TokenCount); // prompt plus the one token that returned
        Assert.Equal(1UL, again.Sampling[0].PhiloxOffset);
    }

    // Random workloads: the patching simulator under sync and async scheduling, with a pool small enough to preempt.

    private static List<(int, Request)> Workload(int seed, int count, out BlockManager blocks)
    {
        var rng = new Random(seed);
        blocks = Hybrid(12 + rng.Next(30), 2 + rng.Next(6), blockSize: 4);
        var arrivals = new List<(int, Request)>();
        var at = 0;
        for (ulong i = 1; i <= (ulong)count; i++)
        {
            at += rng.Next(4);
            var priority = (RequestPriority)(rng.Next(3) - 1);
            DateTimeOffset? deadline = rng.Next(3) == 0 ? new DateTimeOffset(2026, 11, 1, 0, 0, 0, TimeSpan.Zero).AddSeconds(rng.Next(100)) : null;
            var req = Req(i, 1 + rng.Next(14), maxTokens: 1 + rng.Next(18), priority: priority, deadline: deadline, seed: (ulong)rng.Next());
            arrivals.Add((at, rng.Next(4) == 0 ? req with { StopTokenIds = [.. Enumerable.Range(0, 30).Select(k => 1 + ((k * 167) % 4999))] } : req));
        }

        return arrivals;
    }

    private static PlannerOptions Options(int seed) =>
        new() { MaxTokensPerStep = 8 + (seed % 3 * 8), MaxSequencesPerStep = 2 + (seed % 5), MaxChunkTokens = 3 + (seed % 7) };

    public static TheoryData<int> Seeds() => [.. Enumerable.Range(1, 40)];

    [Theory]
    [MemberData(nameof(Seeds))]
    public void AsyncGivesTheSyncTokenStreamUnderAPreemptionStorm(int seed)
    {
        var arrivals = Workload(seed, 24, out var b1);
        _ = Workload(seed, 24, out var b2);
        var options = Options(seed);
        var sync = AsyncHarness.Run(b1, options, arrivals, new PatchingSimulator(), maxInFlight: 1);
        var async = AsyncHarness.Run(b2, options with { AsyncScheduling = true }, arrivals, new PatchingSimulator(), maxInFlight: 2);

        Assert.Equal(sync.Outputs.Keys.Order(), async.Outputs.Keys.Order());
        foreach (var (id, tokens) in sync.Outputs)
        {
            Assert.Equal(tokens, async.Outputs[id]);
        }

        Assert.Equal(sync.Finished.OrderBy(f => f.Key), async.Finished.OrderBy(f => f.Key));
        Assert.Equal(0, sync.ChainedRows);
    }

    [Fact]
    public void TheRandomWorkloadsChainRowsPreemptAndHideTheStopLag()
    {
        int chained = 0, preemptions = 0, ignored = 0;
        foreach (var seed in Enumerable.Range(1, 40))
        {
            var arrivals = Workload(seed, 24, out var b1);
            _ = Workload(seed, 24, out var b2);
            var options = Options(seed);
            var sync = AsyncHarness.Run(b1, options, arrivals, new PatchingSimulator(), maxInFlight: 1);
            var async = AsyncHarness.Run(b2, options with { AsyncScheduling = true }, arrivals, new PatchingSimulator(), maxInFlight: 2);
            chained += async.ChainedRows;
            preemptions += async.Preemptions;
            ignored += async.IgnoredRows;
            Assert.Equal(0, sync.ChainedRows);
        }

        Assert.True(chained > 1000, $"chained rows {chained}");
        Assert.True(preemptions > 20, $"preemptions {preemptions}");
        Assert.True(ignored > 10, $"rows ignored after a stop {ignored}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(17)]
    public void AnAsyncRunIsDeterministicAndReplaysFromItsRecording(int seed)
    {
        var arrivals = Workload(seed, 20, out var b1);
        var options = new PlannerOptions { MaxTokensPerStep = 16, MaxSequencesPerStep = 4, MaxChunkTokens = 5, AsyncScheduling = true };
        _ = Workload(seed, 20, out var b2);
        _ = Workload(seed, 20, out var b3);
        var first = AsyncHarness.Run(b1, options, arrivals, new PatchingSimulator(), maxInFlight: 2);
        var second = AsyncHarness.Run(b2, options, arrivals, new PatchingSimulator(), maxInFlight: 2);
        Assert.Equal(first.Transcript, second.Transcript);
        Assert.True(first.ChainedRows > 0);

        // The recorded inputs (requests, plan calls and the worker's results) replay on a fresh planner to the same plans
        // without a worker. The text codec does not carry request stop ids, so the recording is replayed as objects.
        var replayed = new PlannerReplay(new StepPlanner(b3, options)).Run(first.Inputs);
        Assert.Equal(first.Transcript, replayed);
    }

    [Fact]
    public void AsyncPlansDigestTheirInputStep()
    {
        var p = new StepPlanner(Hybrid(64, 4), Async);
        p.Add(Req(1, 6));
        _ = p.Plan();
        var chained = p.Plan()!;
        Assert.Equal(64, chained.Digest.Length);
        var q = new StepPlanner(Hybrid(64, 4), Async);
        q.Add(Req(1, 6));
        _ = q.Plan();
        Assert.Equal(chained.Digest, q.Plan()!.Digest);
    }
}

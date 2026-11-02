using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Tests.Scheduling;

/// <summary>
/// Async scheduling (R-032) against the two tiny CI models on <see cref="Ops.CpuOps"/>: the token streams and finish reasons of
/// an async run (two plans in flight, decode inputs patched by the worker) equal those of a synchronous run (one plan in
/// flight), with a pool small enough to preempt; and the async run's recording replays to the same plans with no worker.
/// </summary>
public sealed class TinyModelAsyncTests
{
    private const int MaxTokens = 48;
    private const int MaxSequences = 6;

    private static BlockManager Blocks(bool hybrid, int attentionBlocks, int slots) =>
        new(TinyPlanWorker.StateGroups(hybrid), [.. TinyPlanWorker.StateGroups(hybrid).Select(g => new GroupCapacity(g.Index, g.Kind == StateGroupKind.Recurrent ? slots : attentionBlocks, 0))]);

    private static List<(int, Request)> Workload(int seed, int count, int vocab)
    {
        var rng = new Random(seed);
        var arrivals = new List<(int, Request)>();
        var at = 0;
        for (ulong i = 1; i <= (ulong)count; i++)
        {
            at += rng.Next(3);
            arrivals.Add((at, new Request
            {
                Id = new RequestId(i),
                PromptTokenIds = [.. Enumerable.Range(0, 3 + rng.Next(38)).Select(_ => rng.Next(vocab))],
                Sampling = new SamplingParameters { MaxTokens = 4 + rng.Next(20), Temperature = 0 },
                Priority = (RequestPriority)(rng.Next(3) - 1),
                StopTokenIds = [.. Enumerable.Range(0, 10).Select(_ => rng.Next(vocab))],
            }));
        }

        return arrivals;
    }

    private static (AsyncRun Sync, AsyncRun Async, TinyPlanWorker AsyncWorker) RunBoth(bool hybrid, int seed, int attentionBlocks, int slots)
    {
        var options = new PlannerOptions { MaxTokensPerStep = MaxTokens, MaxSequencesPerStep = MaxSequences, MaxChunkTokens = 24 };
        using var syncWorker = new TinyPlanWorker(hybrid, attentionBlocks, slots, MaxTokens, MaxSequences);
        var arrivals = Workload(seed, 10, syncWorker.Vocab);
        var sync = AsyncHarness.Run(Blocks(hybrid, attentionBlocks, slots), options, arrivals, syncWorker, maxInFlight: 1);
        var asyncWorker = new TinyPlanWorker(hybrid, attentionBlocks, slots, MaxTokens, MaxSequences);
        var async = AsyncHarness.Run(Blocks(hybrid, attentionBlocks, slots), options with { AsyncScheduling = true }, arrivals, asyncWorker, maxInFlight: 2);
        return (sync, async, asyncWorker);
    }

    private static void AssertSameStreams(AsyncRun sync, AsyncRun async)
    {
        Assert.Equal(sync.Outputs.Keys.Order(), async.Outputs.Keys.Order());
        foreach (var (id, tokens) in sync.Outputs)
        {
            Assert.Equal(tokens, async.Outputs[id]);
        }

        Assert.Equal(sync.Finished.OrderBy(f => f.Key), async.Finished.OrderBy(f => f.Key));
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(false, 2)]
    [InlineData(true, 1)]
    [InlineData(true, 2)]
    public void AsyncGivesTheSyncTokenStreams(bool hybrid, int seed)
    {
        var (sync, async, worker) = RunBoth(hybrid, seed, attentionBlocks: 64, slots: 8);
        using (worker)
        {
            AssertSameStreams(sync, async);
            Assert.True(async.ChainedRows > 20, $"chained rows {async.ChainedRows}");
            Assert.Equal(async.ChainedRows, worker.Patched);
            Assert.Contains(sync.Finished.Values, r => r == FinishReason.Eos);
            Assert.Contains(sync.Finished.Values, r => r == FinishReason.Length);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void APreemptionStormKeepsTheStreamsIdentical(bool hybrid)
    {
        // 6 blocks of 16 tokens (or 3 recurrent slots) for 10 requests of up to 63 tokens each.
        var (sync, async, worker) = RunBoth(hybrid, 7, attentionBlocks: 6, slots: 3);
        using (worker)
        {
            AssertSameStreams(sync, async);
            Assert.True(async.Preemptions > 0 && sync.Preemptions > 0, $"preemptions sync {sync.Preemptions} async {async.Preemptions}");
            Assert.True(async.ChainedRows > 0);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheRecordedAsyncRunReplaysToTheSamePlans(bool hybrid)
    {
        var (_, async, worker) = RunBoth(hybrid, 3, attentionBlocks: 12, slots: 4);
        worker.Dispose();
        var options = new PlannerOptions { MaxTokensPerStep = MaxTokens, MaxSequencesPerStep = MaxSequences, MaxChunkTokens = 24, AsyncScheduling = true };
        var replayed = new PlannerReplay(new StepPlanner(Blocks(hybrid, 12, 4), options)).Run(async.Inputs);
        Assert.Equal(async.Transcript, replayed);
        Assert.Contains(async.Transcript, l => l.StartsWith("plan ", StringComparison.Ordinal) && !l.EndsWith("none", StringComparison.Ordinal));
    }

    [Fact]
    public void AWorkerThatDidNotRunTheInputStepRefusesThePlan()
    {
        var options = new PlannerOptions { MaxTokensPerStep = MaxTokens, MaxSequencesPerStep = MaxSequences, AsyncScheduling = true };
        var planner = new StepPlanner(Blocks(false, 8, 0), options);
        using var worker = new TinyPlanWorker(false, 8, 0, MaxTokens, MaxSequences);
        planner.Add(new Request { Id = new RequestId(1), PromptTokenIds = [1, 2, 3], Sampling = new SamplingParameters { MaxTokens = 4, Temperature = 0 } });
        _ = planner.Plan(); // never reaches the worker
        var chained = planner.Plan()!;
        Assert.Equal(1UL, chained.InputStepId);
        Assert.Throws<InvalidOperationException>(() => worker.Execute(chained));
        Assert.Equal(SequenceOptions.SampledInput | SequenceOptions.Sample, chained.Sequences[0].Flags);
    }
}

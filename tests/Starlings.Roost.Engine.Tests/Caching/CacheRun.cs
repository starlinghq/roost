using System.Collections.Immutable;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Caching;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Engine.Tests.Blocks;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Tests.Caching;

internal sealed record CacheRunResult(
    Dictionary<ulong, List<int>> Outputs,
    long ComputedTokens,
    PrefixCacheStats? Stats,
    ImmutableArray<string> Transcript,
    int Restores,
    int SnapshotWrites,
    int Preemptions,
    IPlanWorker Worker);

/// <summary>Executes plans, in order, against real contents.</summary>
internal interface IPlanWorker
{
    long ComputedTokens { get; }

    StepOutcome Execute(PlannedStep plan);
}

/// <summary>Runs requests, in waves, through a planner and a <see cref="ContentWorker"/>, checking every invariant after every input.</summary>
internal static class CacheRun
{
    public const int BlockSize = 4;

    public static BlockManager Attention(int blocks, int blockSize = BlockSize) =>
        new([TestGroups.Full(0, blockSize)], [new GroupCapacity(0, blocks, 0)]);

    public static BlockManager Hybrid(int blocks, int slots, int snapshotSlots, int blockSize = BlockSize) =>
        new([TestGroups.Full(0, blockSize), TestGroups.Recurrent(1, snapshotSlots: snapshotSlots)],
            [new GroupCapacity(0, blocks, 0), new GroupCapacity(1, slots, snapshotSlots)]);

    public static Request Req(ulong id, IEnumerable<int> prompt, int maxTokens = 6, ulong lora = 0) => new()
    {
        Id = new RequestId(id),
        PromptTokenIds = [.. prompt],
        Sampling = new SamplingParameters { MaxTokens = maxTokens, Temperature = 0 },
        Lora = new LoraHandle(lora),
    };

    /// <summary>Tokens <paramref name="from"/>..<paramref name="from"/>+count-1, so two prompts share exactly the tokens they share.</summary>
    public static IEnumerable<int> Seq(int from, int count) => Enumerable.Range(from, count).Select(t => (t % 4000) + 1);

    /// <summary>
    /// Runs <paramref name="parent"/> until it is decoding, forks it into <paramref name="child"/>, finishes both, then runs
    /// <paramref name="later"/> (which may share the parent's prefix). Every invariant is checked after every input.
    /// </summary>
    public static CacheRunResult RunWithFork(Func<BlockManager> makeBlocks, PlannerOptions options, Request parent, Request child, Request[] later, int maxInFlight)
    {
        var blocks = makeBlocks();
        var planner = new StepPlanner(blocks, options);
        var replay = new PlannerReplay(planner);
        var worker = new ContentWorker(blocks);
        var outputs = new Dictionary<ulong, List<int>>();
        int restores = 0, writes = 0;
        var forked = false;
        replay.Apply(new PlannerInput.Add(parent));
        var pending = new Queue<StepOutcome>();
        var added = false;
        for (var guard = 0; guard < 100_000; guard++)
        {
            if (!forked && planner.TryGetInfo(parent.Id.Value, out var info) && info.Resident && info.ComputedTokens >= parent.PromptTokenIds.Length && info.ComputedTokens < info.TokenCount)
            {
                replay.Apply(new PlannerInput.Fork(parent.Id.Value, child));
                forked = replay.Transcript.Any(l => l.StartsWith($"fork {parent.Id.Value} {child.Id.Value} Forked", StringComparison.Ordinal));
            }

            PlannedStep? plan = null;
            if (pending.Count < maxInFlight)
            {
                plan = replay.Apply(new PlannerInput.Plan());
                if (plan is not null)
                {
                    restores += plan.CacheOps.Count(o => o.Kind == CacheOpKind.SnapshotRestore);
                    writes += plan.CacheOps.Count(o => o.Kind == CacheOpKind.SnapshotWrite);
                    pending.Enqueue(worker.Execute(plan));
                }
            }

            if (plan is null)
            {
                if (pending.Count == 0)
                {
                    if (forked && !added)
                    {
                        added = true;
                        foreach (var r in later)
                        {
                            replay.Apply(new PlannerInput.Add(r));
                        }

                        continue;
                    }

                    break;
                }

                var outcome = pending.Dequeue();
                foreach (var row in outcome.Sequences.Where(x => !x.Tokens.IsDefaultOrEmpty))
                {
                    if (!outputs.TryGetValue(row.Handle, out var list))
                    {
                        outputs[row.Handle] = list = [];
                    }

                    list.Add(row.Tokens[0]);
                }

                replay.Apply(new PlannerInput.Result(outcome));
            }
        }

        Assert.True(forked, "the fork never happened");
        Assert.Equal(0, planner.SequenceCount);
        return new CacheRunResult(outputs, worker.ComputedTokens, planner.PrefixCache?.Stats, replay.Transcript, restores, writes,
            replay.Transcript.Count(l => l.Contains("Preempted", StringComparison.Ordinal)), worker);
    }

    public static CacheRunResult Run(Func<BlockManager> makeBlocks, PlannerOptions options, IReadOnlyList<Request[]> waves, int maxInFlight = 1, Func<BlockManager, IPlanWorker>? makeWorker = null)
    {
        var blocks = makeBlocks();
        var planner = new StepPlanner(blocks, options);
        var replay = new PlannerReplay(planner);
        var worker = makeWorker is null ? new ContentWorker(blocks) : makeWorker(blocks);
        var outputs = new Dictionary<ulong, List<int>>();
        int restores = 0, writes = 0;
        foreach (var wave in waves)
        {
            foreach (var r in wave)
            {
                replay.Apply(new PlannerInput.Add(r));
            }

            var pending = new Queue<StepOutcome>();
            for (var guard = 0; guard < 100_000; guard++)
            {
                PlannedStep? plan = null;
                if (pending.Count < maxInFlight)
                {
                    plan = replay.Apply(new PlannerInput.Plan());
                    if (plan is not null)
                    {
                        restores += plan.CacheOps.Count(o => o.Kind == CacheOpKind.SnapshotRestore);
                        writes += plan.CacheOps.Count(o => o.Kind == CacheOpKind.SnapshotWrite);
                        pending.Enqueue(worker.Execute(plan));
                    }
                }

                if (plan is null)
                {
                    if (pending.Count == 0)
                    {
                        break;
                    }

                    var outcome = pending.Dequeue();
                    foreach (var row in outcome.Sequences.Where(x => !x.Tokens.IsDefaultOrEmpty))
                    {
                        if (!outputs.TryGetValue(row.Handle, out var list))
                        {
                            outputs[row.Handle] = list = [];
                        }

                        list.Add(row.Tokens[0]);
                    }

                    replay.Apply(new PlannerInput.Result(outcome));
                }
            }

            Assert.Equal(0, planner.SequenceCount);
        }

        (worker as IDisposable)?.Dispose();
        return new CacheRunResult(
            outputs,
            worker.ComputedTokens,
            planner.PrefixCache?.Stats,
            replay.Transcript,
            restores,
            writes,
            replay.Transcript.Count(l => l.Contains("Preempted", StringComparison.Ordinal)),
            worker);
    }
}

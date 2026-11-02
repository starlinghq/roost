using System.Collections.Immutable;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Caching;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.State;

namespace Starlings.Roost.EngineGates.Tests.Support;

/// <summary>An engine configuration: the worker's pools and the planner's options.</summary>
/// <param name="AttentionBlocks">Blocks in the attention pool.</param>
/// <param name="RecurrentSlots">Slots in the recurrent pool (hybrid only).</param>
/// <param name="SnapshotSlots">Slots in the snapshot pool (hybrid only; the prefix cache writes snapshots there).</param>
/// <param name="Planner">The planner's options.</param>
/// <param name="MaxInFlight">Plans issued before the oldest result is applied (1: strictly one at a time).</param>
internal sealed record EngineSetup(int AttentionBlocks, int RecurrentSlots, int SnapshotSlots, PlannerOptions Planner, int MaxInFlight = 1)
{
    /// <summary>A roomy single-request engine: every prompt is prefilled in one call, no cache.</summary>
    public static EngineSetup OneCall => new(64, 4, 0, new PlannerOptions { MaxTokensPerStep = 256, MaxSequencesPerStep = 4 });
}

/// <summary>What one engine run produced.</summary>
/// <param name="Tokens">Generated tokens per request id.</param>
/// <param name="Logits">The float32 logits row that selected each generated token, per request id, in order.</param>
/// <param name="Preemptions">Preemption notices the planner raised.</param>
/// <param name="SnapshotRestores">Snapshot restore ops the worker ran.</param>
/// <param name="SnapshotWrites">Snapshot write ops the worker ran.</param>
/// <param name="Cache">Prefix-cache counters, when the cache is on.</param>
/// <param name="ChunkSizes">Token counts of every prefill row (a row that did not end in a sample, or the first sample of a request).</param>
/// <param name="MaxRowsPerStep">The most rows any step carried.</param>
/// <param name="Steps">Steps run.</param>
internal sealed record EngineRun(
    Dictionary<ulong, List<int>> Tokens,
    Dictionary<ulong, List<float[]>> Logits,
    int Preemptions,
    int SnapshotRestores,
    int SnapshotWrites,
    PrefixCacheStats? Cache,
    List<int> ChunkSizes,
    int MaxRowsPerStep,
    int Steps);

/// <summary>
/// Drives the real engine planner (<see cref="StepPlanner"/> over a <see cref="BlockManager"/>, with its
/// <see cref="PrefixCache"/> when configured) against a <see cref="TinyCpuModel"/>: every plan's cache ops run on the pools,
/// then its forward, and the sampled tokens go back to the planner as the step result. Requests arrive in waves; a wave
/// is added once the previous one has drained. The planner's invariants are checked after every input.
/// </summary>
internal static class EngineRunner
{
    public static EngineRun Run(TinyProfile profile, EngineSetup setup, IReadOnlyList<IReadOnlyList<Request>> waves)
    {
        var groups = TinyCpuModel.GroupsOf(profile);
        var capacities = groups
            .Select(g => g.Kind == StateGroupKind.Recurrent
                ? new GroupCapacity(g.Index, setup.RecurrentSlots, setup.SnapshotSlots)
                : new GroupCapacity(g.Index, setup.AttentionBlocks, 0))
            .ToImmutableArray();
        var planner = new StepPlanner(new BlockManager(groups, capacities), setup.Planner);
        using var model = new TinyCpuModel(profile, setup.AttentionBlocks, setup.RecurrentSlots, setup.SnapshotSlots, setup.Planner.MaxTokensPerStep, setup.Planner.MaxSequencesPerStep);
        int attention = groups.First(g => g.Kind != StateGroupKind.Recurrent).Index;

        var tokens = new Dictionary<ulong, List<int>>();
        var logits = new Dictionary<ulong, List<float[]>>();
        var chunks = new List<int>();
        int preemptions = 0, restores = 0, writes = 0, maxRows = 0, steps = 0;
        foreach (var wave in waves)
        {
            foreach (var r in wave)
            {
                var added = planner.Add(r);
                Assert.True(added == AddResult.Added, $"request {r.Id.Value}: {added}");
                planner.CheckInvariants();
            }

            var pending = new Queue<StepOutcome>();
            for (int guard = 0; ; guard++)
            {
                Assert.True(guard < 100_000, "the engine did not drain");
                PlannedStep? plan = pending.Count < setup.MaxInFlight ? planner.Plan() : null;
                planner.CheckInvariants();
                if (plan is not null)
                {
                    steps++;
                    maxRows = Math.Max(maxRows, plan.Sequences.Length);
                    foreach (var op in plan.CacheOps)
                    {
                        restores += op.Kind == CacheOpKind.SnapshotRestore ? 1 : 0;
                        writes += op.Kind == CacheOpKind.SnapshotWrite ? 1 : 0;
                        model.Apply(op);
                    }

                    var rows = new List<ForwardRow>(plan.Sequences.Length);
                    for (int i = 0; i < plan.Sequences.Length; i++)
                    {
                        var s = plan.Sequences[i];
                        bool sample = (s.Flags & SequenceOptions.Sample) != 0;
                        rows.Add(new ForwardRow((int)s.TokenStart, (int)s.TokenCount, (int)s.PositionOffset, plan.BlocksOf(i, attention).ToArray(), s.RecurrentSlot, sample));
                        if (s.TokenCount > 1 || !tokens.ContainsKey(s.SequenceHandle))
                        {
                            chunks.Add((int)s.TokenCount);
                        }
                    }

                    var rowLogits = model.Forward(plan.TokenIds.AsSpan(), rows);
                    var results = ImmutableArray.CreateBuilder<SequenceResult>(rows.Count);
                    int next = 0;
                    for (int i = 0; i < rows.Count; i++)
                    {
                        ulong handle = plan.Sequences[i].SequenceHandle;
                        if (!rows[i].Sample)
                        {
                            results.Add(new SequenceResult(handle, [], FinishReason.None));
                            continue;
                        }

                        var row = rowLogits[next++];
                        int token = TinyCpuModel.Argmax(row);
                        results.Add(new SequenceResult(handle, [token], FinishReason.None));
                        Record(tokens, logits, handle, token, row);
                    }

                    pending.Enqueue(new StepOutcome(plan.StepId, results.MoveToImmutable()));
                    continue;
                }

                if (pending.Count == 0)
                {
                    break;
                }

                planner.ApplyResult(pending.Dequeue());
                planner.CheckInvariants();
                preemptions += planner.DrainNotices().Count(n => n.Kind == PlannerNoticeKind.Preempted);
            }

            preemptions += planner.DrainNotices().Count(n => n.Kind == PlannerNoticeKind.Preempted);
            Assert.Equal(0, planner.SequenceCount);
        }

        return new EngineRun(tokens, logits, preemptions, restores, writes, planner.PrefixCache?.Stats, chunks, maxRows, steps);
    }

    /// <summary>A greedy request.</summary>
    /// <param name="id">Request id (also the sequence handle).</param>
    /// <param name="prompt">Prompt ids.</param>
    /// <param name="maxTokens">Tokens to generate.</param>
    /// <returns>The request.</returns>
    public static Request Req(ulong id, IEnumerable<int> prompt, int maxTokens) => new()
    {
        Id = new RequestId(id),
        PromptTokenIds = [.. prompt],
        Sampling = new SamplingParameters { MaxTokens = maxTokens, Temperature = 0 },
    };

    private static void Record(Dictionary<ulong, List<int>> tokens, Dictionary<ulong, List<float[]>> logits, ulong handle, int token, float[] row)
    {
        // The planner keeps every sampled token, a preempted sequence's too (StepPlanner.ApplyResult), and recomputes
        // prompt plus output before it samples again, so each sample is the next new token.
        if (!tokens.TryGetValue(handle, out var list))
        {
            tokens[handle] = list = [];
            logits[handle] = [];
        }

        list.Add(token);
        logits[handle].Add(row);
    }
}

using System.Collections.Immutable;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Engine.Tests.Scheduling;

/// <summary>A worker that executes plans in order, as a C2 worker does (one step at a time, in step id order).</summary>
internal interface IOrderedWorker
{
    StepOutcome Execute(PlannedStep plan);
}

/// <summary>
/// The recorded worker for async scheduling (R-032): like <see cref="PlannerHarness.Simulate"/> the token a row samples is a
/// function of the sequence and the position, but here also of the row's last input token, so an input that was not patched
/// (or patched from the wrong step) changes the text. It patches <see cref="SequenceOptions.SampledInput"/> rows from the ids it
/// sampled in the step it ran last and refuses a plan that names another step, as <c>SampledInputPatcher</c> does on a worker.
/// Stop ids and the length limit are checked on the worker, as <c>StepResultWriter</c> does.
/// </summary>
internal sealed class PatchingSimulator : IOrderedWorker
{
    private readonly Dictionary<ulong, int> _sampled = [];
    private ulong _lastStep;

    public int Patched { get; private set; }

    public StepOutcome Execute(PlannedStep plan)
    {
        var tokens = plan.TokenIds.ToArray();
        if (plan.InputStepId != 0 && plan.InputStepId != _lastStep)
        {
            throw new InvalidOperationException($"Plan {plan.StepId} takes its inputs from step {plan.InputStepId}; the worker ran {_lastStep} last.");
        }

        foreach (var s in plan.Sequences)
        {
            if ((s.Flags & SequenceOptions.SampledInput) != 0)
            {
                var at = (int)(s.TokenStart + s.TokenCount - 1);
                Assert.Equal(SharedLayout.PendingTokenId, tokens[at]);
                tokens[at] = _sampled[s.SequenceHandle];
                Patched++;
            }
        }

        foreach (var t in tokens)
        {
            Assert.True(t > 0, "an unpatched placeholder reached the forward");
        }

        _sampled.Clear();
        var rows = ImmutableArray.CreateBuilder<SequenceResult>();
        for (var r = 0; r < plan.Sequences.Length; r++)
        {
            var s = plan.Sequences[r];
            if ((s.Flags & SequenceOptions.Sample) == 0)
            {
                rows.Add(new SequenceResult(s.SequenceHandle, [], FinishReason.None));
                continue;
            }

            var pos = s.PositionOffset + s.TokenCount;
            var input = (ulong)tokens[(int)(s.TokenStart + s.TokenCount - 1)];
            var token = (int)(((s.SequenceHandle * 2654435761UL) + (pos * 40503UL) + (input * 97UL)) % 4999UL) + 1;
            _sampled[s.SequenceHandle] = token;
            var stops = plan.StopTokenIds.AsSpan((int)s.StopStart, (int)s.StopCount);
            var finish = token % 40 == 0 || stops.Contains(token) ? FinishReason.Eos : s.MaxNewTokensLeft == 1 ? FinishReason.Length : FinishReason.None;
            rows.Add(new SequenceResult(s.SequenceHandle, [token], finish));
        }

        _lastStep = plan.StepId;
        return new StepOutcome(plan.StepId, rows.ToImmutable());
    }
}

/// <summary>What one driven run produced.</summary>
/// <param name="Outputs">Per request, the tokens the caller saw, in order.</param>
/// <param name="Finished">Per request, how it finished.</param>
/// <param name="Inputs">Every planner call, for replay.</param>
/// <param name="Transcript">The replay transcript.</param>
/// <param name="Plans">Plans built.</param>
/// <param name="ChainedRows">Rows marked <see cref="SequenceOptions.SampledInput"/>.</param>
/// <param name="Preemptions">Preemption notices.</param>
/// <param name="IgnoredRows">Rows of a result whose sequence had already finished (the stop-id lag).</param>
internal sealed record AsyncRun(
    Dictionary<ulong, List<int>> Outputs,
    Dictionary<ulong, FinishReason> Finished,
    List<PlannerInput> Inputs,
    ImmutableArray<string> Transcript,
    int Plans,
    int ChainedRows,
    int Preemptions,
    int IgnoredRows);

/// <summary>
/// Drives a planner the way an engine loop does: up to <c>maxInFlight</c> plans outstanding (2 is the double-buffered plan
/// ring), each executed by the worker in order and its result applied in order. A request arrives once the run has applied
/// <c>arrivals[i].AfterResults</c> results. The caller sees a row's token only while its sequence is live, which is how the
/// frontend hides the one-step stop lag.
/// </summary>
internal static class AsyncHarness
{
    public static AsyncRun Run(
        BlockManager blocks, PlannerOptions options, IReadOnlyList<(int AfterResults, Request Request)> arrivals, IOrderedWorker worker, int maxInFlight, int maxSteps = 200_000)
    {
        var planner = new StepPlanner(blocks, options);
        var replay = new PlannerReplay(planner);
        var inputs = new List<PlannerInput>();
        var outputs = new Dictionary<ulong, List<int>>();
        var finished = new Dictionary<ulong, FinishReason>();
        var pending = new Queue<StepOutcome>();
        int results = 0, next = 0, plans = 0, chained = 0, ignored = 0;

        void Apply(PlannerInput input)
        {
            inputs.Add(input);
            replay.Apply(input);
        }

        for (var guard = 0; guard < maxSteps; guard++)
        {
            while (next < arrivals.Count && arrivals[next].AfterResults <= results)
            {
                Apply(new PlannerInput.Add(arrivals[next++].Request));
            }

            PlannedStep? plan = null;
            if (pending.Count < maxInFlight)
            {
                inputs.Add(new PlannerInput.Plan());
                plan = replay.Apply(inputs[^1]);
                if (plan is not null)
                {
                    plans++;
                    chained += plan.Sequences.Count(s => (s.Flags & SequenceOptions.SampledInput) != 0);
                    pending.Enqueue(worker.Execute(plan));
                }
            }

            if (plan is not null)
            {
                continue;
            }

            if (pending.Count == 0)
            {
                if (next < arrivals.Count)
                {
                    results = Math.Max(results, arrivals[next].AfterResults); // idle: the next arrival comes now
                    continue;
                }

                Assert.Equal(0, planner.SequenceCount);
                Assert.Equal(0, planner.ParkedBlocks);
                return new AsyncRun(outputs, finished, inputs, replay.Transcript, plans, chained,
                    replay.Transcript.Count(l => l.Contains("Preempted", StringComparison.Ordinal)), ignored);
            }

            var outcome = pending.Dequeue();
            foreach (var row in outcome.Sequences)
            {
                if (!planner.TryGetInfo(row.Handle, out _))
                {
                    ignored += row.Tokens.IsDefaultOrEmpty ? 0 : 1;
                    continue;
                }

                if (!row.Tokens.IsDefaultOrEmpty)
                {
                    if (!outputs.TryGetValue(row.Handle, out var list))
                    {
                        outputs[row.Handle] = list = [];
                    }

                    list.Add(row.Tokens[0]);
                }
            }

            Apply(new PlannerInput.Result(outcome));
            foreach (var f in FinishedIn(replay.Transcript.Last(l => l.StartsWith("result ", StringComparison.Ordinal))))
            {
                finished[f.Handle] = f.Reason;
            }

            results++;
        }

        throw new InvalidOperationException("The run did not drain.");
    }

    /// <summary>Reads the finished list of a transcript line <c>result N finished=[h:Reason,...]</c>.</summary>
    private static IEnumerable<SequenceFinished> FinishedIn(string line)
    {
        var list = line[(line.IndexOf('[', StringComparison.Ordinal) + 1)..^1];
        foreach (var item in list.Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            var f = item.Split(':');
            yield return new SequenceFinished(ulong.Parse(f[0], System.Globalization.CultureInfo.InvariantCulture), Enum.Parse<FinishReason>(f[1]));
        }
    }
}

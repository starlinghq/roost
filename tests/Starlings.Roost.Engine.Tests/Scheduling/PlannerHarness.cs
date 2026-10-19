using System.Collections.Immutable;
using System.Globalization;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Engine.Tests.Blocks;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Tests.Scheduling;

/// <summary>Builders, a deterministic simulated worker, and the text codec for recorded runs.</summary>
internal static class PlannerHarness
{
    /// <summary>Full-attention group (block size 4) plus a recurrent group, sized explicitly.</summary>
    public static BlockManager Hybrid(int attentionBlocks, int slots, int blockSize = 4)
    {
        ImmutableArray<StateGroupDescriptor> groups = [TestGroups.Full(0, blockSize), TestGroups.Recurrent(1, snapshotSlots: 0)];
        return new BlockManager(groups, [new GroupCapacity(0, attentionBlocks, 0), new GroupCapacity(1, slots, 0)]);
    }

    public static Request Req(ulong id, int promptLength, int? maxTokens = 8, RequestPriority priority = RequestPriority.Normal, DateTimeOffset? deadline = null, ulong? seed = null) => new()
    {
        Id = new RequestId(id),
        PromptTokenIds = [.. Enumerable.Range(0, promptLength).Select(i => (int)(((id * 31) + (ulong)i) % 5000) + 1)],
        Sampling = new SamplingParameters { MaxTokens = maxTokens, Temperature = 0.7f, Seed = seed },
        Priority = priority,
        Deadline = deadline,
    };

    /// <summary>
    /// The recorded worker: the token a row samples is a function of its handle and the position it samples at, so any
    /// schedule (preempted or not, chunked or not) must produce the same text. About one in 40 tokens is an EOS.
    /// </summary>
    public static StepOutcome Simulate(PlannedStep plan)
    {
        var rows = ImmutableArray.CreateBuilder<SequenceResult>();
        foreach (var s in plan.Sequences)
        {
            if ((s.Flags & SequenceOptions.Sample) == 0)
            {
                rows.Add(new SequenceResult(s.SequenceHandle, [], FinishReason.None));
                continue;
            }

            var pos = s.PositionOffset + s.TokenCount;
            var token = (int)(((s.SequenceHandle * 2654435761UL) + (pos * 40503UL)) % 4999UL) + 1;
            rows.Add(new SequenceResult(s.SequenceHandle, [token], token % 40 == 0 ? FinishReason.Eos : FinishReason.None));
        }

        return new StepOutcome(plan.StepId, rows.ToImmutable());
    }

    /// <summary>Drives a planner with the simulated worker, keeping up to <paramref name="maxInFlight"/> plans uncompleted.</summary>
    public static Dictionary<ulong, List<int>> Drain(PlannerReplay replay, List<PlannerInput> log, Dictionary<ulong, List<int>> outputs, int maxInFlight, int maxSteps = 100_000)
    {
        var pending = new Queue<StepOutcome>();
        for (var step = 0; step < maxSteps; step++)
        {
            PlannedStep? plan = null;
            if (pending.Count < maxInFlight)
            {
                log.Add(new PlannerInput.Plan());
                plan = replay.Apply(log[^1]);
                if (plan is not null)
                {
                    pending.Enqueue(Simulate(plan));
                }
            }

            if (plan is null)
            {
                if (pending.Count == 0)
                {
                    return outputs;
                }

                Complete(replay, log, outputs, pending.Dequeue());
            }
        }

        throw new InvalidOperationException("The planner did not drain.");
    }

    public static void Complete(PlannerReplay replay, List<PlannerInput> log, Dictionary<ulong, List<int>> outputs, StepOutcome outcome)
    {
        foreach (var r in outcome.Sequences)
        {
            if (!r.Tokens.IsDefaultOrEmpty)
            {
                if (!outputs.TryGetValue(r.Handle, out var list))
                {
                    outputs[r.Handle] = list = [];
                }

                list.Add(r.Tokens[0]);
            }
        }

        log.Add(new PlannerInput.Result(outcome));
        replay.Apply(log[^1]);
    }

    // Text codec: one input per line.
    public static string Encode(IEnumerable<PlannerInput> inputs) => string.Join('\n', inputs.Select(EncodeOne)) + "\n";

    private static string EncodeOne(PlannerInput input) => input switch
    {
        PlannerInput.Add a => "add " + EncodeRequest(a.Request),
        PlannerInput.Fork f => $"fork {f.Parent} " + EncodeRequest(f.Branch),
        PlannerInput.Cancel c => $"cancel {c.Handle}",
        PlannerInput.Plan => "plan",
        PlannerInput.Restart => "restart",
        PlannerInput.Result r => $"result {r.Outcome.StepId} " + string.Join(';', r.Outcome.Sequences.Select(s =>
            $"{s.Handle}:{(int)s.Finish}:{(s.Tokens.IsDefaultOrEmpty ? "-" : s.Tokens[0].ToString(CultureInfo.InvariantCulture))}")),
        _ => throw new ArgumentException("Unknown input."),
    };

    private static string EncodeRequest(Request r) => string.Create(CultureInfo.InvariantCulture,
        $"{r.Id.Value} {(int)r.Priority} {(r.Deadline is { } d ? d.UtcTicks.ToString(CultureInfo.InvariantCulture) : "-")} {(r.Sampling.MaxTokens is { } m ? m.ToString(CultureInfo.InvariantCulture) : "-")} {(r.Sampling.Seed is { } s ? s.ToString(CultureInfo.InvariantCulture) : "-")} {string.Join(',', r.PromptTokenIds)}");

    public static List<PlannerInput> Decode(string text)
    {
        var inputs = new List<PlannerInput>();
        foreach (var line in text.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var p = line.Split(' ');
            inputs.Add(p[0] switch
            {
                "add" => new PlannerInput.Add(DecodeRequest(p, 1)),
                "fork" => new PlannerInput.Fork(ulong.Parse(p[1], CultureInfo.InvariantCulture), DecodeRequest(p, 2)),
                "cancel" => new PlannerInput.Cancel(ulong.Parse(p[1], CultureInfo.InvariantCulture)),
                "plan" => new PlannerInput.Plan(),
                "restart" => new PlannerInput.Restart(),
                "result" => new PlannerInput.Result(new StepOutcome(ulong.Parse(p[1], CultureInfo.InvariantCulture),
                    [.. (p.Length > 2 ? p[2] : string.Empty).Split(';', StringSplitOptions.RemoveEmptyEntries).Select(DecodeResult)])),
                _ => throw new FormatException(line),
            });
        }

        return inputs;
    }

    private static SequenceResult DecodeResult(string s)
    {
        var f = s.Split(':');
        return new SequenceResult(ulong.Parse(f[0], CultureInfo.InvariantCulture), f[2] == "-" ? [] : [int.Parse(f[2], CultureInfo.InvariantCulture)], (FinishReason)int.Parse(f[1], CultureInfo.InvariantCulture));
    }

    private static Request DecodeRequest(string[] p, int at) => new()
    {
        Id = new RequestId(ulong.Parse(p[at], CultureInfo.InvariantCulture)),
        Priority = (RequestPriority)int.Parse(p[at + 1], CultureInfo.InvariantCulture),
        Deadline = p[at + 2] == "-" ? null : new DateTimeOffset(long.Parse(p[at + 2], CultureInfo.InvariantCulture), TimeSpan.Zero),
        Sampling = new SamplingParameters
        {
            MaxTokens = p[at + 3] == "-" ? null : int.Parse(p[at + 3], CultureInfo.InvariantCulture),
            Seed = p[at + 4] == "-" ? null : ulong.Parse(p[at + 4], CultureInfo.InvariantCulture),
            Temperature = 0.7f,
        },
        PromptTokenIds = [.. p[at + 5].Split(',', StringSplitOptions.RemoveEmptyEntries).Select(x => int.Parse(x, CultureInfo.InvariantCulture))],
    };
}

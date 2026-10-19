using System.Runtime.CompilerServices;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Requests;
using static Starlings.Roost.Engine.Tests.Scheduling.PlannerHarness;

namespace Starlings.Roost.Engine.Tests.Scheduling;

/// <summary>
/// Replay tests (R-029): a run is a list of planner inputs, including the results the recorded worker returned. Replaying
/// the list on a fresh planner must give the same plans, byte for byte (SHA-256 digests in the transcript). The committed
/// recording in <c>Recordings/</c> pins the policy: a change to the planner's decisions changes the transcript and fails
/// here until the maintainer of the planner regenerates it with <c>ROOST_UPDATE_RECORDINGS=1</c> and reviews the diff.
/// </summary>
public sealed class PlannerReplayTests
{
    private static string SourceDirectory([CallerFilePath] string here = "") => Path.Combine(Path.GetDirectoryName(here)!, "Recordings");

    private static string OutputDirectory => Path.Combine(AppContext.BaseDirectory, "Scheduling", "Recordings");

    /// <summary>A scripted mixed workload: priorities, deadlines, chunked prompts, a cancel, a fork and a worker restart, over pools small enough to preempt.</summary>
    private static (List<PlannerInput> Log, IReadOnlyList<string> Transcript) RunMixedWorkload()
    {
        var planner = new StepPlanner(Hybrid(attentionBlocks: 30, slots: 4), new PlannerOptions { MaxTokensPerStep = 20, MaxSequencesPerStep = 4 });
        var replay = new PlannerReplay(planner);
        var log = new List<PlannerInput>();
        var outputs = new Dictionary<ulong, List<int>>();
        var t = new DateTimeOffset(2026, 10, 19, 9, 0, 0, TimeSpan.Zero);

        void Do(PlannerInput i)
        {
            log.Add(i);
            replay.Apply(i);
        }

        for (ulong i = 1; i <= 5; i++)
        {
            Do(new PlannerInput.Add(Req(i, 6 + (int)(i * 4), maxTokens: 12, priority: i == 1 ? RequestPriority.High : (RequestPriority)((int)(i % 3) - 1), deadline: i is 1 or 2 or 4 ? t.AddSeconds(i) : null)));
        }

        var pending = new Queue<StepOutcome>();
        var forked = false;
        for (var round = 0; round < 400; round++)
        {
            if (round == 6)
            {
                Do(new PlannerInput.Add(Req(6, 30, maxTokens: 10, priority: RequestPriority.High)));
            }

            if (round == 9)
            {
                Do(new PlannerInput.Cancel(3));
            }

            if (!forked && round is >= 2 and < 20)
            {
                Do(new PlannerInput.Fork(1, Req(7, 1, maxTokens: 8, seed: 5)));
                forked = replay.Transcript[^1].EndsWith("Forked", StringComparison.Ordinal);
            }

            if (round == 20)
            {
                Do(new PlannerInput.Restart());
                pending.Clear();
            }

            PlannedStep? plan = null;
            if (pending.Count < 2)
            {
                var plan1 = new PlannerInput.Plan();
                log.Add(plan1);
                plan = replay.Apply(plan1);
                if (plan is not null)
                {
                    pending.Enqueue(Simulate(plan));
                }
            }

            if (plan is null)
            {
                if (pending.Count == 0)
                {
                    break;
                }

                Complete(replay, log, outputs, pending.Dequeue());
            }
        }

        Assert.True(forked, "the script must fork once");
        Assert.Equal(0, planner.SequenceCount);
        return (log, replay.Transcript);
    }

    [Fact]
    public void ALiveRunReplaysToTheSameTranscript()
    {
        var (log, transcript) = RunMixedWorkload();
        Assert.Contains(transcript, l => l.Contains("Preempted", StringComparison.Ordinal));
        Assert.Contains(transcript, l => l == "restart");

        var fresh = new PlannerReplay(new StepPlanner(Hybrid(30, 4), new PlannerOptions { MaxTokensPerStep = 20, MaxSequencesPerStep = 4 }));
        Assert.Equal(transcript, fresh.Run(log));
    }

    [Fact]
    public void TheRecordingRoundTripsThroughTheTextCodec()
    {
        var (log, transcript) = RunMixedWorkload();
        var decoded = Decode(Encode(log));
        var fresh = new PlannerReplay(new StepPlanner(Hybrid(30, 4), new PlannerOptions { MaxTokensPerStep = 20, MaxSequencesPerStep = 4 }));
        Assert.Equal(transcript, fresh.Run(decoded));
    }

    [Fact]
    public void TheCommittedRecordingStillReplaysToTheCommittedPlans()
    {
        var (log, transcript) = RunMixedWorkload();
        if (Environment.GetEnvironmentVariable("ROOST_UPDATE_RECORDINGS") == "1")
        {
            File.WriteAllText(Path.Combine(SourceDirectory(), "mixed-workload.inputs.txt"), Encode(log));
            File.WriteAllText(Path.Combine(SourceDirectory(), "mixed-workload.plans.txt"), string.Join('\n', transcript) + "\n");
        }

        var inputsPath = Path.Combine(OutputDirectory, "mixed-workload.inputs.txt");
        var plansPath = Path.Combine(OutputDirectory, "mixed-workload.plans.txt");

        var recorded = Decode(File.ReadAllText(inputsPath));
        Assert.Equal(Encode(log), Encode(recorded)); // the simulated worker and the script still produce this recording

        var fresh = new PlannerReplay(new StepPlanner(Hybrid(30, 4), new PlannerOptions { MaxTokensPerStep = 20, MaxSequencesPerStep = 4 }));
        var replayed = fresh.Run(recorded);
        Assert.Equal(File.ReadAllText(plansPath).TrimEnd('\n').Split('\n'), replayed);
    }

    public static TheoryData<int> Seeds() => [.. Enumerable.Range(1, 60)];

    [Theory]
    [MemberData(nameof(Seeds))]
    public void RandomWorkloadsKeepEveryInvariantAndReplayIdentically(int seed) => _ = RunRandom(seed);

    [Fact]
    public void TheRandomWorkloadsReachPreemptionAbortForkAndRestart()
    {
        var all = Enumerable.Range(1, 60).SelectMany(RunRandom).ToList();
        Assert.Contains(all, l => l.Contains("Preempted", StringComparison.Ordinal));
        Assert.Contains(all, l => l.Contains("Aborted", StringComparison.Ordinal));
        Assert.Contains(all, l => l.EndsWith("Forked", StringComparison.Ordinal));
        Assert.Contains(all, l => l == "restart");
        Assert.Contains(all, l => l.Contains("ops=", StringComparison.Ordinal) && !l.Contains("ops=0", StringComparison.Ordinal));
        Assert.Contains(all, l => l.Contains("Length]", StringComparison.Ordinal) || l.Contains(":Length", StringComparison.Ordinal));
    }

    private static IReadOnlyList<string> RunRandom(int seed)
    {
        var rng = new Random(seed);
        var options = new PlannerOptions { MaxTokensPerStep = 8 + rng.Next(40), MaxSequencesPerStep = 1 + rng.Next(6) };
        var blocks = 10 + rng.Next(40);
        var slots = 2 + rng.Next(5);
        var planner = new StepPlanner(Hybrid(blocks, slots), options);
        var replay = new PlannerReplay(planner); // checks CheckInvariants after every input
        var log = new List<PlannerInput>();
        var outputs = new Dictionary<ulong, List<int>>();
        var pending = new Queue<StepOutcome>();
        ulong next = 1;
        var maxInFlight = 1 + rng.Next(3);

        void Do(PlannerInput i)
        {
            log.Add(i);
            replay.Apply(i);
        }

        for (var step = 0; step < 400; step++)
        {
            var roll = rng.Next(100);
            if (roll < 20 && next < 40)
            {
                Do(new PlannerInput.Add(Req(next++, 1 + rng.Next(30), maxTokens: 1 + rng.Next(12), priority: (RequestPriority)(rng.Next(3) - 1),
                    deadline: rng.Next(2) == 0 ? null : new DateTimeOffset(2026, 10, 19, 0, 0, rng.Next(60), TimeSpan.Zero))));
            }
            else if (roll < 24 && next > 1)
            {
                Do(new PlannerInput.Cancel((ulong)(1 + rng.Next((int)next))));
            }
            else if (roll < 28 && next > 1 && next < 40)
            {
                Do(new PlannerInput.Fork((ulong)(1 + rng.Next((int)next)), Req(next++, 1, maxTokens: 1 + rng.Next(8), seed: (ulong)rng.Next(100))));
            }
            else if (roll < 29)
            {
                Do(new PlannerInput.Restart());
                pending.Clear();
            }
            else if (roll < 65 && pending.Count < maxInFlight)
            {
                var plan = new PlannerInput.Plan();
                log.Add(plan);
                var made = replay.Apply(plan);
                if (made is not null)
                {
                    Assert.True(made.Sequences.Length <= options.MaxSequencesPerStep);
                    Assert.True(made.TokenIds.Length <= options.MaxTokensPerStep);
                    pending.Enqueue(Simulate(made));
                }
            }
            else if (pending.Count > 0)
            {
                Complete(replay, log, outputs, pending.Dequeue());
            }
        }

        while (pending.Count > 0)
        {
            Complete(replay, log, outputs, pending.Dequeue());
        }

        Drain(replay, log, outputs, maxInFlight);
        planner.CheckInvariants();
        Assert.Equal(0, planner.SequenceCount);
        Assert.Equal(0, planner.ParkedBlocks);

        var fresh = new PlannerReplay(new StepPlanner(Hybrid(blocks, slots), options));
        Assert.Equal(replay.Transcript, fresh.Run(Decode(Encode(log))));
        return replay.Transcript;
    }
}

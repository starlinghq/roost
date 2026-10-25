using System.Collections.Immutable;
using Starlings.Generation.Roost.Driver;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.Worker.Eager;

namespace Starlings.Generation.Roost.Tests;

/// <summary>The plan writer against the eager worker's reader, with several sequences per step (what M1 batching will send).</summary>
public sealed class PlanBufferTests
{
    private const int Rows = 3;

    [Fact]
    public void PlannerPlansReadBackThroughTheEagerReaderAndResultsReadBackThroughTheDriver()
    {
        var groups = RoostTinyHybrid.CreateStateGroups();
        var planner = new StepPlanner(
            new BlockManager(groups, [new GroupCapacity(0, 16, 0), new GroupCapacity(1, Rows, 0)]),
            new PlannerOptions { MaxTokensPerStep = 40, MaxSequencesPerStep = Rows, ModelStopTokenIds = [TestRig.StopId] });
        for (var i = 0; i < Rows; i++)
        {
            Assert.Equal(AddResult.Added, planner.Add(new Request
            {
                Id = new RequestId((ulong)(i + 1)),
                PromptTokenIds = [.. Enumerable.Range(10 * i, 17 + i)],
                Sampling = new SamplingParameters { Temperature = 0f, MaxTokens = 3 },
            }));
        }

        var limits = TestRig.Limits() with { MaxTokens = 40, MaxSequences = Rows, MaxSelected = Rows, RecurrentSlots = Rows, MaxKvBlocks = 16 };
        var buffer = new byte[PlanBuffer.SizeFor(Rows, 40, groups.Length, 16, 1)];
        var result = new byte[64 * 1024];
        var finished = 0;
        var steps = 0;
        while (finished < Rows && steps++ < 50)
        {
            var step = planner.Plan();
            Assert.NotNull(step);
            var written = PlanBuffer.Write(buffer, step);
            Assert.InRange(written, 1, buffer.Length);

            var parsed = StepPlanReader.Parse(buffer, limits);
            Assert.Equal(step.StepId, parsed.StepId);
            Assert.Equal(step.TokenIds, parsed.TokenIds);
            Assert.Equal(step.Positions, parsed.Positions);
            Assert.Equal(step.Sequences.Length, parsed.Sequences.Length);
            for (var r = 0; r < parsed.Sequences.Length; r++)
            {
                var p = parsed.Sequences[r];
                Assert.Equal(step.Sequences[r].SequenceHandle, p.Handle);
                Assert.Equal(step.BlocksOf(r, 0)[..p.Blocks.Length].ToArray(), p.Blocks);
                Assert.Equal([TestRig.StopId], parsed.StopIds.AsSpan(p.StopStart, p.StopCount).ToArray());
            }

            var emitted = parsed.Sequences.Where(static s => s.Sample).Select(static s => (int)(s.Handle * 7)).ToArray();
            int bytes = StepResultWriter.Write(result, parsed, emitted, default);
            var outcome = PlanBuffer.ReadResult(result.AsSpan(0, bytes), step.StepId);
            Assert.Equal(parsed.Sequences.Select(static s => s.Handle), outcome.Sequences.Select(static s => s.Handle));
            Assert.Equal(emitted, outcome.Sequences.SelectMany(static s => s.Tokens).ToArray());
            finished += planner.ApplyResult(outcome).Length;
        }

        Assert.Equal(Rows, finished);
        Assert.True(steps > 3);
    }

    [Fact]
    public void ATooSmallBufferAndAMismatchedResultAreRefused()
    {
        var planner = new StepPlanner(
            new BlockManager(RoostTinyHybrid.CreateStateGroups(), [new GroupCapacity(0, 4, 0), new GroupCapacity(1, 1, 0)]),
            new PlannerOptions { MaxTokensPerStep = 8, MaxSequencesPerStep = 1 });
        _ = planner.Add(new Request { Id = new RequestId(1), PromptTokenIds = [1, 2, 3], Sampling = new SamplingParameters { Temperature = 0f, MaxTokens = 1 } });
        var step = planner.Plan()!;

        Assert.Throws<InvalidOperationException>(() => PlanBuffer.Write(new byte[128], step));

        var buffer = new byte[4096];
        _ = PlanBuffer.Write(buffer, step);
        var parsed = StepPlanReader.Parse(buffer, TestRig.Limits());
        var result = new byte[4096];
        _ = StepResultWriter.Write(result, parsed, [5], default);
        Assert.Throws<InvalidDataException>(() => PlanBuffer.ReadResult(result, step.StepId + 1));
        result[0] ^= 0xFF;
        Assert.Throws<InvalidDataException>(() => PlanBuffer.ReadResult(result, step.StepId));
    }

    [Fact]
    public void SizeForBoundsTheLargestPlan()
    {
        var bound = PlanBuffer.SizeFor(1, TestRig.MaxTokensPerStep, 2, TestRig.AttentionBlocks, 1);
        Assert.True(bound < 4096, $"a 32-token, one-row plan should fit in a page; bound {bound}");
        Assert.Equal(0, bound % SharedLayout.Alignment);
    }
}

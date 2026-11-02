using Starlings.Roost.Plans;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Tests.Support;

namespace Starlings.Roost.Worker.Tests;

/// <summary>Async scheduling (R-032) on the worker: the reader's checks of sampled-input rows and the host-side patcher.</summary>
public sealed class SampledInputTests
{
    private static readonly PlanLimits Cuda = new(MaxTokens: 64, MaxSequences: 3, MaxKvBlocks: 12, MaxSelected: 3, Vocab: 1000, BlockSize: 16, AttentionBlocks: 12, RecurrentSlots: 3, AttentionGroup: 0, GroupCount: 2);
    private static readonly PlanLimits Cpu = Cuda with { SampledInputs = true };

    private static TestSequence Decode(ulong handle, int position, int[] blocks, int slot, bool sampledInput = true) =>
        new(handle, [sampledInput ? SharedLayout.PendingTokenId : 5], position, blocks, slot) { SampledInput = sampledInput };

    [Fact]
    public void TheCudaWorkerRefusesASampledInputRowAndSaysWhy()
    {
        var plan = PlanWriter.WriteWithInput(8, 7, 2, Decode(1, 20, [0, 1], 0));
        var e = Assert.Throws<PlanRejectedException>(() => StepPlanReader.Parse(plan, Cuda));
        Assert.Equal(PlanRejection.Unsupported, e.Reason);
        Assert.Contains("sampled-input", e.Message, StringComparison.Ordinal);
        Assert.Contains("stream F", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWorkerThatPatchesReadsTheRowWithItsPlaceholder()
    {
        var parsed = StepPlanReader.Parse(PlanWriter.WriteWithInput(8, 7, 2, Decode(1, 20, [0, 1], 0), new TestSequence(2, [9], 16, [2, 3], 1)), Cpu);
        Assert.Equal(7UL, parsed.InputStepId);
        Assert.True(parsed.Sequences[0].SampledInput);
        Assert.False(parsed.Sequences[1].SampledInput);
        Assert.Equal([SharedLayout.PendingTokenId, 9], parsed.TokenIds);
    }

    [Fact]
    public void ThePlaceholderIsStillOutOfRangeWithoutTheFlag()
    {
        var plan = PlanWriter.WriteWithInput(8, 0, 2, new TestSequence(1, [SharedLayout.PendingTokenId], 20, [0, 1], 0));
        Assert.Equal(PlanRejection.OutOfRange, Assert.Throws<PlanRejectedException>(() => StepPlanReader.Parse(plan, Cpu)).Reason);
    }

    public static TheoryData<string> MalformedCases() => ["prefill", "two tokens", "real id", "no input step", "input step not earlier", "input step without rows"];

    [Theory]
    [MemberData(nameof(MalformedCases))]
    public void RefusesAMalformedSampledInputRow(string name)
    {
        var plan = name switch
        {
            "prefill" => PlanWriter.WriteWithInput(8, 7, 2, new TestSequence(1, [SharedLayout.PendingTokenId], 0, [0], 0) { SampledInput = true }),
            "two tokens" => PlanWriter.WriteWithInput(8, 7, 2, new TestSequence(1, [4, SharedLayout.PendingTokenId], 20, [0, 1], 0) { SampledInput = true }),
            "real id" => PlanWriter.WriteWithInput(8, 7, 2, new TestSequence(1, [4], 20, [0, 1], 0) { SampledInput = true }),
            "no input step" => PlanWriter.WriteWithInput(8, 0, 2, Decode(1, 20, [0, 1], 0)),
            "input step not earlier" => PlanWriter.WriteWithInput(8, 8, 2, Decode(1, 20, [0, 1], 0)),
            "input step without rows" => PlanWriter.WriteWithInput(8, 7, 2, Decode(1, 20, [0, 1], 0, sampledInput: false)),
            _ => throw new ArgumentException(name),
        };
        Assert.Equal(PlanRejection.Malformed, Assert.Throws<PlanRejectedException>(() => StepPlanReader.Parse(plan, Cpu)).Reason);
    }

    [Fact]
    public void PatchesEachRowFromTheIdItsSequenceSampledInTheLastStep()
    {
        var patcher = new SampledInputPatcher(1000);
        var step7 = StepPlanReader.Parse(PlanWriter.Write(7, 2, new TestSequence(1, [3, 4, 5], 0, [0], 0), new TestSequence(2, [6, 7], 0, [2], 1), new TestSequence(3, [8], 0, [4], 2)), Cpu);
        patcher.Record(step7, [101, 202, 303]);
        Assert.Equal(7UL, patcher.LastStep);

        var step8 = StepPlanReader.Parse(PlanWriter.WriteWithInput(8, 7, 2, Decode(2, 2, [2], 1), new TestSequence(3, [9], 1, [4], 2), Decode(1, 3, [0], 0)), Cpu);
        patcher.Patch(step8);
        Assert.Equal([202, 9, 101], step8.TokenIds);
    }

    [Fact]
    public void RefusesAStaleOrMissingInputWithoutPatching()
    {
        var patcher = new SampledInputPatcher(1000);
        var step7 = StepPlanReader.Parse(PlanWriter.Write(7, 2, new TestSequence(1, [3], 0, [0], 0)), Cpu);
        var nine = StepPlanReader.Parse(PlanWriter.WriteWithInput(9, 8, 2, Decode(1, 1, [0], 0)), Cpu);

        // Nothing recorded yet.
        Assert.Equal(PlanRejection.OutOfRange, Assert.Throws<PlanRejectedException>(() => patcher.Patch(nine)).Reason);

        // Step 8 never ran on this worker: its ids would be step 7's, which are stale.
        patcher.Record(step7, [42]);
        Assert.Equal(PlanRejection.OutOfRange, Assert.Throws<PlanRejectedException>(() => patcher.Patch(nine)).Reason);
        Assert.Equal(SharedLayout.PendingTokenId, nine.TokenIds[0]);

        // A sequence that did not sample in the input step, next to one that did: nothing is patched.
        var eight = StepPlanReader.Parse(PlanWriter.WriteWithInput(8, 7, 2, Decode(1, 1, [0], 0), Decode(2, 1, [2], 1)), Cpu);
        Assert.Equal(PlanRejection.OutOfRange, Assert.Throws<PlanRejectedException>(() => patcher.Patch(eight)).Reason);
        Assert.Equal([SharedLayout.PendingTokenId, SharedLayout.PendingTokenId], eight.TokenIds);

        // After a failed step the ids are gone.
        patcher.Forget();
        var good = StepPlanReader.Parse(PlanWriter.WriteWithInput(8, 7, 2, Decode(1, 1, [0], 0)), Cpu);
        Assert.Throws<PlanRejectedException>(() => patcher.Patch(good));
    }

    [Fact]
    public void RecordChecksItsInput()
    {
        var patcher = new SampledInputPatcher(1000);
        var step = StepPlanReader.Parse(PlanWriter.Write(7, 2, new TestSequence(1, [3], 0, [0], 0), new TestSequence(2, [3], 0, [1], 1) { Sample = false }), Cpu);
        Assert.Throws<ArgumentException>(() => patcher.Record(step, [1, 2]));
        Assert.Throws<ArgumentException>(() => patcher.Record(step, [1000]));
        Assert.Equal(0UL, patcher.LastStep);
        patcher.Record(step, [999]);
        Assert.Equal(7UL, patcher.LastStep);
    }
}

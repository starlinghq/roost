using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Plans;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Tests.Support;

namespace Starlings.Roost.Worker.Tests;

public sealed class StepPlanReaderTests
{
    private static readonly PlanLimits Limits = new(MaxTokens: 64, MaxSequences: 2, MaxKvBlocks: 8, MaxSelected: 2, Vocab: 1000, BlockSize: 16, AttentionBlocks: 8, RecurrentSlots: 2, AttentionGroup: 0, GroupCount: 2);

    private static byte[] Plan(params TestSequence[] sequences) => PlanWriter.Write(7, 2, sequences);

    private static PlanRejection Rejection(byte[] plan)
    {
        var e = Assert.Throws<PlanRejectedException>(() => StepPlanReader.Parse(plan, Limits));
        return e.Reason;
    }

    [Fact]
    public void ReadsAPrefillAndADecodeSequence()
    {
        var plan = StepPlanReader.Parse(
            Plan(
                new TestSequence(11, [5, 6, 7, 8, 9], 0, [3], 0),
                new TestSequence(12, [4], 16, [0, 1], 1) { Stops = [2, 3], MaxNewTokensLeft = 1 }),
            Limits);
        Assert.Equal(7UL, plan.StepId);
        Assert.Equal([5, 6, 7, 8, 9, 4], plan.TokenIds);
        Assert.Equal([0, 1, 2, 3, 4, 16], plan.Positions);
        Assert.Equal(2, plan.Selected);
        Assert.Equal(3, plan.KvBlocks);
        Assert.Equal([3], plan.Sequences[0].Blocks);
        Assert.Equal(17, plan.Sequences[1].KvLength);
        Assert.Equal([0, 1], plan.Sequences[1].Blocks);
        Assert.Equal(1, plan.Sequences[1].RecurrentSlot);
        Assert.Equal(2, plan.Sequences[1].StopCount);
    }

    [Fact]
    public void KeepsOnlyTheBlocksTheKeysNeed()
    {
        // 5 keys need one block; the plan lists three (blocks held for the request's later growth).
        var plan = StepPlanReader.Parse(Plan(new TestSequence(1, [1, 2, 3, 4, 5], 0, [2, 3, 4], 0)), Limits);
        Assert.Equal([2], plan.Sequences[0].Blocks);
    }

    [Fact]
    public void RefusesATokenIdOutsideTheVocabulary()
    {
        Assert.Equal(PlanRejection.OutOfRange, Rejection(Plan(new TestSequence(1, [1000], 0, [0], 0))));
        Assert.Equal(PlanRejection.OutOfRange, Rejection(Plan(new TestSequence(1, [-1], 0, [0], 0))));
    }

    [Fact]
    public void RefusesABlockIdOutsideThePool()
    {
        Assert.Equal(PlanRejection.OutOfRange, Rejection(Plan(new TestSequence(1, [1], 0, [8], 0))));
        Assert.Equal(PlanRejection.OutOfRange, Rejection(Plan(new TestSequence(1, [1], 0, [-1], 0))));
    }

    [Fact]
    public void RefusesTooFewBlocksForTheKeys()
    {
        Assert.Equal(PlanRejection.OutOfRange, Rejection(Plan(new TestSequence(1, [1], 16, [0], 0))));
    }

    [Fact]
    public void RefusesARecurrentSlotOutsideThePool()
    {
        Assert.Equal(PlanRejection.OutOfRange, Rejection(Plan(new TestSequence(1, [1], 0, [0], 2))));
        Assert.Equal(PlanRejection.OutOfRange, Rejection(Plan(new TestSequence(1, [1], 0, [0], -1))));
    }

    [Fact]
    public void RefusesPositionsPastThePool()
    {
        Assert.Equal(PlanRejection.OutOfRange, Rejection(Plan(new TestSequence(1, [1], 8 * 16, [0, 1, 2, 3, 4, 5, 6, 7], 0))));
    }

    [Fact]
    public void RefusesAStepLargerThanTheWorker()
    {
        Assert.Equal(PlanRejection.TooLarge, Rejection(Plan(new TestSequence(1, new int[65], 0, [0, 1, 2, 3, 4], 0))));
        Assert.Equal(PlanRejection.TooLarge, Rejection(Plan(new TestSequence(1, [1], 0, [0], 0), new TestSequence(2, [1], 0, [1], 1), new TestSequence(3, [1], 0, [2], 1))));
    }

    [Fact]
    public void RefusesSamplingOtherThanGreedy()
    {
        var hot = new PlanSampling { Temperature = 0.8f, TopP = 1f, RepetitionPenalty = 1f };
        Assert.Equal(PlanRejection.Unsupported, Rejection(Plan(new TestSequence(1, [1], 0, [0], 0) { Sampling = hot })));
        var penalised = new PlanSampling { Flags = SamplingOptions.Greedy, RepetitionPenalty = 1.1f };
        Assert.Equal(PlanRejection.Unsupported, Rejection(Plan(new TestSequence(1, [1], 0, [0], 0) { Sampling = penalised })));
        var logprobs = new PlanSampling { Flags = SamplingOptions.Greedy, RepetitionPenalty = 1f, Logprobs = 5 };
        Assert.Equal(PlanRejection.Unsupported, Rejection(Plan(new TestSequence(1, [1], 0, [0], 0) { Sampling = logprobs })));
    }

    [Fact]
    public void AcceptsAChunkThatDoesNotSample()
    {
        var plan = StepPlanReader.Parse(Plan(new TestSequence(1, [1, 2], 0, [0], 0, Sample: false)), Limits);
        Assert.Equal(0, plan.Selected);
    }

    [Fact]
    public void RefusesNonConsecutivePositions()
    {
        var bytes = Plan(new TestSequence(1, [1, 2, 3], 0, [0], 0));
        var header = MemoryMarshal.Read<StepPlanHeader>(bytes);
        MemoryMarshal.Write(bytes.AsSpan((int)header.Positions.Offset + 4), 5);
        Assert.Equal(PlanRejection.Malformed, Rejection(bytes));
    }

    [Fact]
    public void RefusesWorkItDoesNotDo()
    {
        var bytes = Plan(new TestSequence(1, [1], 0, [0], 0));
        var header = MemoryMarshal.Read<StepPlanHeader>(bytes);
        header.CacheOps = new TableRef { Offset = header.TotalBytes, Count = 1, Stride = (uint)Unsafe.SizeOf<PlanCacheOp>() };
        MemoryMarshal.Write(bytes, header);
        Assert.Equal(PlanRejection.Unsupported, Rejection(bytes));
    }

    [Fact]
    public void RefusesAMalformedBuffer()
    {
        var good = Plan(new TestSequence(1, [1], 0, [0], 0));
        Assert.Equal(PlanRejection.Malformed, Rejection(good[..16]));

        var badMagic = (byte[])good.Clone();
        badMagic[0] ^= 0xFF;
        Assert.Equal(PlanRejection.Malformed, Rejection(badMagic));

        var truncated = good[..^70];
        var header = MemoryMarshal.Read<StepPlanHeader>(truncated);
        header.TotalBytes = (ulong)truncated.Length;
        MemoryMarshal.Write(truncated, header);
        Assert.Equal(PlanRejection.Malformed, Rejection(truncated));

        var wrongGroups = (byte[])good.Clone();
        var h2 = MemoryMarshal.Read<StepPlanHeader>(wrongGroups);
        h2.GroupCount = 3;
        MemoryMarshal.Write(wrongGroups, h2);
        Assert.Equal(PlanRejection.Malformed, Rejection(wrongGroups));

        var offsetPastEnd = (byte[])good.Clone();
        var h3 = MemoryMarshal.Read<StepPlanHeader>(offsetPastEnd);
        h3.TokenIds = h3.TokenIds with { Offset = (ulong)offsetPastEnd.Length + 64 };
        MemoryMarshal.Write(offsetPastEnd, h3);
        Assert.Equal(PlanRejection.Malformed, Rejection(offsetPastEnd));
    }

    [Fact]
    public void ARandomBufferNeverEscapesAsAnyOtherException()
    {
        var rng = new Random(25);
        var good = Plan(new TestSequence(1, [1, 2, 3], 0, [0], 0), new TestSequence(2, [4], 16, [1, 2], 1));
        for (int i = 0; i < 2000; i++)
        {
            var bytes = (byte[])good.Clone();
            int flips = 1 + rng.Next(4);
            for (int f = 0; f < flips; f++)
            {
                bytes[rng.Next(bytes.Length)] = (byte)rng.Next(256);
            }

            try
            {
                var plan = StepPlanReader.Parse(bytes, Limits);

                // Whatever was accepted must be inside the limits: nothing reaches the device out of range.
                Assert.All(plan.TokenIds, t => Assert.InRange(t, 0, Limits.Vocab - 1));
                Assert.All(plan.Sequences, s =>
                {
                    Assert.All(s.Blocks, b => Assert.InRange(b, 0, Limits.AttentionBlocks - 1));
                    Assert.InRange(s.RecurrentSlot, 0, Limits.RecurrentSlots - 1);
                });
            }
            catch (PlanRejectedException)
            {
            }
        }
    }
}

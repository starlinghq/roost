using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Models.Qwen35;
using Starlings.Roost.Ops;
using Starlings.Roost.Plans;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Tests.Support;

namespace Starlings.Roost.Worker.Tests;

public sealed class PlanAndResultTests
{
    private static readonly PlanLimits Limits = new(64, 2, 8, 2, 248320, 16, 8, 2, 0, 2);
    private static readonly Qwen35EagerModel Model = new();

    private static int I32(Span<byte> region, EagerLayout l, string name, int index) =>
        MemoryMarshal.Read<int>(region.Slice((int)l.Offsets[name] + (index * 4), 4));

    private static long I64(Span<byte> region, EagerLayout l, string name, int index) =>
        MemoryMarshal.Read<long>(region.Slice((int)l.Offsets[name] + (index * 8), 8));

    [Fact]
    public void WritesThePlanInputsOfAPrefillAndADecodeSequence()
    {
        var plan = StepPlanReader.Parse(
            PlanWriter.Write(1, 2, new TestSequence(1, [10, 11, 12], 0, [5], 0), new TestSequence(2, [13], 17, [2, 7], 1)),
            Limits);
        var layout = Model.Layout(64, 2, 8);
        var region = new byte[layout.PlanBytes];
        PlanRegionWriter.Fill(region, layout, plan, 16);

        Assert.Equal([10, 11, 12, 13], Enumerable.Range(0, 4).Select(i => I32(region, layout, "Ids", i)));
        Assert.Equal([0, 1, 2, 17], Enumerable.Range(0, 4).Select(i => I32(region, layout, "Positions", i)));
        // Slot = block * 16 + position % 16: sequence 1 writes block 5; sequence 2's position 17 is offset 1 of its second block (7).
        Assert.Equal([80, 81, 82, (7 * 16) + 1], Enumerable.Range(0, 4).Select(i => I64(region, layout, "SlotMapping", i)));
        Assert.Equal([0, 3, 4], Enumerable.Range(0, 3).Select(i => I32(region, layout, "QIndptr", i)));
        Assert.Equal([0, 3, 4], Enumerable.Range(0, 3).Select(i => I32(region, layout, "CuSeqlens", i)));
        Assert.Equal([0, 1, 3], Enumerable.Range(0, 3).Select(i => I32(region, layout, "KvIndptr", i)));
        Assert.Equal([5, 2, 7], Enumerable.Range(0, 3).Select(i => I32(region, layout, "KvIndices", i)));
        Assert.Equal([3, 2], Enumerable.Range(0, 2).Select(i => I32(region, layout, "KvLastPageLength", i)));
        Assert.Equal([0, 1], Enumerable.Range(0, 2).Select(i => I32(region, layout, "StateSlots", i)));
        Assert.Equal([2, 3], Enumerable.Range(0, 2).Select(i => I32(region, layout, "Rows", i)));
    }

    [Fact]
    public void SelectsOnlyTheRowsOfSequencesThatSample()
    {
        var plan = StepPlanReader.Parse(
            PlanWriter.Write(1, 2, new TestSequence(1, [10, 11, 12], 0, [5], 0, Sample: false), new TestSequence(2, [13, 14], 0, [2], 1)),
            Limits);
        var layout = Model.Layout(64, 2, 8);
        var region = new byte[layout.PlanBytes];
        PlanRegionWriter.Fill(region, layout, plan, 16);
        Assert.Equal(1, plan.Selected);
        Assert.Equal(4, I32(region, layout, "Rows", 0));
    }

    [Fact]
    public void TheLayoutHoldsHalfAsManyFloatRowsAsTheGeneratedBufferHoldsBf16Rows()
    {
        var layout = Model.Layout(64, 1, 8);
        Assert.Equal(32, layout.MaxSelected);
        Assert.Equal(new Qwen35Model.ForwardLayout(64, 1, 8).ArenaBytes, layout.ArenaBytes);
        Assert.Equal(DType.I64, layout.Dtypes["SlotMapping"]);
        Assert.Equal(DType.I32, layout.Dtypes["Rows"]);
    }

    [Fact]
    public void ResultCarriesTokensFinishReasonsAndTimings()
    {
        var plan = StepPlanReader.Parse(
            PlanWriter.Write(9, 2, new TestSequence(1, [10, 11], 0, [0], 0) { Stops = [42], MaxNewTokensLeft = 5 }, new TestSequence(2, [13], 3, [1], 1) { MaxNewTokensLeft = 1 }, new TestSequence(3, [14], 3, [2], 1, Sample: false)),
            Limits with { MaxSequences = 3, MaxSelected = 3, RecurrentSlots = 3 });
        var buffer = new byte[StepResultWriter.SizeFor(3, 2) + 128];
        int written = StepResultWriter.Write(buffer, plan, [42, 7], new DeviceTimings { ForwardNs = 123 });
        Assert.Equal(StepResultWriter.SizeFor(3, 2), written);

        var h = MemoryMarshal.Read<StepResultHeader>(buffer);
        Assert.Equal(SharedLayout.MagicResult, h.Magic);
        Assert.Equal(9UL, h.StepId);
        Assert.Equal(FaultClass.None, h.Fault);
        Assert.Equal((ulong)written, h.TotalBytes);
        Assert.Equal(2U, h.TokenCount);
        var seqs = Enumerable.Range(0, 3).Select(i => MemoryMarshal.Read<ResultSequence>(buffer.AsSpan((int)h.Sequences.Offset + (i * Unsafe.SizeOf<ResultSequence>())))).ToArray();
        Assert.Equal([1UL, 2UL, 3UL], seqs.Select(static s => s.SequenceHandle));
        Assert.Equal([1U, 1U, 0U], seqs.Select(static s => s.TokenCount));
        Assert.Equal([FinishReason.Eos, FinishReason.Length, FinishReason.None], seqs.Select(static s => s.Finish));
        Assert.Equal(42, MemoryMarshal.Read<int>(buffer.AsSpan((int)h.TokenIds.Offset)));
        Assert.Equal(7, MemoryMarshal.Read<int>(buffer.AsSpan((int)h.TokenIds.Offset + 4)));
        Assert.Equal(123UL, MemoryMarshal.Read<DeviceTimings>(buffer.AsSpan((int)h.Timings.Offset)).ForwardNs);
        Assert.All(new[] { h.Sequences.Offset, h.TokenIds.Offset, h.Timings.Offset }, o => Assert.Equal(0UL, o % 64));
    }

    [Fact]
    public void ResultRefusesABufferThatIsTooSmall()
    {
        var plan = StepPlanReader.Parse(PlanWriter.Write(1, 2, new TestSequence(1, [1], 0, [0], 0)), Limits);
        Assert.Throws<ArgumentException>(() => StepResultWriter.Write(new byte[64], plan, [1], default));
    }

    [Fact]
    public void CudaOpsReportsWhatItRuns()
    {
        Assert.True(CudaOps.Supports(OpId.Gemm));
        Assert.True(CudaOps.Supports(OpId.Argmax));
        Assert.True(CudaOps.Supports(OpId.GatedDeltaRule));
        Assert.True(CudaOps.Supports(OpId.GatedDeltaRuleGates));
        Assert.True(CudaOps.Supports(OpId.GatedRmsNorm));
        Assert.True(CudaOps.Supports(OpId.SigmoidMul));
        Assert.False(CudaOps.Supports(OpId.Sample));
        Assert.False(CudaOps.Supports(OpId.GeluAndMul));
        Assert.False(CudaOps.Supports(OpId.AllReduce));
        Qwen35Model.CheckBackend<CudaOps>();
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 4)]
    [InlineData(5, 8)]
    [InlineData(64, 64)]
    [InlineData(65, 128)]
    public void ShapeBucketsArePowersOfTwo(long rows, uint bucket) => Assert.Equal(bucket, CudaOps.NextPowerOfTwo(rows));

    [Fact]
    public void CudaOpsOutsideABoundSessionIsATypedError()
    {
        var e = Assert.Throws<InvalidOperationException>(() => CudaOps.SiluAndMul(default));
        Assert.Contains("CudaOpsSession", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheModelAdapterMapsEverySlotToCheckpointTensors()
    {
        Assert.Equal(Qwen35Model.WeightSlots.Length, Model.WeightSlots.Count);
        Assert.Equal(Qwen35Model.StateSlots.Length, Model.StateSlots.Count);
        Assert.Equal(12, Model.StateSlots.Count(static s => !s.Recurrent));
        Assert.Equal(36, Model.StateSlots.Count(static s => s.Recurrent));
        Assert.All(Model.WeightSlots, s => Assert.NotEmpty(s.Sources));
        // gate_up concatenates two tensors, norms are offset by one, q_proj is split: those have assemblers, plain copies do not.
        Assert.Equal(2, Model.WeightSlots.Single(static s => s.Name == "l0.gate_up").Sources.Count);
        Assert.NotNull(Model.WeightSlots.Single(static s => s.Name == "l0.gate_up").Assembler);
        Assert.NotNull(Model.WeightSlots.Single(static s => s.Name == "final_norm").Assembler);
        Assert.NotNull(Model.WeightSlots.Single(static s => s.Name == "l3.q_proj").Assembler);
        Assert.Null(Model.WeightSlots.Single(static s => s.Name == "embed").Assembler);
        Assert.Equal(248320, Model.VocabSize);
    }

    [Fact]
    public void AssemblersBuildTheSlotsTheWeightMapDescribes()
    {
        var norm = Model.WeightSlots.Single(static s => s.Name == "final_norm");
        var src = new ushort[1024];
        for (int i = 0; i < src.Length; i++)
        {
            src[i] = Qwen35WeightMap.ToBFloat16Bits((i - 512) / 256f);
        }

        var dst = new byte[norm.Bytes];
        unsafe
        {
            fixed (ushort* p = src)
            {
                norm.Assembler!([new WeightBytes((nint)p, src.Length * 2)], dst);
            }
        }

        var got = MemoryMarshal.Cast<byte, ushort>(dst);
        for (int i = 0; i < 1024; i++)
        {
            Assert.Equal(Qwen35WeightMap.ToBFloat16Bits(1f + Qwen35WeightMap.BFloat16ToSingle(src[i])), got[i]);
        }
    }
}

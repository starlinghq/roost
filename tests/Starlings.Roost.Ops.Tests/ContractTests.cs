using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

public sealed class ContractTests
{
    private static bool Reports<TOps>(OpId op)
        where TOps : struct, IRoostOps => TOps.Supports(op);

    private static bool FusedSupported<TOps>()
        where TOps : struct, IRoostOps => TOps.SupportsAddRmsNorm;

    private static void Sliding<TOps>()
        where TOps : struct, IRoostOps => TOps.SlidingWindowAttention(default);

    private static void Fused<TOps>()
        where TOps : struct, IRoostOps => TOps.AddRmsNorm(default);

    [Fact]
    public void CpuOpsSupportsTheM0OpsAndTheFourHybridOpsOnly()
    {
        foreach (OpId op in Enum.GetValues<OpId>().Where(o => o != OpId.None))
        {
            bool expected = OpCatalog.IsM0(op) || op is OpId.GatedDeltaRule or OpId.CausalConv1d or OpId.RecurrentStateCopy or OpId.RecurrentStateSnapshot;
            Assert.Equal(expected, Reports<CpuOps>(op));
        }

        Assert.False(FusedSupported<CpuOps>());
    }

    [Fact]
    public void ReservedOpsOutsideTheHybridSetStillThrowNotSupported()
    {
        Assert.Throws<NotSupportedException>(() => Sliding<CpuOps>());
        Assert.Throws<NotSupportedException>(() => Fused<CpuOps>());
    }

    [Fact]
    public void CallsAllocateNoManagedMemory()
    {
        using var f = new Bag();
        var x = f.Add(Mem.F32(Enumerable.Range(0, 64).Select(i => (float)Math.Sin(i)).ToArray()));
        var w = f.Add(Mem.F32(Enumerable.Repeat(1f, 8).ToArray()));
        var y = f.Add(Mem.Zeros(DType.F32, 64));
        var pos = f.Add(Mem.I32(0, 1, 2, 3, 4, 5, 6, 7));
        var ids = f.Add(Mem.Zeros(DType.I32, 8));
        var logits = f.Add(Mem.F32(Enumerable.Range(0, 64).Select(i => (float)Math.Cos(i)).ToArray()));
        var temp = f.Add(Mem.F32(Enumerable.Repeat(0.9f, 8).ToArray()));
        var topK = f.Add(Mem.I32(Enumerable.Repeat(5, 8).ToArray()));
        var topP = f.Add(Mem.F32(Enumerable.Repeat(0.9f, 8).ToArray()));
        var minP = f.Add(Mem.F32(Enumerable.Repeat(0.01f, 8).ToArray()));
        var seeds = f.Add(Mem.U64(1, 2, 3, 4, 5, 6, 7, 8));
        var offsets = f.Add(Mem.U64(0, 0, 0, 0, 0, 0, 0, 0));
        var gq = f.Add(Mem.F32(Enumerable.Range(0, 16).Select(i => (float)Math.Sin(i)).ToArray()));
        var gg = f.Add(Mem.F32(Enumerable.Repeat(-0.1f, 4).ToArray()));
        var gb = f.Add(Mem.F32(Enumerable.Repeat(0.5f, 4).ToArray()));
        var go = f.Add(Mem.Zeros(DType.F32, 16));
        var gstate = f.Add(Mem.Zeros(DType.F32, 4 * 4));
        var one = f.Add(Mem.I32(0));
        var cu = f.Add(Mem.I32(0, 4));
        var cw = f.Add(Mem.F32(1, 2, 3, 4));
        var cstate = f.Add(Mem.Zeros(DType.F32, 3));
        var cin = f.Add(Mem.F32(1, 2, 3, 4));
        var cout = f.Add(Mem.Zeros(DType.F32, 4));
        var pairs = f.Add(Mem.I32(0, 1));
        var aq = f.Add(Mem.F32(Enumerable.Range(0, 8).Select(i => (float)Math.Sin(i)).ToArray()));
        var ak = f.Add(Mem.F32(Enumerable.Range(0, 32).Select(i => (float)Math.Cos(i)).ToArray()));
        var ao = f.Add(Mem.Zeros(DType.F32, 8));
        var aIndptrQ = f.Add(Mem.I32(0, 1));
        var aIndptrKv = f.Add(Mem.I32(0, 2));
        var aIdx = f.Add(Mem.I32(0, 1));
        var aLast = f.Add(Mem.I32(2));

        void Everything()
        {
            CpuOps.RmsNorm(new RmsNormArgs { Input = x.Span, Weight = w.Span, Output = y.Span, Rows = 8, Dim = 8, Eps = 1e-6f, DType = DType.F32 });
            CpuOps.QkNorm(new QkNormArgs { Q = x.Span, K = y.Span, QWeight = w.Span, KWeight = w.Span, Tokens = 4, QHeads = 1, KvHeads = 1, HeadDim = 8, Eps = 1e-6f, DType = DType.F32 });
            CpuOps.Rope(new RopeArgs { Q = x.Span, K = y.Span, Positions = pos.Span, Tokens = 8, QHeads = 1, KvHeads = 1, HeadDim = 8, Theta = 10000f, DType = DType.F32 });
            CpuOps.Argmax(new ArgmaxArgs { Logits = logits.Span, Ids = ids.Span, Rows = 8, Vocab = 8 });
            CpuOps.Sample(new SampleArgs { Logits = logits.Span, Temperature = temp.Span, TopK = topK.Span, TopP = topP.Span, MinP = minP.Span, Seeds = seeds.Span, Offsets = offsets.Span, Ids = ids.Span, Rows = 8, Vocab = 8, Stride = 8 });
            CpuOps.GatedDeltaRule(new GatedDeltaRuleArgs
            {
                Q = gq.Span, K = gq.Span, V = gq.Span, G = gg.Span, Beta = gb.Span, Output = go.Span, State = gstate.Span, StateSlots = one.Span, CuSeqlens = cu.Span,
                Batch = 1, Tokens = 4, KHeads = 1, VHeads = 1, KDim = 4, VDim = 4, Scale = 0.5f, UseInitialState = true, L2NormQk = true, DType = DType.F32,
            });
            CpuOps.CausalConv1d(new CausalConv1dArgs
            {
                Input = cin.Span, Weight = cw.Span, Output = cout.Span, ConvState = cstate.Span, StateSlots = one.Span, CuSeqlens = cu.Span,
                Batch = 1, Tokens = 4, Channels = 1, Width = 4, Silu = true, DType = DType.F32,
            });
            CpuOps.PagedAttention(new PagedAttentionArgs
            {
                Q = aq.Span, Output = ao.Span, KCache = ak.Span, VCache = ak.Span, QIndptr = aIndptrQ.Span, KvIndptr = aIndptrKv.Span, KvIndices = aIdx.Span,
                KvLastPageLength = aLast.Span, Mode = AttentionMode.Decode, Batch = 1, Tokens = 1, QHeads = 2, KvHeads = 1, HeadDim = 4, BlockSize = 4,
                KvLayout = TensorLayout.PagedNhd, WindowLeft = -1, Causal = true, KvDType = DType.F32, DType = DType.F32,
            });
            CpuOps.CacheCopy(new CacheCopyArgs { Source = x.Span, Destination = x.Span, Pairs = pairs.Span, BlockBytes = 8, Count = 1 });
        }

        Everything();
        Everything();
        long before = GC.GetAllocatedBytesForCurrentThread();
        Everything();
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }
}

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
    public void GatedDeltaRuleRejectsANonFloat32StatePool()
    {
        using var q = Mem.F32(1, 1);
        using var g = Mem.F32(0f);
        using var o = Mem.Zeros(DType.F32, 1);
        using var st = Mem.Zeros(DType.F64, 1);
        using var slot = Mem.I32(0);
        using var cu = Mem.I32(0, 1);
        var args = new GatedDeltaRuleArgs
        {
            Q = q.Span, K = q.Span, V = q.Span, G = g.Span, Beta = g.Span, Output = o.Span, State = st.Span, StateSlots = slot.Span, CuSeqlens = cu.Span,
            Batch = 1, Tokens = 1, KHeads = 1, VHeads = 1, KDim = 1, VDim = 1, DType = DType.F32,
        };
        Assert.Throws<ArgumentException>(() => CpuOps.GatedDeltaRule(args));
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

        var sx = f.Add(Mem.F32(1, 2, 3, 4));
        var sy = f.Add(Mem.Zeros(DType.F32, 2));
        var tbl = f.Add(Mem.F32(Enumerable.Range(0, 24).Select(i => (float)i).ToArray()));
        var eout = f.Add(Mem.Zeros(DType.F32, 16));
        var eids = f.Add(Mem.I32(0, 1, 2, 3));
        var gA = f.Add(Mem.F32(1, 2, 3, 4, 5, 6));
        var gB = f.Add(Mem.F32(1, 2, 3, 4, 5, 6));
        var gD = f.Add(Mem.Zeros(DType.F32, 4));
        var lmRows = f.Add(Mem.I32(0));
        var lmW = f.Add(Mem.F32(1, 0, 0, 1, 1, 1));
        var lmHid = f.Add(Mem.F32(1, 2));
        var lmOut = f.Add(Mem.Zeros(DType.F32, 4));
        var slotMap = f.Add(Mem.Zeros(DType.I64, 2));
        var kvPool = f.Add(Mem.Zeros(DType.F32, 16));
        var kvIn = f.Add(Mem.F32(1, 2, 3, 4));
        var statePool = f.Add(Mem.Zeros(DType.F32, 8));
        var snapPool = f.Add(Mem.Zeros(DType.F32, 8));
        var snapPairs = f.Add(Mem.I32(0, 1));

        void Everything()
        {
            CpuOps.SiluAndMul(new SiluAndMulArgs { Input = sx.Span, Output = sy.Span, Rows = 1, Dim = 2, DType = DType.F32 });
            CpuOps.GeluAndMul(new GeluAndMulArgs { Input = sx.Span, Output = sy.Span, Rows = 1, Dim = 2, Approximation = GeluApproximation.Tanh, DType = DType.F32 });
            CpuOps.GeluAndMul(new GeluAndMulArgs { Input = sx.Span, Output = sy.Span, Rows = 1, Dim = 2, Approximation = GeluApproximation.Exact, DType = DType.F32 });
            CpuOps.Gemm(new GemmArgs { A = gA.Span, B = gB.Span, D = gD.Span, M = 2, N = 2, K = 3, Alpha = 1f, Beta = 0f, DType = DType.F32, WeightDType = DType.F32 });
            CpuOps.Embedding(new EmbeddingArgs { Ids = eids.Span, Table = tbl.Span, Output = eout.Span, Tokens = 4, Dim = 4, Vocab = 6, DType = DType.F32 });
            CpuOps.LmHead(new LmHeadArgs { Hidden = lmHid.Span, Rows = lmRows.Span, Weight = lmW.Span, Logits = lmOut.Span, Selected = 1, Dim = 2, Vocab = 3, VocabPadded = 4, DType = DType.F32 });
            CpuOps.Residual(new ResidualArgs { A = x.Span, B = y.Span, Output = x.Span, Elements = 8, DType = DType.F32 });
            CpuOps.CacheWrite(new CacheWriteArgs { K = kvIn.Span, V = kvIn.Span, KCache = kvPool.Span, VCache = kvPool.Span, SlotMapping = slotMap.Span, Tokens = 2, KvHeads = 1, HeadDim = 2, BlockSize = 2, KvLayout = TensorLayout.PagedNhd, KScale = 1f, VScale = 1f, KvDType = DType.F32 });
            CpuOps.RecurrentStateCopy(new RecurrentStateCopyArgs { Pool = statePool.Span, Pairs = snapPairs.Span, SlotBytes = 16, Count = 1 });
            CpuOps.RecurrentStateSnapshot(new RecurrentStateSnapshotArgs { Pool = statePool.Span, Snapshots = snapPool.Span, Pairs = snapPairs.Span, SlotBytes = 16, Count = 1, Restore = false });
            CpuOps.RecurrentStateSnapshot(new RecurrentStateSnapshotArgs { Pool = statePool.Span, Snapshots = snapPool.Span, Pairs = snapPairs.Span, SlotBytes = 16, Count = 1, Restore = true });
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

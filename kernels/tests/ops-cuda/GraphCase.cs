using Starlings.Roost.Abi;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>
/// ROOST_FLAG_GRAPH_CAPTURABLE: a decode step (RmsNorm, Gemm, Rope, PagedAttention, CacheWrite) is captured into a CUDA graph after
/// planning, replayed, and must match the eager run bit for bit. Calls that allocate or synchronise fail under capture.
/// </summary>
internal static unsafe class GraphCase
{
    public static void Run(Ctx c)
    {
        const int batch = 4, hidden = 1024, qh = 8, kh = 2, hd = 256, ps = 16;
        Seq[] seqs = [new(1, 40), new(1, 17), new(1, 300), new(1, 64)];
        using AttentionCases.Batch x = AttentionCases.Make(c, seqs, ps, TensorLayout.PagedNhd);
        using var h = new Buf(DType.BF16, batch * hidden).Randn(c.Rng, 2.0);
        using var nw = new Buf(DType.BF16, hidden).Randn(c.Rng, 0.2, 1.0);
        using var normed = new Buf(DType.BF16, batch * hidden);
        using var wq = new Buf(DType.BF16, (long)qh * hd * hidden).Randn(c.Rng, 1.0 / Math.Sqrt(hidden));
        using var q = new Buf(DType.BF16, (long)batch * qh * hd);
        using var k = new Buf(DType.BF16, (long)batch * kh * hd).Randn(c.Rng);
        using var v = new Buf(DType.BF16, (long)batch * kh * hd).Randn(c.Rng);
        using var pos = Buf.Ints(39, 16, 299, 63);
        using var o = new Buf(DType.BF16, (long)batch * qh * hd);
        // The new token's slot: the last position of each sequence, in its (shuffled) page.
        var slots = new long[batch];
        for (int b = 0; b < batch; b++)
        {
            int page = x.KvIndices.As<int>()[x.KvIndptr.As<int>()[b + 1] - 1];
            slots[b] = ((long)page * ps) + x.LastLen.As<int>()[b] - 1;
        }

        using var slot = Buf.Longs(slots);
        h.Up();
        nw.Up();
        wq.Up();
        k.Up();
        v.Up();
        pos.Up();
        slot.Up();
        AttentionCases.Upload(x);

        var norm = new RmsNormParams { StructSize = (uint)sizeof(RmsNormParams), Input = h.DevSpan, Weight = nw.DevSpan, Output = normed.DevSpan, Rows = batch, Dim = hidden, Eps = 1e-6f };
        var gemm = new GemmParams { StructSize = (uint)sizeof(GemmParams), A = normed.DevSpan, B = wq.DevSpan, D = q.DevSpan, M = batch, N = (ulong)(qh * hd), K = hidden, TransposeB = 1, Alpha = 1f, ShapeBucket = 8 };
        var rope = new RopeParams { StructSize = (uint)sizeof(RopeParams), Q = q.DevSpan, K = k.DevSpan, Positions = pos.DevSpan, Tokens = batch, QHeads = qh, KvHeads = kh, HeadDim = hd, RotaryDim = 64, Theta = 1e7f };
        var write = new CacheWriteParams { StructSize = (uint)sizeof(CacheWriteParams), K = k.DevSpan, V = v.DevSpan, KCache = x.K.DevSpan, VCache = x.V.DevSpan, SlotMapping = slot.DevSpan, Tokens = batch, KvHeads = kh, HeadDim = hd, PageSize = ps, KvLayout = TensorLayout.PagedNhd, KScale = 1f, VScale = 1f };
        // Attention reads the Gemm output. The Batch keeps its own Q buffer (disposed with it); q is disposed by its own using.
        PagedAttentionParams attn = AttentionCases.Params(x, o, AttentionMode.Decode, true);
        attn.Q = q.DevSpan;
        using WorkspaceSet wsGemm = c.Lib.MakeWorkspace("gemm-cublaslt", ref gemm);
        using WorkspaceSet wsAttn = c.Lib.MakeWorkspace("paged-attention-flashinfer-fa2", ref attn);
        c.Lib.PlanWith("gemm-cublaslt", ref gemm, wsGemm, c.Stream);
        c.Lib.PlanWith("paged-attention-flashinfer-fa2", ref attn, wsAttn, c.Stream);

        void Step(bool sync)
        {
            c.Lib.CheckStatus(c.Lib.Try("rms-norm-flashinfer", ref norm, c.Stream), "rms-norm");
            c.Lib.RunWith("gemm-cublaslt", ref gemm, wsGemm, c.Stream, sync: false);
            c.Lib.CheckStatus(c.Lib.Try("rope-roost", ref rope, c.Stream), "rope");
            c.Lib.CheckStatus(c.Lib.Try("cache-write-roost", ref write, c.Stream), "cache-write");
            c.Lib.RunWith("paged-attention-flashinfer-fa2", ref attn, wsAttn, c.Stream, sync: false);
            if (sync)
            {
                Cuda.Sync(c.Stream);
            }
        }

        // Eager reference. The Rope and CacheWrite steps modify q, k and the pool, so snapshot the inputs and restore them.
        using var kSave = k.Clone();
        using var poolK = x.K.Clone();
        using var poolV = x.V.Clone();
        Step(sync: true);
        using var eager = Ctx.Fetch(o);
        void Restore()
        {
            kSave.Up();
            Cuda.ToDevice(x.K.Dev, poolK.Host, poolK.Bytes);
            Cuda.ToDevice(x.V.Dev, poolV.Host, poolV.Bytes);
            o.FillDevice(0xFF);
        }

        Restore();
        (nint graph, nint exec) = (0, 0);
        try
        {
            (graph, exec) = Cuda.Capture(c.Stream, () => Step(sync: false));
        }
        catch (Exception e)
        {
            c.Check("Graph: decode step (RmsNorm, Gemm, Rope, CacheWrite, PagedAttention) captured", false, e.Message);
            Cuda.ClearError();
            return;
        }

        c.Check("Graph: decode step (RmsNorm, Gemm, Rope, CacheWrite, PagedAttention) captured", true);
        Cuda.Launch(exec, c.Stream);
        Cuda.Sync(c.Stream);
        using var replay = Ctx.Fetch(o);
        c.Check("Graph: replay output equals the eager step bitwise", Compare.BytesEqual(eager, replay));
        Restore();
        Cuda.Launch(exec, c.Stream);
        Cuda.Sync(c.Stream);
        using var replay2 = Ctx.Fetch(o);
        c.Check("Graph: second replay (state restored) equals the first", Compare.BytesEqual(replay, replay2));
        Cuda.Destroy(graph, exec);
    }
}

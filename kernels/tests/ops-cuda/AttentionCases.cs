using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Roost.KernelOps.Tests.Cuda;

internal sealed record Seq(int QLen, int KvLen);

/// <summary>Paged GQA attention (FlashInfer FA2 decode and prefill, head_dim 256, 8 query heads over 2 KV heads) against CpuOps.</summary>
internal static unsafe class AttentionCases
{
    private const int QHeads = 8, KvHeads = 2, HeadDim = 256;

    // Proposed: Q, K, V and O are bf16 and FlashInfer rounds the softmax weights to bf16 before the PV product, so each
    // output carries an error of up to about 2^-8 * sum(p |v|) <= 2^-8 * max|V|, regardless of how small the output is, plus
    // the final rounding (one bf16 ulp, 2^-7 relative). The absolute part is therefore scaled by max|V| of the pool.
    public static Tol AttnTol => new("attention-bf16", Math.Pow(2, -7), Math.Pow(2, -8));

    internal static double MaxAbs(Buf b)
    {
        double m = 0;
        for (long i = 0; i < b.Count; i++)
        {
            m = Math.Max(m, Math.Abs(b.Get(i)));
        }

        return m;
    }

    public static void Run(Ctx c)
    {
        foreach ((TensorLayout layout, int ps) in new[] { (TensorLayout.PagedNhd, 16), (TensorLayout.PagedHnd, 16), (TensorLayout.PagedNhd, 32), (TensorLayout.PagedHnd, 64) })
        {
            Decode(c, $"mixed4 {layout} ps{ps}", [5, 37, 130, 16], ps, layout);
        }

        Decode(c, "single-token-history", [1], 16, TensorLayout.PagedNhd);
        Decode(c, "page-multiples", [16, 32, 48, 64], 16, TensorLayout.PagedNhd);
        Decode(c, "single-6000 (split-KV)", [6000], 16, TensorLayout.PagedNhd, determinism: true);
        Decode(c, "mixed8", [900, 17, 1, 2048, 333, 64, 1024, 777], 16, TensorLayout.PagedNhd, determinism: true);
        Decode(c, "mixed8 HND ps64", [900, 17, 1, 2048, 333, 64, 1024, 777], 64, TensorLayout.PagedHnd);
        var many = new int[64];
        for (int i = 0; i < many.Length; i++)
        {
            many[i] = c.Rng.Int(1, 301);
        }

        Decode(c, "batch64", many, 16, TensorLayout.PagedNhd);

        Prefill(c, "one-token", [new(1, 1)], 16, TensorLayout.PagedNhd, true);
        Prefill(c, "fresh-100", [new(100, 100)], 16, TensorLayout.PagedNhd, true);
        Prefill(c, "fresh-100 HND", [new(100, 100)], 16, TensorLayout.PagedHnd, true);
        Prefill(c, "fresh-700 ps64", [new(700, 700)], 64, TensorLayout.PagedNhd, true, determinism: true);
        Prefill(c, "fresh-1024", [new(1024, 1024)], 16, TensorLayout.PagedNhd, true);
        Prefill(c, "chunk-128-of-1000", [new(128, 1000)], 16, TensorLayout.PagedNhd, true, determinism: true);
        Prefill(c, "chunk-128-of-1000 HND ps32", [new(128, 1000)], 32, TensorLayout.PagedHnd, true);
        Prefill(c, "multi-seq", [new(17, 17), new(64, 300), new(1, 50), new(200, 200)], 16, TensorLayout.PagedNhd, true);
        Prefill(c, "non-causal-32-of-200", [new(32, 200)], 16, TensorLayout.PagedNhd, false);
        Prefill(c, "partial-last-page", [new(33, 33), new(5, 21)], 16, TensorLayout.PagedNhd, true);
        Rejections(c);
    }

    private static void Decode(Ctx c, string label, int[] kvLens, int ps, TensorLayout layout, bool determinism = false)
        => Attention(c, "decode " + label, AttentionMode.Decode, kvLens.Select(l => new Seq(1, l)).ToArray(), ps, layout, true, determinism);

    private static void Prefill(Ctx c, string label, Seq[] seqs, int ps, TensorLayout layout, bool causal, bool determinism = false)
        => Attention(c, "prefill " + label, AttentionMode.Prefill, seqs, ps, layout, causal, determinism);

    internal sealed class Batch : IDisposable
    {
        public Buf Q = null!, K = null!, V = null!, QIndptr = null!, KvIndptr = null!, KvIndices = null!, LastLen = null!;
        public int Tokens, BatchSize, PageSize;
        public TensorLayout Layout;

        public void Dispose()
        {
            foreach (Buf b in new[] { Q, K, V, QIndptr, KvIndptr, KvIndices, LastLen })
            {
                b.Dispose();
            }
        }
    }

    internal static Batch Make(Ctx c, Seq[] seqs, int ps, TensorLayout layout, double qStd = 1.0)
    {
        int batch = seqs.Length;
        int tokens = seqs.Sum(s => s.QLen);
        var pagesPer = seqs.Select(s => (s.KvLen + ps - 1) / ps).ToArray();
        int total = pagesPer.Sum();
        int poolPages = total + 3;
        var perm = Enumerable.Range(0, poolPages).ToArray();
        for (int i = perm.Length - 1; i > 0; i--)
        {
            int j = c.Rng.Int(0, i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }

        var qIndptr = new int[batch + 1];
        var kvIndptr = new int[batch + 1];
        var last = new int[batch];
        for (int b = 0; b < batch; b++)
        {
            qIndptr[b + 1] = qIndptr[b] + seqs[b].QLen;
            kvIndptr[b + 1] = kvIndptr[b] + pagesPer[b];
            last[b] = seqs[b].KvLen - ((pagesPer[b] - 1) * ps);
        }

        long poolElems = (long)poolPages * ps * KvHeads * HeadDim;
        return new Batch
        {
            Q = new Buf(DType.BF16, (long)tokens * QHeads * HeadDim).Randn(c.Rng, qStd),
            K = new Buf(DType.BF16, poolElems).Randn(c.Rng, 1.0),
            V = new Buf(DType.BF16, poolElems).Randn(c.Rng, 1.0),
            QIndptr = Buf.Ints(qIndptr),
            KvIndptr = Buf.Ints(kvIndptr),
            KvIndices = Buf.Ints(perm.Take(total).ToArray()),
            LastLen = Buf.Ints(last),
            Tokens = tokens,
            BatchSize = batch,
            PageSize = ps,
            Layout = layout,
        };
    }

    internal static PagedAttentionParams Params(Batch x, Buf output, AttentionMode mode, bool causal)
    {
        return new PagedAttentionParams
        {
            StructSize = (uint)sizeof(PagedAttentionParams),
            Q = x.Q.DevSpan,
            Output = output.DevSpan,
            KCache = x.K.DevSpan,
            VCache = x.V.DevSpan,
            QIndptr = x.QIndptr.DevSpan,
            KvIndptr = x.KvIndptr.DevSpan,
            KvIndices = x.KvIndices.DevSpan,
            KvLastPageLen = x.LastLen.DevSpan,
            QIndptrHost = x.QIndptr.Host,
            KvIndptrHost = x.KvIndptr.Host,
            KvLastPageLenHost = x.LastLen.Host,
            Mode = mode,
            Batch = (uint)x.BatchSize,
            Tokens = (uint)x.Tokens,
            QHeads = QHeads,
            KvHeads = KvHeads,
            HeadDim = HeadDim,
            PageSize = (uint)x.PageSize,
            KvLayout = x.Layout,
            SmScale = 0f,
            LogitsSoftCap = 0f,
            WindowLeft = -1,
            Causal = causal ? 1u : 0u,
        };
    }

    internal static void Upload(Batch x)
    {
        x.Q.Up();
        x.K.Up();
        x.V.Up();
        x.QIndptr.Up();
        x.KvIndptr.Up();
        x.KvIndices.Up();
        x.LastLen.Up();
    }

    internal static void Reference(Batch x, Buf want, AttentionMode mode, bool causal)
    {
        CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = x.Q.HostSpan,
            Output = want.HostSpan,
            KCache = x.K.HostSpan,
            VCache = x.V.HostSpan,
            QIndptr = x.QIndptr.HostSpan,
            KvIndptr = x.KvIndptr.HostSpan,
            KvIndices = x.KvIndices.HostSpan,
            KvLastPageLength = x.LastLen.HostSpan,
            Mode = mode,
            Batch = x.BatchSize,
            Tokens = x.Tokens,
            QHeads = QHeads,
            KvHeads = KvHeads,
            HeadDim = HeadDim,
            BlockSize = x.PageSize,
            KvLayout = x.Layout,
            Scale = 0f,
            SoftCap = 0f,
            WindowLeft = -1,
            Causal = causal,
            KvDType = DType.BF16,
            DType = DType.BF16,
        });
    }

    private static void Attention(Ctx c, string label, AttentionMode mode, Seq[] seqs, int ps, TensorLayout layout, bool causal, bool determinism)
    {
        using Batch x = Make(c, seqs, ps, layout);
        using var o = new Buf(DType.BF16, (long)x.Tokens * QHeads * HeadDim);
        Upload(x);
        o.FillDevice(0xFF);
        PagedAttentionParams p = Params(x, o, mode, causal);
        using WorkspaceSet ws = c.Lib.MakeWorkspace("paged-attention-flashinfer-fa2", ref p);
        int unplanned = c.Lib.TryRunWith("paged-attention-flashinfer-fa2", ref p, ws, c.Stream);
        c.Lib.PlanWith("paged-attention-flashinfer-fa2", ref p, ws, c.Stream);
        c.Lib.RunWith("paged-attention-flashinfer-fa2", ref p, ws, c.Stream);
        using var got = Ctx.Fetch(o);
        using var want = new Buf(DType.BF16, o.Count);
        Reference(x, want, mode, causal);
        string name = $"PagedAttention {label} q={string.Join(",", seqs.Take(4).Select(s => s.QLen))}{(seqs.Length > 4 ? ",.." : string.Empty)} kv={string.Join(",", seqs.Take(4).Select(s => s.KvLen))}{(seqs.Length > 4 ? ",.." : string.Empty)} ws={ws.Sizes.DeviceBytes / 1024}KiB";
        c.Report(Compare.Close(name, got, want, AttnTol, scale: MaxAbs(x.V)));
        if (unplanned != (int)KernelStatus.NotPlanned)
        {
            c.Check($"PagedAttention {label}: run before plan -> NOT_PLANNED", false, $"status {unplanned}");
        }

        // C1: the *_host tables are read by the workspace query and plan() only; run() works with them NULL.
        PagedAttentionParams noHost = p;
        noHost.QIndptrHost = 0;
        noHost.KvIndptrHost = 0;
        noHost.KvLastPageLenHost = 0;
        o.FillDevice(0xFF);
        c.Lib.RunWith("paged-attention-flashinfer-fa2", ref noHost, ws, c.Stream);
        using var withoutHost = Ctx.Fetch(o);
        c.Check($"PagedAttention {label}: run() with NULL host tables equals the planned run", Compare.BytesEqual(got, withoutHost));
        if (determinism)
        {
            o.FillDevice(0xFF);
            c.Lib.RunWith("paged-attention-flashinfer-fa2", ref p, ws, c.Stream);
            using var again = Ctx.Fetch(o);
            c.Check($"PagedAttention {label}: second run bit-identical", Compare.BytesEqual(got, again));
        }
    }

    private static void Rejections(Ctx c)
    {
        using Batch x = Make(c, [new(1, 40), new(1, 17)], 16, TensorLayout.PagedNhd);
        using var o = new Buf(DType.BF16, (long)x.Tokens * QHeads * HeadDim);
        Upload(x);
        PagedAttentionParams good = Params(x, o, AttentionMode.Decode, true);
        const string v = "paged-attention-flashinfer-fa2";
        using WorkspaceSet ws = c.Lib.MakeWorkspace(v, ref good);
        c.Lib.PlanWith(v, ref good, ws, c.Stream);
        var hd = good;
        hd.HeadDim = 128;
        c.Check("PagedAttention head_dim 128 -> UNSUPPORTED", c.Lib.TryRunWith(v, ref hd, ws, c.Stream) == (int)KernelStatus.Unsupported);
        var group = good;
        group.QHeads = 16;
        c.Check("PagedAttention group size 8 -> UNSUPPORTED", c.Lib.TryRunWith(v, ref group, ws, c.Stream) == (int)KernelStatus.Unsupported);
        var window = good;
        window.WindowLeft = 128;
        c.Check("PagedAttention sliding window -> UNSUPPORTED", c.Lib.TryRunWith(v, ref window, ws, c.Stream) == (int)KernelStatus.Unsupported);
        var cap = good;
        cap.LogitsSoftCap = 30f;
        c.Check("PagedAttention soft cap -> UNSUPPORTED", c.Lib.TryRunWith(v, ref cap, ws, c.Stream) == (int)KernelStatus.Unsupported);
        var tokens = good;
        tokens.Tokens = 3;
        c.Check("PagedAttention decode with tokens != batch -> INVALID_ARGUMENT", c.Lib.TryRunWith(v, ref tokens, ws, c.Stream) == (int)KernelStatus.InvalidArgument);
        var mode = good;
        mode.Mode = AttentionMode.Prefill;
        c.Check("PagedAttention plan made for decode, run as prefill -> INVALID_ARGUMENT", c.Lib.TryRunWith(v, ref mode, ws, c.Stream) == (int)KernelStatus.InvalidArgument);
        var noHostPlan = good;
        noHostPlan.KvIndptrHost = 0;
        fixed (Workspace* wp = &ws.Value)
        {
            c.Check("PagedAttention plan without host tables -> INVALID_ARGUMENT", c.Lib.Plan(v, &noHostPlan, wp, c.Stream) == (int)KernelStatus.InvalidArgument);
        }

        var shortIndices = good;
        shortIndices.KvIndices.Length = 1;
        c.Check("PagedAttention kv_indices shorter than the planned pages -> INVALID_ARGUMENT", c.Lib.TryRunWith(v, ref shortIndices, ws, c.Stream) == (int)KernelStatus.InvalidArgument);
        var f16 = good;
        f16.Q.Dtype = DType.F16;
        c.Check("PagedAttention f16 q -> INVALID_ARGUMENT", c.Lib.TryRunWith(v, ref f16, ws, c.Stream) == (int)KernelStatus.InvalidArgument);
        var layout = good;
        layout.KvLayout = TensorLayout.RowMajor;
        c.Check("PagedAttention row-major KV -> INVALID_ARGUMENT", c.Lib.TryRunWith(v, ref layout, ws, c.Stream) == (int)KernelStatus.InvalidArgument);
        var small = new WorkspaceSet(new WorkspaceSizes { DeviceBytes = 64, PinnedHostBytes = 64, PlanBytes = ws.Sizes.PlanBytes });
        int tooSmall;
        fixed (Workspace* w = &small.Value)
        {
            tooSmall = c.Lib.Plan(v, &good, w, c.Stream);
        }

        c.Check("PagedAttention plan into a small workspace -> WORKSPACE_TOO_SMALL", tooSmall == (int)KernelStatus.WorkspaceTooSmall);
        small.Dispose();
        Cuda.Sync(c.Stream);
        Cuda.ClearError();
    }
}

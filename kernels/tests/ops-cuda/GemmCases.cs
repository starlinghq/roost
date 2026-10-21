using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>Gemm and LmHead (cuBLASLt with the algorithm pinned per shape bucket) against CpuOps, plus the pinning properties.</summary>
internal static unsafe class GemmCases
{
    public static Tol GemmTol(DType dt) => dt == DType.BF16 ? new Tol("gemm-bf16", Math.Pow(2, -6), Math.Pow(2, -9)) : new Tol("gemm-f16", Math.Pow(2, -9), Math.Pow(2, -12));

    public static Tol LogitsTol => new("lm-head-f32", Math.Pow(2, -12), Math.Pow(2, -13));

    public static void Run(Ctx c)
    {
        // Qwen3.5-0.8B projections of a full-attention layer (hidden 1024, q_proj 2 x 8 x 256 with the output gate, kv 2 x 256, mlp 3584).
        (string Name, int N, int K)[] shapes =
        [
            ("q_proj", 4096, 1024),
            ("k_proj", 512, 1024),
            ("o_proj", 1024, 2048),
            ("gate_up", 7168, 1024),
            ("down", 1024, 3584),
        ];
        foreach (DType dt in new[] { DType.BF16, DType.F16 })
        {
            foreach ((string name, int n, int k) in shapes)
            {
                foreach (int m in new[] { 1, 3, 17, 64 })
                {
                    Gemm(c, dt, name, m, n, k, transB: true, bias: false, alpha: 1f, beta: 0f, bucket: 0);
                }
            }
        }

        Gemm(c, DType.BF16, "q_proj-m200", 200, 4096, 1024, true, false, 1f, 0f, 256);
        Gemm(c, DType.BF16, "down-nontransposed", 17, 1024, 3584, false, false, 1f, 0f, 0);
        Gemm(c, DType.BF16, "bias", 17, 1024, 2048, true, true, 1f, 0f, 0);
        Gemm(c, DType.BF16, "alpha-beta-c", 17, 1024, 2048, true, false, 0.5f, 0.25f, 0);
        // bf16 with both a C input and a bias is refused (R-019 finding, see Rejections); the f16 combination is served and checked.
        Gemm(c, DType.F16, "alpha-beta-c-bias", 5, 512, 1024, false, true, 2f, 1f, 64);
        Gemm(c, DType.F16, "bias", 17, 1024, 2048, true, true, 1f, 0f, 0);
        Gemm(c, DType.BF16, "small-k8", 4, 64, 8, true, false, 1f, 0f, 0);
        Pinning(c);
        LmHead(c, DType.BF16, 4096, 1024, 4096, 1, 0f);
        LmHead(c, DType.BF16, 4096, 1024, 4096, 3, 0f);
        LmHead(c, DType.BF16, 4000, 1024, 4096, 5, 30f);
        LmHead(c, DType.F16, 4096, 1024, 4128, 2, 0f);
        LmHeadPinning(c);
        Rejections(c);
    }

    internal static string AlgoText(WorkspaceSet ws)
    {
        byte* p = (byte*)ws.Plan;
        if (p == null)
        {
            return string.Empty;
        }

        // PlanRecord of kernels/src/gemm.cu: magic u32 @0, op u32 @4, n u64 @8, k u64 @16, bucket_m u64 @24, ..., workspace u64 @48,
        // algo_id i32 @56, tile_id i32 @60, stages_id i32 @64, split_k i32 @68.
        return $"algo={Marshal.ReadInt32((nint)p + 56)} tile={Marshal.ReadInt32((nint)p + 60)} stages={Marshal.ReadInt32((nint)p + 64)} splitk={Marshal.ReadInt32((nint)p + 68)} scratch={Marshal.ReadInt64((nint)p + 48)}";
    }

    public static void Gemm(Ctx c, DType dt, string label, int m, int n, int k, bool transB, bool bias, float alpha, float beta, int bucket, Buf? weight = null)
    {
        using var a = new Buf(dt, (long)m * k).Randn(c.Rng);
        Buf b = weight ?? new Buf(dt, (long)n * k).Randn(c.Rng, 1.0 / Math.Sqrt(k));
        using var biasBuf = new Buf(dt, n).Randn(c.Rng, 0.5);
        using var cBuf = new Buf(dt, (long)m * n).Randn(c.Rng);
        using var d = new Buf(dt, (long)m * n);
        a.Up();
        b.Up();
        biasBuf.Up();
        cBuf.Up();
        d.FillDevice(0xAB);
        var p = new GemmParams
        {
            StructSize = (uint)sizeof(GemmParams),
            A = a.DevSpan,
            B = b.DevSpan,
            C = beta != 0f ? cBuf.DevSpan : default,
            D = d.DevSpan,
            Bias = bias ? biasBuf.DevSpan : default,
            M = (ulong)m,
            N = (ulong)n,
            K = (ulong)k,
            TransposeB = transB ? 1u : 0u,
            Alpha = alpha,
            Beta = beta,
            ShapeBucket = (uint)bucket,
        };
        using WorkspaceSet ws = c.Lib.MakeWorkspace("gemm-cublaslt", ref p);
        int unplanned = c.Lib.TryRunWith("gemm-cublaslt", ref p, ws, c.Stream);
        c.Lib.PlanWith("gemm-cublaslt", ref p, ws, c.Stream);
        c.Lib.RunWith("gemm-cublaslt", ref p, ws, c.Stream);
        using var got = Ctx.Fetch(d);
        using var want = new Buf(dt, (long)m * n);
        CpuOps.Gemm(new GemmArgs
        {
            A = a.HostSpan,
            B = b.HostSpan,
            C = beta != 0f ? cBuf.HostSpan : default,
            D = want.HostSpan,
            Bias = bias ? biasBuf.HostSpan : default,
            M = m,
            N = n,
            K = k,
            TransposeB = transB,
            Alpha = alpha,
            Beta = beta,
            ShapeBucket = bucket,
            DType = dt,
            WeightDType = dt,
        });
        c.Report(Compare.Close($"Gemm {dt} {label} m={m} n={n} k={k} tB={(transB ? 1 : 0)} bias={(bias ? 1 : 0)} a/b={alpha}/{beta} bucket={bucket}", got, want, GemmTol(dt)));
        if (m == 1 && label is "q_proj" or "down")
        {
            c.Line($"     {AlgoText(ws)} ({label}, m=1)");
        }

        if (unplanned != (int)KernelStatus.NotPlanned)
        {
            c.Check($"Gemm {label} run before plan -> NOT_PLANNED", false, $"status {unplanned}");
        }

        if (weight is null)
        {
            b.Dispose();
        }
    }

    // Pinning (S2): with a plan made for a bucket, the rows of a small-M call match the bucket-size call bit for bit; a
    // second run is bit-identical (determinism).
    private static void Pinning(Ctx c)
    {
        foreach ((int n, int k) in new[] { (1024, 3584), (4096, 1024), (7168, 1024), (1024, 2048), (512, 1024) })
        {
            const int bucket = 64;
            using var a = new Buf(DType.BF16, (long)bucket * k).Randn(c.Rng);
            using var b = new Buf(DType.BF16, (long)n * k).Randn(c.Rng, 1.0 / Math.Sqrt(k));
            a.Up();
            b.Up();
            using var full = new Buf(DType.BF16, (long)bucket * n);
            using var partial = new Buf(DType.BF16, (long)bucket * n);
            var p = new GemmParams { StructSize = (uint)sizeof(GemmParams), A = a.DevSpan, B = b.DevSpan, D = full.DevSpan, M = bucket, N = (ulong)n, K = (ulong)k, TransposeB = 1, Alpha = 1f, ShapeBucket = bucket };
            using WorkspaceSet ws = c.Lib.MakeWorkspace("gemm-cublaslt", ref p);
            c.Lib.PlanWith("gemm-cublaslt", ref p, ws, c.Stream);
            c.Lib.RunWith("gemm-cublaslt", ref p, ws, c.Stream);
            using var first = Ctx.Fetch(full);
            c.Lib.RunWith("gemm-cublaslt", ref p, ws, c.Stream);
            using var second = Ctx.Fetch(full);
            c.Check($"Gemm n={n} k={k} bucket 64: two runs are bit-identical", Compare.BytesEqual(first, second), AlgoText(ws));
            bool all = true;
            string failed = string.Empty;
            foreach (int m in new[] { 1, 2, 5, 17, 33, 63 })
            {
                partial.FillDevice(0);
                var q = p;
                q.M = (ulong)m;
                q.D = partial.DevSpan;
                c.Lib.RunWith("gemm-cublaslt", ref q, ws, c.Stream);
                using var got = Ctx.Fetch(partial);
                bool same = Compare.BytesEqual(got, first, 0, (long)m * n * 2);
                all &= same;
                if (!same)
                {
                    failed += $" m={m}";
                }
            }

            c.Check($"Gemm n={n} k={k} bucket 64: rows of m in {{1,2,5,17,33,63}} equal the m=64 rows bitwise", all, failed);
            var over = p;
            over.M = bucket + 1;
            c.Check($"Gemm n={n} k={k}: m above the planned bucket -> INVALID_ARGUMENT", c.Lib.TryRunWith("gemm-cublaslt", ref over, ws, c.Stream) == (int)KernelStatus.InvalidArgument);
        }
    }

    public static void LmHead(Ctx c, DType dt, int vocab, int dim, int vocabPadded, int selected, float softCap, Buf? weight = null, Buf? hidden = null, int tokens = 0, string label = "")
    {
        tokens = tokens == 0 ? selected + 3 : tokens;
        Buf w = weight ?? new Buf(dt, (long)vocab * dim).Randn(c.Rng, 0.05);
        Buf h = hidden ?? new Buf(dt, (long)tokens * dim).Randn(c.Rng, 1.0);
        using var rows = new Buf(DType.I32, selected);
        for (int i = 0; i < selected; i++)
        {
            rows.As<int>()[i] = tokens - 1 - i;
        }

        using var logits = new Buf(DType.F32, (long)selected * vocabPadded);
        new Span<byte>((void*)logits.Host, (int)logits.Bytes).Fill(0xAB);
        using var want = logits.Clone();
        h.Up();
        w.Up();
        rows.Up();
        logits.Up();
        var p = new LmHeadParams
        {
            StructSize = (uint)sizeof(LmHeadParams),
            Hidden = h.DevSpan,
            Rows = rows.DevSpan,
            Weight = w.DevSpan,
            Logits = logits.DevSpan,
            Selected = (uint)selected,
            Dim = (uint)dim,
            Vocab = (uint)vocab,
            VocabPadded = (uint)vocabPadded,
            SoftCap = softCap,
        };
        using WorkspaceSet ws = c.Lib.MakeWorkspace("lm-head-cublaslt", ref p);
        c.Lib.PlanWith("lm-head-cublaslt", ref p, ws, c.Stream);
        c.Lib.RunWith("lm-head-cublaslt", ref p, ws, c.Stream);
        using var got = Ctx.Fetch(logits);
        CpuOps.LmHead(new LmHeadArgs { Hidden = h.HostSpan, Rows = rows.HostSpan, Weight = w.HostSpan, Logits = want.HostSpan, Selected = selected, Dim = dim, Vocab = vocab, VocabPadded = vocabPadded, SoftCap = softCap, DType = dt });
        using var gotc = Ctx.Compact(got, selected, vocab, vocabPadded);
        using var wantc = Ctx.Compact(want, selected, vocab, vocabPadded);
        c.Report(Compare.Close($"LmHead {dt} {label}vocab={vocab} dim={dim} padded={vocabPadded} selected={selected} cap={softCap}", gotc, wantc, LogitsTol));
        bool padOk = true;
        var raw = got.As<byte>();
        for (int r = 0; r < selected; r++)
        {
            for (int b = vocab * 4; b < vocabPadded * 4; b++)
            {
                padOk &= raw[(r * vocabPadded * 4) + b] == 0xAB;
            }
        }

        c.Check("LmHead padded logits columns untouched", padOk);
        if (weight is null)
        {
            w.Dispose();
        }

        if (hidden is null)
        {
            h.Dispose();
        }
    }

    // LmHead takes the same pinned-algorithm path: a plan for a bucket of 8 rows gives the first rows of a 1, 3 or 5 row call the same bytes.
    private static void LmHeadPinning(Ctx c)
    {
        const int vocab = 4096, dim = 1024, bucket = 8;
        using var w = new Buf(DType.BF16, (long)vocab * dim).Randn(c.Rng, 0.05);
        using var h = new Buf(DType.BF16, (long)bucket * dim).Randn(c.Rng);
        using var rows = Buf.Ints(0, 1, 2, 3, 4, 5, 6, 7);
        using var logits = new Buf(DType.F32, (long)bucket * vocab);
        w.Up();
        h.Up();
        rows.Up();
        var p = new LmHeadParams { StructSize = (uint)sizeof(LmHeadParams), Hidden = h.DevSpan, Rows = rows.DevSpan, Weight = w.DevSpan, Logits = logits.DevSpan, Selected = bucket, Dim = dim, Vocab = vocab, VocabPadded = vocab, ShapeBucket = bucket };
        using WorkspaceSet ws = c.Lib.MakeWorkspace("lm-head-cublaslt", ref p);
        c.Lib.PlanWith("lm-head-cublaslt", ref p, ws, c.Stream);
        c.Lib.RunWith("lm-head-cublaslt", ref p, ws, c.Stream);
        using var full = Ctx.Fetch(logits);
        logits.FillDevice(0);
        c.Lib.RunWith("lm-head-cublaslt", ref p, ws, c.Stream);
        using var again = Ctx.Fetch(logits);
        bool ok = Compare.BytesEqual(full, again);
        foreach (int n in new[] { 1, 3, 5 })
        {
            logits.FillDevice(0);
            var q = p;
            q.Selected = (uint)n;
            c.Lib.RunWith("lm-head-cublaslt", ref q, ws, c.Stream);
            using var part = Ctx.Fetch(logits);
            ok &= Compare.BytesEqual(full, part, 0, (long)n * vocab * 4);
        }

        c.Check("LmHead bucket 8: two runs, and rows {1,3,5} alone, are bitwise equal to the 8-row call", ok, AlgoText(ws));
    }

    private static void Rejections(Ctx c)
    {
        using var a = new Buf(DType.BF16, 16 * 64);
        using var b = new Buf(DType.BF16, 64 * 64);
        using var d = new Buf(DType.BF16, 16 * 64);
        a.Up();
        b.Up();
        d.Up();
        var good = new GemmParams { StructSize = (uint)sizeof(GemmParams), A = a.DevSpan, B = b.DevSpan, D = d.DevSpan, M = 16, N = 64, K = 64, TransposeB = 1, Alpha = 1f };
        using WorkspaceSet ws = c.Lib.MakeWorkspace("gemm-cublaslt", ref good);
        c.Lib.PlanWith("gemm-cublaslt", ref good, ws, c.Stream);
        var ta = good;
        ta.TransposeA = 1;
        c.Check("Gemm transpose_a -> UNSUPPORTED", c.Lib.TryRunWith("gemm-cublaslt", ref ta, ws, c.Stream) == (int)KernelStatus.Unsupported);
        var f32 = good;
        f32.A.Dtype = DType.F32;
        c.Check("Gemm f32 -> UNSUPPORTED", c.Lib.TryRunWith("gemm-cublaslt", ref f32, ws, c.Stream) == (int)KernelStatus.Unsupported);
        var shape = good;
        shape.N = 128;
        c.Check("Gemm run with another n than planned -> INVALID_ARGUMENT", c.Lib.TryRunWith("gemm-cublaslt", ref shape, ws, c.Stream) == (int)KernelStatus.InvalidArgument);
        var odd = good;
        odd.A.Address += 2;
        c.Check("Gemm misaligned a -> INVALID_ARGUMENT", c.Lib.TryRunWith("gemm-cublaslt", ref odd, ws, c.Stream) == (int)KernelStatus.InvalidArgument);
        var small = good;
        small.D.Length = 10;
        c.Check("Gemm short d -> INVALID_ARGUMENT", c.Lib.TryRunWith("gemm-cublaslt", ref small, ws, c.Stream) == (int)KernelStatus.InvalidArgument);
        // R-019: bf16 with beta != 0 and a bias is outside the gemm-bf16 class on the pinned algorithms; plan() and run() both return the typed
        // UNSUPPORTED before any launch, and D is not written. The same call in f16 is planned.
        using var cBuf = new Buf(DType.BF16, 16 * 64);
        using var biasBuf = new Buf(DType.BF16, 64);
        cBuf.Up();
        biasBuf.Up();
        d.FillDevice(0xAB);
        var cb = good;
        cb.C = cBuf.DevSpan;
        cb.Beta = 1f;
        cb.Bias = biasBuf.DevSpan;
        int cbRun = c.Lib.TryRunWith("gemm-cublaslt", ref cb, ws, c.Stream);
        int cbPlan;
        fixed (Workspace* w = &ws.Value)
        {
            cbPlan = c.Lib.Plan("gemm-cublaslt", &cb, w, c.Stream);
        }

        Cuda.Sync(c.Stream);
        using var untouched = Ctx.Fetch(d);
        bool unwritten = true;
        foreach (byte x in untouched.As<byte>())
        {
            unwritten &= x == 0xAB;
        }

        c.Check("Gemm bf16 with C (beta != 0) and bias -> UNSUPPORTED from plan and from run, D untouched", cbPlan == (int)KernelStatus.Unsupported && cbRun == (int)KernelStatus.Unsupported && unwritten, $"plan {cbPlan}, run {cbRun}, d untouched {unwritten}");
        var tiny = new WorkspaceSet(new WorkspaceSizes { DeviceBytes = 1024, PlanBytes = 4 });
        int tooSmall;
        fixed (Workspace* w = &tiny.Value)
        {
            tooSmall = c.Lib.Plan("gemm-cublaslt", &good, w, c.Stream);
        }

        c.Check("Gemm plan into a too-small plan record -> WORKSPACE_TOO_SMALL", tooSmall == (int)KernelStatus.WorkspaceTooSmall);
        tiny.Dispose();
        Cuda.Sync(c.Stream);
        Cuda.ClearError();
    }
}

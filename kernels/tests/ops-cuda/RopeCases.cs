using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>Rope, default and llama3, with the rotary_dim != head_dim shape Qwen3.5 uses (64 of 256, theta 1e7).</summary>
internal static unsafe class RopeCases
{
    public static void Run(Ctx c)
    {
        foreach (DType dt in new[] { DType.BF16, DType.F16 })
        {
            // Qwen3.5-0.8B full-attention layers: 8 query heads, 2 KV heads, head_dim 256, partial_rotary_factor 0.25, theta 1e7.
            Rope(c, dt, "qwen35", 1, 8, 2, 256, 64, RopeKind.Default, false, 1e7f, MaxPos: 4096);
            Rope(c, dt, "qwen35", 17, 8, 2, 256, 64, RopeKind.Default, false, 1e7f, MaxPos: 4096);
            Rope(c, dt, "qwen35", 129, 8, 2, 256, 64, RopeKind.Default, false, 1e7f, MaxPos: 4096);
            Rope(c, dt, "qwen35-longpos", 129, 8, 2, 256, 64, RopeKind.Default, false, 1e7f, MaxPos: 262144);
            Rope(c, dt, "full-rotary", 33, 8, 2, 256, 0, RopeKind.Default, false, 1e6f, MaxPos: 4096);
            Rope(c, dt, "interleaved", 33, 8, 2, 256, 64, RopeKind.Default, true, 1e7f, MaxPos: 4096);
            Rope(c, dt, "hd64", 33, 8, 2, 64, 0, RopeKind.Default, false, 1e4f, MaxPos: 4096);
            Rope(c, dt, "hd128", 33, 16, 8, 128, 0, RopeKind.Default, false, 1e6f, MaxPos: 4096);
            Rope(c, dt, "llama3-hd128", 33, 8, 2, 128, 0, RopeKind.Llama3, false, 5e5f, MaxPos: 8192);
            Rope(c, dt, "llama3-hd256-rot64", 33, 8, 2, 256, 64, RopeKind.Llama3, false, 5e5f, MaxPos: 8192);
            Rope(c, dt, "huge-positions", 65, 8, 2, 256, 64, RopeKind.Default, false, 1e7f, MaxPos: int.MaxValue);
            Rope(c, dt, "llama3-longpos", 65, 8, 2, 128, 0, RopeKind.Llama3, false, 5e5f, MaxPos: 131072);
        }

        Reject(c);
    }

    private static void Rope(Ctx c, DType dt, string label, int tokens, int qh, int kh, int hd, int rot, RopeKind kind, bool interleaved, float theta, int MaxPos)
    {
        using var q = new Buf(dt, (long)tokens * qh * hd).Randn(c.Rng, 1.0);
        using var k = new Buf(dt, (long)tokens * kh * hd).Randn(c.Rng, 1.0);
        using var pos = new Buf(DType.I32, tokens);
        for (int i = 0; i < tokens; i++)
        {
            pos.As<int>()[i] = c.Rng.Int(0, MaxPos);
        }

        using var qw = q.Clone();
        using var kw = k.Clone();
        q.Up();
        k.Up();
        pos.Up();
        var p = new RopeParams
        {
            StructSize = (uint)sizeof(RopeParams),
            Q = q.DevSpan,
            K = k.DevSpan,
            Positions = pos.DevSpan,
            Tokens = (uint)tokens,
            QHeads = (uint)qh,
            KvHeads = (uint)kh,
            HeadDim = (uint)hd,
            RotaryDim = (uint)rot,
            Kind = kind,
            Interleaved = interleaved ? 1u : 0u,
            Theta = theta,
            Llama3Factor = 8f,
            Llama3LowFreqFactor = 1f,
            Llama3HighFreqFactor = 4f,
            Llama3OriginalMaxPosition = 8192,
        };
        c.Lib.Exec("rope-roost", ref p, c.Stream);
        using var gq = Ctx.Fetch(q);
        using var gk = Ctx.Fetch(k);
        CpuOps.Rope(new RopeArgs
        {
            Q = qw.HostSpan,
            K = kw.HostSpan,
            Positions = pos.HostSpan,
            Tokens = tokens,
            QHeads = qh,
            KvHeads = kh,
            HeadDim = hd,
            RotaryDim = rot,
            Kind = kind,
            Interleaved = interleaved,
            Theta = theta,
            Llama3Factor = 8f,
            Llama3LowFreqFactor = 1f,
            Llama3HighFreqFactor = 4f,
            Llama3OriginalMaxPosition = 8192,
            DType = dt,
        });
        string name = $"Rope {dt} {label} tokens={tokens} {qh}/{kh}x{hd} rot={(rot == 0 ? hd : rot)} maxpos={MaxPos}";
        c.Report(Compare.Close(name + " q", gq, qw, Tol.Elementwise(dt)));
        c.Report(Compare.Close(name + " k", gk, kw, Tol.Elementwise(dt)));
    }

    private static void Reject(Ctx c)
    {
        using var q = new Buf(DType.BF16, 4 * 8 * 256);
        using var k = new Buf(DType.BF16, 4 * 2 * 256);
        using var pos = new Buf(DType.I32, 4);
        q.Up();
        k.Up();
        pos.Up();
        var good = new RopeParams
        {
            StructSize = (uint)sizeof(RopeParams),
            Q = q.DevSpan,
            K = k.DevSpan,
            Positions = pos.DevSpan,
            Tokens = 4,
            QHeads = 8,
            KvHeads = 2,
            HeadDim = 256,
            RotaryDim = 64,
            Theta = 1e7f,
        };
        var odd = good;
        odd.RotaryDim = 63;
        c.Check("Rope odd rotary_dim -> INVALID_ARGUMENT", c.Lib.Try("rope-roost", ref odd, c.Stream) == (int)KernelStatus.InvalidArgument);
        var big = good;
        big.RotaryDim = 512;
        c.Check("Rope rotary_dim > head_dim -> INVALID_ARGUMENT", c.Lib.Try("rope-roost", ref big, c.Stream) == (int)KernelStatus.InvalidArgument);
        var hd = good;
        hd.HeadDim = 1024;
        hd.RotaryDim = 600;
        c.Check("Rope rotary_dim 600 -> UNSUPPORTED", c.Lib.Try("rope-roost", ref hd, c.Stream) == (int)KernelStatus.Unsupported);
        var l3 = good;
        l3.Kind = RopeKind.Llama3;
        c.Check("Rope llama3 without factors -> INVALID_ARGUMENT", c.Lib.Try("rope-roost", ref l3, c.Stream) == (int)KernelStatus.InvalidArgument);
        var posType = good;
        posType.Positions.Dtype = DType.I64;
        c.Check("Rope int64 positions -> INVALID_ARGUMENT", c.Lib.Try("rope-roost", ref posType, c.Stream) == (int)KernelStatus.InvalidArgument);
        Cuda.Sync(c.Stream);
        Cuda.ClearError();
    }
}

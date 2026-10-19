using Starlings.Roost.Abi;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>
/// ROOST_FLAG_DETERMINISTIC and ROOST_FLAG_BATCH_INVARIANT evidence for the row-wise ops: the same input gives the same bytes
/// twice, and the first rows of a batch are bitwise identical when the call covers only those rows.
/// </summary>
internal static unsafe class Invariance
{
    public static void Run(Ctx c)
    {
        foreach (DType dt in new[] { DType.BF16, DType.F16 })
        {
            RmsNorm(c, dt);
            QkNorm(c, dt);
            SiluAndMul(c, dt);
            Rope(c, dt);
        }
    }

    private static string Name(string op, DType dt) => $"{op} {dt}: two runs, and rows {{1,17}} alone, are bitwise equal to the 64-row call";

    private static void RmsNorm(Ctx c, DType dt)
    {
        const int rows = 64, dim = 1024;
        using var x = new Buf(dt, rows * dim).Randn(c.Rng, 2.0);
        using var w = new Buf(dt, dim).Randn(c.Rng, 0.2, 1.0);
        using var y = new Buf(dt, rows * dim);
        x.Up();
        w.Up();
        var p = new RmsNormParams { StructSize = (uint)sizeof(RmsNormParams), Input = x.DevSpan, Weight = w.DevSpan, Output = y.DevSpan, Rows = rows, Dim = dim, Eps = 1e-6f };
        c.Lib.Exec("rms-norm-flashinfer", ref p, c.Stream);
        using var full = Ctx.Fetch(y);
        bool ok = true;
        y.FillDevice(0);
        c.Lib.Exec("rms-norm-flashinfer", ref p, c.Stream);
        using var again = Ctx.Fetch(y);
        ok &= Compare.BytesEqual(full, again);
        foreach (int n in new[] { 1, 17 })
        {
            y.FillDevice(0);
            var q = p;
            q.Rows = (uint)n;
            c.Lib.Exec("rms-norm-flashinfer", ref q, c.Stream);
            using var part = Ctx.Fetch(y);
            ok &= Compare.BytesEqual(full, part, 0, (long)n * dim * 2);
        }

        c.Check(Name("RmsNorm", dt), ok);
    }

    private static void QkNorm(Ctx c, DType dt)
    {
        const int tokens = 64, qh = 8, kh = 2, hd = 256;
        using var q0 = new Buf(dt, tokens * qh * hd).Randn(c.Rng, 1.5);
        using var k0 = new Buf(dt, tokens * kh * hd).Randn(c.Rng, 1.5);
        using var qw = new Buf(dt, hd).Randn(c.Rng, 0.2, 1.0);
        using var kw = new Buf(dt, hd).Randn(c.Rng, 0.2, 1.0);
        using var q = new Buf(dt, q0.Count);
        using var k = new Buf(dt, k0.Count);
        qw.Up();
        kw.Up();
        Buf[] RunWith(int n)
        {
            Cuda.ToDevice(q.Dev, q0.Host, q0.Bytes);
            Cuda.ToDevice(k.Dev, k0.Host, k0.Bytes);
            var p = new QkNormParams { StructSize = (uint)sizeof(QkNormParams), Q = q.DevSpan, K = k.DevSpan, QWeight = qw.DevSpan, KWeight = kw.DevSpan, Tokens = (uint)n, QHeads = qh, KvHeads = kh, HeadDim = hd, Eps = 1e-6f };
            c.Lib.Exec("qk-norm-flashinfer", ref p, c.Stream);
            return [Ctx.Fetch(q), Ctx.Fetch(k)];
        }

        Buf[] full = RunWith(tokens), again = RunWith(tokens);
        bool ok = Compare.BytesEqual(full[0], again[0]) && Compare.BytesEqual(full[1], again[1]);
        foreach (int n in new[] { 1, 17 })
        {
            Buf[] part = RunWith(n);
            ok &= Compare.BytesEqual(full[0], part[0], 0, (long)n * qh * hd * 2) && Compare.BytesEqual(full[1], part[1], 0, (long)n * kh * hd * 2);
            part[0].Dispose();
            part[1].Dispose();
        }

        c.Check(Name("QkNorm", dt), ok);
        foreach (Buf b in full.Concat(again))
        {
            b.Dispose();
        }
    }

    private static void SiluAndMul(Ctx c, DType dt)
    {
        const int rows = 64, dim = 3584;
        using var x = new Buf(dt, 2L * rows * dim).Randn(c.Rng, 2.0);
        using var y = new Buf(dt, (long)rows * dim);
        x.Up();
        var p = new ActAndMulParams { StructSize = (uint)sizeof(ActAndMulParams), Input = x.DevSpan, Output = y.DevSpan, Rows = rows, Dim = dim };
        c.Lib.Exec("silu-and-mul-flashinfer", ref p, c.Stream);
        using var full = Ctx.Fetch(y);
        y.FillDevice(0);
        c.Lib.Exec("silu-and-mul-flashinfer", ref p, c.Stream);
        using var again = Ctx.Fetch(y);
        bool ok = Compare.BytesEqual(full, again);
        foreach (int n in new[] { 1, 17 })
        {
            y.FillDevice(0);
            var q = p;
            q.Rows = (uint)n;
            c.Lib.Exec("silu-and-mul-flashinfer", ref q, c.Stream);
            using var part = Ctx.Fetch(y);
            ok &= Compare.BytesEqual(full, part, 0, (long)n * dim * 2);
        }

        c.Check(Name("SiluAndMul", dt), ok);
    }

    private static void Rope(Ctx c, DType dt)
    {
        const int tokens = 64, qh = 8, kh = 2, hd = 256;
        using var q0 = new Buf(dt, tokens * qh * hd).Randn(c.Rng);
        using var k0 = new Buf(dt, tokens * kh * hd).Randn(c.Rng);
        using var pos = new Buf(DType.I32, tokens);
        for (int i = 0; i < tokens; i++)
        {
            pos.As<int>()[i] = c.Rng.Int(0, 100000);
        }

        using var q = new Buf(dt, q0.Count);
        using var k = new Buf(dt, k0.Count);
        pos.Up();
        Buf[] RunWith(int n)
        {
            Cuda.ToDevice(q.Dev, q0.Host, q0.Bytes);
            Cuda.ToDevice(k.Dev, k0.Host, k0.Bytes);
            var p = new RopeParams { StructSize = (uint)sizeof(RopeParams), Q = q.DevSpan, K = k.DevSpan, Positions = pos.DevSpan, Tokens = (uint)n, QHeads = qh, KvHeads = kh, HeadDim = hd, RotaryDim = 64, Theta = 1e7f };
            c.Lib.Exec("rope-roost", ref p, c.Stream);
            return [Ctx.Fetch(q), Ctx.Fetch(k)];
        }

        Buf[] full = RunWith(tokens), again = RunWith(tokens);
        bool ok = Compare.BytesEqual(full[0], again[0]) && Compare.BytesEqual(full[1], again[1]);
        foreach (int n in new[] { 1, 17 })
        {
            Buf[] part = RunWith(n);
            ok &= Compare.BytesEqual(full[0], part[0], 0, (long)n * qh * hd * 2) && Compare.BytesEqual(full[1], part[1], 0, (long)n * kh * hd * 2);
            part[0].Dispose();
            part[1].Dispose();
        }

        c.Check(Name("Rope", dt), ok);
        foreach (Buf b in full.Concat(again))
        {
            b.Dispose();
        }
    }
}

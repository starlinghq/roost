using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>RmsNorm, QkNorm, SiluAndMul: row-wise ops compared with CpuOps.</summary>
internal static unsafe class NormCases
{
    private const float Eps = 1e-6f; // Qwen3.5-0.8B rms_norm_eps

    public static void Run(Ctx c)
    {
        foreach (DType dt in new[] { DType.BF16, DType.F16 })
        {
            foreach (int rows in new[] { 1, 7, 64, 257 })
            {
                RmsNorm(c, dt, rows, 1024, 0, 0);
            }

            RmsNorm(c, dt, 5, 1024, 1280, 1152);
            RmsNorm(c, dt, 3, 3584, 0, 0);
            QkNorm(c, dt, 1, 0, 0);
            QkNorm(c, dt, 5, 0, 0);
            QkNorm(c, dt, 33, 0, 0);
            QkNorm(c, dt, 9, 2560, 704);
            SiluAndMul(c, dt, 1, 3584);
            SiluAndMul(c, dt, 8, 3584);
            SiluAndMul(c, dt, 33, 3584);
            SiluAndMul(c, dt, 3, 1000);
            SiluAndMul(c, dt, 3, 1001);
        }

        Rejections(c);
    }

    private static void RmsNorm(Ctx c, DType dt, int rows, int dim, int inStride, int outStride)
    {
        int istr = inStride == 0 ? dim : inStride, ostr = outStride == 0 ? dim : outStride;
        using var x = new Buf(dt, (long)rows * istr).Randn(c.Rng, 2.0);
        using var w = new Buf(dt, dim).Randn(c.Rng, 0.2, 1.0);
        using var y = new Buf(dt, (long)rows * ostr);
        x.Up();
        w.Up();
        y.FillDevice(0xAB);
        var p = new RmsNormParams
        {
            StructSize = (uint)sizeof(RmsNormParams),
            Input = x.DevSpan,
            Weight = w.DevSpan,
            Output = y.DevSpan,
            Rows = (uint)rows,
            Dim = (uint)dim,
            InputStride = (uint)inStride,
            OutputStride = (uint)outStride,
            Eps = Eps,
        };
        c.Lib.Exec("rms-norm-flashinfer", ref p, c.Stream);
        using var got = Ctx.Fetch(y);
        using var xc = Ctx.Compact(x, rows, dim, istr);
        using var want = new Buf(dt, (long)rows * dim);
        CpuOps.RmsNorm(new RmsNormArgs { Input = xc.HostSpan, Weight = w.HostSpan, Output = want.HostSpan, Rows = rows, Dim = dim, Eps = Eps, DType = dt });
        using var gotc = Ctx.Compact(got, rows, dim, ostr);
        c.Report(Compare.Close($"RmsNorm {dt} rows={rows} dim={dim} strides={istr}/{ostr}", gotc, want, Tol.Elementwise(dt)));
        if (ostr > dim)
        {
            bool padOk = true;
            var raw = got.As<byte>();
            int es = Elem.Size(dt);
            for (int r = 0; r < rows; r++)
            {
                for (int b = dim * es; b < ostr * es; b++)
                {
                    padOk &= raw[(r * ostr * es) + b] == 0xAB;
                }
            }

            c.Check($"RmsNorm {dt} output padding untouched", padOk);
        }
    }

    private static void QkNorm(Ctx c, DType dt, int tokens, int qStride, int kStride)
    {
        const int qh = 8, kh = 2, hd = 256;
        int qs = qStride == 0 ? qh * hd : qStride, ks = kStride == 0 ? kh * hd : kStride;
        using var q = new Buf(dt, (long)tokens * qs).Randn(c.Rng, 1.5);
        using var k = new Buf(dt, (long)tokens * ks).Randn(c.Rng, 1.5);
        using var qw = new Buf(dt, hd).Randn(c.Rng, 0.2, 1.0);
        using var kw = new Buf(dt, hd).Randn(c.Rng, 0.2, 1.0);
        q.Up();
        k.Up();
        qw.Up();
        kw.Up();
        var p = new QkNormParams
        {
            StructSize = (uint)sizeof(QkNormParams),
            Q = q.DevSpan,
            K = k.DevSpan,
            QWeight = qw.DevSpan,
            KWeight = kw.DevSpan,
            Tokens = (uint)tokens,
            QHeads = qh,
            KvHeads = kh,
            HeadDim = hd,
            QStride = (uint)qStride,
            KStride = (uint)kStride,
            Eps = Eps,
        };
        c.Lib.Exec("qk-norm-flashinfer", ref p, c.Stream);
        using var gq = Ctx.Fetch(q);
        using var gk = Ctx.Fetch(k);
        using var qc = Ctx.Compact(q, tokens, qh * hd, qs);
        using var kc = Ctx.Compact(k, tokens, kh * hd, ks);
        CpuOps.QkNorm(new QkNormArgs { Q = qc.HostSpan, K = kc.HostSpan, QWeight = qw.HostSpan, KWeight = kw.HostSpan, Tokens = tokens, QHeads = qh, KvHeads = kh, HeadDim = hd, Eps = Eps, DType = dt });
        using var gqc = Ctx.Compact(gq, tokens, qh * hd, qs);
        using var gkc = Ctx.Compact(gk, tokens, kh * hd, ks);
        c.Report(Compare.Close($"QkNorm {dt} q tokens={tokens} strides={qs}/{ks}", gqc, qc, Tol.Elementwise(dt)));
        c.Report(Compare.Close($"QkNorm {dt} k tokens={tokens} strides={qs}/{ks}", gkc, kc, Tol.Elementwise(dt)));
    }

    private static void SiluAndMul(Ctx c, DType dt, int rows, int dim)
    {
        using var x = new Buf(dt, 2L * rows * dim).Randn(c.Rng, 2.0);
        using var y = new Buf(dt, (long)rows * dim);
        x.Up();
        y.FillDevice(0xAB);
        var p = new ActAndMulParams { StructSize = (uint)sizeof(ActAndMulParams), Input = x.DevSpan, Output = y.DevSpan, Rows = (uint)rows, Dim = (uint)dim };
        c.Lib.Exec("silu-and-mul-flashinfer", ref p, c.Stream);
        using var got = Ctx.Fetch(y);
        using var want = new Buf(dt, (long)rows * dim);
        CpuOps.SiluAndMul(new SiluAndMulArgs { Input = x.HostSpan, Output = want.HostSpan, Rows = rows, Dim = dim, DType = dt });
        c.Report(Compare.Close($"SiluAndMul {dt} rows={rows} dim={dim}", got, want, Tol.Elementwise(dt)));
    }

    private static void Rejections(Ctx c)
    {
        using var x = new Buf(DType.BF16, 64 * 8);
        using var w = new Buf(DType.BF16, 64);
        using var y = new Buf(DType.BF16, 64 * 8);
        x.Up();
        w.Up();
        y.Up();
        var good = new RmsNormParams { StructSize = (uint)sizeof(RmsNormParams), Input = x.DevSpan, Weight = w.DevSpan, Output = y.DevSpan, Rows = 8, Dim = 64, Eps = Eps };
        var shortStruct = good;
        shortStruct.StructSize = 8;
        c.Check("RmsNorm short struct_size -> ABI_MISMATCH", c.Lib.Try("rms-norm-flashinfer", ref shortStruct, c.Stream) == (int)KernelStatus.AbiMismatch);
        var badMinor = good;
        badMinor.AbiMinor = 9;
        c.Check("RmsNorm abi_minor above the library's -> ABI_MISMATCH", c.Lib.Try("rms-norm-flashinfer", ref badMinor, c.Stream) == (int)KernelStatus.AbiMismatch);
        var shortSpan = good;
        shortSpan.Input.Length = 10;
        c.Check("RmsNorm short span -> INVALID_ARGUMENT", c.Lib.Try("rms-norm-flashinfer", ref shortSpan, c.Stream) == (int)KernelStatus.InvalidArgument);
        var wrongType = good;
        wrongType.Weight.Dtype = DType.F32;
        c.Check("RmsNorm weight dtype mismatch -> INVALID_ARGUMENT", c.Lib.Try("rms-norm-flashinfer", ref wrongType, c.Stream) == (int)KernelStatus.InvalidArgument);
        var reserved = good;
        reserved.Reserved0 = 1;
        c.Check("RmsNorm non-zero reserved -> INVALID_ARGUMENT", c.Lib.Try("rms-norm-flashinfer", ref reserved, c.Stream) == (int)KernelStatus.InvalidArgument);
        var f32 = good;
        f32.Input.Dtype = DType.F32;
        f32.Input.Length = 1000;
        c.Check("RmsNorm f32 -> UNSUPPORTED", c.Lib.Try("rms-norm-flashinfer", ref f32, c.Stream) == (int)KernelStatus.Unsupported);
        var misaligned = good;
        misaligned.Input.Address += 2;
        c.Check("RmsNorm misaligned input -> INVALID_ARGUMENT", c.Lib.Try("rms-norm-flashinfer", ref misaligned, c.Stream) == (int)KernelStatus.InvalidArgument);
        Cuda.Sync(c.Stream);
        Cuda.ClearError();
    }
}

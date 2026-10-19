using Starlings.Roost;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Roost.HybridConformance;

/// <summary>GatedDeltaRuleGates, GatedRmsNorm and SigmoidMul (contract version 3) against CpuOps.</summary>
internal static unsafe class GatingCases
{
    // Qwen3.5-0.8B (config.json at 2fc06364715b967f1860aea9cf38778875588b17): 16 value heads of 128 in the linear layers,
    // 8 attention heads of head_dim 256 in the full-attention layers, rms_norm_eps 1e-6.
    private const int VHeads = 16;
    private const int VDim = 128;
    private const int AttnDim = 8 * 256;
    private const float Eps = 1e-6f;

    // Limits (see kernels/README.md, Gating ops): a float32 output differs from the double reference by float rounding of
    // a few operations, a bf16 output by at most one bf16 step (2^-8 to 2^-7 relative to the result; 2^-7 is the limit).
    private const double Bf16Limit = 1.0 / 128;
    private const double F32Limit = 1e-5;

    public static void Run(Library lib, Report r, nint stream)
    {
        // Qwen3.5: a and b bf16 (in_proj_a, in_proj_b outputs), A_log f32, dt_bias bf16, g f32, beta bf16.
        Gates(lib, r, stream, "gates/qwen35 prefill 37 tokens, contiguous", 37, 0, 0, 0, DType.BF16, DType.F32, DType.BF16, DType.F32, DType.BF16, 6.0, 1);
        Gates(lib, r, stream, "gates/qwen35 prefill 300 tokens, a and b halves of a [tokens, 32] view", 300, 32, 0, 16, DType.BF16, DType.F32, DType.BF16, DType.F32, DType.BF16, 6.0, 2);
        Gates(lib, r, stream, "gates/qwen35 decode 1 token", 1, 0, 0, 0, DType.BF16, DType.F32, DType.BF16, DType.F32, DType.BF16, 6.0, 3);
        Gates(lib, r, stream, "gates/qwen35 softplus above the threshold (a up to 40)", 64, 0, 0, 0, DType.BF16, DType.F32, DType.BF16, DType.F32, DType.BF16, 40.0, 4);
        Gates(lib, r, stream, "gates/f32 activations, bf16 A_log and g, f32 beta, strided", 41, 40, 3, 21, DType.F32, DType.BF16, DType.F32, DType.BF16, DType.F32, 8.0, 5);

        // Qwen3.5 linear-attention output norm: rows = tokens * 16, dim 128, f32 weight.
        Norm(lib, r, stream, "norm/qwen35 prefill 37 tokens", 37 * VHeads, VDim, DType.BF16, DType.F32, false, 11);
        Norm(lib, r, stream, "norm/qwen35 decode 1 token", VHeads, VDim, DType.BF16, DType.F32, false, 12);
        Norm(lib, r, stream, "norm/qwen35 300 tokens in place", 300 * VHeads, VDim, DType.BF16, DType.F32, true, 13);
        Norm(lib, r, stream, "norm/f32 activations, bf16 weight, dim 100", 33, 100, DType.F32, DType.BF16, false, 14);
        Norm(lib, r, stream, "norm/f32 dim 3000 (row wider than a block)", 9, 3000, DType.F32, DType.F32, true, 15);

        // Qwen3.5 attention output gate: rows = tokens, dim = 8 heads * 256.
        Mul(lib, r, stream, "sigmoid-mul/qwen35 prefill 37 tokens", 37, AttnDim, DType.BF16, false, 21);
        Mul(lib, r, stream, "sigmoid-mul/qwen35 decode 1 token", 1, AttnDim, DType.BF16, false, 22);
        Mul(lib, r, stream, "sigmoid-mul/qwen35 300 tokens in place", 300, AttnDim, DType.BF16, true, 23);
        Mul(lib, r, stream, "sigmoid-mul/f32 odd size", 7, 1001, DType.F32, false, 24);

        if (!lib.GraphMode)
        {
            Rejections(lib, r, stream);
        }
    }

    private static float[] Rand(Random rng, long n, double scale) =>
        Enumerable.Range(0, (int)n).Select(_ => (float)(((rng.NextDouble() * 2) - 1) * scale)).ToArray();

    private static Buf Make(DType t, float[] v) => t == DType.BF16 ? Buf.FromBf16(v) : Buf.FromF32(v);

    private static (double Worst, double Abs) Worst(Buf got, byte[] gotBytes, Buf want, byte[] wantBytes, double floor)
    {
        var e = Compare.Errors(i => got.At(gotBytes, i), i => want.At(wantBytes, i), got.Elements, floor);
        return (e.MaxScaled, e.MaxAbs);
    }

    private static double Limit(DType t) => t == DType.BF16 ? Bf16Limit : F32Limit;

    private static void Gates(Library lib, Report r, nint stream, string name, int tokens, int fusedStride, int aCol, int bCol,
        DType act, DType alogT, DType dtT, DType gT, DType betaT, double aScale, int seed)
    {
        var entry = lib.Find("gated-delta-rule-gates-cuda")!.Value;
        var rng = new Random(seed);
        if (fusedStride != 0 && (aCol + VHeads > fusedStride || bCol + VHeads > fusedStride))
        {
            throw new ArgumentException("a and b columns do not fit the fused stride");
        }

        // fusedStride 0: a and b are separate contiguous [tokens, 16] tensors. Otherwise both are column views of one
        // [tokens, fusedStride] buffer, a at column aCol (stride = fusedStride) and b at column bCol.
        var aStride = fusedStride;
        var bStride = fusedStride;
        Buf a, b;
        nint aAddrHost, bAddrHost, aAddrDev, bAddrDev;
        ulong aLen, bLen;
        if (fusedStride == 0)
        {
            a = Make(act, Rand(rng, (long)tokens * VHeads, aScale));
            b = Make(act, Rand(rng, (long)tokens * VHeads, 6.0));
            aAddrHost = a.Host;
            bAddrHost = b.Host;
            aAddrDev = a.Device;
            bAddrDev = b.Device;
            aLen = (ulong)a.Elements;
            bLen = (ulong)b.Elements;
        }
        else
        {
            // One fused buffer holds both; its a part gets aScale values and its b part the b range.
            var v = Rand(rng, (long)tokens * fusedStride, 6.0);
            for (var t = 0; t < tokens; t++)
            {
                for (var h = 0; h < VHeads; h++)
                {
                    v[(t * fusedStride) + aCol + h] *= (float)(aScale / 6.0);
                }
            }

            a = Make(act, v);
            b = a;
            var es = Buf.Size(act);
            aAddrHost = a.Host + (aCol * es);
            bAddrHost = a.Host + (bCol * es);
            aAddrDev = a.Device + (aCol * es);
            bAddrDev = a.Device + (bCol * es);
            aLen = (ulong)(a.Elements - aCol);
            bLen = (ulong)(a.Elements - bCol);
        }

        using var aOwner = a;
        using var bOwner = ReferenceEquals(a, b) ? null : b;
        using var aLog = Make(alogT, Rand(rng, VHeads, 3.0));
        using var dt = Make(dtT, Rand(rng, VHeads, 3.0));
        using var g = new Buf(gT, (long)tokens * VHeads);
        using var beta = new Buf(betaT, (long)tokens * VHeads);
        a.ToDevice();
        if (!ReferenceEquals(a, b))
        {
            b.ToDevice();
        }

        aLog.ToDevice();
        dt.ToDevice();
        Cuda.Fill(g.Device, 0xFF, g.Bytes);
        Cuda.Fill(beta.Device, 0xFF, beta.Bytes);

        var cpuG = new Buf(gT, g.Elements);
        var cpuBeta = new Buf(betaT, beta.Elements);
        CpuOps.GatedDeltaRuleGates(new GatedDeltaRuleGatesArgs
        {
            A = new DeviceSpan { Address = aAddrHost, Length = aLen, Dtype = act },
            B = new DeviceSpan { Address = bAddrHost, Length = bLen, Dtype = act },
            ALog = aLog.HostSpan(),
            DtBias = dt.HostSpan(),
            G = cpuG.HostSpan(),
            Beta = cpuBeta.HostSpan(),
            Tokens = tokens,
            VHeads = VHeads,
            AStride = aStride,
            BStride = bStride,
            DType = act,
        });

        lib.Run(entry, new GatedDeltaRuleGatesParams
        {
            StructSize = (uint)sizeof(GatedDeltaRuleGatesParams),
            AbiMinor = 2,
            A = new DeviceSpan { Address = aAddrDev, Length = aLen, Dtype = act },
            B = new DeviceSpan { Address = bAddrDev, Length = bLen, Dtype = act },
            ALog = aLog.DevSpan(),
            DtBias = dt.DevSpan(),
            G = g.DevSpan(),
            Beta = beta.DevSpan(),
            Tokens = (uint)tokens,
            VHeads = VHeads,
            AStride = (uint)aStride,
            BStride = (uint)bStride,
        }, null, stream);
        Cuda.Sync(stream);

        var gotG = g.FromDevice();
        var gotBeta = beta.FromDevice();
        var (gs, ga) = Worst(g, gotG, cpuG, cpuG.HostBytes(), 1e-3);
        var (bs, ba) = Worst(beta, gotBeta, cpuBeta, cpuBeta.HostBytes(), 1e-3);
        var ok = gs <= Limit(gT) && bs <= Limit(betaT);
        r.Check(ok, name, $"tokens={tokens} g max_scaled={gs:E2} (limit {Limit(gT):E2}) max_abs={ga:E2}; beta max_scaled={bs:E2} (limit {Limit(betaT):E2}) max_abs={ba:E2}");
        cpuG.Dispose();
        cpuBeta.Dispose();
    }

    private static void Norm(Library lib, Report r, nint stream, string name, int rows, int dim, DType act, DType wt, bool inPlace, int seed)
    {
        var entry = lib.Find("gated-rms-norm-cuda")!.Value;
        var rng = new Random(seed);
        long n = (long)rows * dim;
        using var input = Make(act, Rand(rng, n, 2.0));
        using var gate = Make(act, Rand(rng, n, 4.0));
        // linear_attn.norm.weight: layer 0 of the checkpoint holds 0.52 to 1.06.
        using var weight = Make(wt, Rand(rng, dim, 0.3).Select(x => 0.8f + x).ToArray());
        using var output = new Buf(act, n);
        input.ToDevice();
        gate.ToDevice();
        weight.ToDevice();
        Cuda.Fill(output.Device, 0xFF, output.Bytes);

        var cpuOut = new Buf(act, n);
        var cpuIn = input.Clone();
        CpuOps.GatedRmsNorm(new GatedRmsNormArgs
        {
            Input = cpuIn.HostSpan(),
            Gate = gate.HostSpan(),
            Weight = weight.HostSpan(),
            Output = cpuOut.HostSpan(),
            Rows = rows,
            Dim = dim,
            Eps = Eps,
            DType = act,
        });

        lib.Run(entry, new GatedRmsNormParams
        {
            StructSize = (uint)sizeof(GatedRmsNormParams),
            AbiMinor = 2,
            Input = input.DevSpan(),
            Gate = gate.DevSpan(),
            Weight = weight.DevSpan(),
            Output = inPlace ? input.DevSpan() : output.DevSpan(),
            Rows = (uint)rows,
            Dim = (uint)dim,
            Eps = Eps,
        }, null, stream);
        Cuda.Sync(stream);

        var got = inPlace ? input.FromDevice() : output.FromDevice();
        var (s, a) = Worst(cpuOut, got, cpuOut, cpuOut.HostBytes(), act == DType.BF16 ? 1e-2 : 1e-6);
        var ok = s <= Limit(act);
        r.Check(ok, name, $"rows={rows} dim={dim} in_place={inPlace} max_scaled={s:E2} (limit {Limit(act):E2}) max_abs={a:E2}");
        cpuOut.Dispose();
        cpuIn.Dispose();
    }

    private static void Mul(Library lib, Report r, nint stream, string name, int rows, int dim, DType act, bool inPlace, int seed)
    {
        var entry = lib.Find("sigmoid-mul-cuda")!.Value;
        var rng = new Random(seed);
        long n = (long)rows * dim;
        using var input = Make(act, Rand(rng, n, 2.0));
        using var gate = Make(act, Rand(rng, n, 8.0));
        using var output = new Buf(act, n);
        input.ToDevice();
        gate.ToDevice();
        Cuda.Fill(output.Device, 0xFF, output.Bytes);

        var cpuOut = new Buf(act, n);
        CpuOps.SigmoidMul(new SigmoidMulArgs { Input = input.HostSpan(), Gate = gate.HostSpan(), Output = cpuOut.HostSpan(), Rows = rows, Dim = dim, DType = act });

        lib.Run(entry, new SigmoidMulParams
        {
            StructSize = (uint)sizeof(SigmoidMulParams),
            AbiMinor = 2,
            Input = input.DevSpan(),
            Gate = gate.DevSpan(),
            Output = inPlace ? input.DevSpan() : output.DevSpan(),
            Rows = (uint)rows,
            Dim = (uint)dim,
        }, null, stream);
        Cuda.Sync(stream);

        var got = inPlace ? input.FromDevice() : output.FromDevice();
        var (s, a) = Worst(cpuOut, got, cpuOut, cpuOut.HostBytes(), act == DType.BF16 ? 1e-3 : 1e-6);
        var ok = s <= Limit(act);
        r.Check(ok, name, $"rows={rows} dim={dim} in_place={inPlace} max_scaled={s:E2} (limit {Limit(act):E2}) max_abs={a:E2}");
        cpuOut.Dispose();
    }

    // Arguments the library must refuse before it launches anything, with the typed status the contract names.
    private static void Rejections(Library lib, Report r, nint stream)
    {
        var gatesEntry = lib.Find("gated-delta-rule-gates-cuda")!.Value;
        var normEntry = lib.Find("gated-rms-norm-cuda")!.Value;
        var mulEntry = lib.Find("sigmoid-mul-cuda")!.Value;
        const int Tokens = 5;
        using var a = Buf.FromBf16(Rand(new Random(1), Tokens * VHeads, 1));
        using var b = Buf.FromBf16(Rand(new Random(2), Tokens * VHeads, 1));
        using var aLog = Buf.FromF32(Rand(new Random(3), VHeads, 1));
        using var dt = Buf.FromBf16(Rand(new Random(4), VHeads, 1));
        using var g = new Buf(DType.F32, Tokens * VHeads);
        using var beta = new Buf(DType.BF16, Tokens * VHeads);
        var good = new GatedDeltaRuleGatesParams
        {
            StructSize = (uint)sizeof(GatedDeltaRuleGatesParams),
            AbiMinor = 2,
            A = a.DevSpan(),
            B = b.DevSpan(),
            ALog = aLog.DevSpan(),
            DtBias = dt.DevSpan(),
            G = g.DevSpan(),
            Beta = beta.DevSpan(),
            Tokens = Tokens,
            VHeads = VHeads,
        };
        void ExpectGates(string what, GatedDeltaRuleGatesParams p, int status)
        {
            var rc = lib.RunRaw(gatesEntry, p, null, 0);
            r.Check(rc == status, $"gating/reject/gates {what}", $"status {rc} (want {status}): {lib.LastError()}");
        }

        var p1 = good;
        p1.StructSize = 32;
        ExpectGates("struct_size too small", p1, 5);
        p1 = good;
        p1.AStride = VHeads - 1;
        ExpectGates("a_stride below v_heads", p1, 1);
        p1 = good;
        p1.A.Length -= 1;
        ExpectGates("a shorter than its reach", p1, 1);
        p1 = good;
        p1.G.Length -= 1;
        ExpectGates("g shorter than tokens * v_heads", p1, 1);
        p1 = good;
        p1.ALog.Dtype = DType.F16;
        ExpectGates("f16 A_log", p1, 2);
        p1 = good;
        p1.Beta.Address = p1.A.Address + 2;
        ExpectGates("beta overlapping a", p1, 1);
        p1 = good;
        p1.G.Address += 2;
        ExpectGates("g not aligned to f32", p1, 1);

        const int Rows = 4;
        const int Dim = 64;
        using var input = Buf.FromBf16(Rand(new Random(5), Rows * Dim, 1));
        using var gate = Buf.FromBf16(Rand(new Random(6), Rows * Dim, 1));
        using var weight = Buf.FromF32(Rand(new Random(7), Dim, 1));
        using var output = new Buf(DType.BF16, Rows * Dim);
        var norm = new GatedRmsNormParams
        {
            StructSize = (uint)sizeof(GatedRmsNormParams),
            AbiMinor = 2,
            Input = input.DevSpan(),
            Gate = gate.DevSpan(),
            Weight = weight.DevSpan(),
            Output = output.DevSpan(),
            Rows = Rows,
            Dim = Dim,
            Eps = Eps,
        };
        void ExpectNorm(string what, GatedRmsNormParams p, int status)
        {
            var rc = lib.RunRaw(normEntry, p, null, 0);
            r.Check(rc == status, $"gating/reject/norm {what}", $"status {rc} (want {status}): {lib.LastError()}");
        }

        var n1 = norm;
        n1.Eps = float.NaN;
        ExpectNorm("eps NaN", n1, 1);
        n1 = norm;
        n1.Eps = -1;
        ExpectNorm("negative eps", n1, 1);
        n1 = norm;
        n1.Dim = 0;
        ExpectNorm("dim 0", n1, 1);
        n1 = norm;
        n1.Weight.Length = Dim - 1;
        ExpectNorm("weight shorter than dim", n1, 1);
        n1 = norm;
        n1.Gate.Dtype = DType.F32;
        ExpectNorm("gate dtype differs from input", n1, 1);
        n1 = norm;
        n1.Input.Dtype = DType.F16;
        n1.Gate.Dtype = DType.F16;
        n1.Output.Dtype = DType.F16;
        ExpectNorm("f16 activations", n1, 2);
        n1 = norm;
        n1.Output.Address = input.Device + 8;
        ExpectNorm("output partially overlapping input", n1, 1);
        n1 = norm;
        n1.StructSize = 16;
        ExpectNorm("struct_size too small", n1, 5);

        var mul = new SigmoidMulParams
        {
            StructSize = (uint)sizeof(SigmoidMulParams),
            AbiMinor = 2,
            Input = input.DevSpan(),
            Gate = gate.DevSpan(),
            Output = output.DevSpan(),
            Rows = Rows,
            Dim = Dim,
        };
        void ExpectMul(string what, SigmoidMulParams p, int status)
        {
            var rc = lib.RunRaw(mulEntry, p, null, 0);
            r.Check(rc == status, $"gating/reject/sigmoid-mul {what}", $"status {rc} (want {status}): {lib.LastError()}");
        }

        var m1 = mul;
        m1.Output.Length -= 1;
        ExpectMul("output shorter than rows * dim", m1, 1);
        m1 = mul;
        m1.Gate.Address = 0;
        ExpectMul("NULL gate", m1, 1);
        m1 = mul;
        m1.Output.Address = gate.Device + 2;
        ExpectMul("output partially overlapping gate", m1, 1);
        m1 = mul;
        m1.StructSize = 8;
        ExpectMul("struct_size too small", m1, 5);
        Cuda.Sync(stream);
    }
}

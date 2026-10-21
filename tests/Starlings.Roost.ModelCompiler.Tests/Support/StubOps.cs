using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>
/// A test stub backend: float32 math for a few ops, host addresses in DeviceSpan, and NotSupportedException for
/// everything else. It is not the reference (that is <see cref="CpuOps"/>). The tests keep it only where a backend that
/// lacks ops is the point: the hybrid <c>CheckBackend</c> test needs a backend without GatedDeltaRule and
/// CausalConv1d. No numeric comparison runs on it.
/// </summary>
public readonly struct StubCpuOps : IRoostOps
{
    private static unsafe float* F(in DeviceSpan s)
    {
        if (s.Dtype != DType.F32)
        {
            throw new InvalidOperationException("StubCpuOps handles float32 buffers only, got " + s.Dtype);
        }

        return (float*)s.Address;
    }

    private static unsafe int* I(in DeviceSpan s) => (int*)s.Address;

    public static unsafe void RmsNorm(in RmsNormArgs a)
    {
        float* x = F(a.Input), w = F(a.Weight), o = F(a.Output);
        for (int r = 0; r < a.Rows; r++)
        {
            double ss = 0;
            for (int d = 0; d < a.Dim; d++)
            {
                ss += (double)x[(r * a.Dim) + d] * x[(r * a.Dim) + d];
            }

            float inv = (float)(1.0 / Math.Sqrt((ss / a.Dim) + a.Eps));
            for (int d = 0; d < a.Dim; d++)
            {
                o[(r * a.Dim) + d] = x[(r * a.Dim) + d] * inv * w[d];
            }
        }
    }

    public static void QkNorm(in QkNormArgs a) => throw new NotSupportedException(nameof(QkNorm));

    public static void Rope(in RopeArgs a) => throw new NotSupportedException(nameof(Rope));

    public static void PagedAttention(in PagedAttentionArgs a) => throw new NotSupportedException(nameof(PagedAttention));

    public static unsafe void SiluAndMul(in SiluAndMulArgs a)
    {
        float* x = F(a.Input), o = F(a.Output);
        for (int r = 0; r < a.Rows; r++)
        {
            for (int d = 0; d < a.Dim; d++)
            {
                float g = x[(r * 2 * a.Dim) + d];
                float u = x[(r * 2 * a.Dim) + a.Dim + d];
                o[(r * a.Dim) + d] = g / (1f + MathF.Exp(-g)) * u;
            }
        }
    }

    public static void GeluAndMul(in GeluAndMulArgs a) => throw new NotSupportedException(nameof(GeluAndMul));

    public static unsafe void Gemm(in GemmArgs a)
    {
        float* x = F(a.A), b = F(a.B), o = F(a.D);
        for (long m = 0; m < a.M; m++)
        {
            for (long n = 0; n < a.N; n++)
            {
                double acc = 0;
                for (long k = 0; k < a.K; k++)
                {
                    float bv = a.TransposeB ? b[(n * a.K) + k] : b[(k * a.N) + n];
                    acc += (double)x[(m * a.K) + k] * bv;
                }

                o[(m * a.N) + n] = (float)(a.Alpha * acc);
            }
        }
    }

    public static unsafe void Embedding(in EmbeddingArgs a)
    {
        int* ids = I(a.Ids);
        float* table = F(a.Table), o = F(a.Output);
        for (int t = 0; t < a.Tokens; t++)
        {
            for (int d = 0; d < a.Dim; d++)
            {
                o[(t * a.Dim) + d] = table[((long)ids[t] * a.Dim) + d];
            }
        }
    }

    public static unsafe void LmHead(in LmHeadArgs a)
    {
        float* h = F(a.Hidden), w = F(a.Weight), o = F(a.Logits);
        int* rows = I(a.Rows);
        for (int s = 0; s < a.Selected; s++)
        {
            for (int v = 0; v < a.Vocab; v++)
            {
                double acc = 0;
                for (int d = 0; d < a.Dim; d++)
                {
                    acc += (double)h[(rows[s] * a.Dim) + d] * w[((long)v * a.Dim) + d];
                }

                o[((long)s * a.VocabPadded) + v] = (float)acc;
            }
        }
    }

    public static unsafe void Residual(in ResidualArgs a)
    {
        float* x = F(a.A), y = F(a.B), o = F(a.Output);
        for (long i = 0; i < a.Elements; i++)
        {
            o[i] = x[i] + y[i];
        }
    }

    public static void CacheWrite(in CacheWriteArgs a) => throw new NotSupportedException(nameof(CacheWrite));

    public static void CacheCopy(in CacheCopyArgs a) => throw new NotSupportedException(nameof(CacheCopy));

    public static void Argmax(in ArgmaxArgs a) => throw new NotSupportedException(nameof(Argmax));

    public static void Sample(in SampleArgs a) => throw new NotSupportedException(nameof(Sample));
}

/// <summary>
/// <see cref="CpuOps"/> plus a fused AddRmsNorm, so the fused path and the generated Residual + RmsNorm fallback can be
/// compared on the reference backend. CpuOps does not implement the fused ops (<c>SupportsAddRmsNorm</c> is false), so
/// this test backend supplies one, written independently of CpuOps.RmsNorm and Residual: in double, one pass per row.
/// Every other op is CpuOps itself.
/// </summary>
public readonly struct CpuFusedOps : IRoostOps
{
    public static bool Supports(OpId op) => CpuOps.Supports(op);

    public static bool SupportsAddRmsNorm => true;

    public static unsafe void AddRmsNorm(in AddRmsNormArgs a)
    {
        // Input := rmsnorm(Input + Residual) * Weight; Residual := Input + Residual.
        float* x = (float*)a.Input.Address, r = (float*)a.Residual.Address, w = (float*)a.Weight.Address;
        for (int row = 0; row < a.Rows; row++)
        {
            double ss = 0;
            for (int d = 0; d < a.Dim; d++)
            {
                float sum = x[(row * a.Dim) + d] + r[(row * a.Dim) + d];
                r[(row * a.Dim) + d] = sum;
                ss += (double)sum * sum;
            }

            double inv = 1.0 / Math.Sqrt((ss / a.Dim) + a.Eps);
            for (int d = 0; d < a.Dim; d++)
            {
                x[(row * a.Dim) + d] = (float)(r[(row * a.Dim) + d] * inv * w[d]);
            }
        }
    }

    public static void RmsNorm(in RmsNormArgs a) => CpuOps.RmsNorm(a);

    public static void QkNorm(in QkNormArgs a) => CpuOps.QkNorm(a);

    public static void Rope(in RopeArgs a) => CpuOps.Rope(a);

    public static void PagedAttention(in PagedAttentionArgs a) => CpuOps.PagedAttention(a);

    public static void SiluAndMul(in SiluAndMulArgs a) => CpuOps.SiluAndMul(a);

    public static void GeluAndMul(in GeluAndMulArgs a) => CpuOps.GeluAndMul(a);

    public static void Gemm(in GemmArgs a) => CpuOps.Gemm(a);

    public static void Embedding(in EmbeddingArgs a) => CpuOps.Embedding(a);

    public static void LmHead(in LmHeadArgs a) => CpuOps.LmHead(a);

    public static void Residual(in ResidualArgs a) => CpuOps.Residual(a);

    public static void CacheWrite(in CacheWriteArgs a) => CpuOps.CacheWrite(a);

    public static void CacheCopy(in CacheCopyArgs a) => CpuOps.CacheCopy(a);

    public static void Argmax(in ArgmaxArgs a) => CpuOps.Argmax(a);

    public static void Sample(in SampleArgs a) => CpuOps.Sample(a);
}

/// <summary>
/// A backend that does no math and records the sequence of calls (op id and the first buffer address) into
/// preallocated thread-static storage, so the hybrid model's forward can be checked for order, wiring and
/// allocations without implementing attention or Gated DeltaNet.
/// </summary>
public readonly struct RecordingOps : IRoostOps
{
    [ThreadStatic]
    private static OpId[]? _ops;

    [ThreadStatic]
    private static nint[]? _first;

    [ThreadStatic]
    private static nint[]? _second;

    [ThreadStatic]
    private static int _count;

    public static int Count => _count;

    public static OpId[] Ops => _ops ??= new OpId[4096];

    public static nint[] FirstAddress => _first ??= new nint[4096];

    public static nint[] SecondAddress => _second ??= new nint[4096];

    public static void Reset() => _count = 0;

    public static bool Supports(OpId op) => op is OpId.GatedDeltaRule or OpId.CausalConv1d or OpId.GatedDeltaRuleGates or OpId.GatedRmsNorm or OpId.SigmoidMul || OpCatalog.IsM0(op);

    private static void Log(OpId op, in DeviceSpan first, in DeviceSpan second = default)
    {
        var ops = Ops;
        var addr = FirstAddress;
        var addr2 = SecondAddress;
        if (_count < ops.Length)
        {
            ops[_count] = op;
            addr[_count] = first.Address;
            addr2[_count] = second.Address;
            _count++;
        }
    }

    public static void RmsNorm(in RmsNormArgs a) => Log(OpId.RmsNorm, a.Input);

    public static void QkNorm(in QkNormArgs a) => Log(OpId.QkNorm, a.Q);

    public static void Rope(in RopeArgs a) => Log(OpId.Rope, a.Q);

    public static void PagedAttention(in PagedAttentionArgs a) => Log(OpId.PagedAttention, a.Q, a.KCache);

    public static void SiluAndMul(in SiluAndMulArgs a) => Log(OpId.SiluAndMul, a.Input);

    public static void GeluAndMul(in GeluAndMulArgs a) => Log(OpId.GeluAndMul, a.Input);

    public static void Gemm(in GemmArgs a) => Log(OpId.Gemm, a.A);

    public static void Embedding(in EmbeddingArgs a) => Log(OpId.Embedding, a.Ids);

    public static void LmHead(in LmHeadArgs a) => Log(OpId.LmHead, a.Hidden);

    public static void Residual(in ResidualArgs a) => Log(OpId.Residual, a.A);

    public static void CacheWrite(in CacheWriteArgs a) => Log(OpId.CacheWrite, a.K, a.KCache);

    public static void CacheCopy(in CacheCopyArgs a) => Log(OpId.CacheCopy, a.Source);

    public static void Argmax(in ArgmaxArgs a) => Log(OpId.Argmax, a.Logits);

    public static void Sample(in SampleArgs a) => Log(OpId.Sample, a.Logits);

    public static void GatedDeltaRule(in GatedDeltaRuleArgs a) => Log(OpId.GatedDeltaRule, a.Q, a.State);

    public static void CausalConv1d(in CausalConv1dArgs a) => Log(OpId.CausalConv1d, a.Input, a.ConvState);

    public static void GatedDeltaRuleGates(in GatedDeltaRuleGatesArgs a) => Log(OpId.GatedDeltaRuleGates, a.A, a.ALog);

    public static void GatedRmsNorm(in GatedRmsNormArgs a) => Log(OpId.GatedRmsNorm, a.Input, a.Gate);

    public static void SigmoidMul(in SigmoidMulArgs a) => Log(OpId.SigmoidMul, a.Input, a.Gate);
}

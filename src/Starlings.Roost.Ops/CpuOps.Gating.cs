using Starlings.Roost.Abi;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

public readonly partial struct CpuOps
{
    // torch.nn.functional.softplus(x) with its defaults (beta 1, threshold 20) returns x itself above the threshold.
    private const double SoftplusThreshold = 20.0;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>The two gate lines of transformers 5.20.1 <c>Qwen3_5GatedDeltaNet.forward</c>, in double, per token t and value head h:</para>
    /// <code>
    /// G[t, h]    = -exp(ALog[h]) * softplus(A[t, h] + DtBias[h])   softplus(x) = x &gt; 20 ? x : log(1 + exp(x))
    /// Beta[t, h] = 1 / (1 + exp(-B[t, h]))
    /// </code>
    /// <para>A and B are read as DType at their row strides (0 means VHeads); ALog and DtBias are read, and G and Beta written, as their span's
    /// Dtype (DType when the span has none), each result rounded once.</para>
    /// </remarks>
    public static void GatedDeltaRuleGates(in GatedDeltaRuleGatesArgs a)
    {
        int vh = a.VHeads;
        long aStride = RowStride(a.AStride, vh, nameof(a.AStride));
        long bStride = RowStride(a.BStride, vh, nameof(a.BStride));
        Need(a.A, StridedExtent(a.Tokens, aStride, vh), nameof(a.A));
        Need(a.B, StridedExtent(a.Tokens, bStride, vh), nameof(a.B));
        Need(a.ALog, vh, nameof(a.ALog));
        Need(a.DtBias, vh, nameof(a.DtBias));
        Need(a.G, (long)a.Tokens * vh, nameof(a.G));
        Need(a.Beta, (long)a.Tokens * vh, nameof(a.Beta));
        DType alt = SpanType(a.ALog, a.DType);
        DType dbt = SpanType(a.DtBias, a.DType);
        DType gt = SpanType(a.G, a.DType);
        DType bt = SpanType(a.Beta, a.DType);
        for (long t = 0; t < a.Tokens; t++)
        {
            for (int h = 0; h < vh; h++)
            {
                double x = Load(a.A.Address, a.DType, (t * aStride) + h) + Load(a.DtBias.Address, dbt, h);
                double softplus = x > SoftplusThreshold ? x : Math.Log(1.0 + Math.Exp(x));
                Store(a.G.Address, gt, (t * vh) + h, -Math.Exp(Load(a.ALog.Address, alt, h)) * softplus);
                Store(a.Beta.Address, bt, (t * vh) + h, Sigmoid(Load(a.B.Address, a.DType, (t * bStride) + h)));
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>transformers 5.20.1 <c>Qwen3_5RMSNormGated.forward</c> (norm before gate), in double, per row of Dim:
    /// <c>Output = x / sqrt(sum(x^2) / Dim + Eps) * Weight * silu(Gate)</c>, rounded once. The reference rounds the normalised row to the
    /// input type before multiplying by the weight; in float32 that rounding is the only difference. Input, Gate and Output are DType; Weight
    /// is read as its span's Dtype (DType when it has none). Input may alias Output.</para>
    /// </remarks>
    public static void GatedRmsNorm(in GatedRmsNormArgs a)
    {
        long n = (long)a.Rows * a.Dim;
        Need(a.Input, n, nameof(a.Input));
        Need(a.Gate, n, nameof(a.Gate));
        Need(a.Weight, a.Dim, nameof(a.Weight));
        Need(a.Output, n, nameof(a.Output));
        DType t = a.DType;
        DType wt = SpanType(a.Weight, t);
        for (long r = 0; r < a.Rows; r++)
        {
            long o = r * a.Dim;
            double ss = 0;
            for (int i = 0; i < a.Dim; i++)
            {
                double v = Load(a.Input.Address, t, o + i);
                ss += v * v;
            }

            double inv = 1.0 / Math.Sqrt((ss / a.Dim) + a.Eps);
            for (int i = 0; i < a.Dim; i++)
            {
                double y = Load(a.Input.Address, t, o + i) * inv * Load(a.Weight.Address, wt, i) * Silu(Load(a.Gate.Address, t, o + i));
                Store(a.Output.Address, t, o + i, y);
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks><c>Output = Input / (1 + exp(-Gate))</c> elementwise in double, rounded once (transformers 5.20.1 <c>Qwen3_5Attention.forward</c>,
    /// <c>attn_output * torch.sigmoid(gate)</c>). All three are DType. Input may alias Output.</remarks>
    public static void SigmoidMul(in SigmoidMulArgs a)
    {
        long n = (long)a.Rows * a.Dim;
        Need(a.Input, n, nameof(a.Input));
        Need(a.Gate, n, nameof(a.Gate));
        Need(a.Output, n, nameof(a.Output));
        DType t = a.DType;
        for (long i = 0; i < n; i++)
        {
            Store(a.Output.Address, t, i, Load(a.Input.Address, t, i) * Sigmoid(Load(a.Gate.Address, t, i)));
        }
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static DType SpanType(in DeviceSpan s, DType fallback) => s.Dtype == DType.Undefined ? fallback : s.Dtype;
}

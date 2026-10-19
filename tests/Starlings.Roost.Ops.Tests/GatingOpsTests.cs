using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>The Qwen3.5 gating ops of contract version 3 on hand-computed cases.</summary>
public sealed class GatingOpsTests
{
    private static readonly double s_ln2 = Math.Log(2);
    private static readonly double s_ln3 = Math.Log(3);

    [Fact]
    public void GatesHandComputed()
    {
        // Two tokens, two value heads. x = A + DtBias is 0, 25, -1 and exactly 20: softplus(0) = ln 2, softplus(25) = 25
        // (above torch's threshold of 20 softplus returns x), softplus(-1) = ln(1 + 1/e), softplus(20) = ln(1 + e^20).
        // exp(ALog) is 1 for head 0 and 2 for head 1. Beta = sigmoid(B): sigmoid(0) = 1/2, sigmoid(ln 3) = 3/4.
        using var a = Mem.F64(1, 24, 0, 19);
        using var b = Mem.F64(0, s_ln3, -s_ln3, 0);
        using var aLog = Mem.F64(0, s_ln2);
        using var dtBias = Mem.F64(-1, 1);
        using var g = Mem.Zeros(DType.F64, 4);
        using var beta = Mem.Zeros(DType.F64, 4);
        CpuOps.GatedDeltaRuleGates(Gates(a, b, aLog, dtBias, g, beta, 2, 2));
        Near.Eq([-s_ln2, -2 * 25.0, -Math.Log(1 + Math.Exp(-1)), -2 * Math.Log(1 + Math.Exp(20))], g.ReadF64(), 1e-14);
        Near.Eq([0.5, 0.75, 0.25, 0.5], beta.ReadF64(), 1e-15);
    }

    [Fact]
    public void GatesReadStridedViewsAndWriteEachSpansType()
    {
        // A and B are columns of one [tokens, 5] row: A at column 0 and B at column 2, stride 5; the other columns are
        // junk that must not be read. G is float32 and Beta bfloat16, as Qwen3.5 binds them; A_log is float32 and
        // DtBias has no dtype, so it is read as DType.
        using var fused = Mem.F64(0, 0, 0, 0, 99, 0, 0, 0, 0, 99);
        using var b = new Bag();
        var bView = new DeviceSpan { Address = fused.Address + (2 * sizeof(double)), Length = 8, Dtype = DType.F64 };
        var aLog = b.Add(Mem.F32(0f, (float)s_ln2));
        var dtBias = b.Add(Mem.F64(0, 0));
        var g = b.Add(Mem.Zeros(DType.F32, 4));
        var beta = b.Add(Mem.Zeros(DType.BF16, 4));
        var args = new GatedDeltaRuleGatesArgs
        {
            A = fused.Span, B = bView, ALog = aLog.Span, DtBias = dtBias.Span with { Dtype = DType.Undefined }, G = g.Span, Beta = beta.Span,
            Tokens = 2, VHeads = 2, AStride = 5, BStride = 5, DType = DType.F64,
        };
        CpuOps.GatedDeltaRuleGates(args);
        double expG = -s_ln2;
        Near.Eq([expG, 2 * expG, expG, 2 * expG], [.. g.ReadF32().Select(x => (double)x)], 1e-6);
        Assert.Equal([0.5f, 0.5f, 0.5f, 0.5f], beta.ReadBf16());
    }

    [Fact]
    public void GatesRejectAShortViewAndAStrideBelowTheRowWidth()
    {
        using var b = new Bag();
        var a = b.Add(Mem.F64(new double[7]));
        var v = b.Add(Mem.F64(new double[8]));
        var two = b.Add(Mem.F64(0, 0));
        var o = b.Add(Mem.Zeros(DType.F64, 4));
        var o2 = b.Add(Mem.Zeros(DType.F64, 4));
        // Stride 5 over two tokens needs 5 + 2 = 7 elements: A has them, so only the stride check can reject it.
        var e = Assert.Throws<ArgumentException>(() => CpuOps.GatedDeltaRuleGates(Gates(a, v, two, two, o, o2, 2, 2) with { AStride = 1 }));
        Assert.Equal(nameof(GatedDeltaRuleGatesArgs.AStride), e.ParamName);
        Assert.Throws<ArgumentException>(() => CpuOps.GatedDeltaRuleGates(Gates(a, v, two, two, o, o2, 2, 2) with { AStride = 6 }));
        CpuOps.GatedDeltaRuleGates(Gates(a, v, two, two, o, o2, 2, 2) with { AStride = 5 });
    }

    [Fact]
    public void GatedRmsNormHandComputed()
    {
        // Eps 0. Row [3, 4]: mean(x^2) = 12.5. Row [-1, 2]: mean 2.5. Weight [1, 0.5] multiplies as stored (not 1 + w).
        // silu(0) = 0, silu(ln 3) = ln 3 * 3/4. The norm comes first: the gate does not enter mean(x^2).
        using var x = Mem.F64(3, 4, -1, 2);
        using var gate = Mem.F64(0, s_ln3, s_ln3, s_ln3);
        using var w = Mem.F64(1, 0.5);
        using var o = Mem.Zeros(DType.F64, 4);
        CpuOps.GatedRmsNorm(Norm(x, gate, w, o, 2, 2, 0f));
        double silu3 = s_ln3 * 0.75;
        Near.Eq([0, 4 / Math.Sqrt(12.5) * 0.5 * silu3, -1 / Math.Sqrt(2.5) * silu3, 2 / Math.Sqrt(2.5) * 0.5 * silu3], o.ReadF64(), 1e-15);
    }

    [Fact]
    public void GatedRmsNormEpsilonAndInPlace()
    {
        // In place, eps 1: row [1, 1] has mean 1, so rsqrt(2); weight 2 and gate 10 (silu(10) = 10 / (1 + e^-10)).
        using var x = Mem.F64(1, 1);
        using var gate = Mem.F64(10, 10);
        using var w = Mem.F64(2, 2);
        CpuOps.GatedRmsNorm(Norm(x, gate, w, x, 1, 2, 1f));
        double expected = 1 / Math.Sqrt(2) * 2 * (10 / (1 + Math.Exp(-10)));
        Near.Eq([expected, expected], x.ReadF64(), 1e-14);
    }

    [Fact]
    public void GatedRmsNormReadsAFloat32WeightUnderBfloat16Activations()
    {
        // Qwen3.5 keeps linear_attn.norm.weight in float32 while activations are bfloat16. Row [1, 1] normalises to 1
        // (eps 0); silu(16) = 16 / (1 + e^-16), which rounds to 16 in bfloat16. Weight [1.5, 0.25] read as float32 gives
        // [24, 4]; read as bfloat16 it would be the low halves of those floats, which are zero.
        using var x = Mem.Bf16(1, 1);
        using var gate = Mem.Bf16(16, 16);
        using var w = Mem.F32(1.5f, 0.25f);
        using var o = Mem.Zeros(DType.BF16, 2);
        CpuOps.GatedRmsNorm(Norm(x, gate, w, o, 1, 2, 0f) with { DType = DType.BF16 });
        Assert.Equal([24f, 4f], o.ReadBf16());
    }

    [Fact]
    public void SigmoidMulHandComputedAndInPlace()
    {
        using var x = Mem.F64(2, -4, 6, 1);
        using var gate = Mem.F64(0, s_ln3, -s_ln3, -800);
        using var o = Mem.Zeros(DType.F64, 4);
        CpuOps.SigmoidMul(new SigmoidMulArgs { Input = x.Span, Gate = gate.Span, Output = o.Span, Rows = 2, Dim = 2, DType = DType.F64 });
        Near.Eq([1, -3, 1.5, 0], o.ReadF64(), 1e-15);
        CpuOps.SigmoidMul(new SigmoidMulArgs { Input = x.Span, Gate = gate.Span, Output = x.Span, Rows = 2, Dim = 2, DType = DType.F64 });
        Near.Eq([1, -3, 1.5, 0], x.ReadF64(), 1e-15);
    }

    [Fact]
    public void GatingOpsRejectShortSpans()
    {
        using var b = new Bag();
        var three = b.Add(Mem.F64(1, 2, 3));
        var four = b.Add(Mem.F64(1, 2, 3, 4));
        Assert.Throws<ArgumentException>(() => CpuOps.SigmoidMul(new SigmoidMulArgs { Input = four.Span, Gate = three.Span, Output = four.Span, Rows = 2, Dim = 2, DType = DType.F64 }));
        Assert.Throws<ArgumentException>(() => CpuOps.GatedRmsNorm(Norm(four, four, three, four, 1, 4, 0f)));
        Assert.Throws<ArgumentException>(() => CpuOps.GatedDeltaRuleGates(Gates(four, four, four, four, three, four, 2, 2)));
    }

    private static GatedDeltaRuleGatesArgs Gates(Mem a, Mem b, Mem aLog, Mem dtBias, Mem g, Mem beta, int tokens, int vHeads) => new()
    {
        A = a.Span, B = b.Span, ALog = aLog.Span, DtBias = dtBias.Span, G = g.Span, Beta = beta.Span, Tokens = tokens, VHeads = vHeads, DType = DType.F64,
    };

    private static GatedRmsNormArgs Norm(Mem x, Mem gate, Mem w, Mem o, int rows, int dim, float eps) => new()
    {
        Input = x.Span, Gate = gate.Span, Weight = w.Span, Output = o.Span, Rows = rows, Dim = dim, Eps = eps, DType = DType.F64,
    };
}

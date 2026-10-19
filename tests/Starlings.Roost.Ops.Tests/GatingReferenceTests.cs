using System.Text.Json;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>
/// The Qwen3.5 gating ops of contract version 3, and GatedDeltaRule with its in-op l2 norm, against transformers 5.20.1
/// itself on seeded random inputs.
/// </summary>
/// <remarks>
/// <c>Reference/qwen35-gating-cases.json</c> is written by <c>Reference/make_qwen35_gating_cases.py</c>, which runs
/// <c>Qwen3_5GatedDeltaNet</c> and <c>Qwen3_5Attention</c> (modeling_qwen3_5.py sha256 b48635f7…, the version
/// evidence/pipeline pins) in float32 with random weights and records the tensors each op reads and writes. The reference
/// computes in float32 and CpuOps in double, so the elementwise ops are compared to a few float32 ulps and the delta rule
/// (chunked, with a triangular solve) to 1e-5 of the output's largest magnitude.
/// </remarks>
public sealed class GatingReferenceTests
{
    private static readonly Lazy<JsonElement> s_cases = new(() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Reference", "qwen35-gating-cases.json"))).RootElement);

    private static JsonElement Linear => s_cases.Value.GetProperty("linear");

    private static JsonElement Attention => s_cases.Value.GetProperty("attention");

    [Fact]
    public void TheCasesComeFromThePinnedReference()
    {
        Assert.Equal("5.20.1", s_cases.Value.GetProperty("transformers").GetString());
        Assert.Equal("b48635f7388fbfd1b214b8c19275f8c9854fd02c4f3a3376a2339b753a2a5457", s_cases.Value.GetProperty("modeling_qwen3_5_sha256").GetString());
    }

    [Fact]
    public void GatesMatchTransformers()
    {
        var (g, beta) = Gates(Linear);
        Rel(Arr(Linear, "g"), g, 4e-7);
        Rel(Arr(Linear, "beta"), beta, 4e-7);
    }

    [Fact]
    public void TheGateCasesCoverBothSidesOfTheSoftplusThreshold()
    {
        double[] a = Arr(Linear, "a");
        double dt = Arr(Linear, "dt_bias")[0];
        int vh = Int(Linear, "v_heads");
        var x = Enumerable.Range(0, Int(Linear, "tokens")).Select(t => a[t * vh] + dt).ToArray();
        Assert.Contains(x, v => v > 20);
        Assert.Contains(x, v => v < -20);
        Assert.Contains(x, v => v is > 0 and < 20);
    }

    [Fact]
    public void GatedDeltaRuleNormalisesQAndKInsideTheOp()
    {
        // q, k and v as transformers passes them to the delta rule, before its l2 norm; L2NormQk on and Scale 0 give the
        // reference output. Without L2NormQk the output is far off, so the norm is not applied anywhere else.
        double[] core = Arr(Linear, "core");
        double tol = 1e-5 * core.Max(Math.Abs);
        Abs(core, DeltaRule(Linear, Arr(Linear, "g"), Arr(Linear, "beta"), l2: true), tol);
        double[] unnormalised = DeltaRule(Linear, Arr(Linear, "g"), Arr(Linear, "beta"), l2: false);
        Assert.True(core.Zip(unnormalised).Max(p => Math.Abs(p.First - p.Second)) > 100 * tol);
    }

    [Fact]
    public void GatedRmsNormMatchesTransformers()
    {
        double[] expected = Arr(Linear, "norm_out");
        Rel(expected, GatedNorm(Linear, Arr(Linear, "core"), Arr(Linear, "norm_weight")), 4e-7);

        // Discrimination: a zero-centred weight (1 + w), or the gate applied before the norm, is far from the reference.
        double[] onePlus = GatedNorm(Linear, Arr(Linear, "core"), [.. Arr(Linear, "norm_weight").Select(w => 1 + w)]);
        Assert.True(expected.Zip(onePlus).Max(p => Math.Abs(p.First - p.Second)) > 0.1);
        double[] gateFirst = GateThenNorm(Linear);
        Assert.True(expected.Zip(gateFirst).Max(p => Math.Abs(p.First - p.Second)) > 0.1);
    }

    [Fact]
    public void GatesThenDeltaRuleThenGatedNormMatchTransformers()
    {
        // The whole linear-attention core on CpuOps from the raw a and b projections, z and the conv output's q, k, v.
        var (g, beta) = Gates(Linear);
        double[] core = DeltaRule(Linear, g, beta, l2: true);
        double[] expected = Arr(Linear, "norm_out");
        Abs(expected, GatedNorm(Linear, core, Arr(Linear, "norm_weight")), 1e-5 * expected.Max(Math.Abs));
    }

    [Fact]
    public void SigmoidMulMatchesTheAttentionOutputGate()
    {
        double[] attn = Arr(Attention, "attn");
        double[] gate = Arr(Attention, "gate");
        int rows = Int(Attention, "tokens");
        int dim = Int(Attention, "heads") * Int(Attention, "head_dim");
        using var b = new Bag();
        var x = b.Add(Mem.F32(attn));
        var gm = b.Add(Mem.F32(gate));
        var o = b.Add(Mem.Zeros(DType.F32, rows * dim));
        CpuOps.SigmoidMul(new SigmoidMulArgs { Input = x.Span, Gate = gm.Span, Output = o.Span, Rows = rows, Dim = dim, DType = DType.F32 });
        double[] actual = [.. o.ReadF32().Select(v => (double)v)];
        Rel(Arr(Attention, "gated"), actual, 4e-7);
    }

    private static (double[] G, double[] Beta) Gates(JsonElement c)
    {
        int tokens = Int(c, "tokens");
        int vh = Int(c, "v_heads");
        using var b = new Bag();
        var a = b.Add(Mem.F32(Arr(c, "a")));
        var bm = b.Add(Mem.F32(Arr(c, "b")));
        var aLog = b.Add(Mem.F32(Arr(c, "a_log")));
        var dt = b.Add(Mem.F32(Arr(c, "dt_bias")));
        var g = b.Add(Mem.Zeros(DType.F64, tokens * vh));
        var beta = b.Add(Mem.Zeros(DType.F64, tokens * vh));
        CpuOps.GatedDeltaRuleGates(new GatedDeltaRuleGatesArgs
        {
            A = a.Span, B = bm.Span, ALog = aLog.Span, DtBias = dt.Span, G = g.Span, Beta = beta.Span, Tokens = tokens, VHeads = vh, DType = DType.F32,
        });
        return (g.ReadF64(), beta.ReadF64());
    }

    private static double[] DeltaRule(JsonElement c, double[] g, double[] beta, bool l2)
    {
        int tokens = Int(c, "tokens");
        int kh = Int(c, "k_heads");
        int vh = Int(c, "v_heads");
        int kd = Int(c, "k_dim");
        int vd = Int(c, "v_dim");
        using var b = new Bag();
        var q = b.Add(Mem.F32(Arr(c, "q")));
        var k = b.Add(Mem.F32(Arr(c, "k")));
        var v = b.Add(Mem.F32(Arr(c, "v")));
        var gm = b.Add(Mem.F64(g));
        var bm = b.Add(Mem.F64(beta));
        var state = b.Add(Mem.Zeros(DType.F32, vh * kd * vd));
        var slots = b.Add(Mem.I32(0));
        var cu = b.Add(Mem.I32(0, tokens));
        // CpuOps reads Q, K, V and Output as DType (float32 here) and G and Beta as their spans' type (double).
        var o32 = b.Add(Mem.Zeros(DType.F32, tokens * vh * vd));
        CpuOps.GatedDeltaRule(new GatedDeltaRuleArgs
        {
            Q = q.Span, K = k.Span, V = v.Span, G = gm.Span, Beta = bm.Span, Output = o32.Span, State = state.Span, StateSlots = slots.Span, CuSeqlens = cu.Span,
            Batch = 1, Tokens = tokens, KHeads = kh, VHeads = vh, KDim = kd, VDim = vd, Scale = 0, UseInitialState = false, L2NormQk = l2, DType = DType.F32,
        });
        return [.. o32.ReadF32().Select(x => (double)x)];
    }

    private static double[] GatedNorm(JsonElement c, double[] core, double[] weight)
    {
        int rows = Int(c, "tokens") * Int(c, "v_heads");
        int dim = Int(c, "v_dim");
        using var b = new Bag();
        var x = b.Add(Mem.F32(core));
        var z = b.Add(Mem.F32(Arr(c, "z")));
        var w = b.Add(Mem.F32(weight));
        var o = b.Add(Mem.Zeros(DType.F32, rows * dim));
        CpuOps.GatedRmsNorm(new GatedRmsNormArgs
        {
            Input = x.Span, Gate = z.Span, Weight = w.Span, Output = o.Span, Rows = rows, Dim = dim, Eps = (float)c.GetProperty("eps").GetDouble(), DType = DType.F32,
        });
        return [.. o.ReadF32().Select(v => (double)v)];
    }

    // The other order, rmsnorm(x * silu(z)) * w, written out directly as a discriminating alternative.
    private static double[] GateThenNorm(JsonElement c)
    {
        double[] core = Arr(c, "core");
        double[] z = Arr(c, "z");
        double[] w = Arr(c, "norm_weight");
        int dim = Int(c, "v_dim");
        double eps = c.GetProperty("eps").GetDouble();
        double[] gated = [.. core.Zip(z).Select(p => p.First * p.Second / (1 + Math.Exp(-p.Second)))];
        var r = new double[gated.Length];
        for (int row = 0; row < gated.Length / dim; row++)
        {
            double inv = 1 / Math.Sqrt((gated.Skip(row * dim).Take(dim).Sum(v => v * v) / dim) + eps);
            for (int i = 0; i < dim; i++)
            {
                r[(row * dim) + i] = gated[(row * dim) + i] * inv * w[i];
            }
        }

        return r;
    }

    private static double[] Arr(JsonElement c, string name) => [.. c.GetProperty(name).EnumerateArray().Select(e => e.GetDouble())];

    private static int Int(JsonElement c, string name) => c.GetProperty(name).GetInt32();

    // |expected - actual| <= rel * max(|expected|, 1e-6) per element.
    private static void Rel(double[] expected, double[] actual, double rel)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double tol = rel * Math.Max(Math.Abs(expected[i]), 1e-6);
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol, $"[{i}] expected {expected[i]:R} actual {actual[i]:R}");
        }
    }

    private static void Abs(double[] expected, double[] actual, double tol) => Near.Eq(expected, actual, tol);
}

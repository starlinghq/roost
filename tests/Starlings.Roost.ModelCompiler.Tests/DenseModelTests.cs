using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.ModelCompiler.Tests.Models;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>The dense tiny model: slot order, plan inputs, graph, numerics against an independent transcription, allocations.</summary>
public sealed unsafe class DenseModelTests : IDisposable
{
    private const int Tokens = 5;
    private static readonly int[] Ids = [3, 17, 63, 0, 42];
    private static readonly int[] Rows = [4, 2];

    private readonly Native _native = new();
    private readonly DeviceSpan[] _weights;
    private readonly float[][] _weightData;
    private readonly DenseModel.ForwardLayout _layout = new(maxTokens: 8, maxSequences: 2, maxKvBlocks: 0);
    private readonly DeviceSpan _arena;
    private readonly DeviceSpan _planRegion;

    public DenseModelTests()
    {
        var rng = new Random(7);
        _weights = new DeviceSpan[DenseModel.WeightSlots.Length];
        _weightData = new float[_weights.Length][];
        for (int i = 0; i < _weights.Length; i++)
        {
            var slot = DenseModel.WeightSlots[i];
            _weights[i] = _native.Alloc(slot.Elements, slot.Dtype);
            _weightData[i] = new float[slot.Elements];
            bool norm = slot.Name.EndsWith("norm", StringComparison.Ordinal);
            for (int j = 0; j < slot.Elements; j++)
            {
                _weightData[i][j] = norm ? 1f + (((float)rng.NextDouble() - 0.5f) * 0.2f) : ((float)rng.NextDouble() - 0.5f) * 0.6f;
                ((float*)_weights[i].Address)[j] = _weightData[i][j];
            }
        }

        _arena = _native.AllocBytes(_layout.ArenaBytes);
        _planRegion = _native.AllocBytes(_layout.PlanBytes);
    }

    public void Dispose() => _native.Dispose();

    private DenseModel.PlanInputs BindPlan()
    {
        var plan = DenseModel.PlanInputs.Bind(_planRegion, _layout, Tokens, Rows.Length, 1, 0);
        for (int i = 0; i < Ids.Length; i++)
        {
            ((int*)plan.Ids.Address)[i] = Ids[i];
        }

        for (int i = 0; i < Rows.Length; i++)
        {
            ((int*)plan.Rows.Address)[i] = Rows[i];
        }

        return plan;
    }

    private float[] RunForward<TOps>()
        where TOps : struct, Starlings.Roost.Ops.IRoostOps
    {
        var ctx = new DenseModel.StepContext(_layout, _arena, _weights, default, BindPlan(), Tokens, Rows.Length, 1, 0);
        DenseModel.Forward<TOps>(in ctx);
        var logits = DenseModel.Output_logits(in ctx);
        var r = new float[logits.Length];
        new ReadOnlySpan<float>((void*)logits.Address, r.Length).CopyTo(r);
        return r;
    }

    [Fact]
    public void WeightSlotsFollowDeclarationOrderLayerByLayer()
    {
        string[] expected =
        [
            "embed",
            "l0.norm", "l0.gate_up", "l0.down",
            "l1.norm", "l1.gate_up", "l1.down",
            "final_norm", "lm_head",
        ];
        Assert.Equal(expected, DenseModel.WeightSlots.Select(s => s.Name).ToArray());
        Assert.Equal("model.layers.1.mlp.gate_up_proj.weight", DenseModel.WeightSlots[5].Source);
        Assert.Equal(64L * 16, DenseModel.WeightSlots[0].Elements);
        Assert.Equal(64L * 16, DenseModel.WeightSlots[2].Elements);
        Assert.Equal(16L * 32, DenseModel.WeightSlots[3].Elements);
        Assert.Equal("64,16", DenseModel.WeightSlots[0].Shape);
        Assert.Equal([-1, 0, 0, 0, 1, 1, 1, -1, -1], DenseModel.WeightSlots.Select(s => s.Layer).ToArray());
        Assert.All(DenseModel.WeightSlots, s => Assert.Equal(DType.F32, s.Dtype));
    }

    [Fact]
    public void PlanInputsAreTheOnesTheOpsReadAndAreLaidOutAligned()
    {
        Assert.Equal(["Ids", "Rows"], DenseModel.PlanInputs.Layout.Select(p => p.Name).ToArray());
        Assert.Equal(("Ids", DType.I32, "tokens"), DenseModel.PlanInputs.Layout[0]);
        Assert.Equal(("Rows", DType.I32, "selected"), DenseModel.PlanInputs.Layout[1]);
        Assert.Equal(0, _layout.IdsOffset);
        Assert.Equal(0, _layout.RowsOffset % 64);
        Assert.Equal(_layout.RowsOffset + _layout.CapacityTokens * 4, _layout.PlanBytes);
        Assert.Equal(64, _layout.CapacityTokens);
        Assert.Equal(0, _layout.FixedRegionOffset % 256);
        var plan = BindPlan();
        Assert.Equal((ulong)Tokens, plan.Ids.Length);
        Assert.Equal((ulong)Rows.Length, plan.Rows.Length);
        Assert.Equal(DType.I32, plan.Ids.Dtype);
    }

    [Fact]
    public void NoStateGroupsForAModelWithoutAttention()
    {
        Assert.Empty(DenseModel.CreateStateGroups());
        Assert.Empty(DenseModel.StateSlots);
    }

    [Fact]
    public void GraphCarriesShapesDtypesKindsAndOrigins()
    {
        var g = DenseModel.BuildGraph();
        Assert.True(g.IsReference);
        Assert.Equal("tiny-dense", g.Name);
        Assert.Equal(
            ["Embedding", "RmsNorm", "Gemm", "SiluAndMul", "Gemm", "Residual", "RmsNorm", "Gemm", "SiluAndMul", "Gemm", "Residual", "RmsNorm", "LmHead"],
            g.Nodes.Select(n => n.Op).ToArray());
        Assert.All(g.Nodes, n => Assert.Equal([n.Id], n.Origin.ToArray()));
        Assert.All(g.Values.Where(v => v.Kind != ValueKind.Input), v => Assert.Equal(GraphDType.F32, v.DType));

        GraphValue Value(string name) => g.Values.Single(v => v.Name == name);
        Assert.Equal(ValueKind.Weight, Value("l1.down").Kind);
        Assert.Equal("16,32", string.Join(",", Value("l1.down").Shape.Select(d => d.ToString())));
        Assert.Equal(ValueKind.Input, Value("plan.Ids").Kind);
        Assert.Equal(GraphDType.I32, Value("plan.Ids").DType);
        Assert.Equal("tokens", Value("plan.Ids").Shape[0].Symbol);
        Assert.Equal(ValueKind.Intermediate, Value("h1").Kind);
        Assert.Equal("tokens,16", string.Join(",", Value("h1").Shape.Select(d => d.ToString())));
        Assert.Equal(ValueKind.Output, Value("logits").Kind);
        Assert.Equal("selected,64", string.Join(",", Value("logits").Shape.Select(d => d.ToString())));
        Assert.Equal(GraphLayout.RowMajor, Value("h1").Layout);
        Assert.Same(Sharding.Replicated, Value("h1").Sharding);

        // Value ids index the value list, weights first in slot order.
        Assert.All(g.Values, v => Assert.Same(v, g.Values[v.Id]));
        Assert.Equal(DenseModel.WeightSlots.Select(s => s.Name), g.Values.Take(9).Select(v => v.Name));

        var gemm = g.Nodes[2];
        Assert.Equal("A,B", gemm.Attributes["in"]);
        Assert.Equal("D", gemm.Attributes["out"]);
        Assert.Equal("true", gemm.Attributes["arg.TransposeB"]);
        Assert.Equal("(64)", gemm.Attributes["arg.N"]);
        Assert.Equal("tokens", gemm.Attributes["arg.M"]);
        Assert.Equal("0", gemm.Attributes["layer"]);
        Assert.Equal("1", g.Nodes[7].Attributes["layer"]);
        Assert.False(g.Nodes[0].Attributes.ContainsKey("layer"));
        Assert.Equal("Ids,Table".Split(',').Order().ToArray(), g.Nodes[0].Attributes["in"].Split(',').Order().ToArray());
    }

    [Fact]
    public void ArenaSharesSpaceBetweenDeadValues()
    {
        var g = DenseModel.BuildGraph();
        long naive = g.Values
            .Where(v => v.Kind is ValueKind.Intermediate && v.Shape[0].Symbol == "tokens")
            .Sum(v => v.Shape.Skip(1).Aggregate(1L, (a, d) => a * d.Size) * 4);
        Assert.True(DenseModel.ForwardLayout.PerTokenBytes < naive, $"arena {DenseModel.ForwardLayout.PerTokenBytes} B/token vs {naive} B/token without reuse");
        // The widest live set is the gate/up output (2 * 32 floats) plus the activation (32) and the residual stream.
        Assert.True(DenseModel.ForwardLayout.PerTokenBytes >= (64 + 32) * 4);
        Assert.Equal(0, DenseModel.ForwardLayout.FixedBytes);
        Assert.True(_layout.ArenaBytes >= DenseModel.ForwardLayout.PerTokenBytes * _layout.CapacityTokens);
    }

    [Fact]
    public void ForwardMatchesAnIndependentDoubleTranscription()
    {
        float[] got = RunForward<StubCpuOps>();
        double[] want = Reference();
        Assert.Equal(Rows.Length * 64, got.Length);
        Assert.Equal(want.Length, got.Length);
        double worst = 0;
        for (int i = 0; i < want.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(got[i] - want[i]) / Math.Max(1e-3, Math.Abs(want[i])));
        }

        Assert.True(worst < 1e-4, $"max relative difference {worst}");
        Assert.Contains(got, x => Math.Abs(x) > 1e-3f);
    }

    [Fact]
    public void ForwardEqualsTheGraphRunOnTheSameBackend()
    {
        float[] got = RunForward<StubCpuOps>();
        var g = DenseModel.BuildGraph();
        var inputs = new Dictionary<int, double[]>();
        for (int i = 0; i < _weightData.Length; i++)
        {
            inputs[i] = _weightData[i].Select(x => (double)x).ToArray();
        }

        inputs[g.Values.Single(v => v.Name == "plan.Ids").Id] = Ids.Select(x => (double)x).ToArray();
        inputs[g.Values.Single(v => v.Name == "plan.Rows").Id] = Rows.Select(x => (double)x).ToArray();
        var evaluator = new GraphEvaluator<StubCpuOps>(new Dictionary<string, long> { ["tokens"] = Tokens, ["selected"] = Rows.Length, ["seqs"] = 1, ["kv_blocks"] = 0 });
        var values = evaluator.Evaluate(g, inputs);
        double[] graphLogits = values[g.Values.Single(v => v.Name == "logits").Id];
        Assert.Equal(got.Select(x => (double)x).ToArray(), graphLogits);
    }

    [Fact]
    public void ForwardAllocatesNothingOnceWarm()
    {
        var plan = BindPlan();
        void Step()
        {
            var ctx = new DenseModel.StepContext(_layout, _arena, _weights, default, plan, Tokens, Rows.Length, 1, 0);
            DenseModel.Forward<StubCpuOps>(in ctx);
        }

        long[] rounds = AllocationProbe.Rounds(Step);
        Assert.Equal(0, rounds.Min());
    }

    [Fact]
    public void AllocationProbeSeesAnAllocatingStep()
    {
        object? sink = null;
        void Allocating() => sink = new float[16];

        Assert.True(AllocationProbe.Rounds(Allocating).Min() >= 2000 * 64);
        GC.KeepAlive(sink);
    }

    private double[] Reference()
    {
        const int H = 16, I = 32, V = 64;
        double[][] w = _weightData.Select(a => a.Select(x => (double)x).ToArray()).ToArray();
        double[] h = new double[Tokens * H];
        for (int t = 0; t < Tokens; t++)
        {
            Array.Copy(w[0], Ids[t] * H, h, t * H, H);
        }

        double[] Norm(double[] x, double[] g)
        {
            var o = new double[x.Length];
            for (int t = 0; t < Tokens; t++)
            {
                double ss = 0;
                for (int d = 0; d < H; d++)
                {
                    ss += x[(t * H) + d] * x[(t * H) + d];
                }

                double inv = 1 / Math.Sqrt((ss / H) + 1e-6);
                for (int d = 0; d < H; d++)
                {
                    o[(t * H) + d] = x[(t * H) + d] * inv * g[d];
                }
            }

            return o;
        }

        double[] MatMul(double[] x, double[] b, int k, int n)
        {
            var o = new double[Tokens * n];
            for (int t = 0; t < Tokens; t++)
            {
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int q = 0; q < k; q++)
                    {
                        acc += x[(t * k) + q] * b[(j * k) + q];
                    }

                    o[(t * n) + j] = acc;
                }
            }

            return o;
        }

        for (int l = 0; l < 2; l++)
        {
            double[] x = Norm(h, w[1 + (l * 3)]);
            double[] up = MatMul(x, w[2 + (l * 3)], H, 2 * I);
            double[] act = new double[Tokens * I];
            for (int t = 0; t < Tokens; t++)
            {
                for (int d = 0; d < I; d++)
                {
                    double gate = up[(t * 2 * I) + d];
                    act[(t * I) + d] = gate / (1 + Math.Exp(-gate)) * up[(t * 2 * I) + I + d];
                }
            }

            double[] mlp = MatMul(act, w[3 + (l * 3)], I, H);
            for (int i = 0; i < h.Length; i++)
            {
                h[i] += mlp[i];
            }
        }

        double[] fin = Norm(h, w[7]);
        var logits = new double[Rows.Length * V];
        for (int s = 0; s < Rows.Length; s++)
        {
            for (int v = 0; v < V; v++)
            {
                double acc = 0;
                for (int d = 0; d < H; d++)
                {
                    acc += fin[(Rows[s] * H) + d] * w[8][(v * H) + d];
                }

                logits[(s * V) + v] = acc;
            }
        }

        return logits;
    }
}

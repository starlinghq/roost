using Starlings.Roost.Graph.Passes;
using Starlings.Roost.ModelCompiler.Checking;
using Starlings.Roost.ModelCompiler.Tests;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Graph.Tests;

/// <summary>
/// The AddRmsNorm fusion pass (R-071) on the three model definitions, checked by the C7 checker against the unfused
/// graph on <see cref="CpuOps"/>. CpuOps defines AddRmsNorm as Residual then RmsNorm, so the comparison is exact: zero
/// absolute and relative tolerance on every value every fused and unfused node writes, logits included.
/// </summary>
public sealed class FuseAddRmsNormPassTests
{
    private static readonly TestTolerances Exact = new(0, 0);

    public static TheoryData<string, int, int> Models => new()
    {
        // Model, layers, RmsNorm nodes in the reference graph (two per layer and the final norm).
        { "roost-tiny", 4, 9 },
        { "roost-tiny-hybrid", 8, 17 },
        { "qwen3.5-0.8b", 24, 49 },
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void EveryResidualFusesWithTheNormOfItsSum(string model, int layers, int norms)
    {
        var reference = FusionHarness.Build(model);
        int nodeCount = reference.Nodes.Count;
        var fused = new FuseAddRmsNormPass().Run(reference, FusionHarness.Cpu);

        // Two residuals per layer; each takes the next norm with it. Only the first layer's input norm, which reads the
        // embedding rather than a residual sum, stays unfused.
        var adds = fused.Nodes.Where(n => n.Op == "AddRmsNorm").ToList();
        Assert.Equal(2 * layers, adds.Count);
        Assert.Equal(norms - adds.Count, fused.Nodes.Count(n => n.Op == "RmsNorm"));
        Assert.Equal(0, fused.Nodes.Count(n => n.Op == "Residual"));
        Assert.Equal(nodeCount - adds.Count, fused.Nodes.Count);
        Assert.Equal("l0.input_norm", fused.Nodes.Single(n => n.Op == "RmsNorm").Attributes["node"]);

        foreach (var add in adds)
        {
            var residual = reference.Nodes[add.Origin[0]];
            var norm = reference.Nodes[add.Origin[1]];
            Assert.Equal("Residual", residual.Op);
            Assert.Equal("RmsNorm", norm.Op);
            Assert.Equal(residual.Attributes["node"] + "+" + norm.Attributes["node"], add.Attributes["node"]);

            // Input := B (overwritten with the norm), Residual := A (overwritten with the sum); both keep their reference ids.
            Assert.Equal([residual.Inputs[1], residual.Inputs[0], norm.Inputs[1]], add.Inputs.ToArray());
            Assert.Equal([norm.Outputs[0], residual.Outputs[0]], add.Outputs.ToArray());
            Assert.Equal(norm.Attributes["arg.Eps"], add.Attributes["arg.Eps"]);
            Assert.Equal(norm.Attributes["arg.DType"], add.Attributes["arg.DType"]);
        }

        // Ids are positions, every other node keeps its origin, and the input graph is untouched.
        Assert.Equal(Enumerable.Range(0, fused.Nodes.Count), fused.Nodes.Select(n => n.Id));
        Assert.All(fused.Nodes.Where(n => n.Op != "AddRmsNorm"), n => Assert.Equal(n.Op, reference.Nodes[n.Origin.Single()].Op));
        Assert.False(fused.IsReference);
        Assert.Same(reference.Values, fused.Values);
        Assert.Equal(nodeCount, reference.Nodes.Count);
        Assert.True(reference.IsReference);

        // Nothing is left to fuse.
        Assert.Equal(fused.Nodes.Count, new FuseAddRmsNormPass().Run(fused, FusionHarness.Cpu).Nodes.Count);
    }

    [Theory]
    [InlineData("roost-tiny")]
    [InlineData("roost-tiny-hybrid")]
    public void TheFusedGraphEqualsTheUnfusedGraphExactlyOnCpuOps(string model)
    {
        var reference = FusionHarness.Build(model);
        var result = Check(new FuseAddRmsNormPass(), reference);
        Assert.True(result.Passed, Describe(result));
        Assert.Equal(reference.Nodes.Count - reference.Nodes.Count(n => n.Op == "Residual"), result.NodesChecked);
        AssertNonTrivialLogits(reference);
    }

    [Fact]
    public void TheFusedQwen35GraphEqualsTheUnfusedGraphExactlyOnCpuOps()
    {
        // Qwen3.5-0.8B has 0.75e9 parameters, too many to generate and run twice per CI job, so the check runs over a slice
        // of the definition's own graph: layer 2 (Gated DeltaNet), layer 3 (gated full attention) and the input norm of
        // layer 4, which covers all four residual-norm pairs of a linear and an attention layer at the checkpoint's widths.
        var reference = FusionHarness.Window(FusionHarness.Build("qwen3.5-0.8b"), "l2.input_norm", "l4.input_norm");
        var fused = new FuseAddRmsNormPass().Run(reference, FusionHarness.Cpu);
        Assert.Equal(4, fused.Nodes.Count(n => n.Op == "AddRmsNorm"));
        Assert.Contains(reference.Nodes, n => n.Op == "GatedDeltaRule");
        Assert.Contains(reference.Nodes, n => n.Op == "PagedAttention");

        var result = Check(new FuseAddRmsNormPass(), reference);
        Assert.True(result.Passed, Describe(result));
        Assert.Equal(fused.Nodes.Count, result.NodesChecked);

        // The compared values are not trivially equal: layer 4's normalised input is finite and not all zero.
        var values = new GraphEvaluator<CpuOps>(FusionHarness.Symbols).Evaluate(fused, FusionHarness.Inputs(reference));
        double[] last = values[fused.Nodes[^1].Outputs[0]];
        Assert.Contains(last, x => Math.Abs(x) > 1e-3);
        Assert.All(last, x => Assert.True(double.IsFinite(x)));
    }

    [Fact]
    public void ABrokenFusionIsCaughtOnTheBf16Model()
    {
        var reference = FusionHarness.Build("roost-tiny-hybrid");
        var result = Check(new EpsilonOverridePass("0.5"), reference);
        Assert.False(result.Passed);
        Assert.All(result.Mismatches.Take(1), m => Assert.Equal(["Residual", "RmsNorm"], m.ReferenceNodes.Select(i => reference.Nodes[i].Op).ToArray()));
    }

    [Fact]
    public void AGpuTargetGetsTheFusionOnlyWithAConformancePassedVariant()
    {
        var reference = FusionHarness.Build("roost-tiny");
        var without = new GraphTarget(89, 1, 1, 1, new Dictionary<string, IReadOnlyList<string>> { ["AddRmsNorm"] = [] });
        var with = new GraphTarget(89, 1, 1, 1, new Dictionary<string, IReadOnlyList<string>> { ["AddRmsNorm"] = ["v0"] });
        Assert.Equal(reference.Nodes.Select(n => n.Op), new FuseAddRmsNormPass().Run(reference, without).Nodes.Select(n => n.Op));
        Assert.Equal(8, new FuseAddRmsNormPass().Run(reference, with).Nodes.Count(n => n.Op == "AddRmsNorm"));
    }

    [Fact]
    public void APairIsLeftUnfusedWhenTheFusedOpWouldChangeTheResult()
    {
        // B read after the residual: the fused op would overwrite it.
        Assert.Equal(0, Fusions(Synthetic(readBAfter: true)));

        // A tensor-parallel partial sum must be reduced before the norm.
        Assert.Equal(0, Fusions(Synthetic(partialSum: true)));

        // The residual reads a column view.
        Assert.Equal(0, Fusions(Synthetic(view: true)));

        // The two nodes disagree on the activation type.
        Assert.Equal(0, Fusions(Synthetic(normType: "F32")));

        // The norm input is a different value.
        Assert.Equal(0, Fusions(Synthetic(normReadsA: true)));

        // The control: the same graph without any of these fuses.
        Assert.Equal(1, Fusions(Synthetic()));
    }

    [Fact]
    public void ArgumentsAreValidated()
    {
        var pass = new FuseAddRmsNormPass();
        Assert.Equal("fuse-add-rmsnorm", pass.Name);
        Assert.Equal(PassKind.Fusion, pass.Kind);
        Assert.Throws<ArgumentNullException>(() => pass.Run(null!, FusionHarness.Cpu));
        Assert.Throws<ArgumentNullException>(() => pass.Run(Synthetic(), null!));
    }

    private static int Fusions(OpGraph g) => new FuseAddRmsNormPass().Run(g, FusionHarness.Cpu).Nodes.Count(n => n.Op == "AddRmsNorm");

    private static GraphCheckResult Check(IGraphPass pass, OpGraph reference) =>
        new GraphChecker().CheckPass(
            pass,
            reference,
            FusionHarness.Cpu,
            new GraphEvaluator<CpuOps>(FusionHarness.Symbols),
            FusionHarness.Inputs(reference),
            Exact);

    private static void AssertNonTrivialLogits(OpGraph reference)
    {
        var values = new GraphEvaluator<CpuOps>(FusionHarness.Symbols).Evaluate(reference, FusionHarness.Inputs(reference));
        double[] logits = values[reference.Nodes[^1].Outputs[0]];
        Assert.Contains(logits, x => Math.Abs(x) > 1e-3);
        Assert.All(logits, x => Assert.True(double.IsFinite(x)));
    }

    private static string Describe(GraphCheckResult r) =>
        string.Join("; ", r.Mismatches.Take(5).Select(m => $"node {m.CandidateNode} (from {string.Join(",", m.ReferenceNodes)}) abs {m.MaxAbsoluteError:R} rel {m.MaxRelativeError:R}"));

    // x, y: [tokens, 8] intermediates from a Gemm; s = Residual(x, y); n = RmsNorm(s, w); optionally a later read of y.
    private static OpGraph Synthetic(bool readBAfter = false, bool partialSum = false, bool view = false, string normType = "BF16", bool normReadsA = false)
    {
        ShapeDim[] shape = [ShapeDim.Named("tokens"), ShapeDim.Of(8)];
        var sharding = partialSum ? new Sharding(MeshAxis.Tensor, ShardKind.PartialSum, -1) : Sharding.Replicated;
        GraphValue V(int id, string name, ValueKind kind, IReadOnlyList<ShapeDim> s) => new(id, name, kind, s, GraphDType.BF16, GraphLayout.RowMajor, id == 3 ? sharding : Sharding.Replicated);
        var values = new List<GraphValue>
        {
            V(0, "w", ValueKind.Weight, [ShapeDim.Of(8)]),
            V(1, "proj", ValueKind.Weight, [ShapeDim.Of(8), ShapeDim.Of(8)]),
            V(2, "x", ValueKind.Intermediate, shape),
            V(3, "y", ValueKind.Intermediate, shape),
            V(4, "s", ValueKind.Intermediate, shape),
            V(5, "n", ValueKind.Intermediate, shape),
            V(6, "z", ValueKind.Intermediate, shape),
        };
        Dictionary<string, string> A(string node, string inF, string outF, params (string Key, string Value)[] args)
        {
            var d = new Dictionary<string, string> { ["in"] = inF, ["out"] = outF, ["node"] = node };
            foreach (var (k, v) in args)
            {
                d[k] = v;
            }

            return d;
        }

        var residualAttrs = A("res", "A,B", "Output", ("arg.Elements", "tokens*8"), ("arg.DType", "BF16"));
        if (view)
        {
            residualAttrs["view.B"] = "0:8";
        }

        var nodes = new List<OpNode>
        {
            new(0, "Residual", [2, 3], [4], residualAttrs, [0], null),
            new(1, "RmsNorm", [normReadsA ? 2 : 4, 0], [5], A("norm", "Input,Weight", "Output", ("arg.Rows", "tokens"), ("arg.Dim", "8"), ("arg.Eps", "1E-06"), ("arg.DType", normType)), [1], null),
        };
        if (readBAfter)
        {
            nodes.Add(new(2, "Gemm", [3, 1], [6], A("later", "A,B", "D"), [2], null));
        }

        return new OpGraph("synthetic", values, nodes, true);
    }
}

using Starlings.Roost.Graph;
using Starlings.Roost.ModelCompiler.Checking;
using Starlings.Roost.ModelCompiler.Tests.Models;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>
/// The C7 rule: every pass output is checked op by op against the unfused graph on the CPU backend. The reference
/// backend (R-018) is not merged yet, so the stub from this project stands in; the checker is backend-agnostic.
/// </summary>
public sealed class GraphCheckerTests
{
    private static readonly GraphTarget Target = new(0, 1, 1, 1, new Dictionary<string, IReadOnlyList<string>>());
    private static readonly TestTolerances Tolerances = new(1e-5, 1e-4);

    private readonly OpGraph _reference = DenseModel.BuildGraph();
    private readonly Dictionary<int, double[]> _inputs = new();
    private readonly Dictionary<string, long> _symbols = new() { ["tokens"] = 5, ["selected"] = 2, ["seqs"] = 1, ["kv_blocks"] = 0 };

    public GraphCheckerTests()
    {
        var rng = new Random(3);
        foreach (var v in _reference.Values.Where(v => v.Kind is ValueKind.Weight))
        {
            long n = v.Shape.Aggregate(1L, (a, d) => a * d.Size);
            bool norm = v.Name.EndsWith("norm", StringComparison.Ordinal);
            _inputs[v.Id] = Enumerable.Range(0, (int)n).Select(_ => norm ? 1 + ((rng.NextDouble() - 0.5) * 0.2) : (rng.NextDouble() - 0.5) * 0.6).Select(x => (double)(float)x).ToArray();
        }

        _inputs[_reference.Values.Single(v => v.Name == "plan.Ids").Id] = [3, 17, 63, 0, 42];
        _inputs[_reference.Values.Single(v => v.Name == "plan.Rows").Id] = [4, 2];
    }

    private GraphCheckResult Check(OpGraph candidate, bool fusedBackend = true) =>
        new GraphChecker().Check(
            _reference,
            candidate,
            fusedBackend ? new GraphEvaluator<StubFusedOps>(_symbols) : new GraphEvaluator<StubCpuOps>(_symbols),
            _inputs,
            Tolerances);

    [Fact]
    public void AnIdentityPassPassesAndEveryNodeIsChecked()
    {
        var result = Check(new IdentityPass().Run(_reference, Target));
        Assert.True(result.Passed);
        Assert.Equal(_reference.Nodes.Count, result.NodesChecked);
    }

    [Fact]
    public void AFusionPassIsCheckedNodeByNodeAgainstTheUnfusedGraph()
    {
        var fused = new FuseAddRmsNormPass().Run(_reference, Target);
        Assert.Equal(_reference.Nodes.Count - 2, fused.Nodes.Count);
        var fusedNodes = fused.Nodes.Where(n => n.Op == "AddRmsNorm").ToList();
        Assert.Equal(2, fusedNodes.Count);
        Assert.Equal([5, 6], fusedNodes[0].Origin.ToArray());
        Assert.Equal([10, 11], fusedNodes[1].Origin.ToArray());

        var result = Check(fused);
        Assert.True(result.Passed, string.Join("; ", result.Mismatches.Select(m => $"node {m.CandidateNode} abs {m.MaxAbsoluteError}")));
        Assert.Equal(fused.Nodes.Count, result.NodesChecked);
    }

    [Fact]
    public void ABrokenPassIsCaughtAndNamesTheNodesItReplaced()
    {
        var broken = new FuseAddRmsNormPass("0.5").Run(_reference, Target);
        var result = Check(broken);
        Assert.False(result.Passed);
        var m = result.Mismatches[0];
        Assert.Equal("AddRmsNorm", broken.Nodes[m.CandidateNode].Op);
        Assert.Equal([5, 6], m.ReferenceNodes.ToArray());
        Assert.True(m.MaxAbsoluteError > 1e-3);
        Assert.True(m.MaxRelativeError > 1e-3);
        // The error then flows through every later node, which are all reported too.
        Assert.True(result.Mismatches.Count > 1);
    }

    [Fact]
    public void ADroppedOutputIsAMismatch()
    {
        var candidate = new IdentityPass().Run(_reference, Target);
        var nodes = candidate.Nodes.Where(n => n.Op != "LmHead").ToList();
        var result = Check(new OpGraph("cut", candidate.Values, nodes, false));
        Assert.False(result.Passed);
        Assert.Contains(result.Mismatches, m => m.CandidateNode == -1);
    }

    [Fact]
    public void ANodeThatProducesAValueTheReferenceLacksIsAMismatch()
    {
        var extra = new List<GraphValue>(_reference.Values);
        var nodes = _reference.Nodes.Select(n => new OpNode(n.Id, n.Op, n.Inputs, n.Outputs, n.Attributes, n.Origin, n.Variant)).ToList();
        var result = new GraphChecker().Check(
            _reference,
            new OpGraph("x", extra, nodes, false),
            new PartialEvaluator(new GraphEvaluator<StubFusedOps>(_symbols)),
            _inputs,
            Tolerances);
        Assert.False(result.Passed);
    }

    [Fact]
    public void ToleranceDecidesWhetherASmallDifferencePasses()
    {
        var fused = new FuseAddRmsNormPass("1e-3").Run(_reference, Target);
        Assert.False(new GraphChecker().Check(_reference, fused, new GraphEvaluator<StubFusedOps>(_symbols), _inputs, new TestTolerances(1e-9, 1e-9)).Passed);
        Assert.True(new GraphChecker().Check(_reference, fused, new GraphEvaluator<StubFusedOps>(_symbols), _inputs, new TestTolerances(1e-1, 1e-1)).Passed);
    }

    [Fact]
    public void ArgumentsAreValidated()
    {
        var checker = new GraphChecker();
        var ev = new GraphEvaluator<StubFusedOps>(_symbols);
        Assert.Throws<ArgumentNullException>(() => checker.Check(null!, _reference, ev, _inputs, Tolerances));
        Assert.Throws<ArgumentNullException>(() => checker.Check(_reference, null!, ev, _inputs, Tolerances));
        Assert.Throws<ArgumentNullException>(() => checker.Check(_reference, _reference, null!, _inputs, Tolerances));
        Assert.Throws<ArgumentNullException>(() => checker.Check(_reference, _reference, ev, _inputs, null!));
    }

    /// <summary>Drops one value from the evaluator's answer, as an evaluator that lost a result would.</summary>
    private sealed class PartialEvaluator(IGraphEvaluator inner) : IGraphEvaluator
    {
        private int _calls;

        public IReadOnlyDictionary<int, double[]> Evaluate(OpGraph graph, IReadOnlyDictionary<int, double[]> inputs)
        {
            var all = new Dictionary<int, double[]>(inner.Evaluate(graph, inputs));
            if (_calls++ == 0)
            {
                // The reference answer lacks the logits; the candidate's has them.
                all.Remove(graph.Nodes[^1].Outputs[0]);
            }

            return all;
        }
    }
}

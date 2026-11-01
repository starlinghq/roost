using Starlings.Roost.Graph;
using Starlings.Roost.Graph.Passes;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>Test tolerances for float32 (in-memory values; nothing under evidence/).</summary>
public sealed class TestTolerances(double absolute, double relative) : IToleranceSource
{
    public (double Absolute, double Relative) Lookup(string op, GraphDType dtype) => (absolute, relative);
}

/// <summary>
/// The real AddRmsNorm fusion (<see cref="FuseAddRmsNormPass"/>, R-071) with the epsilon of every fused node replaced,
/// so a test can break a pass on purpose and watch the checker catch it.
/// </summary>
public sealed class EpsilonOverridePass(string epsilon) : IGraphPass
{
    private readonly FuseAddRmsNormPass _inner = new();

    public string Name => _inner.Name + "-eps-" + epsilon;

    public PassKind Kind => PassKind.Fusion;

    public OpGraph Run(OpGraph graph, GraphTarget target)
    {
        var fused = _inner.Run(graph, target);
        var nodes = fused.Nodes.Select(n => n.Op != FuseAddRmsNormPass.FusedOp
            ? n
            : new OpNode(n.Id, n.Op, n.Inputs, n.Outputs, new Dictionary<string, string>(n.Attributes) { ["arg.Eps"] = epsilon }, n.Origin, n.Variant)).ToList();
        return new OpGraph(fused.Name, fused.Values, nodes, false);
    }
}

/// <summary>A pass that renames nothing and fuses nothing; its output must equal the input.</summary>
public sealed class IdentityPass : IGraphPass
{
    public string Name => "identity";

    public PassKind Kind => PassKind.Fusion;

    public OpGraph Run(OpGraph graph, GraphTarget target) =>
        new(graph.Name, graph.Values, graph.Nodes.Select(n => new OpNode(n.Id, n.Op, n.Inputs, n.Outputs, n.Attributes, n.Origin, n.Variant)).ToList(), false);
}

using Starlings.Roost.Graph;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>Test tolerances for float32 (in-memory values; nothing under evidence/).</summary>
public sealed class TestTolerances(double absolute, double relative) : IToleranceSource
{
    public (double Absolute, double Relative) Lookup(string op, GraphDType dtype) => (absolute, relative);
}

/// <summary>
/// A test fusion pass, standing in for the real ones that stream L writes (R-071): Residual followed by the RmsNorm of
/// its result becomes AddRmsNorm. <see cref="Epsilon"/> lets a test break it on purpose.
/// </summary>
public sealed class FuseAddRmsNormPass(string? epsilon = null) : IGraphPass
{
    public string Name => "fuse-add-rmsnorm";

    public PassKind Kind => PassKind.Fusion;

    public string? Epsilon { get; } = epsilon;

    public OpGraph Run(OpGraph graph, GraphTarget target)
    {
        var nodes = new List<OpNode>();
        var src = graph.Nodes;
        for (int i = 0; i < src.Count; i++)
        {
            var r = src[i];
            if (r.Op == "Residual" && i + 1 < src.Count && src[i + 1].Op == "RmsNorm" && src[i + 1].Inputs[0] == r.Outputs[0])
            {
                var norm = src[i + 1];
                var attrs = new Dictionary<string, string>
                {
                    ["in"] = "Input,Residual,Weight",
                    ["out"] = "Input,Residual",
                    ["arg.Rows"] = norm.Attributes["arg.Rows"],
                    ["arg.Dim"] = norm.Attributes["arg.Dim"],
                    ["arg.Eps"] = Epsilon ?? norm.Attributes["arg.Eps"],
                    ["arg.DType"] = norm.Attributes["arg.DType"],
                    ["node"] = norm.Attributes["node"],
                };
                // Input := the add's second operand (overwritten with the normalised sum); Residual := its first (overwritten with the sum).
                nodes.Add(new OpNode(
                    nodes.Count,
                    "AddRmsNorm",
                    [r.Inputs[1], r.Inputs[0], norm.Inputs[1]],
                    [norm.Outputs[0], r.Outputs[0]],
                    attrs,
                    [r.Id, norm.Id],
                    null));
                i++;
            }
            else
            {
                nodes.Add(new OpNode(nodes.Count, r.Op, r.Inputs, r.Outputs, r.Attributes, [r.Id], r.Variant));
            }
        }

        return new OpGraph(graph.Name + "+" + Name, graph.Values, nodes, false);
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

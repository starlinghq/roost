using System;
using System.Collections.Generic;

namespace Starlings.Roost.Graph.Passes;

/// <summary>
/// Fusion pass (R-071): a <c>Residual</c> and the <c>RmsNorm</c> of its sum become one <c>AddRmsNorm</c>
/// (<c>Residual += Input; Input = RmsNorm(Residual)</c>, contract C3).
/// </summary>
/// <remarks>
/// <para>
/// For a node <c>Residual(A, B) =&gt; S</c> and the first later node <c>RmsNorm(Input = S, Weight = W) =&gt; Y</c>, the pass
/// emits, at the Residual's position, <c>AddRmsNorm(Input = B =&gt; Y, Residual = A =&gt; S, Weight = W)</c> with the
/// RmsNorm's scalar arguments. Both overwritten values keep their reference ids, so the C7 checker compares Y and S with
/// the unfused graph. The node's <see cref="OpNode.Origin"/> is the union of the two nodes' origins.
/// </para>
/// <para>
/// A pair is fused only when the result is the same computation: A and B are distinct intermediates that no later node
/// reads (the fused op overwrites them in place); A, B, S and Y have one shape and element type; W is a weight; neither
/// node reads a column view; the two nodes name the same activation type; and no value is a tensor-parallel partial
/// sum. Anything else is left unfused. The pass emits nothing for a target without the op (see
/// <see cref="GraphTarget.AvailableVariants"/>; the CPU reference, SM 0, always has it).
/// </para>
/// </remarks>
public sealed class FuseAddRmsNormPass : IGraphPass
{
    /// <summary>The C3 op the pass emits.</summary>
    public const string FusedOp = "AddRmsNorm";

    /// <inheritdoc/>
    public string Name => "fuse-add-rmsnorm";

    /// <inheritdoc/>
    public PassKind Kind => PassKind.Fusion;

    /// <inheritdoc/>
    public OpGraph Run(OpGraph graph, GraphTarget target)
    {
        if (graph is null)
        {
            throw new ArgumentNullException(nameof(graph));
        }

        if (target is null)
        {
            throw new ArgumentNullException(nameof(target));
        }

        var src = graph.Nodes;
        var normOf = new int[src.Count];
        var fusedInto = new bool[src.Count];
        for (int i = 0; i < src.Count; i++)
        {
            normOf[i] = -1;
        }

        if (GraphProtocol.Provides(target, FusedOp))
        {
            for (int i = 0; i < src.Count; i++)
            {
                int j = Match(graph, i);
                if (j >= 0 && !fusedInto[j])
                {
                    normOf[i] = j;
                    fusedInto[j] = true;
                }
            }
        }

        var nodes = new List<OpNode>(src.Count);
        for (int i = 0; i < src.Count; i++)
        {
            if (fusedInto[i])
            {
                continue;
            }

            nodes.Add(normOf[i] >= 0 ? Fuse(src[i], src[normOf[i]], nodes.Count) : GraphProtocol.Renumber(src[i], nodes.Count));
        }

        return new OpGraph(graph.Name + "+" + Name, graph.Values, nodes, false);
    }

    // Returns the index of the RmsNorm that the Residual at index i fuses with, or -1.
    private static int Match(OpGraph graph, int i)
    {
        var nodes = graph.Nodes;
        var r = nodes[i];
        if (r.Op != "Residual" || GraphProtocol.HasView(r) || r.Outputs.Count != 1)
        {
            return -1;
        }

        int a = GraphProtocol.Input(r, "A");
        int b = GraphProtocol.Input(r, "B");
        int s = GraphProtocol.Output(r, "Output");
        if (a < 0 || b < 0 || s < 0 || a == b || r.Inputs.Count != 2)
        {
            return -1;
        }

        int j = -1;
        for (int k = i + 1; k < nodes.Count; k++)
        {
            if (nodes[k].Op == "RmsNorm" && GraphProtocol.Input(nodes[k], "Input") == s)
            {
                j = k;
                break;
            }
        }

        if (j < 0)
        {
            return -1;
        }

        var n = nodes[j];
        int w = GraphProtocol.Input(n, "Weight");
        int y = GraphProtocol.Output(n, "Output");
        if (GraphProtocol.HasView(n) || n.Outputs.Count != 1 || n.Inputs.Count != 2 || w < 0 || y < 0)
        {
            return -1;
        }

        var values = graph.Values;
        if (values[a].Kind != ValueKind.Intermediate || values[b].Kind != ValueKind.Intermediate || values[w].Kind != ValueKind.Weight)
        {
            return -1;
        }

        if (!GraphProtocol.SameShape(values[a], values[s]) || !GraphProtocol.SameShape(values[b], values[s]) || !GraphProtocol.SameShape(values[y], values[s]))
        {
            return -1;
        }

        if (values[a].Sharding.Kind == ShardKind.PartialSum || values[b].Sharding.Kind == ShardKind.PartialSum || values[s].Sharding.Kind == ShardKind.PartialSum)
        {
            return -1;
        }

        r.Attributes.TryGetValue("arg.DType", out string? rType);
        n.Attributes.TryGetValue("arg.DType", out string? nType);
        if (!string.Equals(rType, nType, StringComparison.Ordinal))
        {
            return -1;
        }

        // The fused op overwrites A and B, so neither may be read again.
        for (int k = i + 1; k < nodes.Count; k++)
        {
            foreach (int v in nodes[k].Inputs)
            {
                if (v == a || v == b)
                {
                    return -1;
                }
            }
        }

        return j;
    }

    private static OpNode Fuse(OpNode residual, OpNode norm, int id)
    {
        var attrs = new Dictionary<string, string>
        {
            ["in"] = "Input,Residual,Weight",
            ["out"] = "Input,Residual",
        };
        foreach (var kv in norm.Attributes)
        {
            if (kv.Key.StartsWith("arg.", StringComparison.Ordinal))
            {
                attrs[kv.Key] = kv.Value;
            }
        }

        attrs["node"] = NodeName(residual) + "+" + NodeName(norm);
        if (residual.Attributes.TryGetValue("layer", out string? layer))
        {
            attrs["layer"] = layer;
        }

        return new OpNode(
            id,
            FusedOp,
            [GraphProtocol.Input(residual, "B"), GraphProtocol.Input(residual, "A"), GraphProtocol.Input(norm, "Weight")],
            [GraphProtocol.Output(norm, "Output"), GraphProtocol.Output(residual, "Output")],
            attrs,
            GraphProtocol.Origins(residual, norm),
            null);
    }

    private static string NodeName(OpNode node) => node.Attributes.TryGetValue("node", out string? name) ? name : node.Op;
}

using System;
using System.Collections.Generic;

namespace Starlings.Roost.Graph.Passes;

/// <summary>
/// Reads the node attributes of the graph protocol the model compiler writes (src/Starlings.Roost.ModelCompiler/README.md,
/// "The graph protocol"): <c>in</c> and <c>out</c> name the argument field of each input and output position, <c>arg.FIELD</c>
/// holds a scalar argument, <c>view.FIELD</c> marks a column view of an input, and <c>node</c> and <c>layer</c> name the node.
/// </summary>
internal static class GraphProtocol
{
    /// <summary>The argument field names of a node's inputs (<paramref name="key"/> <c>in</c>) or outputs (<c>out</c>).</summary>
    public static string[] Fields(OpNode node, string key) =>
        node.Attributes.TryGetValue(key, out string? text) && text.Length > 0 ? text.Split(',') : Array.Empty<string>();

    /// <summary>The value id a node reads through <paramref name="field"/>, or -1.</summary>
    public static int Input(OpNode node, string field)
    {
        int i = Array.IndexOf(Fields(node, "in"), field);
        return i >= 0 && i < node.Inputs.Count ? node.Inputs[i] : -1;
    }

    /// <summary>The value id a node writes through <paramref name="field"/>, or -1.</summary>
    public static int Output(OpNode node, string field)
    {
        int i = Array.IndexOf(Fields(node, "out"), field);
        return i >= 0 && i < node.Outputs.Count ? node.Outputs[i] : -1;
    }

    /// <summary>Whether any input of the node is a column view.</summary>
    public static bool HasView(OpNode node)
    {
        foreach (var key in node.Attributes.Keys)
        {
            if (key.StartsWith("view.", StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Whether two values have the same shape and element type.</summary>
    public static bool SameShape(GraphValue x, GraphValue y)
    {
        if (x.DType != y.DType || x.Shape.Count != y.Shape.Count)
        {
            return false;
        }

        for (int i = 0; i < x.Shape.Count; i++)
        {
            if (x.Shape[i] != y.Shape[i])
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Whether a fusion pass may emit <paramref name="op"/> for <paramref name="target"/>: the CPU reference (SM 0)
    /// implements it, and a GPU target needs at least one conformance-passed variant of it.
    /// </summary>
    public static bool Provides(GraphTarget target, string op) =>
        target.Sm == 0 || (target.AvailableVariants.TryGetValue(op, out var variants) && variants.Count > 0);

    /// <summary>The union of two origin lists, in order, without repeats.</summary>
    public static IReadOnlyList<int> Origins(OpNode first, OpNode second)
    {
        var all = new List<int>(first.Origin);
        foreach (int o in second.Origin)
        {
            if (!all.Contains(o))
            {
                all.Add(o);
            }
        }

        return all;
    }

    /// <summary>A copy of a node at a new position, keeping its origin.</summary>
    public static OpNode Renumber(OpNode node, int id) =>
        new(id, node.Op, node.Inputs, node.Outputs, node.Attributes, node.Origin, node.Variant);
}

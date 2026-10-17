using System.Collections.Generic;

namespace Spike.Generator
{
    /// <summary>A node of the C7 op graph: a C3 op with inputs and an output shape [tokens, OutDim].</summary>
    internal sealed class GNode
    {
        public GNode(string name, string op, string[] inputs, string? weight, int outDim)
        {
            Name = name;
            Op = op;
            Inputs = inputs;
            Weight = weight;
            OutDim = outDim;
        }

        public string Name { get; }
        public string Op { get; }
        public string[] Inputs { get; }
        public string? Weight { get; }
        public int OutDim { get; }

        /// <summary>Second output, set on fused nodes (the residual sum that later nodes still read).</summary>
        public string? Output2 { get; set; }
    }

    internal sealed class Graph
    {
        public List<GNode> Nodes { get; } = new List<GNode>();
    }

    /// <summary>C7 pass interface: a pure graph to graph rewrite.</summary>
    internal interface IGraphPass
    {
        string Name { get; }

        Graph Run(Graph input);
    }

    /// <summary>Fuses Residual followed directly by an RmsNorm of its result into AddRmsNorm.</summary>
    internal sealed class AddRmsNormFusion : IGraphPass
    {
        public string Name => "fuse-add-rmsnorm";

        public Graph Run(Graph input)
        {
            var g = new Graph();
            var n = input.Nodes;
            for (int i = 0; i < n.Count; i++)
            {
                var r = n[i];
                if (r.Op == "Residual" && i + 1 < n.Count && n[i + 1].Op == "RmsNorm" && n[i + 1].Inputs[0] == r.Name)
                {
                    var norm = n[i + 1];
                    g.Nodes.Add(new GNode(norm.Name, "AddRmsNorm", r.Inputs, norm.Weight, norm.OutDim) { Output2 = r.Name });
                    i++;
                }
                else
                {
                    g.Nodes.Add(r);
                }
            }

            return g;
        }
    }
}

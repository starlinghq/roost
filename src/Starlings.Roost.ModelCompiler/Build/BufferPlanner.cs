using System;
using System.Collections.Generic;
using System.Linq;
using Starlings.Roost.Graph;

namespace Starlings.Roost.ModelCompiler;

/// <summary>Where one value lives in the workspace arena.</summary>
internal sealed class Placement
{
    public Placement(int valueId, bool tokenMajor, string? rows, long offset, long elementsPerRow, GraphDType dtype)
    {
        ValueId = valueId;
        TokenMajor = tokenMajor;
        Rows = rows;
        Offset = offset;
        ElementsPerRow = elementsPerRow;
        Dtype = dtype;
    }

    public int ValueId { get; }

    /// <summary>True when the value has a leading tokens or selected dimension.</summary>
    public bool TokenMajor { get; }

    /// <summary>The leading step symbol (tokens or selected) for a token-major value.</summary>
    public string? Rows { get; }

    /// <summary>Token-major: bytes per token of capacity from the start of the token region. Fixed: bytes from the start of the fixed region.</summary>
    public long Offset { get; }

    /// <summary>Token-major: elements per row. Fixed: elements in all.</summary>
    public long ElementsPerRow { get; }

    public GraphDType Dtype { get; }
}

internal sealed class BufferPlan
{
    public BufferPlan(Dictionary<int, Placement> placements, long perTokenBytes, long fixedBytes)
    {
        Placements = placements;
        PerTokenBytes = perTokenBytes;
        FixedBytes = fixedBytes;
    }

    public Dictionary<int, Placement> Placements { get; }

    /// <summary>Token region size per token of capacity, after buffer reuse.</summary>
    public long PerTokenBytes { get; }

    /// <summary>Fixed region size in bytes, after buffer reuse; each buffer starts on a 256-byte boundary.</summary>
    public long FixedBytes { get; }
}

/// <summary>
/// Assigns every activation a place in one arena by live range. Values that are dead share space; a value an op
/// overwrites in place shares its input's place. Offsets of token-major values scale with the token capacity, so
/// the whole arena is one allocation sized once per worker and the forward allocates nothing.
/// </summary>
internal static class BufferPlanner
{
    internal const int FixedAlignment = 256;

    public static BufferPlan Plan(OpGraph graph)
    {
        int n = graph.Nodes.Count;
        var values = graph.Values;
        var arena = new HashSet<int>(values.Where(v => v.Kind == ValueKind.Intermediate || v.Kind == ValueKind.Output).Select(v => v.Id));
        var def = new Dictionary<int, int>();
        var last = new Dictionary<int, int>();
        var parent = new Dictionary<int, int>();
        foreach (int v in arena)
        {
            parent[v] = v;
        }

        int Find(int x)
        {
            while (parent[x] != x)
            {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }

            return x;
        }

        for (int i = 0; i < n; i++)
        {
            var node = graph.Nodes[i];
            foreach (int o in node.Outputs)
            {
                def[o] = i;
            }

            foreach (int v in node.Inputs)
            {
                if (arena.Contains(v))
                {
                    last[v] = i;
                }
            }

            string[] inF = Split(node, "in");
            string[] outF = Split(node, "out");
            for (int a = 0; a < outF.Length; a++)
            {
                int ia = Array.IndexOf(inF, outF[a]);
                if (ia >= 0 && arena.Contains(node.Inputs[ia]) && arena.Contains(node.Outputs[a]))
                {
                    parent[Find(node.Outputs[a])] = Find(node.Inputs[ia]);
                }
            }
        }

        foreach (var v in values.Where(v => v.Kind == ValueKind.Output))
        {
            last[v.Id] = n;
        }

        var groups = new Dictionary<int, Group>();
        foreach (int v in arena.OrderBy(x => x))
        {
            int root = Find(v);
            if (!groups.TryGetValue(root, out var g))
            {
                g = new Group();
                groups[root] = g;
            }

            g.Members.Add(v);
            g.Def = Math.Min(g.Def, def.TryGetValue(v, out int d) ? d : n);
            g.Last = Math.Max(g.Last, last.TryGetValue(v, out int l) ? l : (def.TryGetValue(v, out int d2) ? d2 : 0));
            var info = Shape(values[v]);
            g.Size = Math.Max(g.Size, info.TokenMajor ? info.Elements * ModelBuilder.ByteSize(values[v].DType) : Align(info.Elements * ModelBuilder.ByteSize(values[v].DType), FixedAlignment));
            g.TokenMajor = info.TokenMajor;
        }

        var placements = new Dictionary<int, Placement>();
        long tokenTotal = Allocate(groups.Values.Where(g => g.TokenMajor).OrderBy(g => g.Def).ThenBy(g => g.Members[0]).ToList());
        long fixedTotal = Allocate(groups.Values.Where(g => !g.TokenMajor).OrderBy(g => g.Def).ThenBy(g => g.Members[0]).ToList());
        foreach (var g in groups.Values)
        {
            foreach (int v in g.Members)
            {
                var info = Shape(values[v]);
                placements[v] = new Placement(v, info.TokenMajor, info.Rows, g.Offset, info.Elements, values[v].DType);
            }
        }

        return new BufferPlan(placements, tokenTotal, fixedTotal);
    }

    private static (bool TokenMajor, string? Rows, long Elements) Shape(GraphValue v)
    {
        bool tokenMajor = v.Shape.Count > 0 && v.Shape[0].IsSymbolic;
        long e = 1;
        for (int i = tokenMajor ? 1 : 0; i < v.Shape.Count; i++)
        {
            e *= v.Shape[i].Size;
        }

        return (tokenMajor, tokenMajor ? v.Shape[0].Symbol : null, e);
    }

    private static long Align(long x, long a) => (x + a - 1) / a * a;

    private static string[] Split(OpNode node, string key) =>
        node.Attributes.TryGetValue(key, out string? s) && s.Length > 0 ? s.Split(',') : Array.Empty<string>();

    private static long Allocate(List<Group> ordered)
    {
        var active = new List<Group>();
        long total = 0;
        foreach (var g in ordered)
        {
            active.RemoveAll(a => a.Last < g.Def);
            long candidate = 0;
            foreach (var a in active.OrderBy(x => x.Offset))
            {
                if (candidate + g.Size <= a.Offset)
                {
                    break;
                }

                candidate = Math.Max(candidate, a.Offset + a.Size);
            }

            g.Offset = candidate;
            active.Add(g);
            total = Math.Max(total, candidate + g.Size);
        }

        return total;
    }

    private sealed class Group
    {
        public List<int> Members { get; } = new List<int>();

        public int Def { get; set; } = int.MaxValue;

        public int Last { get; set; }

        public long Size { get; set; }

        public long Offset { get; set; }

        public bool TokenMajor { get; set; }
    }
}

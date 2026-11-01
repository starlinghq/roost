using System.Globalization;
using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.ModelCompiler;
using Starlings.Roost.Models.Qwen35;
using Starlings.Roost.Models.Tiny;

namespace Starlings.Roost.Graph.Tests;

/// <summary>
/// The reference graphs of the three model definitions and seeded inputs for them: weights from
/// <see cref="TinyWeights.Generate"/> (also for Qwen3.5, whose real weights are not in CI), random KV and recurrent
/// state, and one prefill chunk of four tokens at positions 5 to 8 of a sequence whose first five tokens are already in
/// the cache (block 2) and in recurrent-state slot 1.
/// </summary>
internal static class FusionHarness
{
    public const int Tokens = 4;
    public const ulong Seed = 0x5EED_0071;

    public static readonly GraphTarget Cpu = new(0, 1, 1, 1, new Dictionary<string, IReadOnlyList<string>>());

    public static readonly IReadOnlyDictionary<string, long> Symbols = new Dictionary<string, long>
    {
        ["tokens"] = Tokens,
        ["selected"] = 1,
        ["seqs"] = 1,
        ["kv_blocks"] = 1,
        ["blocks"] = 4,
        ["slots"] = 2,
    };

    public static OpGraph Build(string model) => model switch
    {
        "roost-tiny" => RoostTiny.BuildGraph(),
        "roost-tiny-hybrid" => RoostTinyHybrid.BuildGraph(),
        "qwen3.5-0.8b" => Qwen35Model.BuildGraph(),
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null),
    };

    public static long Elements(GraphValue v) =>
        v.Shape.Aggregate(1L, (n, d) => n * (d.IsSymbolic ? DimExpression.Evaluate(d.Symbol!, Symbols) : d.Size));

    /// <summary>Inputs for every weight, state slot and plan input of <paramref name="graph"/>, and for every intermediate it reads but does not produce.</summary>
    public static Dictionary<int, double[]> Inputs(OpGraph graph)
    {
        var produced = graph.Nodes.SelectMany(n => n.Outputs).ToHashSet();
        var read = graph.Nodes.SelectMany(n => n.Inputs).ToHashSet();
        var inputs = new Dictionary<int, double[]>();
        foreach (var v in graph.Values)
        {
            var dtype = (DType)(uint)v.DType;
            if (v.Kind == ValueKind.Weight)
            {
                if (read.Contains(v.Id))
                {
                    inputs[v.Id] = TinyWeights.Generate(Seed, v.Name, string.Join(",", v.Shape.Select(d => d.Size.ToString(CultureInfo.InvariantCulture))), dtype);
                }
            }
            else if (v.Kind == ValueKind.State || (v.Kind == ValueKind.Intermediate && read.Contains(v.Id) && !produced.Contains(v.Id)))
            {
                inputs[v.Id] = Uniform(v, dtype, v.Kind == ValueKind.State ? 0.5 : 1.0);
            }
            else if (v.Kind == ValueKind.Input && Plan(v.Name) is { } plan)
            {
                inputs[v.Id] = plan;
            }
        }

        return inputs;
    }

    private static double[] Uniform(GraphValue v, DType dtype, double amplitude)
    {
        var rng = new Random(HashCode.Combine(v.Name.Length, (int)Seed, v.Id));
        var values = new double[Elements(v)];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = TinyWeights.Round(((rng.NextDouble() * 2) - 1) * amplitude, dtype);
        }

        return values;
    }

    private static double[]? Plan(string name)
    {
        const int past = 5, block = 2, blockSize = 16;
        var positions = Enumerable.Range(past, Tokens).Select(p => (double)p).ToArray();
        return name switch
        {
            "plan.Ids" => [3, 17, 100, 42],
            "plan.Positions" => positions,
            "plan.SlotMapping" => positions.Select(p => (block * blockSize) + p).ToArray(),
            "plan.QIndptr" => [0, Tokens],
            "plan.KvIndptr" => [0, 1],
            "plan.KvIndices" => [block],
            "plan.KvLastPageLength" => [past + Tokens],
            "plan.Rows" => [Tokens - 1],
            "plan.StateSlots" => [1],
            "plan.CuSeqlens" => [0, Tokens],
            _ => null,
        };
    }

    /// <summary>
    /// The nodes from <paramref name="firstNode"/> to <paramref name="lastNode"/> (definition names) as a reference graph
    /// of their own. Value ids are kept; values the window neither reads nor writes become one-element placeholders, so a
    /// slice of a large model is evaluated without allocating the rest of it.
    /// </summary>
    public static OpGraph Window(OpGraph graph, string firstNode, string lastNode)
    {
        int first = graph.Nodes.ToList().FindIndex(n => n.Attributes["node"] == firstNode);
        int last = graph.Nodes.ToList().FindIndex(n => n.Attributes["node"] == lastNode);
        Assert.True(first >= 0 && last >= first, $"window {firstNode}..{lastNode}");
        var kept = graph.Nodes.Skip(first).Take(last - first + 1).ToList();
        var used = kept.SelectMany(n => n.Inputs.Concat(n.Outputs)).ToHashSet();
        var values = graph.Values
            .Select(v => used.Contains(v.Id)
                ? v
                : new GraphValue(v.Id, v.Name, ValueKind.Intermediate, [], GraphDType.F32, v.Layout, v.Sharding))
            .ToList();
        var nodes = kept.Select((n, i) => new OpNode(i, n.Op, n.Inputs, n.Outputs, n.Attributes, [i], n.Variant)).ToList();
        return new OpGraph(graph.Name + "[" + firstNode + ".." + lastNode + "]", values, nodes, true);
    }
}

using System;
using System.Collections.Generic;
using Starlings.Roost.Graph;

namespace Starlings.Roost.ModelCompiler;

internal sealed class WeightSlotInfo
{
    public WeightSlotInfo(string name, string? source, GraphDType dtype, long[] shape, int? layer, LocationInfo? location)
    {
        Name = name;
        Source = source;
        Dtype = dtype;
        Shape = shape;
        Layer = layer;
        Location = location;
        long n = 1;
        foreach (long d in shape)
        {
            n *= d;
        }

        Elements = n;
    }

    public string Name { get; }

    public string? Source { get; }

    public GraphDType Dtype { get; }

    public long[] Shape { get; }

    public long Elements { get; }

    public int? Layer { get; }

    public LocationInfo? Location { get; }
}

internal sealed class StateSlotInfo
{
    public StateSlotInfo(string name, int group, int layer, string role, GraphDType dtype, long[] blockShape)
    {
        Name = name;
        Group = group;
        Layer = layer;
        Role = role;
        Dtype = dtype;
        BlockShape = blockShape;
        long n = 1;
        foreach (long d in blockShape)
        {
            n *= d;
        }

        ElementsPerBlock = n;
    }

    public string Name { get; }

    public int Group { get; }

    public int Layer { get; }

    public string Role { get; }

    public GraphDType Dtype { get; }

    /// <summary>Shape of one block (paged groups) or one slot (recurrent), without the leading pool dimension.</summary>
    public long[] BlockShape { get; }

    public long ElementsPerBlock { get; }
}

internal sealed class GroupInfo
{
    public GroupInfo(int index, string name, string kind, int[] layers, GraphDType dtype)
    {
        Index = index;
        Name = name;
        Kind = kind;
        Layers = layers;
        Dtype = dtype;
    }

    public int Index { get; }

    public string Name { get; }

    public string Kind { get; }

    public int[] Layers { get; }

    public GraphDType Dtype { get; }

    public long KvHeads { get; set; }

    public long HeadSize { get; set; }

    public long BlockSize { get; set; } = 1;

    public long Window { get; set; }

    public long SnapshotSlots { get; set; }

    public long[] StateShape { get; set; } = Array.Empty<long>();

    public long[] ConvShape { get; set; } = Array.Empty<long>();

    public GraphDType ConvDtype { get; set; }

    public long BytesPerBlock { get; set; }
}

internal sealed class PlanInputInfo
{
    public PlanInputInfo(string field, GraphDType dtype, string count)
    {
        Field = field;
        Dtype = dtype;
        Count = count;
    }

    /// <summary>The op-argument field name that selects this input, for example Ids.</summary>
    public string Field { get; }

    public GraphDType Dtype { get; }

    /// <summary>The element count as a function of the step: tokens, selected, seqs, seqs+1 or kv_blocks.</summary>
    public string Count { get; }
}

internal sealed class BuiltModel
{
    public BuiltModel(
        OpGraph graph,
        List<WeightSlotInfo> weights,
        List<StateSlotInfo> stateSlots,
        List<GroupInfo> groups,
        List<PlanInputInfo> planInputs,
        List<int> outputValues,
        List<string> usedOps,
        GraphDType dtype)
    {
        Graph = graph;
        Weights = weights;
        StateSlots = stateSlots;
        Groups = groups;
        PlanInputs = planInputs;
        OutputValues = outputValues;
        UsedOps = usedOps;
        Dtype = dtype;
    }

    public OpGraph Graph { get; }

    public List<WeightSlotInfo> Weights { get; }

    public List<StateSlotInfo> StateSlots { get; }

    public List<GroupInfo> Groups { get; }

    /// <summary>The plan inputs the graph reads, in the fixed order of the plan input table.</summary>
    public List<PlanInputInfo> PlanInputs { get; }

    public List<int> OutputValues { get; }

    public List<string> UsedOps { get; }

    public GraphDType Dtype { get; }
}

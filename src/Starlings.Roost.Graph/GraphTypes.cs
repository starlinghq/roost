using System;
using System.Collections.Generic;

namespace Starlings.Roost.Graph;

/// <summary>Element types of graph values. Values equal <c>Starlings.Roost.Abi.DType</c> (C1 <c>ROOST_DTYPE_*</c>).</summary>
public enum GraphDType
{
    /// <summary>Undefined.</summary>
    Undefined = 0,

    /// <summary>binary32.</summary>
    F32 = 1,

    /// <summary>binary16.</summary>
    F16 = 2,

    /// <summary>bfloat16.</summary>
    BF16 = 3,

    /// <summary>binary64.</summary>
    F64 = 4,

    /// <summary>FP8 E4M3.</summary>
    F8E4M3 = 5,

    /// <summary>FP8 E5M2.</summary>
    F8E5M2 = 6,

    /// <summary>int8.</summary>
    I8 = 7,

    /// <summary>uint8.</summary>
    U8 = 8,

    /// <summary>int16.</summary>
    I16 = 9,

    /// <summary>int32.</summary>
    I32 = 10,

    /// <summary>uint32.</summary>
    U32 = 11,

    /// <summary>int64.</summary>
    I64 = 12,

    /// <summary>uint64.</summary>
    U64 = 13,

    /// <summary>Packed signed 4-bit.</summary>
    I4 = 14,

    /// <summary>Packed unsigned 4-bit.</summary>
    U4 = 15,

    /// <summary>FP4 E2M1.</summary>
    F4E2M1 = 16,
}

/// <summary>Layouts of graph values. Values equal <c>Starlings.Roost.Abi.TensorLayout</c>.</summary>
public enum GraphLayout
{
    /// <summary>Row-major.</summary>
    RowMajor = 0,

    /// <summary>Column-major.</summary>
    ColMajor = 1,

    /// <summary>Paged KV pool, NHD.</summary>
    PagedNhd = 2,

    /// <summary>Paged KV pool, HND.</summary>
    PagedHnd = 3,
}

/// <summary>One dimension of a shape: a constant, or a symbol bound per step (tokens, batch).</summary>
public readonly struct ShapeDim : IEquatable<ShapeDim>
{
    private ShapeDim(long size, string? symbol)
    {
        Size = size;
        Symbol = symbol;
    }

    /// <summary>The constant size, or -1 for a symbolic dimension.</summary>
    public long Size { get; }

    /// <summary>The symbol, for example <c>tokens</c>; null for a constant.</summary>
    public string? Symbol { get; }

    /// <summary>Whether the dimension is symbolic.</summary>
    public bool IsSymbolic => Symbol is not null;

    /// <summary>A constant dimension.</summary>
    /// <param name="size">The size, at least zero.</param>
    /// <returns>The dimension.</returns>
    public static ShapeDim Of(long size) =>
        size >= 0 ? new ShapeDim(size, null) : throw new ArgumentOutOfRangeException(nameof(size));

    /// <summary>A symbolic dimension.</summary>
    /// <param name="symbol">The symbol.</param>
    /// <returns>The dimension.</returns>
    public static ShapeDim Named(string symbol) => new(-1, symbol ?? throw new ArgumentNullException(nameof(symbol)));

    /// <inheritdoc/>
    public bool Equals(ShapeDim other) => Size == other.Size && string.Equals(Symbol, other.Symbol, StringComparison.Ordinal);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is ShapeDim other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => Symbol is null ? Size.GetHashCode() : StringComparer.Ordinal.GetHashCode(Symbol);

    /// <inheritdoc/>
    public override string ToString() => Symbol ?? Size.ToString(System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>Equality.</summary>
    /// <param name="left">Left.</param>
    /// <param name="right">Right.</param>
    /// <returns>True when equal.</returns>
    public static bool operator ==(ShapeDim left, ShapeDim right) => left.Equals(right);

    /// <summary>Inequality.</summary>
    /// <param name="left">Left.</param>
    /// <param name="right">Right.</param>
    /// <returns>True when different.</returns>
    public static bool operator !=(ShapeDim left, ShapeDim right) => !left.Equals(right);
}

/// <summary>The mesh axis a value is sharded over (C9).</summary>
public enum MeshAxis
{
    /// <summary>Not sharded.</summary>
    None = 0,

    /// <summary>Tensor parallel.</summary>
    Tensor,

    /// <summary>Pipeline parallel (stage assignment).</summary>
    Pipeline,

    /// <summary>Data parallel.</summary>
    Data,

    /// <summary>Expert parallel.</summary>
    Expert,
}

/// <summary>How a value is distributed over a mesh axis.</summary>
public enum ShardKind
{
    /// <summary>Every rank holds all of it.</summary>
    Replicated = 0,

    /// <summary>Split along <see cref="Sharding.Dimension"/>.</summary>
    Split,

    /// <summary>Every rank holds a partial sum that a collective reduces.</summary>
    PartialSum,
}

/// <summary>A sharding annotation (from M4; every value is <see cref="Replicated"/> before then).</summary>
public sealed class Sharding
{
    /// <summary>Creates an annotation.</summary>
    /// <param name="axis">The mesh axis.</param>
    /// <param name="kind">The distribution.</param>
    /// <param name="dimension">The split dimension, or -1.</param>
    public Sharding(MeshAxis axis, ShardKind kind, int dimension)
    {
        Axis = axis;
        Kind = kind;
        Dimension = dimension;
    }

    /// <summary>Not sharded.</summary>
    public static Sharding Replicated { get; } = new(MeshAxis.None, ShardKind.Replicated, -1);

    /// <summary>The mesh axis.</summary>
    public MeshAxis Axis { get; }

    /// <summary>The distribution.</summary>
    public ShardKind Kind { get; }

    /// <summary>The split dimension for <see cref="ShardKind.Split"/>, else -1.</summary>
    public int Dimension { get; }
}

/// <summary>What a graph value is.</summary>
public enum ValueKind
{
    /// <summary>Produced by a node.</summary>
    Intermediate = 0,

    /// <summary>A step input from the plan (ids, positions, block tables).</summary>
    Input,

    /// <summary>A weight, loaded once.</summary>
    Weight,

    /// <summary>State that persists across steps (KV pools, recurrent state; C4).</summary>
    State,

    /// <summary>A graph output (logits, sampled ids, head outputs).</summary>
    Output,
}

/// <summary>A typed value of the graph: shape, element type, layout and sharding.</summary>
public sealed class GraphValue
{
    /// <summary>Creates a value.</summary>
    /// <param name="id">Index in <see cref="OpGraph.Values"/>.</param>
    /// <param name="name">Stable name (weight slot name for weights).</param>
    /// <param name="kind">What it is.</param>
    /// <param name="shape">Shape, outermost first.</param>
    /// <param name="dtype">Element type.</param>
    /// <param name="layout">Layout.</param>
    /// <param name="sharding">Sharding; <see cref="Sharding.Replicated"/> before M4.</param>
    public GraphValue(int id, string name, ValueKind kind, IReadOnlyList<ShapeDim> shape, GraphDType dtype, GraphLayout layout, Sharding sharding)
    {
        Id = id;
        Name = name;
        Kind = kind;
        Shape = shape;
        DType = dtype;
        Layout = layout;
        Sharding = sharding;
    }

    /// <summary>Index in <see cref="OpGraph.Values"/>.</summary>
    public int Id { get; }

    /// <summary>Stable name.</summary>
    public string Name { get; }

    /// <summary>What it is.</summary>
    public ValueKind Kind { get; }

    /// <summary>Shape, outermost first.</summary>
    public IReadOnlyList<ShapeDim> Shape { get; }

    /// <summary>Element type.</summary>
    public GraphDType DType { get; }

    /// <summary>Layout.</summary>
    public GraphLayout Layout { get; }

    /// <summary>Sharding.</summary>
    public Sharding Sharding { get; }
}

/// <summary>One call of a C3 op.</summary>
public sealed class OpNode
{
    /// <summary>Creates a node.</summary>
    /// <param name="id">Index in <see cref="OpGraph.Nodes"/>.</param>
    /// <param name="op">The C3 member name, for example <c>RmsNorm</c>; checked against <c>IRoostOps</c> by the generator.</param>
    /// <param name="inputs">Value ids read, in argument order.</param>
    /// <param name="outputs">Value ids written.</param>
    /// <param name="attributes">Scalar arguments by args field name (eps, theta, window), invariant-culture text.</param>
    /// <param name="origin">Ids of the unfused reference nodes this node replaces; itself for an unfused node.</param>
    /// <param name="variant">Kernel variant id chosen by variant selection, or null.</param>
    public OpNode(
        int id,
        string op,
        IReadOnlyList<int> inputs,
        IReadOnlyList<int> outputs,
        IReadOnlyDictionary<string, string> attributes,
        IReadOnlyList<int> origin,
        string? variant)
    {
        Id = id;
        Op = op;
        Inputs = inputs;
        Outputs = outputs;
        Attributes = attributes;
        Origin = origin;
        Variant = variant;
    }

    /// <summary>Index in <see cref="OpGraph.Nodes"/>.</summary>
    public int Id { get; }

    /// <summary>The C3 member name.</summary>
    public string Op { get; }

    /// <summary>Value ids read.</summary>
    public IReadOnlyList<int> Inputs { get; }

    /// <summary>Value ids written.</summary>
    public IReadOnlyList<int> Outputs { get; }

    /// <summary>Scalar arguments by args field name.</summary>
    public IReadOnlyDictionary<string, string> Attributes { get; }

    /// <summary>
    /// Ids of the nodes of the unfused reference graph this node stands for. The checker compares this node's outputs
    /// with the outputs of those nodes on <c>CpuOps</c>.
    /// </summary>
    public IReadOnlyList<int> Origin { get; }

    /// <summary>Kernel variant id, or null before variant selection.</summary>
    public string? Variant { get; }
}

/// <summary>A typed DAG of C3 ops: nodes in a valid execution order, and the values they read and write.</summary>
public sealed class OpGraph
{
    /// <summary>Creates a graph.</summary>
    /// <param name="name">Model or profile name.</param>
    /// <param name="values">Values, indexed by id.</param>
    /// <param name="nodes">Nodes in execution order, indexed by id.</param>
    /// <param name="isReference">True for the unfused graph built from the model definition.</param>
    public OpGraph(string name, IReadOnlyList<GraphValue> values, IReadOnlyList<OpNode> nodes, bool isReference)
    {
        Name = name;
        Values = values;
        Nodes = nodes;
        IsReference = isReference;
    }

    /// <summary>Model or profile name.</summary>
    public string Name { get; }

    /// <summary>Values, indexed by id.</summary>
    public IReadOnlyList<GraphValue> Values { get; }

    /// <summary>Nodes in execution order.</summary>
    public IReadOnlyList<OpNode> Nodes { get; }

    /// <summary>True for the unfused reference graph; every pass output is checked against it.</summary>
    public bool IsReference { get; }
}

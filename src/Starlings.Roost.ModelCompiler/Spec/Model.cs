using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.CodeAnalysis.Text;

namespace Starlings.Roost.ModelCompiler;

/// <summary>Structural equality for arrays inside pipeline models, so incremental caching works.</summary>
internal readonly struct EquatableArray<T> : IEquatable<EquatableArray<T>>, IEnumerable<T>
    where T : IEquatable<T>
{
    private readonly T[]? _items;

    public EquatableArray(T[] items) => _items = items;

    public T[] Items => _items ?? Array.Empty<T>();

    public int Length => Items.Length;

    public T this[int index] => Items[index];

    public bool Equals(EquatableArray<T> other)
    {
        var a = Items;
        var b = other.Items;
        if (a.Length != b.Length)
        {
            return false;
        }

        for (int i = 0; i < a.Length; i++)
        {
            if (!a[i].Equals(b[i]))
            {
                return false;
            }
        }

        return true;
    }

    public override bool Equals(object? obj) => obj is EquatableArray<T> o && Equals(o);

    public override int GetHashCode()
    {
        int h = 17;
        foreach (var i in Items)
        {
            h = (h * 31) + i.GetHashCode();
        }

        return h;
    }

    public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)Items).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => Items.GetEnumerator();
}

/// <summary>Where an attribute was written, so a diagnostic points at it.</summary>
internal sealed record LocationInfo(string Path, TextSpan Span, LinePositionSpan Lines);

internal sealed record DimSpec(string Name, string Value, LocationInfo? Location);

internal sealed record WeightSpec(string Name, string Shape, string? Dtype, string? Source, string? Layers, LocationInfo? Location);

internal sealed record GroupSpec(
    string Name,
    string Kind,
    string? Layers,
    string KvHeads,
    string HeadSize,
    string BlockSize,
    string? Dtype,
    string StateShape,
    string ConvShape,
    string? ConvDtype,
    string Window,
    string SnapshotSlots,
    LocationInfo? Location);

internal sealed record NodeSpec(string Name, string Op, string Args, string? Layers, LocationInfo? Location);

internal sealed record OutputSpec(string Name, LocationInfo? Location);

internal sealed record ModelSpec(
    string Namespace,
    string ClassName,
    string ModelName,
    string Dtype,
    EquatableArray<DimSpec> Dims,
    EquatableArray<WeightSpec> Weights,
    EquatableArray<GroupSpec> Groups,
    EquatableArray<NodeSpec> Nodes,
    EquatableArray<OutputSpec> Outputs,
    LocationInfo? Location);

/// <summary>The kind of an op-argument field, as far as the generator fills it.</summary>
internal enum FieldKind
{
    Span,
    Int,
    Long,
    Float,
    Bool,
    Enum,
    DType,
    Other,
}

internal sealed record FieldInfo(string Name, FieldKind Kind, string TypeName, EquatableArray<string> EnumMembers);

/// <summary>One op of IRoostOps: its method, argument struct and fields. Fused and reserved ops are not abstract.</summary>
internal sealed record OpInfo(string Name, string ArgsType, bool IsAbstract, bool HasSupportsFlag, EquatableArray<FieldInfo> Fields);

/// <summary>The op vocabulary read from the IRoostOps symbol in the compilation, plus which contract types it can see.</summary>
internal sealed record Vocabulary(
    bool Found,
    EquatableArray<OpInfo> Ops,
    bool HasGraphTypes,
    bool HasStateTypes,
    bool HasOpId);

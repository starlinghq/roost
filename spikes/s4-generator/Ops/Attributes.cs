using System;

namespace Spike.Ops;

// Declared here for the spike so the Toy compiles without the generator's post-initialisation output being required
// by tools. The generator reads these by metadata name.

/// <summary>Marks a partial static class as a model definition; the generator emits Forward&lt;TOps&gt; for it.</summary>
[AttributeUsage(AttributeTargets.Class)]
public sealed class RoostModelAttribute : Attribute
{
    public int Hidden { get; set; }
    public int Intermediate { get; set; }
    public int Vocab { get; set; }
}

/// <summary>One node of the C7 graph, in topological order. Inputs name earlier nodes.</summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
public sealed class NodeAttribute : Attribute
{
    public NodeAttribute(string name, string op, params string[] inputs)
    {
        Name = name;
        Op = op;
        Inputs = inputs;
    }

    public string Name { get; }
    public string Op { get; }
    public string[] Inputs { get; }

    /// <summary>Weight slot name, when the op has a weight.</summary>
    public string? Weight { get; set; }

    /// <summary>Output feature dimension: H, I, 2I or V. Omitted means same as the first input.</summary>
    public string? Out { get; set; }
}

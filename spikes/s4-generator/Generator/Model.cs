using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.CodeAnalysis.Text;

namespace System.Runtime.CompilerServices
{
    internal static class IsExternalInit { }
}

namespace Spike.Generator
{
    /// <summary>Structural equality for arrays inside pipeline models, so incremental caching works.</summary>
    internal readonly struct EquatableArray<T> : IEquatable<EquatableArray<T>>, IEnumerable<T>
        where T : IEquatable<T>
    {
        private readonly T[] _items;

        public EquatableArray(T[] items) => _items = items;

        public T[] Items => _items ?? Array.Empty<T>();

        public int Length => Items.Length;

        public T this[int index] => Items[index];

        public bool Equals(EquatableArray<T> other)
        {
            var a = Items;
            var b = other.Items;
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
            {
                if (!a[i].Equals(b[i])) return false;
            }

            return true;
        }

        public override bool Equals(object? obj) => obj is EquatableArray<T> o && Equals(o);

        public override int GetHashCode()
        {
            int h = 17;
            foreach (var i in Items) h = (h * 31) + i.GetHashCode();
            return h;
        }

        public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)Items).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => Items.GetEnumerator();
    }

    internal sealed record LocationInfo(string Path, TextSpan Span, LinePositionSpan Lines);

    internal sealed record NodeSpec(string Name, string Op, EquatableArray<string> Inputs, string? Weight, string? Out, LocationInfo? Location);

    internal sealed record ModelSpec(string Namespace, string ClassName, int Hidden, int Intermediate, int Vocab, EquatableArray<NodeSpec> Nodes);

    internal sealed record FieldInfo(string Name, string Type);

    internal sealed record OpInfo(string Name, string ArgsType, bool Optional, EquatableArray<FieldInfo> Fields);

    /// <summary>The op vocabulary read from the IRoostOps symbol in the compilation.</summary>
    internal sealed record OpVocab(bool Found, EquatableArray<OpInfo> Ops);
}

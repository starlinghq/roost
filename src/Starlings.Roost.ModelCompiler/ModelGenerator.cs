using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Starlings.Roost.ModelCompiler;

/// <summary>
/// Incremental generator over Roost model definitions. A static partial class marked <c>[RoostModel]</c> (with
/// <c>[Dim]</c>, <c>[Weight]</c>, <c>[StateGroup]</c>, <c>[Node]</c> and <c>[Output]</c> attributes) becomes a typed op
/// graph (contract C7) and is lowered to <c>Forward&lt;TOps&gt;</c> over <c>IRoostOps</c> (contract C3), together with
/// the weight-loader slot order, the state-group descriptors (C4) and the plan input layout. The op vocabulary is
/// read from the <c>IRoostOps</c> symbol in the compilation, never listed here.
/// </summary>
[Generator(LanguageNames.CSharp)]
public sealed class ModelGenerator : IIncrementalGenerator
{
    internal const string OpsInterface = "Starlings.Roost.Ops.IRoostOps";
    internal const string DeviceSpanType = "Starlings.Roost.DeviceSpan";
    internal const string DTypeEnum = "Starlings.Roost.Abi.DType";

    /// <inheritdoc/>
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        context.RegisterPostInitializationOutput(static ctx => ctx.AddSource("RoostModelAttributes.g.cs", SourceText.From(Attributes.Source, Encoding.UTF8)));

        var vocab = context.CompilationProvider.Select(static (c, _) => ReadVocabulary(c));

        var models = context.SyntaxProvider.ForAttributeWithMetadataName(
            Attributes.Model,
            static (n, _) => n is ClassDeclarationSyntax,
            static (ctx, ct) => ReadModel(ctx, ct));

        context.RegisterSourceOutput(models.Combine(vocab), static (spc, pair) => Generate(spc, pair.Left, pair.Right));
    }

    private static Vocabulary ReadVocabulary(Compilation compilation)
    {
        var iface = compilation.GetTypeByMetadataName(OpsInterface);
        bool graph = compilation.GetTypeByMetadataName("Starlings.Roost.Graph.OpGraph") != null;
        bool state = compilation.GetTypeByMetadataName("Starlings.Roost.State.StateGroupDescriptor") != null;
        bool opId = compilation.GetTypeByMetadataName("Starlings.Roost.Abi.OpId") != null;
        if (iface is null)
        {
            return new Vocabulary(false, default, graph, state, opId);
        }

        var members = iface.GetMembers();
        var flags = new HashSet<string>(
            members.OfType<IPropertySymbol>().Where(p => p.IsStatic && p.Name.StartsWith("Supports", StringComparison.Ordinal)).Select(p => p.Name),
            StringComparer.Ordinal);
        var ops = new List<OpInfo>();
        foreach (var m in members.OfType<IMethodSymbol>())
        {
            if (!m.IsStatic || m.MethodKind != MethodKind.Ordinary || m.DeclaredAccessibility != Accessibility.Public || m.Parameters.Length != 1)
            {
                continue;
            }

            var p = m.Parameters[0];
            if (p.RefKind != RefKind.In || p.Type.TypeKind != TypeKind.Struct)
            {
                continue;
            }

            var fields = new List<FieldInfo>();
            foreach (var f in p.Type.GetMembers().OfType<IFieldSymbol>().Where(f => !f.IsStatic && !f.IsImplicitlyDeclared))
            {
                fields.Add(ReadField(f));
            }

            ops.Add(new OpInfo(
                m.Name,
                p.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
                m.IsAbstract,
                flags.Contains("Supports" + m.Name),
                new EquatableArray<FieldInfo>(fields.ToArray())));
        }

        return new Vocabulary(true, new EquatableArray<OpInfo>(ops.OrderBy(o => o.Name, StringComparer.Ordinal).ToArray()), graph, state, opId);
    }

    private static FieldInfo ReadField(IFieldSymbol f)
    {
        var t = f.Type;
        string full = t.ToDisplayString();
        string typeName = t.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        var none = new EquatableArray<string>(Array.Empty<string>());
        if (full == DeviceSpanType)
        {
            return new FieldInfo(f.Name, FieldKind.Span, typeName, none);
        }

        if (t.TypeKind == TypeKind.Enum)
        {
            var members = t.GetMembers().OfType<IFieldSymbol>().Where(x => x.IsConst).Select(x => x.Name).ToArray();
            return new FieldInfo(f.Name, full == DTypeEnum ? FieldKind.DType : FieldKind.Enum, typeName, new EquatableArray<string>(members));
        }

        FieldKind kind;
        switch (t.SpecialType)
        {
            case SpecialType.System_Int32:
                kind = FieldKind.Int;
                break;
            case SpecialType.System_Int64:
                kind = FieldKind.Long;
                break;
            case SpecialType.System_Single:
                kind = FieldKind.Float;
                break;
            case SpecialType.System_Boolean:
                kind = FieldKind.Bool;
                break;
            default:
                kind = FieldKind.Other;
                break;
        }

        return new FieldInfo(f.Name, kind, typeName, none);
    }

    private static ModelSpec ReadModel(GeneratorAttributeSyntaxContext ctx, CancellationToken ct)
    {
        var cls = (INamedTypeSymbol)ctx.TargetSymbol;
        var modelAttr = ctx.Attributes[0];
        string modelName = modelAttr.ConstructorArguments.Length > 0 ? modelAttr.ConstructorArguments[0].Value as string ?? cls.Name : cls.Name;
        string dtype = "BF16";
        foreach (var kv in modelAttr.NamedArguments)
        {
            if (kv.Key == "Dtype")
            {
                dtype = kv.Value.Value as string ?? dtype;
            }
        }

        var dims = new List<DimSpec>();
        var weights = new List<WeightSpec>();
        var groups = new List<GroupSpec>();
        var nodes = new List<NodeSpec>();
        var outputs = new List<OutputSpec>();
        foreach (var attr in cls.GetAttributes())
        {
            ct.ThrowIfCancellationRequested();
            string? name = attr.AttributeClass?.ToDisplayString();
            if (name is null || !name.StartsWith(Attributes.Namespace, StringComparison.Ordinal))
            {
                continue;
            }

            var loc = LocationOf(attr, ct);
            var args = attr.ConstructorArguments;
            var named = attr.NamedArguments.ToDictionary(k => k.Key, k => k.Value.Value as string, StringComparer.Ordinal);
            string Named(string key, string fallback = "") => named.TryGetValue(key, out var v) && v != null ? v : fallback;
            string? NamedOrNull(string key) => named.TryGetValue(key, out var v) ? v : null;
            string Arg(int i) => args.Length > i ? Convert.ToString(args[i].Value, System.Globalization.CultureInfo.InvariantCulture) ?? string.Empty : string.Empty;

            switch (name)
            {
                case Attributes.Dim:
                    dims.Add(new DimSpec(Arg(0), Arg(1), loc));
                    break;
                case Attributes.Weight:
                    weights.Add(new WeightSpec(Arg(0), Arg(1), NamedOrNull("Dtype"), NamedOrNull("Source"), NamedOrNull("Layers"), loc, attr.NamedArguments.Any(k => k.Key == "Reserved" && k.Value.Value is true)));
                    break;
                case Attributes.StateGroup:
                    groups.Add(new GroupSpec(
                        Arg(0),
                        Arg(1),
                        NamedOrNull("Layers"),
                        Named("KvHeads"),
                        Named("HeadSize"),
                        Named("BlockSize"),
                        NamedOrNull("Dtype"),
                        Named("StateShape"),
                        Named("ConvShape"),
                        NamedOrNull("ConvDtype"),
                        Named("Window"),
                        Named("SnapshotSlots"),
                        loc));
                    break;
                case Attributes.Node:
                    nodes.Add(new NodeSpec(Arg(0), Arg(1), Arg(2), NamedOrNull("Layers"), loc));
                    break;
                case Attributes.Output:
                    outputs.Add(new OutputSpec(Arg(0), loc));
                    break;
            }
        }

        string ns = cls.ContainingNamespace.IsGlobalNamespace ? string.Empty : cls.ContainingNamespace.ToDisplayString();
        return new ModelSpec(
            ns,
            cls.Name,
            modelName,
            dtype,
            new EquatableArray<DimSpec>(dims.ToArray()),
            new EquatableArray<WeightSpec>(weights.ToArray()),
            new EquatableArray<GroupSpec>(groups.ToArray()),
            new EquatableArray<NodeSpec>(nodes.ToArray()),
            new EquatableArray<OutputSpec>(outputs.ToArray()),
            LocationOf(modelAttr, ct));
    }

    private static LocationInfo? LocationOf(AttributeData attr, CancellationToken ct)
    {
        var loc = attr.ApplicationSyntaxReference?.GetSyntax(ct).GetLocation();
        if (loc is null)
        {
            return null;
        }

        var ls = loc.GetLineSpan();
        return new LocationInfo(ls.Path, loc.SourceSpan, ls.Span);
    }

    private static void Generate(SourceProductionContext spc, ModelSpec model, Vocabulary vocab)
    {
        void Report(DiagnosticDescriptor d, LocationInfo? l, string message)
        {
            var loc = l is null ? Location.None : Location.Create(l.Path, l.Span, l.Lines);
            spc.ReportDiagnostic(Diagnostic.Create(d, loc, message));
        }

        if (!vocab.Found)
        {
            Report(Diagnostics.OpsNotFound, model.Location, "The compilation does not reference " + OpsInterface + "; reference Starlings.Roost.Ops");
            return;
        }

        var built = ModelBuilder.Build(model, vocab, Report);
        if (built is null)
        {
            return;
        }

        var plan = BufferPlanner.Plan(built.Graph);
        string source = Emitter.Emit(model, vocab, built, plan);
        string hint = (model.Namespace.Length > 0 ? model.Namespace + "." : string.Empty) + model.ClassName + ".Roost.g.cs";
        spc.AddSource(hint, SourceText.From(source, Encoding.UTF8));
    }
}

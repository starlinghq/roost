using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Spike.Generator
{
    /// <summary>
    /// Incremental generator: [RoostModel] class + [Node] attributes -> typed op graph -> identity lowering and a fused
    /// lowering, both emitted as Forward&lt;TOps&gt;. The op vocabulary is read from the IRoostOps symbol, never listed here.
    /// </summary>
    [Generator(LanguageNames.CSharp)]
    public sealed class ModelGenerator : IIncrementalGenerator
    {
        internal const string OpsInterface = "Spike.Ops.IRoostOps";
        internal const string ModelAttribute = "Spike.Ops.RoostModelAttribute";
        internal const string NodeAttribute = "Spike.Ops.NodeAttribute";

        private static readonly DiagnosticDescriptor UnknownOp = new DiagnosticDescriptor(
            "ROOST001", "Unknown op", "Op '{0}' used by node '{1}' is not a member of IRoostOps (known ops: {2})", "Roost", DiagnosticSeverity.Error, true);

        private static readonly DiagnosticDescriptor UnknownInput = new DiagnosticDescriptor(
            "ROOST002", "Unknown input", "Node '{0}' reads '{1}', which is not an earlier node", "Roost", DiagnosticSeverity.Error, true);

        private static readonly DiagnosticDescriptor BadArgs = new DiagnosticDescriptor(
            "ROOST003", "Unsupported args field", "Op '{0}' has args field '{1}' that the generator cannot fill from a graph node", "Roost", DiagnosticSeverity.Error, true);

        private static readonly DiagnosticDescriptor Arity = new DiagnosticDescriptor(
            "ROOST004", "Wrong input count", "Node '{0}' ({1}) has {2} input(s) but the op takes {3}", "Roost", DiagnosticSeverity.Error, true);

        private static readonly DiagnosticDescriptor NoDim = new DiagnosticDescriptor(
            "ROOST006", "Output dimension required", "Node '{0}' ({1}) reads token ids, so its output dimension cannot be inferred; set Out", "Roost", DiagnosticSeverity.Error, true);

        private static readonly DiagnosticDescriptor NoOps = new DiagnosticDescriptor(
            "ROOST005", "IRoostOps not found", "The compilation does not reference Spike.Ops.IRoostOps", "Roost", DiagnosticSeverity.Error, true);

        public void Initialize(IncrementalGeneratorInitializationContext context)
        {
            var vocab = context.CompilationProvider.Select(static (c, _) => ReadVocab(c));

            var models = context.SyntaxProvider.ForAttributeWithMetadataName(
                ModelAttribute,
                static (n, _) => n is ClassDeclarationSyntax,
                static (ctx, ct) => ReadModel(ctx, ct));

            context.RegisterSourceOutput(models.Combine(vocab), static (spc, pair) => Emit(spc, pair.Left, pair.Right));
        }

        private static OpVocab ReadVocab(Compilation compilation)
        {
            var iface = compilation.GetTypeByMetadataName(OpsInterface);
            if (iface is null) return new OpVocab(false, default);
            var ops = new List<OpInfo>();
            foreach (var m in iface.GetMembers().OfType<IMethodSymbol>())
            {
                if (!m.IsStatic || m.MethodKind != MethodKind.Ordinary || m.DeclaredAccessibility != Accessibility.Public || m.Parameters.Length != 1) continue;
                var argsType = m.Parameters[0].Type;
                var fields = argsType.GetMembers().OfType<IFieldSymbol>().Where(f => !f.IsStatic)
                    .Select(f => new FieldInfo(f.Name, f.Type.Name)).ToArray();
                ops.Add(new OpInfo(m.Name, argsType.Name, m.IsAbstract == false, new EquatableArray<FieldInfo>(fields)));
            }

            return new OpVocab(true, new EquatableArray<OpInfo>(ops.OrderBy(o => o.Name, StringComparer.Ordinal).ToArray()));
        }

        private static ModelSpec ReadModel(GeneratorAttributeSyntaxContext ctx, CancellationToken ct)
        {
            var cls = (INamedTypeSymbol)ctx.TargetSymbol;
            int hidden = 0, inter = 0, vocabSize = 0;
            foreach (var kv in ctx.Attributes[0].NamedArguments)
            {
                int v = kv.Value.Value is int i ? i : 0;
                if (kv.Key == "Hidden") hidden = v;
                else if (kv.Key == "Intermediate") inter = v;
                else if (kv.Key == "Vocab") vocabSize = v;
            }

            var nodes = new List<NodeSpec>();
            foreach (var attr in cls.GetAttributes())
            {
                ct.ThrowIfCancellationRequested();
                if (attr.AttributeClass?.ToDisplayString() != NodeAttribute) continue;
                string name = (string)attr.ConstructorArguments[0].Value!;
                string op = (string)attr.ConstructorArguments[1].Value!;
                var inputs = attr.ConstructorArguments[2].Values.Select(v => (string)v.Value!).ToArray();
                string? weight = null, outDim = null;
                foreach (var kv in attr.NamedArguments)
                {
                    if (kv.Key == "Weight") weight = (string?)kv.Value.Value;
                    else if (kv.Key == "Out") outDim = (string?)kv.Value.Value;
                }

                LocationInfo? loc = null;
                var loc0 = attr.ApplicationSyntaxReference?.GetSyntax(ct).GetLocation();
                if (loc0 != null)
                {
                    var ls = loc0.GetLineSpan();
                    loc = new LocationInfo(ls.Path, loc0.SourceSpan, ls.Span);
                }

                nodes.Add(new NodeSpec(name, op, new EquatableArray<string>(inputs), weight, outDim, loc));
            }

            string ns = cls.ContainingNamespace.IsGlobalNamespace ? string.Empty : cls.ContainingNamespace.ToDisplayString();
            return new ModelSpec(ns, cls.Name, hidden, inter, vocabSize, new EquatableArray<NodeSpec>(nodes.ToArray()));
        }

        private static void Report(SourceProductionContext spc, DiagnosticDescriptor d, LocationInfo? l, params object[] args)
        {
            var loc = l is null ? Location.None : Location.Create(l.Path, l.Span, l.Lines);
            spc.ReportDiagnostic(Diagnostic.Create(d, loc, args));
        }

        private static void Emit(SourceProductionContext spc, ModelSpec model, OpVocab vocab)
        {
            if (!vocab.Found)
            {
                Report(spc, NoOps, null);
                return;
            }

            var ops = vocab.Ops.ToDictionary(o => o.Name);
            string known = string.Join(", ", vocab.Ops.Select(o => o.Name));
            var graph = new Graph();
            var dims = new Dictionary<string, int> { ["tokens"] = 1 };
            bool bad = false;
            var poisoned = new HashSet<string>();
            foreach (var n in model.Nodes)
            {
                if (!ops.TryGetValue(n.Op, out var op))
                {
                    Report(spc, UnknownOp, n.Location, n.Op, n.Name, known);
                    poisoned.Add(n.Name);
                    bad = true;
                    continue;
                }

                foreach (var f in op.Fields)
                {
                    if (!SupportedFields.Contains(f.Name))
                    {
                        Report(spc, BadArgs, n.Location, n.Op, f.Name);
                        bad = true;
                    }
                }

                int want = op.Fields.Count(f => f.Name == "Input" || f.Name == "Input2");
                if (n.Inputs.Length != want)
                {
                    Report(spc, Arity, n.Location, n.Name, n.Op, n.Inputs.Length, want);
                    bad = true;
                    continue;
                }

                bool inputsOk = true;
                foreach (var i in n.Inputs)
                {
                    if (poisoned.Contains(i))
                    {
                        // Downstream of a node that already has a diagnostic: no cascade.
                        poisoned.Add(n.Name);
                        inputsOk = false;
                    }
                    else if (!dims.ContainsKey(i))
                    {
                        Report(spc, UnknownInput, n.Location, n.Name, i);
                        bad = true;
                        inputsOk = false;
                    }
                }

                if (!inputsOk) continue;
                if (n.Out is null && n.Inputs.Length > 0 && n.Inputs[0] == "tokens")
                {
                    Report(spc, NoDim, n.Location, n.Name, n.Op);
                    poisoned.Add(n.Name);
                    bad = true;
                    continue;
                }

                int outDim = n.Out is null ? dims[n.Inputs[0]] : ResolveDim(model, n.Out);
                if (n.Op == "SiluAndMul") outDim = dims[n.Inputs[0]] / 2;
                dims[n.Name] = outDim;
                graph.Nodes.Add(new GNode(n.Name, n.Op, n.Inputs.ToArray(), n.Weight, outDim));
            }

            if (bad) return;

            var fused = new AddRmsNormFusion().Run(graph);
            string src = Lower(model, graph, fused, dims, ops);
            spc.AddSource(model.ClassName + ".g.cs", src);
        }

        private static readonly HashSet<string> SupportedFields = new HashSet<string>
        {
            "Input", "Input2", "Weight", "Output", "Output2", "Rows", "InDim", "OutDim",
        };

        private static int ResolveDim(ModelSpec m, string s)
        {
            switch (s)
            {
                case "H": return m.Hidden;
                case "I": return m.Intermediate;
                case "2I": return 2 * m.Intermediate;
                case "V": return m.Vocab;
                default: return int.Parse(s, CultureInfo.InvariantCulture);
            }
        }

        private static string Id(string s) => new string(s.Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray());

        private static string Lower(ModelSpec m, Graph original, Graph fused, Dictionary<string, int> dims, Dictionary<string, OpInfo> ops)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// <auto-generated/>");
            sb.AppendLine("#nullable enable");
            sb.AppendLine("using global::Spike.Ops;");
            if (m.Namespace.Length > 0) sb.Append("namespace ").Append(m.Namespace).AppendLine(" {");
            sb.Append("public static unsafe partial class ").Append(m.ClassName).AppendLine(" {");

            var weights = original.Nodes.Where(n => n.Weight != null).Select(n => (Slot: n.Weight!, Op: n.Op, Node: n)).GroupBy(w => w.Slot).Select(g => g.First()).ToList();
            sb.Append("    public static readonly string[] WeightSlots = { ").Append(string.Join(", ", weights.Select(w => "\"" + w.Slot + "\""))).AppendLine(" };");
            sb.Append("    public static readonly string[] Passes = { \"identity\", \"").Append(new AddRmsNormFusion().Name).AppendLine("\" };");

            // Workspace: every buffer is allocated once, outside the forward.
            sb.AppendLine("    public sealed class Workspace : global::System.IDisposable {");
            sb.AppendLine("        public readonly int MaxTokens;");
            sb.AppendLine("        public DeviceSpan TokenIds;");
            foreach (var w in weights) sb.Append("        public DeviceSpan W_").Append(Id(w.Slot)).AppendLine(";");
            foreach (var n in original.Nodes) sb.Append("        public DeviceSpan N_").Append(Id(n.Name)).AppendLine(";");
            sb.AppendLine("        private readonly global::System.Collections.Generic.List<nint> _allocs = new();");
            sb.AppendLine("        private DeviceSpan Alloc(long count, DType t) { nint p = (nint)global::System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(count * 4), 64); new global::System.Span<byte>((void*)p, (int)(count * 4)).Clear(); _allocs.Add(p); return new DeviceSpan(p, (int)count, t); }");
            sb.AppendLine("        public Workspace(int maxTokens) {");
            sb.AppendLine("            MaxTokens = maxTokens;");
            sb.AppendLine("            TokenIds = Alloc(maxTokens, DType.I32);");
            foreach (var w in weights)
            {
                string count = w.Op == "RmsNorm" || w.Op == "AddRmsNorm"
                    ? dims[w.Node.Name].ToString(CultureInfo.InvariantCulture)
                    : w.Op == "Embedding" ? "(long)" + m.Vocab.ToString(CultureInfo.InvariantCulture) + " * " + dims[w.Node.Name].ToString(CultureInfo.InvariantCulture)
                    : "(long)" + dims[w.Node.Name].ToString(CultureInfo.InvariantCulture) + " * " + dims[w.Node.Inputs[0]].ToString(CultureInfo.InvariantCulture);
                sb.Append("            W_").Append(Id(w.Slot)).Append(" = Alloc(").Append(count).AppendLine(", DType.F32);");
            }

            foreach (var n in original.Nodes) sb.Append("            N_").Append(Id(n.Name)).Append(" = Alloc((long)maxTokens * ").Append(n.OutDim.ToString(CultureInfo.InvariantCulture)).AppendLine(", DType.F32);");
            sb.AppendLine("            uint s = 12345;");
            for (int i = 0; i < weights.Count; i++)
            {
                string f = "W_" + Id(weights[i].Slot);
                bool norm = weights[i].Op == "RmsNorm" || weights[i].Op == "AddRmsNorm";
                sb.Append("            { float* p = (float*)").Append(f).Append(".Ptr; for (int i = 0; i < ").Append(f).Append(".Length; i++) { s = s * 1664525u + 1013904223u; ")
                  .Append(norm ? "p[i] = 1f + ((s >> 8) / 16777216f - 0.5f) * 0.1f;" : "p[i] = ((s >> 8) / 16777216f - 0.5f) * 0.4f;").AppendLine(" } }");
            }

            sb.AppendLine("        }");
            sb.AppendLine("        public void Dispose() { foreach (var p in _allocs) global::System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)p); _allocs.Clear(); }");
            sb.Append("        public DeviceSpan Output => N_").Append(Id(original.Nodes[original.Nodes.Count - 1].Name)).AppendLine(";");
            sb.AppendLine("    }");

            EmitForward(sb, "Forward", "identity lowering of the unfused graph", original, dims, ops);
            EmitForward(sb, "ForwardFused", "lowering after fuse-add-rmsnorm; falls back to the unfused parts when the backend lacks the op", fused, dims, ops);
            sb.AppendLine("}");
            if (m.Namespace.Length > 0) sb.AppendLine("}");
            return sb.ToString();
        }


        private static void EmitForward(StringBuilder sb, string name, string doc, Graph g, Dictionary<string, int> dims, Dictionary<string, OpInfo> ops)
        {
            sb.Append("    /// <summary>").Append(doc).AppendLine(".</summary>");
            sb.Append("    public static void ").Append(name).AppendLine("<TOps>(Workspace w, int tokens) where TOps : struct, IRoostOps {");
            foreach (var n in g.Nodes)
            {
                if (n.Op == "AddRmsNorm")
                {
                    sb.AppendLine("        if (TOps.SupportsAddRmsNorm) {");
                    EmitCall(sb, "AddRmsNorm", n, g, dims, ops, "            ");
                    sb.AppendLine("        } else {");
                    var add = new GNode(n.Output2!, "Residual", n.Inputs, null, dims[n.Output2!]);
                    var norm = new GNode(n.Name, "RmsNorm", new[] { n.Output2! }, n.Weight, n.OutDim);
                    EmitCall(sb, "Residual", add, g, dims, ops, "            ");
                    EmitCall(sb, "RmsNorm", norm, g, dims, ops, "            ");
                    sb.AppendLine("        }");
                }
                else
                {
                    EmitCall(sb, n.Op, n, g, dims, ops, "        ");
                }
            }

            sb.AppendLine("    }");
        }

        private static string Buf(string node) => node == "tokens" ? "w.TokenIds" : "w.N_" + Id(node);

        private static void EmitCall(StringBuilder sb, string op, GNode n, Graph g, Dictionary<string, int> dims, Dictionary<string, OpInfo> ops, string ind)
        {
            var info = ops[op];
            sb.Append(ind).AppendLine("{");
            sb.Append(ind).Append("    var a = new ").Append(info.ArgsType).AppendLine(" {");
            foreach (var f in info.Fields)
            {
                string? v = null;
                switch (f.Name)
                {
                    case "Input": v = Buf(n.Inputs[0]); break;
                    case "Input2": v = Buf(n.Inputs[1]); break;
                    case "Weight": v = "w.W_" + Id(n.Weight!); break;
                    case "Output": v = Buf(n.Name); break;
                    case "Output2": v = Buf(n.Output2!); break;
                    case "Rows": v = "tokens"; break;
                    case "InDim": v = dims[n.Inputs[0]].ToString(CultureInfo.InvariantCulture); break;
                    case "OutDim": v = n.OutDim.ToString(CultureInfo.InvariantCulture); break;
                }

                sb.Append(ind).Append("        ").Append(f.Name).Append(" = ").Append(v).AppendLine(",");
            }

            sb.Append(ind).AppendLine("    };");
            sb.Append(ind).Append("    TOps.").Append(op).AppendLine("(in a);");
            sb.Append(ind).AppendLine("}");
        }
    }
}

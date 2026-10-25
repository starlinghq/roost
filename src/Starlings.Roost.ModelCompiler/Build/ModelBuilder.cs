using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Starlings.Roost.Graph;

namespace Starlings.Roost.ModelCompiler;

/// <summary>
/// Turns a model definition into the C7 graph. Everything the lowering needs travels in the graph itself: the
/// <c>in</c> and <c>out</c> node attributes name the argument field of each input and output position, and
/// <c>arg.FIELD</c> carries the scalar arguments, so a pass output can be lowered the same way.
/// </summary>
internal sealed class ModelBuilder
{
    /// <summary>The names a step's runtime counts go by in shapes and integer arguments.</summary>
    internal static readonly string[] RuntimeSymbols = { "tokens", "selected", "seqs", "kv_blocks" };

    /// <summary>
    /// Plan inputs: a DeviceSpan field with one of these names that a node leaves unbound reads the step's input of that
    /// name. This is how the plan input layout follows from the C3 argument structs alone.
    /// </summary>
    internal static readonly PlanInputInfo[] PlanTable =
    {
        new PlanInputInfo("Ids", GraphDType.I32, "tokens"),
        new PlanInputInfo("Positions", GraphDType.I32, "tokens"),
        new PlanInputInfo("SlotMapping", GraphDType.I64, "tokens"),
        new PlanInputInfo("QIndptr", GraphDType.I32, "seqs+1"),
        new PlanInputInfo("KvIndptr", GraphDType.I32, "seqs+1"),
        new PlanInputInfo("KvIndices", GraphDType.I32, "kv_blocks"),
        new PlanInputInfo("KvLastPageLength", GraphDType.I32, "seqs"),
        new PlanInputInfo("Rows", GraphDType.I32, "selected"),
        new PlanInputInfo("StateSlots", GraphDType.I32, "seqs"),
        new PlanInputInfo("CuSeqlens", GraphDType.I32, "seqs+1"),
    };

    /// <summary>Fused ops whose unfused form the lowering knows.</summary>
    internal static readonly string[] KnownDecompositions = { "AddRmsNorm" };

    private static readonly string[] KvRoles = { "k", "v" };
    private static readonly char[] Semicolon = { ';' };

    private readonly ModelSpec _spec;
    private readonly Dictionary<string, OpInfo> _ops;
    private readonly Action<DiagnosticDescriptor, LocationInfo?, string> _report;
    private readonly Dictionary<string, long> _dims = new Dictionary<string, long>(StringComparer.Ordinal);
    private readonly List<ValueRec> _values = new List<ValueRec>();
    private readonly Dictionary<string, int> _byName = new Dictionary<string, int>(StringComparer.Ordinal);
    private readonly List<NodeRec> _nodes = new List<NodeRec>();
    private readonly HashSet<string> _poisoned = new HashSet<string>(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _overwrittenBy = new Dictionary<int, string>();
    private readonly HashSet<int> _read = new HashSet<int>();
    private readonly HashSet<string> _reserved = new HashSet<string>(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _planValues = new Dictionary<string, int>(StringComparer.Ordinal);
    private readonly List<WeightSlotInfo> _weights = new List<WeightSlotInfo>();
    private readonly List<StateSlotInfo> _stateSlots = new List<StateSlotInfo>();
    private readonly List<GroupInfo> _groups = new List<GroupInfo>();
    private readonly HashSet<string> _usedOps = new HashSet<string>(StringComparer.Ordinal);
    private GraphDType _dtype;
    private bool _failed;

    private ModelBuilder(ModelSpec spec, Vocabulary vocab, Action<DiagnosticDescriptor, LocationInfo?, string> report)
    {
        _spec = spec;
        _ops = vocab.Ops.ToDictionary(o => o.Name, StringComparer.Ordinal);
        _report = report;
    }

    internal static BuiltModel? Build(ModelSpec spec, Vocabulary vocab, Action<DiagnosticDescriptor, LocationInfo?, string> report)
    {
        return new ModelBuilder(spec, vocab, report).Run(vocab);
    }

    private void Fail(DiagnosticDescriptor d, LocationInfo? loc, string message)
    {
        _failed = true;
        _report(d, loc, message);
    }

    private BuiltModel? Run(Vocabulary vocab)
    {
        if (!TryDType(_spec.Dtype, out _dtype) || IsSubByte(_dtype))
        {
            Fail(Diagnostics.BadDefinition, _spec.Location, "Model dtype '" + _spec.Dtype + "' is not a byte-sized element type name (for example F32, F16, BF16)");
            return null;
        }

        if (!vocab.HasStateTypes && _spec.Groups.Length > 0)
        {
            Fail(Diagnostics.MissingContractType, _spec.Location, "State groups need Starlings.Roost.State.StateGroupDescriptor; reference Starlings.Roost.Abstractions");
        }

        ReadDims();
        ReadWeights();
        ReadGroups();
        ReadNodes();
        ReadOutputs();

        if (_failed)
        {
            return null;
        }

        foreach (var w in _weights)
        {
            if (!_read.Contains(_byName[w.Name]) && !_reserved.Contains(w.Name))
            {
                _report(Diagnostics.UnusedWeight, w.Location, "Weight slot '" + w.Name + "' is never read by a node; the loader still reserves its slot");
            }
        }

        return Finish();
    }

    // ---- dims, weights, state groups ---------------------------------------------------------------------

    private void ReadDims()
    {
        foreach (var d in _spec.Dims)
        {
            if (!IsIdentifier(d.Name) || Array.IndexOf(RuntimeSymbols, d.Name) >= 0 || _dims.ContainsKey(d.Name))
            {
                Fail(Diagnostics.Duplicate, d.Location, "Dim name '" + d.Name + "' is not an identifier, is a step symbol (" + string.Join(", ", RuntimeSymbols) + ") or is declared twice");
                continue;
            }

            if (!DimExpression.TryEvaluate(d.Value, _dims.ToDictionary(k => k.Key, k => k.Value), out long v, out string? err))
            {
                Fail(Diagnostics.BadShape, d.Location, "Dim '" + d.Name + "': " + err);
                continue;
            }

            _dims[d.Name] = v;
        }
    }

    private void ReadWeights()
    {
        var expanded = ExpandBlocks(_spec.Weights.Items, w => w.Layers, w => w.Location, "Weight");
        foreach (var (w, layer) in expanded)
        {
            if (!Subst(w.Name, layer, w.Location, out string name) || !Subst(w.Shape, layer, w.Location, out string shape)
                || !Subst(w.Source ?? string.Empty, layer, w.Location, out string source))
            {
                continue;
            }

            var dtype = _dtype;
            if (w.Dtype != null && !TryDType(w.Dtype, out dtype))
            {
                Fail(Diagnostics.BadDefinition, w.Location, "Weight '" + name + "': dtype '" + w.Dtype + "' is not an element type name");
                continue;
            }

            if (!ParseShape(shape, false, w.Location, "Weight '" + name + "'", out var dims))
            {
                continue;
            }

            if (_byName.ContainsKey(name))
            {
                Fail(Diagnostics.Duplicate, w.Location, "Name '" + name + "' is declared twice");
                continue;
            }

            var info = new WeightSlotInfo(name, source.Length == 0 ? null : source, dtype, dims.Select(d => d.Size).ToArray(), layer, w.Location);
            _weights.Add(info);
            if (w.Reserved)
            {
                _reserved.Add(name);
            }

            AddValue(name, ValueKind.Weight, dims, dtype, GraphLayout.RowMajor);
        }
    }

    private void ReadGroups()
    {
        int index = 0;
        foreach (var g in _spec.Groups)
        {
            if (g.Kind != "Full" && g.Kind != "Sliding" && g.Kind != "Recurrent")
            {
                Fail(Diagnostics.BadStateGroup, g.Location, "State group '" + g.Name + "': kind '" + g.Kind + "' is not supported by the generator (Full, Sliding or Recurrent)");
                continue;
            }

            int[] layers = Array.Empty<int>();
            string? lerr = g.Layers is null ? "Layers is required" : null;
            if (g.Layers != null && !Layers.TryParse(g.Layers, out layers, out lerr))
            {
                layers = Array.Empty<int>();
            }

            if (lerr != null)
            {
                Fail(Diagnostics.BadStateGroup, g.Location, "State group '" + g.Name + "': " + lerr);
                continue;
            }

            if (layers.Length == 0 || !IsIdentifier(g.Name) || _groups.Any(x => x.Name == g.Name))
            {
                Fail(Diagnostics.BadStateGroup, g.Location, "State group '" + g.Name + "' is empty, is not an identifier or is declared twice");
                continue;
            }

            bool recurrent = g.Kind == "Recurrent";
            var dtype = recurrent ? GraphDType.F32 : _dtype;
            if (g.Dtype != null && (!TryDType(g.Dtype, out dtype) || IsSubByte(dtype)))
            {
                Fail(Diagnostics.BadStateGroup, g.Location, "State group '" + g.Name + "': dtype '" + g.Dtype + "' is not a byte-sized element type name");
                continue;
            }

            var info = new GroupInfo(index, g.Name, g.Kind, layers, dtype) { ConvDtype = _dtype };
            if (g.ConvDtype != null)
            {
                var cd = _dtype;
                if (!TryDType(g.ConvDtype, out cd) || IsSubByte(cd))
                {
                    Fail(Diagnostics.BadStateGroup, g.Location, "State group '" + g.Name + "': ConvDtype '" + g.ConvDtype + "' is not a byte-sized element type name");
                    continue;
                }

                info.ConvDtype = cd;
            }

            bool ok = true;
            if (recurrent)
            {
                ok &= ParseShape(g.StateShape, false, g.Location, "State group '" + g.Name + "' StateShape", out var ss) && ss.Count > 0;
                if (ok)
                {
                    info.StateShape = ss.Select(d => d.Size).ToArray();
                }
                else if (g.StateShape.Length == 0)
                {
                    Fail(Diagnostics.BadStateGroup, g.Location, "Recurrent state group '" + g.Name + "' needs StateShape");
                }

                if (g.ConvShape.Length > 0)
                {
                    bool cok = ParseShape(g.ConvShape, false, g.Location, "State group '" + g.Name + "' ConvShape", out var cs);
                    ok &= cok;
                    if (cok)
                    {
                        info.ConvShape = cs.Select(d => d.Size).ToArray();
                    }
                }

                ok &= TryInt(g.SnapshotSlots, true, g.Location, g.Name + " SnapshotSlots", out long snaps);
                info.SnapshotSlots = snaps;
            }
            else
            {
                ok &= TryInt(g.KvHeads, false, g.Location, g.Name + " KvHeads", out long kvh);
                ok &= TryInt(g.HeadSize, false, g.Location, g.Name + " HeadSize", out long hs);
                ok &= TryInt(g.BlockSize, false, g.Location, g.Name + " BlockSize", out long bs);
                info.KvHeads = kvh;
                info.HeadSize = hs;
                info.BlockSize = bs;
                if (g.Kind == "Sliding")
                {
                    ok &= TryInt(g.Window, false, g.Location, g.Name + " Window", out long win);
                    info.Window = win;
                }
            }

            if (!ok)
            {
                _failed = true;
                continue;
            }

            long bytes = 0;
            foreach (int layer in layers)
            {
                if (recurrent)
                {
                    var st = new StateSlotInfo(g.Name + ".l" + layer.ToString(CultureInfo.InvariantCulture) + ".state", index, layer, "state", dtype, info.StateShape);
                    AddState(st);
                    bytes += st.ElementsPerBlock * ByteSize(dtype);
                    if (info.ConvShape.Length > 0)
                    {
                        var cv = new StateSlotInfo(g.Name + ".l" + layer.ToString(CultureInfo.InvariantCulture) + ".conv", index, layer, "conv", info.ConvDtype, info.ConvShape);
                        AddState(cv);
                        bytes += cv.ElementsPerBlock * ByteSize(info.ConvDtype);
                    }
                }
                else
                {
                    long[] shape = { info.BlockSize, info.KvHeads, info.HeadSize };
                    foreach (string role in KvRoles)
                    {
                        var st = new StateSlotInfo(g.Name + ".l" + layer.ToString(CultureInfo.InvariantCulture) + "." + role, index, layer, role, dtype, shape);
                        AddState(st);
                        bytes += st.ElementsPerBlock * ByteSize(dtype);
                    }
                }
            }

            info.BytesPerBlock = bytes;
            _groups.Add(info);
            index++;
        }
    }

    private void AddState(StateSlotInfo st)
    {
        if (_byName.ContainsKey(st.Name))
        {
            Fail(Diagnostics.Duplicate, null, "Name '" + st.Name + "' is declared twice");
            return;
        }

        _stateSlots.Add(st);
        var shape = new List<ShapeDim> { ShapeDim.Named(st.Role == "k" || st.Role == "v" ? "blocks" : "slots") };
        foreach (long d in st.BlockShape)
        {
            shape.Add(ShapeDim.Of(d));
        }

        AddValue(st.Name, ValueKind.State, shape, st.Dtype, st.Role == "k" || st.Role == "v" ? GraphLayout.PagedNhd : GraphLayout.RowMajor);
    }

    // ---- nodes -------------------------------------------------------------------------------------------

    private void ReadNodes()
    {
        foreach (var (n, layer) in ExpandBlocks(_spec.Nodes.Items, x => x.Layers, x => x.Location, "Node"))
        {
            if (Subst(n.Name, layer, n.Location, out string name) && Subst(n.Args, layer, n.Location, out string args))
            {
                AddNode(n, name, args, layer);
            }
        }
    }

    private sealed class Binding
    {
        public Binding(string field, string raw)
        {
            Field = field;
            Raw = raw;
        }

        public string Field { get; }

        public string Raw { get; }
    }

    private void AddNode(NodeSpec spec, string name, string args, int? layer)
    {
        var loc = spec.Location;
        var bindings = new List<Binding>();
        foreach (string part in args.Split(Semicolon, StringSplitOptions.RemoveEmptyEntries))
        {
            string p = part.Trim();
            if (p.Length == 0)
            {
                continue;
            }

            int eq = p.IndexOf('=');
            if (eq <= 0)
            {
                Fail(Diagnostics.BadBinding, loc, "Node '" + name + "': '" + p + "' is not FIELD=binding");
                continue;
            }

            // FIELD=>target splits at the first '='; keep the arrow with the binding.
                string raw = p.Substring(eq + 1).Trim();
                bindings.Add(new Binding(p.Substring(0, eq).Trim(), raw.StartsWith(">", StringComparison.Ordinal) ? "=" + raw : raw));
        }

        if (!_ops.TryGetValue(spec.Op, out var op))
        {
            Fail(Diagnostics.UnknownOp, loc, "Op '" + spec.Op + "' used by node '" + name + "' is not a member of IRoostOps (known ops: " + string.Join(", ", _ops.Keys.OrderBy(k => k, StringComparer.Ordinal)) + ")");
            foreach (var b in bindings)
            {
                int arrow = b.Raw.IndexOf("=>", StringComparison.Ordinal);
                if (arrow >= 0)
                {
                    _poisoned.Add(ParseTarget(b.Raw.Substring(arrow + 2)).Name);
                }
            }

            return;
        }

        if (op.HasSupportsFlag && Array.IndexOf(KnownDecompositions, op.Name) < 0)
        {
            Fail(Diagnostics.NoDecomposition, loc, "Node '" + name + "' uses fused op '" + op.Name + "', whose unfused form the generator does not know; use the unfused ops (the fused form comes from a graph pass)");
            return;
        }

        var fields = op.Fields.ToDictionary(f => f.Name, StringComparer.Ordinal);
        var bound = new HashSet<string>(StringComparer.Ordinal);
        var inFields = new List<string>();
        var inValues = new List<int>();
        var outFields = new List<string>();
        var outPending = new List<(string Field, TargetSpec Target, int? InPlaceOf)>();
        var attrs = new SortedDictionary<string, string>(StringComparer.Ordinal);
        bool bad = false;
        int nodeIndex = _nodes.Count;
        int? stateGroup = null;

        foreach (var b in bindings)
        {
            if (!fields.TryGetValue(b.Field, out var f))
            {
                Fail(Diagnostics.UnknownField, loc, "Op '" + op.Name + "' (node '" + name + "') has no field '" + b.Field + "' (fields: " + string.Join(", ", op.Fields.Select(x => x.Name)) + ")");
                bad = true;
                continue;
            }

            if (!bound.Add(b.Field))
            {
                Fail(Diagnostics.BadBinding, loc, "Node '" + name + "' binds field '" + b.Field + "' twice");
                bad = true;
                continue;
            }

            if (f.Kind == FieldKind.Span)
            {
                string raw = b.Raw;
                int arrow = raw.IndexOf("=>", StringComparison.Ordinal);
                string src = arrow >= 0 ? raw.Substring(0, arrow).Trim() : raw;
                (long Start, long End)? view = null;
                if (arrow < 0 && src.EndsWith("]", StringComparison.Ordinal) && src.IndexOf('[') > 0)
                {
                    int open = src.IndexOf('[');
                    string range = src.Substring(open + 1, src.Length - open - 2);
                    src = src.Substring(0, open).Trim();
                    view = ParseView(range, name, b.Field, loc);
                    if (view is null)
                    {
                        bad = true;
                        continue;
                    }
                }

                TargetSpec? target = arrow >= 0 ? ParseTarget(raw.Substring(arrow + 2)) : (TargetSpec?)null;
                int? srcId = null;
                if (src.Length > 0)
                {
                    srcId = Resolve(src, layer, name, b.Field, loc, out bool silent);
                    if (srcId is null)
                    {
                        bad = true;
                        if (!silent)
                        {
                            _failed = true;
                        }
                    }
                    else
                    {
                        if (view.HasValue && !CheckView(srcId.Value, view.Value, name, b.Field, loc))
                        {
                            bad = true;
                            continue;
                        }

                        inFields.Add(b.Field);
                        inValues.Add(srcId.Value);
                        if (view.HasValue)
                        {
                            attrs["view." + b.Field] = view.Value.Start.ToString(CultureInfo.InvariantCulture) + ":" + view.Value.End.ToString(CultureInfo.InvariantCulture);
                        }

                        if (_values[srcId.Value].Kind == ValueKind.State)
                        {
                            stateGroup = stateGroup ?? _stateSlots[srcId.Value - _weights.Count].Group;
                        }
                    }
                }

                if (target != null)
                {
                    outPending.Add((b.Field, target, srcId));
                    outFields.Add(b.Field);
                }
            }
            else
            {
                string? v = ScalarText(f, b.Raw, name, loc);
                if (v is null)
                {
                    bad = true;
                }
                else
                {
                    attrs["arg." + b.Field] = v;
                }
            }
        }

        // An unbound PagedAttention WindowLeft is 0, a window of one token on CpuOps and refused by the GPU kernel (R-022, R-025).
        if (op.Name == "PagedAttention" && !bound.Contains("WindowLeft"))
        {
            Fail(Diagnostics.UnboundWindowLeft, loc, "Node '" + name + "' (PagedAttention) does not bind WindowLeft; an unbound field is 0, not 'no window'. Bind WindowLeft=-1 for no window");
            bad = true;
        }

        // Fields the node leaves unbound: plan inputs by name, the activation dtype, the KV dtype of a bound state slot.
        foreach (var f in op.Fields)
        {
            if (bound.Contains(f.Name))
            {
                continue;
            }

            if (f.Kind == FieldKind.Span)
            {
                var plan = PlanTable.FirstOrDefault(pi => pi.Field == f.Name);
                if (plan != null)
                {
                    inFields.Add(f.Name);
                    inValues.Add(PlanValue(plan));
                }
            }
            else if (f.Kind == FieldKind.DType && f.Name == "DType")
            {
                attrs["arg.DType"] = _dtype.ToString();
            }
            else if (f.Kind == FieldKind.DType && f.Name == "KvDType" && stateGroup.HasValue)
            {
                var slot = _stateSlots.FirstOrDefault(s => s.Group == stateGroup.Value);
                if (slot != null)
                {
                    attrs["arg.KvDType"] = slot.Dtype.ToString();
                }
            }
        }

        // In-place hazards: a value overwritten earlier must not be read, and one value must not be both
        // overwritten and read through another field of the same node.
        var inPlace = outPending.Where(o => o.InPlaceOf.HasValue).ToList();
        for (int i = 0; i < inValues.Count; i++)
        {
            if (_overwrittenBy.TryGetValue(inValues[i], out string? by))
            {
                Fail(Diagnostics.InPlaceHazard, loc, "Node '" + name + "' reads '" + _values[inValues[i]].Name + "' through field '" + inFields[i] + "', which node '" + by + "' already overwrote in place");
                bad = true;
            }
        }

        foreach (var o in inPlace)
        {
            int id = o.InPlaceOf!.Value;
            if (_values[id].Kind != ValueKind.Intermediate && _values[id].Kind != ValueKind.Output)
            {
                Fail(Diagnostics.BadBinding, loc, "Node '" + name + "' field '" + o.Field + "' writes '" + _values[id].Name + "' in place, but only node outputs can be overwritten");
                bad = true;
            }
            else if (inValues.Where((v, i) => v == id && inFields[i] != o.Field).Any())
            {
                Fail(Diagnostics.InPlaceHazard, loc, "Node '" + name + "' writes '" + _values[id].Name + "' in place through '" + o.Field + "' and also reads it through another field");
                bad = true;
            }
        }

        if (bad)
        {
            foreach (var o in outPending)
            {
                _poisoned.Add(o.Target.Name);
            }

            _failed = true;
            return;
        }

        // Outputs: new SSA values; an in-place output shares the shape and dtype of its input unless given.
        var outValues = new List<int>();
        foreach (var o in outPending)
        {
            var t = o.Target;
            List<ShapeDim> shape;
            GraphDType dt = o.InPlaceOf.HasValue ? _values[o.InPlaceOf.Value].Dtype : _dtype;
            if (t.Dtype != null && (!TryDType(t.Dtype, out dt) || IsSubByte(dt)))
            {
                Fail(Diagnostics.BadShape, loc, "Node '" + name + "': output '" + t.Name + "' dtype '" + t.Dtype + "' is not a byte-sized element type name");
                _poisoned.Add(t.Name);
                continue;
            }

            if (t.Shape != null)
            {
                if (!ParseShape(t.Shape, true, loc, "Node '" + name + "' output '" + t.Name + "'", out shape))
                {
                    _poisoned.Add(t.Name);
                    continue;
                }
            }
            else if (o.InPlaceOf.HasValue)
            {
                shape = _values[o.InPlaceOf.Value].Shape.ToList();
            }
            else
            {
                Fail(Diagnostics.BadShape, loc, "Node '" + name + "': output '" + t.Name + "' needs a shape, for example " + t.Name + "[tokens,hidden]");
                _poisoned.Add(t.Name);
                continue;
            }

            if (!IsIdentifierPath(t.Name) || _byName.ContainsKey(t.Name))
            {
                Fail(Diagnostics.Duplicate, loc, "Name '" + t.Name + "' (output of node '" + name + "') is not a name or is declared twice");
                _poisoned.Add(t.Name);
                continue;
            }

            outValues.Add(AddValue(t.Name, ValueKind.Intermediate, shape, dt, GraphLayout.RowMajor));
        }

        if (outValues.Count != outPending.Count)
        {
            _failed = true;
            return;
        }

        foreach (var o in inPlace)
        {
            _overwrittenBy[o.InPlaceOf!.Value] = name;
        }

        foreach (int v in inValues)
        {
            _read.Add(v);
        }

        attrs["in"] = string.Join(",", inFields);
        attrs["out"] = string.Join(",", outFields);
        attrs["node"] = name;
        if (layer.HasValue)
        {
            attrs["layer"] = layer.Value.ToString(CultureInfo.InvariantCulture);
        }

        _usedOps.Add(op.Name);
        _nodes.Add(new NodeRec(nodeIndex, op.Name, inValues, outValues, attrs));
    }

    /// <summary>Parses the <c>start:end</c> of a column view: two integer expressions over the dims, 0 &lt;= start &lt; end.</summary>
    private (long Start, long End)? ParseView(string range, string node, string field, LocationInfo? loc)
    {
        int colon = range.IndexOf(':');
        if (colon < 0)
        {
            Fail(Diagnostics.BadBinding, loc, "Node '" + node + "' field '" + field + "': a view is written value[start:end] (columns of the last dimension), not '[" + range + "]'");
            return null;
        }

        var dims = _dims.ToDictionary(k => k.Key, k => k.Value);
        long end = 0;
        string? error = null;
        if (!DimExpression.TryEvaluate(range.Substring(0, colon), dims, out long start, out error)
            || !DimExpression.TryEvaluate(range.Substring(colon + 1), dims, out end, out error))
        {
            Fail(Diagnostics.BadBinding, loc, "Node '" + node + "' field '" + field + "' view [" + range + "]: " + error + " (a view's bounds are integer expressions over the dims)");
            return null;
        }

        if (start < 0 || end <= start)
        {
            Fail(Diagnostics.BadBinding, loc, "Node '" + node + "' field '" + field + "': view [" + start.ToString(CultureInfo.InvariantCulture) + ":" + end.ToString(CultureInfo.InvariantCulture) + "] needs 0 <= start < end");
            return null;
        }

        return (start, end);
    }

    /// <summary>A view is a column slice of a token-major activation (a node output): the value must have rank 2 or more and a last dimension of at least <c>end</c>.</summary>
    private bool CheckView(int valueId, (long Start, long End) view, string node, string field, LocationInfo? loc)
    {
        var v = _values[valueId];
        if (v.Kind != ValueKind.Intermediate && v.Kind != ValueKind.Output)
        {
            Fail(Diagnostics.BadBinding, loc, "Node '" + node + "' field '" + field + "' takes a view of '" + v.Name + "', but only node outputs can be viewed");
            return false;
        }

        if (v.Shape.Count < 2 || v.Shape[v.Shape.Count - 1].IsSymbolic)
        {
            Fail(Diagnostics.BadBinding, loc, "Node '" + node + "' field '" + field + "' takes a view of '" + v.Name + "', which needs a fixed last dimension and a leading token dimension");
            return false;
        }

        long width = v.Shape[v.Shape.Count - 1].Size;
        if (view.End > width)
        {
            Fail(Diagnostics.BadBinding, loc, "Node '" + node + "' field '" + field + "': view end " + view.End.ToString(CultureInfo.InvariantCulture) + " is beyond the " + width.ToString(CultureInfo.InvariantCulture) + " columns of '" + v.Name + "'");
            return false;
        }

        return true;
    }

    private sealed class TargetSpec
    {
        public TargetSpec(string name, string? shape, string? dtype)
        {
            Name = name;
            Shape = shape;
            Dtype = dtype;
        }

        public string Name { get; }

        public string? Shape { get; }

        public string? Dtype { get; }
    }

    /// <summary>Parses <c>name[shape]:dtype</c>; the shape and dtype are optional.</summary>
    private static TargetSpec ParseTarget(string text)
    {
        text = text.Trim();
        string? shape = null;
        string? dtype = null;
        int colon = text.LastIndexOf(':');
        int close = text.LastIndexOf(']');
        if (colon > close && colon >= 0)
        {
            dtype = text.Substring(colon + 1).Trim();
            text = text.Substring(0, colon).Trim();
        }

        int open = text.IndexOf('[');
        if (open >= 0 && text.EndsWith("]", StringComparison.Ordinal))
        {
            shape = text.Substring(open + 1, text.Length - open - 2);
            text = text.Substring(0, open).Trim();
        }

        return new TargetSpec(text, shape, dtype);
    }

    private int PlanValue(PlanInputInfo plan)
    {
        string name = "plan." + plan.Field;
        if (_planValues.TryGetValue(name, out int id))
        {
            return id;
        }

        var shape = new List<ShapeDim> { ShapeDim.Named(plan.Count) };
        id = AddValue(name, ValueKind.Input, shape, plan.Dtype, GraphLayout.RowMajor);
        _planValues[name] = id;
        return id;
    }

    /// <summary>Resolves a span source: n: (node output), w: (weight slot), s: (state slot), p: (plan input), or a bare name.</summary>
    private int? Resolve(string src, int? layer, string node, string field, LocationInfo? loc, out bool silent)
    {
        silent = false;
        char kind = '\0';
        string name = src;
        if (src.Length > 2 && src[1] == ':' && "nwsp".IndexOf(src[0]) >= 0)
        {
            kind = src[0];
            name = src.Substring(2);
        }

        if (_poisoned.Contains(name) && (kind == '\0' || kind == 'n'))
        {
            silent = true;
            return null;
        }

        if (kind == 'p')
        {
            var plan = PlanTable.FirstOrDefault(pi => pi.Field == name);
            if (plan != null)
            {
                return PlanValue(plan);
            }
        }
        else if (kind == 's')
        {
            string abs = name;
            if (name.Count(c => c == '.') == 1 && layer.HasValue)
            {
                int dot = name.IndexOf('.');
                abs = name.Substring(0, dot) + ".l" + layer.Value.ToString(CultureInfo.InvariantCulture) + name.Substring(dot);
            }

            if (_byName.TryGetValue(abs, out int sid) && _values[sid].Kind == ValueKind.State)
            {
                return sid;
            }

            name = abs;
        }
        else if (_byName.TryGetValue(name, out int id))
        {
            var k = _values[id].Kind;
            bool ok = kind == '\0'
                || (kind == 'n' && (k == ValueKind.Intermediate || k == ValueKind.Output))
                || (kind == 'w' && k == ValueKind.Weight);
            if (ok && k != ValueKind.Input)
            {
                return id;
            }
        }

        Fail(Diagnostics.UnknownValue, loc, "Node '" + node + "' field '" + field + "' reads '" + src + "', which is not an earlier node output, weight slot or state slot" + (kind == 's' ? " (looked for '" + name + "')" : string.Empty));
        silent = true;
        return null;
    }

    private string? ScalarText(FieldInfo f, string raw, string node, LocationInfo? loc)
    {
        raw = raw.Trim();
        switch (f.Kind)
        {
            case FieldKind.Int:
            case FieldKind.Long:
                return FoldExpression(raw, loc, "Node '" + node + "' field '" + f.Name + "'");
            case FieldKind.Float:
                if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) && !double.IsNaN(d) && !double.IsInfinity(d))
                {
                    return d.ToString("R", CultureInfo.InvariantCulture);
                }

                break;
            case FieldKind.Bool:
                if (raw == "true" || raw == "false")
                {
                    return raw;
                }

                break;
            case FieldKind.Enum:
            case FieldKind.DType:
                if (Array.IndexOf(f.EnumMembers.Items, raw) >= 0)
                {
                    return raw;
                }

                Fail(Diagnostics.BadBinding, loc, "Node '" + node + "' field '" + f.Name + "': '" + raw + "' is not a member of " + f.TypeName + " (" + string.Join(", ", f.EnumMembers.Items) + ")");
                return null;
        }

        Fail(Diagnostics.BadBinding, loc, "Node '" + node + "' field '" + f.Name + "' is " + f.Kind + "; '" + raw + "' is not a valid value (buffers are bound as FIELD=value or FIELD=>name[shape])");
        return null;
    }

    /// <summary>Checks an integer expression and returns it with the dims folded in; step symbols stay as names.</summary>
    private string? FoldExpression(string text, LocationInfo? loc, string what)
    {
        var symbols = _dims.ToDictionary(k => k.Key, k => k.Value);
        foreach (string s in RuntimeSymbols)
        {
            symbols[s] = 1;
        }

        if (!DimExpression.TryEvaluate(text, symbols, out _, out string? err))
        {
            Fail(Diagnostics.BadShape, loc, what + ": " + err);
            return null;
        }

        var sb = new StringBuilder();
        int i = 0;
        while (i < text.Length)
        {
            if (DimExpression.IsNameStart(text[i]))
            {
                int s = i;
                while (i < text.Length && DimExpression.IsNamePart(text[i]))
                {
                    i++;
                }

                string n = text.Substring(s, i - s);
                if (_dims.TryGetValue(n, out long v))
                {
                    sb.Append('(').Append(v.ToString(CultureInfo.InvariantCulture)).Append(')');
                }
                else
                {
                    sb.Append(n);
                }
            }
            else
            {
                sb.Append(text[i]);
                i++;
            }
        }

        return sb.ToString().Replace(" ", string.Empty);
    }

    private void ReadOutputs()
    {
        if (_spec.Outputs.Length == 0)
        {
            Fail(Diagnostics.BadDefinition, _spec.Location, "Model '" + _spec.ModelName + "' declares no [Output]");
        }

        foreach (var o in _spec.Outputs)
        {
            if (_poisoned.Contains(o.Name))
            {
                continue;
            }

            if (_byName.TryGetValue(o.Name, out int id) && (_values[id].Kind == ValueKind.Intermediate || _values[id].Kind == ValueKind.Output))
            {
                _values[id].Kind = ValueKind.Output;
            }
            else
            {
                Fail(Diagnostics.UnknownValue, o.Location, "Output '" + o.Name + "' is not a node output");
            }
        }
    }

    // ---- helpers -----------------------------------------------------------------------------------------

    private int AddValue(string name, ValueKind kind, IReadOnlyList<ShapeDim> shape, GraphDType dtype, GraphLayout layout)
    {
        int id = _values.Count;
        _values.Add(new ValueRec(id, name, kind, shape.ToList(), dtype, layout));
        _byName[name] = id;
        return id;
    }

    private bool Subst(string text, int? layer, LocationInfo? loc, out string result)
    {
        if (layer is null)
        {
            if (text.IndexOf('{') >= 0)
            {
                Fail(Diagnostics.BadLayers, loc, "'" + text + "' uses a layer placeholder but has no Layers");
                result = text;
                return false;
            }

            result = text;
            return true;
        }

        if (!Layers.TrySubstitute(text, layer.Value, out result, out string? err))
        {
            Fail(Diagnostics.BadLayers, loc, err!);
            return false;
        }

        return true;
    }

    /// <summary>
    /// Items without Layers appear once, in place. A run of consecutive items with Layers is a block: it is emitted
    /// layer by layer (ascending), and within a layer in declaration order, so a layer's nodes stay together.
    /// </summary>
    private List<(T Item, int? Layer)> ExpandBlocks<T>(T[] items, Func<T, string?> layersOf, Func<T, LocationInfo?> locOf, string what)
    {
        var result = new List<(T, int?)>();
        int i = 0;
        while (i < items.Length)
        {
            if (layersOf(items[i]) is null)
            {
                result.Add((items[i], null));
                i++;
                continue;
            }

            int j = i;
            var sets = new List<int[]>();
            while (j < items.Length && layersOf(items[j]) != null)
            {
                if (!Layers.TryParse(layersOf(items[j])!, out int[] set, out string? err))
                {
                    Fail(Diagnostics.BadLayers, locOf(items[j]), what + ": " + err);
                    set = Array.Empty<int>();
                }

                sets.Add(set);
                j++;
            }

            var union = new SortedSet<int>(sets.SelectMany(s => s));
            foreach (int layer in union)
            {
                for (int k = i; k < j; k++)
                {
                    if (Array.IndexOf(sets[k - i], layer) >= 0)
                    {
                        result.Add((items[k], layer));
                    }
                }
            }

            i = j;
        }

        return result;
    }

    private bool ParseShape(string text, bool arenaValue, LocationInfo? loc, string what, out List<ShapeDim> shape)
    {
        shape = new List<ShapeDim>();
        if (text.Trim().Length == 0)
        {
            return true;
        }

        var symbols = _dims.ToDictionary(k => k.Key, k => k.Value);
        string[] terms = text.Split(',');
        for (int i = 0; i < terms.Length; i++)
        {
            string term = terms[i].Trim();
            if (Array.IndexOf(RuntimeSymbols, term) >= 0)
            {
                if (!arenaValue)
                {
                    Fail(Diagnostics.BadShape, loc, what + ": '" + term + "' is a step symbol; this shape must be constant");
                    return false;
                }

                if (i != 0 || (term != "tokens" && term != "selected"))
                {
                    Fail(Diagnostics.BadShape, loc, what + ": step symbol '" + term + "' is only allowed as the first dimension, and only tokens or selected (activations are token-major)");
                    return false;
                }

                shape.Add(ShapeDim.Named(term));
                continue;
            }

            if (!DimExpression.TryEvaluate(term, symbols, out long v, out string? err) || v < 0)
            {
                Fail(Diagnostics.BadShape, loc, what + ": " + (err ?? "dimension '" + term + "' is negative"));
                return false;
            }

            shape.Add(ShapeDim.Of(v));
        }

        return true;
    }

    private bool TryInt(string text, bool allowEmpty, LocationInfo? loc, string what, out long value)
    {
        value = 0;
        if (text.Trim().Length == 0)
        {
            if (allowEmpty)
            {
                return true;
            }

            Fail(Diagnostics.BadStateGroup, loc, "State group " + what + " is required");
            return false;
        }

        if (!DimExpression.TryEvaluate(text, _dims.ToDictionary(k => k.Key, k => k.Value), out value, out string? err) || (!allowEmpty && value <= 0) || value < 0)
        {
            Fail(Diagnostics.BadStateGroup, loc, "State group " + what + ": " + (err ?? "'" + text + "' must be positive"));
            return false;
        }

        return true;
    }

    /// <summary>Parses an element type name (F32, BF16, ...); numbers and unknown names are rejected.</summary>
    internal static bool TryDType(string name, out GraphDType dtype)
    {
        dtype = GraphDType.Undefined;
        return Enum.IsDefined(typeof(GraphDType), name) && name != nameof(GraphDType.Undefined) && Enum.TryParse(name, out dtype);
    }

    internal static bool IsSubByte(GraphDType t) => t == GraphDType.I4 || t == GraphDType.U4 || t == GraphDType.F4E2M1;

    internal static int ByteSize(GraphDType t)
    {
        switch (t)
        {
            case GraphDType.F64:
            case GraphDType.I64:
            case GraphDType.U64:
                return 8;
            case GraphDType.F32:
            case GraphDType.I32:
            case GraphDType.U32:
                return 4;
            case GraphDType.F16:
            case GraphDType.BF16:
            case GraphDType.I16:
                return 2;
            default:
                return 1;
        }
    }

    private static bool IsIdentifier(string s) => s.Length > 0 && DimExpression.IsNameStart(s[0]) && s.All(DimExpression.IsNamePart);

    private static bool IsIdentifierPath(string s) => s.Length > 0 && DimExpression.IsNameStart(s[0]) && s.All(c => DimExpression.IsNamePart(c) || c == '.');

    private BuiltModel Finish()
    {
        var values = _values.Select(v => new GraphValue(v.Id, v.Name, v.Kind, v.Shape, v.Dtype, v.Layout, Sharding.Replicated)).ToList();
        var nodes = _nodes.Select(n => new OpNode(
            n.Id,
            n.Op,
            n.Inputs,
            n.Outputs,
            n.Attrs.ToDictionary(k => k.Key, k => k.Value, StringComparer.Ordinal),
            new[] { n.Id },
            null)).ToList();
        var graph = new OpGraph(_spec.ModelName, values, nodes, true);
        var plan = PlanTable.Where(pi => _planValues.ContainsKey("plan." + pi.Field)).ToList();
        var outputs = _values.Where(v => v.Kind == ValueKind.Output).Select(v => v.Id).ToList();
        return new BuiltModel(graph, _weights, _stateSlots, _groups, plan, outputs, _usedOps.OrderBy(o => o, StringComparer.Ordinal).ToList(), _dtype);
    }

    private sealed class ValueRec
    {
        public ValueRec(int id, string name, ValueKind kind, List<ShapeDim> shape, GraphDType dtype, GraphLayout layout)
        {
            Id = id;
            Name = name;
            Kind = kind;
            Shape = shape;
            Dtype = dtype;
            Layout = layout;
        }

        public int Id { get; }

        public string Name { get; }

        public ValueKind Kind { get; set; }

        public List<ShapeDim> Shape { get; }

        public GraphDType Dtype { get; }

        public GraphLayout Layout { get; }
    }

    private sealed class NodeRec
    {
        public NodeRec(int id, string op, List<int> inputs, List<int> outputs, SortedDictionary<string, string> attrs)
        {
            Id = id;
            Op = op;
            Inputs = inputs;
            Outputs = outputs;
            Attrs = attrs;
        }

        public int Id { get; }

        public string Op { get; }

        public List<int> Inputs { get; }

        public List<int> Outputs { get; }

        public SortedDictionary<string, string> Attrs { get; }
    }
}

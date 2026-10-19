using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.ModelCompiler.Checking;
using Starlings.Roost.ModelCompiler.Tests.Models;
using Starlings.Roost.Ops;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>
/// The numeric hybrid-graph check on <see cref="CpuOps"/> (R-020's deferred item): a Qwen3.5-shaped hybrid with a fused
/// q/k/v projection, a convolution over all its channels and q, k and v as strided column views of the convolution
/// output, running its generated forward and its graph through the same backend. Three independent checks: the
/// recurrent block against a plain double transcription, chunked and token-by-token runs against one prefill (state
/// carries across steps), and the graph evaluator against the generated forward.
/// </summary>
public sealed unsafe class FusedHybridModelTests : IDisposable
{
    private const int Hidden = 16;
    private const int KHeads = 2;
    private const int VHeads = 4;
    private const int KDim = 4;
    private const int VDim = 4;
    private const int KeyDim = KHeads * KDim;
    private const int ValueDim = VHeads * VDim;
    private const int ConvDim = (2 * KeyDim) + ValueDim;
    private const int Width = 4;
    private const int Vocab = 32;
    private const int BlockSize = 4;

    private static readonly int[] Ids = [3, 17, 29, 0, 11, 5, 30];
    private static readonly FusedHybridModel.ForwardLayout Layout = new(maxTokens: 8, maxSequences: 1, maxKvBlocks: 8);

    private readonly Native _native = new();
    private readonly DeviceSpan[] _weights;
    private readonly double[][] _weightData;
    private readonly DeviceSpan _arena;
    private readonly DeviceSpan _planRegion;
    private DeviceSpan[] _state;

    public FusedHybridModelTests()
    {
        var rng = new Random(11);
        _weights = new DeviceSpan[FusedHybridModel.WeightSlots.Length];
        _weightData = new double[_weights.Length][];
        for (int i = 0; i < _weights.Length; i++)
        {
            var slot = FusedHybridModel.WeightSlots[i];
            _weights[i] = _native.Alloc(slot.Elements, slot.Dtype);
            _weightData[i] = new double[slot.Elements];
            bool norm = slot.Name.EndsWith("norm", StringComparison.Ordinal);
            bool gate = slot.Name.Contains("in_proj_a", StringComparison.Ordinal);
            for (int j = 0; j < slot.Elements; j++)
            {
                double v = norm ? 1 + ((rng.NextDouble() - 0.5) * 0.2) : (rng.NextDouble() - 0.5) * (gate ? 0.4 : 0.8);
                if (gate)
                {
                    v -= 0.3;
                }

                _weightData[i][j] = (float)v;
                ((float*)_weights[i].Address)[j] = (float)v;
            }
        }

        _arena = _native.AllocBytes(Layout.ArenaBytes);
        _planRegion = _native.AllocBytes(Layout.PlanBytes);
        _state = NewState();
    }

    public void Dispose() => _native.Dispose();

    private DeviceSpan[] NewState()
    {
        var s = new DeviceSpan[FusedHybridModel.StateSlots.Length];
        for (int i = 0; i < s.Length; i++)
        {
            var slot = FusedHybridModel.StateSlots[i];
            long units = slot.Role is "k" or "v" ? 8 : 2;
            s[i] = _native.Alloc(units * slot.ElementsPerBlock, slot.Dtype);
        }

        return s;
    }

    private static void SetInt(DeviceSpan span, int index, long value)
    {
        switch (span.Dtype)
        {
            case DType.I32:
                ((int*)span.Address)[index] = (int)value;
                break;
            case DType.I64:
                ((long*)span.Address)[index] = value;
                break;
            default:
                throw new NotSupportedException(span.Dtype.ToString());
        }
    }

    /// <summary>One step of the single sequence: <paramref name="ids"/> are tokens at positions start.. of its context.</summary>
    private (float[] Logits, float[] Mix) Step(int[] ids, int start)
    {
        int n = ids.Length;
        int context = start + n;
        int pages = (context + BlockSize - 1) / BlockSize;
        var plan = FusedHybridModel.PlanInputs.Bind(_planRegion, Layout, n, n, 1, pages);
        for (int t = 0; t < n; t++)
        {
            SetInt(plan.Ids, t, ids[t]);
            SetInt(plan.Positions, t, start + t);
            SetInt(plan.SlotMapping, t, start + t);
            SetInt(plan.Rows, t, t);
        }

        SetInt(plan.QIndptr, 0, 0);
        SetInt(plan.QIndptr, 1, n);
        SetInt(plan.KvIndptr, 0, 0);
        SetInt(plan.KvIndptr, 1, pages);
        for (int p = 0; p < pages; p++)
        {
            SetInt(plan.KvIndices, p, p);
        }

        SetInt(plan.KvLastPageLength, 0, context - ((pages - 1) * BlockSize));
        SetInt(plan.StateSlots, 0, 1);
        SetInt(plan.CuSeqlens, 0, 0);
        SetInt(plan.CuSeqlens, 1, n);

        var ctx = new FusedHybridModel.StepContext(Layout, _arena, _weights, _state, plan, n, n, 1, pages);
        FusedHybridModel.Forward<CpuOps>(in ctx);
        var logits = FusedHybridModel.Output_logits(in ctx);
        var mix = FusedHybridModel.Output_l0_mix(in ctx);
        return (
            new ReadOnlySpan<float>((void*)logits.Address, n * Vocab).ToArray(),
            new ReadOnlySpan<float>((void*)mix.Address, n * Hidden).ToArray());
    }

    private double[] Weight(string name) => _weightData[Array.FindIndex(FusedHybridModel.WeightSlots, s => s.Name == name)];

    [Fact]
    public void ViewsAreBoundAsColumnSlicesOfTheConvolutionOutput()
    {
        var g = FusedHybridModel.BuildGraph();
        var deltas = g.Nodes.Where(n => n.Op == "GatedDeltaRule").ToArray();
        Assert.Equal(3, deltas.Length);
        foreach (var n in deltas)
        {
            Assert.Equal("0:8", n.Attributes["view.Q"]);
            Assert.Equal("8:16", n.Attributes["view.K"]);
            Assert.Equal("16:32", n.Attributes["view.V"]);
            Assert.Equal("(32)", n.Attributes["arg.QStride"]);
            string[] fields = n.Attributes["in"].Split(',');
            int q = Array.IndexOf(fields, "Q");
            Assert.Equal(n.Inputs[q], n.Inputs[Array.IndexOf(fields, "K")]);
            Assert.Equal(n.Inputs[q], n.Inputs[Array.IndexOf(fields, "V")]);
            // The parent is the convolution output, never a copy.
            Assert.Equal(ConvDim, g.Values[n.Inputs[q]].Shape[1].Size);
            Assert.Equal("CausalConv1d", g.Nodes.Single(c => c.Outputs.Contains(n.Inputs[q])).Op);
        }

        Assert.DoesNotContain(g.Nodes, n => n.Op != "GatedDeltaRule" && n.Attributes.Keys.Any(k => k.StartsWith("view.", StringComparison.Ordinal)));
    }

    [Fact]
    public void TheRecurrentBlockMatchesAPlainTranscription()
    {
        var (_, mix) = Step(Ids, 0);

        // Independent transcription of layer 0's mixer in double: norm, fused projection, depthwise causal conv with SiLU,
        // split by head counts, q/k L2 norm, the delta rule with the key head h / (VHeads / KHeads) of value head h, out_proj.
        int n = Ids.Length;
        double[] embed = Weight("embed");
        double[] norm = Weight("l0.norm");
        double[] qkvW = Weight("l0.in_proj_qkv");
        double[] convW = Weight("l0.conv");
        double[] aW = Weight("l0.in_proj_a");
        double[] bW = Weight("l0.in_proj_b");
        double[] outW = Weight("l0.out_proj");

        var x = new double[n][];
        for (int t = 0; t < n; t++)
        {
            double[] e = embed.AsSpan(Ids[t] * Hidden, Hidden).ToArray();
            double inv = 1 / Math.Sqrt(e.Sum(v => v * v) / Hidden + 1e-6);
            x[t] = e.Select((v, i) => v * inv * norm[i]).ToArray();
        }

        double[] Project(double[] w, int rows, double[] xt) =>
            Enumerable.Range(0, rows).Select(r => Enumerable.Range(0, Hidden).Sum(k => xt[k] * w[(r * Hidden) + k])).ToArray();

        var qkv = x.Select(xt => Project(qkvW, ConvDim, xt)).ToArray();
        var conv = new double[n][];
        for (int t = 0; t < n; t++)
        {
            conv[t] = new double[ConvDim];
            for (int c = 0; c < ConvDim; c++)
            {
                double y = 0;
                for (int j = 0; j < Width; j++)
                {
                    int src = t - (Width - 1) + j;
                    y += convW[(c * Width) + j] * (src >= 0 ? qkv[src][c] : 0);
                }

                conv[t][c] = y / (1 + Math.Exp(-y));
            }
        }

        var state = new double[VHeads][,];
        for (int h = 0; h < VHeads; h++)
        {
            state[h] = new double[KDim, VDim];
        }

        var expected = new double[n * Hidden];
        for (int t = 0; t < n; t++)
        {
            double[] a = Project(aW, VHeads, x[t]);
            double[] b = Project(bW, VHeads, x[t]);
            var o = new double[ValueDim];
            for (int h = 0; h < VHeads; h++)
            {
                int kh = h / (VHeads / KHeads);
                double[] q = conv[t].AsSpan(kh * KDim, KDim).ToArray();
                double[] k = conv[t].AsSpan(KeyDim + (kh * KDim), KDim).ToArray();
                double qn = 1 / Math.Sqrt(q.Sum(v => v * v) + 1e-6);
                double kn = 1 / Math.Sqrt(k.Sum(v => v * v) + 1e-6);
                for (int i = 0; i < KDim; i++)
                {
                    q[i] *= qn * 0.5;
                    k[i] *= kn;
                }

                double decay = Math.Exp(a[h]);
                var d = new double[VDim];
                for (int j = 0; j < VDim; j++)
                {
                    double kv = 0;
                    for (int i = 0; i < KDim; i++)
                    {
                        state[h][i, j] *= decay;
                        kv += state[h][i, j] * k[i];
                    }

                    d[j] = b[h] * (conv[t][(2 * KeyDim) + (h * VDim) + j] - kv);
                }

                for (int j = 0; j < VDim; j++)
                {
                    double acc = 0;
                    for (int i = 0; i < KDim; i++)
                    {
                        state[h][i, j] += k[i] * d[j];
                        acc += state[h][i, j] * q[i];
                    }

                    o[(h * VDim) + j] = acc;
                }
            }

            for (int r = 0; r < Hidden; r++)
            {
                expected[(t * Hidden) + r] = Enumerable.Range(0, ValueDim).Sum(k => o[k] * outW[(r * ValueDim) + k]);
            }
        }

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - mix[i]) <= 1e-4 + (1e-4 * Math.Abs(expected[i])), $"element {i}: expected {expected[i]}, got {mix[i]}");
        }
    }

    [Fact]
    public void ChunkedAndTokenByTokenRunsEqualOnePrefill()
    {
        var (whole, wholeMix) = Step(Ids, 0);
        Assert.All(whole, v => Assert.True(float.IsFinite(v)));

        _state = NewState();
        var (l1, m1) = Step(Ids[..3], 0);
        var (l2, m2) = Step(Ids[3..], 3);
        AssertClose(whole, [.. l1, .. l2]);
        AssertClose(wholeMix, [.. m1, .. m2]);

        _state = NewState();
        var logits = new List<float>();
        for (int t = 0; t < Ids.Length; t++)
        {
            logits.AddRange(Step(Ids[t..(t + 1)], t).Logits);
        }

        AssertClose(whole, [.. logits]);
    }

    [Fact]
    public void TheGraphEvaluatorAndTheGeneratedForwardAgreeAndTheCheckerWatchesTheViews()
    {
        var reference = FusedHybridModel.BuildGraph();
        int n = Ids.Length;
        var symbols = new Dictionary<string, long> { ["tokens"] = n, ["selected"] = n, ["seqs"] = 1, ["kv_blocks"] = 2, ["slots"] = 2, ["blocks"] = 8 };
        var inputs = new Dictionary<int, double[]>();
        foreach (var v in reference.Values)
        {
            if (v.Kind == ValueKind.Weight)
            {
                inputs[v.Id] = Weight(v.Name);
            }
        }

        void Plan(string field, params double[] values) => inputs[reference.Values.Single(v => v.Name == "plan." + field).Id] = values;
        Plan("Ids", Ids.Select(i => (double)i).ToArray());
        Plan("Positions", Enumerable.Range(0, n).Select(i => (double)i).ToArray());
        Plan("SlotMapping", Enumerable.Range(0, n).Select(i => (double)i).ToArray());
        Plan("Rows", Enumerable.Range(0, n).Select(i => (double)i).ToArray());
        Plan("QIndptr", 0, n);
        Plan("KvIndptr", 0, 2);
        Plan("KvIndices", 0, 1);
        Plan("KvLastPageLength", n - BlockSize);
        Plan("StateSlots", 1);
        Plan("CuSeqlens", 0, n);

        var checker = new GraphChecker();
        var tolerances = new TestTolerances(1e-6, 1e-5);
        var evaluator = new GraphEvaluator<CpuOps>(symbols);
        var target = new GraphTarget(0, 1, 1, 1, new Dictionary<string, IReadOnlyList<string>>());

        // The identity pass: every node, views included, checked op by op.
        var identity = checker.CheckPass(new IdentityPass(), reference, target, evaluator, inputs, tolerances);
        Assert.True(identity.Passed);
        Assert.Equal(reference.Nodes.Count, identity.NodesChecked);

        // The evaluator's logits are the generated forward's logits.
        var values = evaluator.Evaluate(reference, inputs);
        var logitsId = reference.Nodes[^1].Outputs[0];
        var (forward, _) = Step(Ids, 0);
        AssertClose(forward, values[logitsId].Select(v => (float)v).ToArray());

        // A pass that moves K's view by one column changes the numbers, and the checker names the node and what it replaced.
        var broken = checker.CheckPass(new ShiftKViewPass(), reference, target, evaluator, inputs, tolerances);
        Assert.False(broken.Passed);
        Assert.Contains(broken.Mismatches, m => reference.Nodes[m.ReferenceNodes[0]].Op == "GatedDeltaRule");
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 + (1e-4 * Math.Abs(expected[i])), $"element {i}: {expected[i]} vs {actual[i]}");
        }
    }

    /// <summary>A deliberately wrong pass: K of every GatedDeltaRule is read one column to the right.</summary>
    private sealed class ShiftKViewPass : IGraphPass
    {
        public string Name => "shift-k-view";

        public PassKind Kind => PassKind.Fusion;

        public OpGraph Run(OpGraph graph, GraphTarget target) =>
            new(
                graph.Name + "+" + Name,
                graph.Values,
                graph.Nodes.Select(n =>
                {
                    if (n.Op != "GatedDeltaRule")
                    {
                        return new OpNode(n.Id, n.Op, n.Inputs, n.Outputs, n.Attributes, [n.Id], n.Variant);
                    }

                    var attrs = new Dictionary<string, string>(n.Attributes) { ["view.K"] = "9:17" };
                    return new OpNode(n.Id, n.Op, n.Inputs, n.Outputs, attrs, [n.Id], n.Variant);
                }).ToList(),
                false);
    }
}

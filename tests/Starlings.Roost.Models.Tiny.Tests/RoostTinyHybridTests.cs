using System.Globalization;
using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.ModelCompiler.Tests;
using Starlings.Roost.Ops;
using Starlings.Roost.State;

namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>
/// roost-tiny-hybrid (Qwen3.5-shaped) structure: weight and state slots, the fused linear-attention projection, plan inputs,
/// graph pattern, the q/k/v views and call order. The numeric half is in <see cref="RoostTinyHybridNumericTests"/>.
/// </summary>
public class RoostTinyHybridTests
{
    private const nint WeightBase = 0x1000_0000;
    private const nint StateBase = 0x2000_0000;
    private const nint ArenaBase = 0x4000_0000;
    private const nint PlanBase = 0x6000_0000;

    private static readonly TinyConfig C = TinyConfig.Hybrid;
    private static readonly RoostTinyHybrid.ForwardLayout Layout = new(maxTokens: 4096, maxSequences: 16, maxKvBlocks: 512);

    private static int[] LinearLayers => Enumerable.Range(0, C.Layers).Where(l => C.LayerType(l) == TinyLayerType.LinearAttention).ToArray();

    private static int[] AttentionLayers => Enumerable.Range(0, C.Layers).Where(l => C.LayerType(l) == TinyLayerType.FullAttention).ToArray();

    [Fact]
    public void LayerPatternIsQwen35sFullAttentionEveryFourthLayer()
    {
        Assert.Equal([0, 1, 2, 4, 5, 6], LinearLayers);
        Assert.Equal([3, 7], AttentionLayers);
        // Qwen3.5-0.8B repeats the same rule to 24 layers: 6 attention and 18 recurrent (decision 15).
        var real = C with { Layers = 24 };
        Assert.Equal(6, Enumerable.Range(0, 24).Count(l => real.LayerType(l) == TinyLayerType.FullAttention));
        Assert.Equal(18, Enumerable.Range(0, 24).Count(l => real.LayerType(l) == TinyLayerType.LinearAttention));
        Assert.Equal(RoostTinyHybrid.ModelName, C.Name);
    }

    [Fact]
    public void LinearAttentionProjectionIsOneFusedSlotAsInTheCheckpoint()
    {
        var slots = RoostTinyHybrid.WeightSlots.ToDictionary(s => s.Name);
        int convDim = C.LinearConvDim;
        Assert.Equal((2 * 4 * 16) + (4 * 16), convDim);

        foreach (int l in LinearLayers)
        {
            // One in_proj_qkv of [q | k | v, hidden] and a convolution over all of those channels; there are no separate q, k or v projections.
            Assert.Equal($"{convDim},{C.Hidden}", slots[$"l{l}.in_proj_qkv"].Shape);
            Assert.Equal($"{convDim},{C.ConvWidth}", slots[$"l{l}.conv"].Shape);
            Assert.Equal($"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight", slots[$"l{l}.in_proj_qkv"].Source);
            Assert.Equal($"model.language_model.layers.{l}.linear_attn.conv1d.weight", slots[$"l{l}.conv"].Source);
            Assert.Equal($"{C.LinearValueHeads},{C.Hidden}", slots[$"l{l}.in_proj_a"].Shape);
            Assert.Equal($"{C.LinearValueHeads},{C.Hidden}", slots[$"l{l}.in_proj_b"].Shape);
            Assert.Equal($"{C.Hidden},{C.LinearValueHeads * C.LinearValueDim}", slots[$"l{l}.out_proj"].Shape);
            Assert.Equal($"{C.LinearValueHeads * C.LinearValueDim},{C.Hidden}", slots[$"l{l}.in_proj_z"].Shape);
            Assert.Equal($"{C.LinearValueHeads}", slots[$"l{l}.A_log"].Shape);
            Assert.Equal($"{C.LinearValueHeads}", slots[$"l{l}.dt_bias"].Shape);
            Assert.Equal($"{C.LinearValueDim}", slots[$"l{l}.linear_norm"].Shape);
            Assert.Equal($"model.language_model.layers.{l}.linear_attn.A_log", slots[$"l{l}.A_log"].Source);
            Assert.Equal($"model.language_model.layers.{l}.linear_attn.norm.weight", slots[$"l{l}.linear_norm"].Source);
            Assert.DoesNotContain($"l{l}.q_proj", slots.Keys);
            Assert.DoesNotContain($"l{l}.lin_q_proj", slots.Keys);
        }

        foreach (int l in AttentionLayers)
        {
            Assert.DoesNotContain($"l{l}.in_proj_qkv", slots.Keys);
            Assert.Equal($"{C.Heads * C.HeadDim},{C.Hidden}", slots[$"l{l}.q_proj"].Shape);
            Assert.Equal($"{C.KvHeads * C.HeadDim},{C.Hidden}", slots[$"l{l}.k_proj"].Shape);
            // The output gate is its own projection (the loader splits the checkpoint's interleaved q_proj, as in R-035).
            Assert.Equal($"{C.Heads * C.HeadDim},{C.Hidden}", slots[$"l{l}.q_gate_proj"].Shape);
            Assert.Equal(slots[$"l{l}.q_proj"].Source, slots[$"l{l}.q_gate_proj"].Source);
        }
    }

    [Fact]
    public void WeightSlotsAreBf16ExceptTheFloat32GatingParametersAndTheHeadIsTiedToTheEmbedding()
    {
        var names = RoostTinyHybrid.WeightSlots.Select(s => s.Name).ToArray();
        Assert.Equal(1 + (LinearLayers.Length * 13) + (AttentionLayers.Length * 11) + 1, names.Length);
        Assert.Equal("embed", names[0]);
        Assert.Equal("final_norm", names[^1]);
        Assert.DoesNotContain("lm_head", names);
        Assert.True(C.TiedHead);
        foreach (var slot in RoostTinyHybrid.WeightSlots)
        {
            // A_log and the gated norm's weight are float32 in the checkpoint; everything else is bfloat16.
            bool f32 = slot.Name.EndsWith(".A_log", StringComparison.Ordinal) || slot.Name.EndsWith(".linear_norm", StringComparison.Ordinal);
            Assert.Equal(f32 ? DType.F32 : DType.BF16, slot.Dtype);
        }

        Assert.Equal(
            ["l0.input_norm", "l0.in_proj_qkv", "l0.conv", "l0.in_proj_z", "l0.in_proj_a", "l0.in_proj_b", "l0.A_log", "l0.dt_bias", "l0.linear_norm", "l0.out_proj", "l0.post_norm", "l0.gate_up", "l0.down"],
            names.Skip(1).Take(13).ToArray());

        var graph = RoostTinyHybrid.BuildGraph();
        var head = graph.Nodes.Single(n => n.Op == "LmHead");
        string[] fields = head.Attributes["in"].Split(',');
        Assert.Equal("embed", graph.Values[head.Inputs[Array.IndexOf(fields, "Weight")]].Name);
    }

    [Fact]
    public void StateGroupsAreOnePagedAttentionGroupAndOneRecurrentGroupWithTheFusedConvolutionState()
    {
        var groups = RoostTinyHybrid.CreateStateGroups();
        Assert.Equal(2, groups.Length);

        var attn = groups[0];
        Assert.Equal(StateGroupKind.Full, attn.Kind);
        Assert.Equal(AttentionLayers.Select(l => new LayerKvShape(l, C.KvHeads, C.HeadDim)).ToArray(), attn.Layers.ToArray());
        Assert.Equal(DType.BF16, attn.DType);
        Assert.Equal(2L * 2 * C.BlockSize * C.KvHeads * C.HeadDim * 2, attn.BytesPerBlock);

        var lin = groups[1];
        Assert.Equal(StateGroupKind.Recurrent, lin.Kind);
        Assert.Equal(1, lin.BlockSize);
        Assert.Equal(DType.F32, lin.DType);
        Assert.Equal([C.LinearValueHeads, C.LinearKeyDim, C.LinearValueDim], lin.RecurrentStateShape.ToArray());
        // Per layer: the float32 delta-rule state plus the bf16 convolution state of ALL fused channels, [conv_dim, width - 1].
        long perLayer = ((long)C.LinearValueHeads * C.LinearKeyDim * C.LinearValueDim * 4) + ((long)C.LinearConvDim * (C.ConvWidth - 1) * 2);
        Assert.Equal(LinearLayers.Length * perLayer, lin.BytesPerBlock);
    }

    [Fact]
    public void StateSlotsAreOrderedByGroupThenLayerThenRole()
    {
        string[] expected =
        [
            "attn.l3.k", "attn.l3.v", "attn.l7.k", "attn.l7.v",
            .. LinearLayers.SelectMany(l => new[] { $"linear.l{l}.state", $"linear.l{l}.conv" }),
        ];
        Assert.Equal(expected, RoostTinyHybrid.StateSlots.Select(s => s.Name).ToArray());
        var conv = RoostTinyHybrid.StateSlots.Single(s => s.Name == "linear.l5.conv");
        Assert.Equal(C.LinearConvDim * (C.ConvWidth - 1L), conv.ElementsPerBlock);
        Assert.Equal(DType.BF16, conv.Dtype);
        Assert.Equal(DType.F32, RoostTinyHybrid.StateSlots.Single(s => s.Name == "linear.l5.state").Dtype);
    }

    [Fact]
    public void PlanInputsAreTheOnesAttentionAndTheRecurrentOpsRead()
    {
        Assert.Equal(
            ["Ids", "Positions", "SlotMapping", "QIndptr", "KvIndptr", "KvIndices", "KvLastPageLength", "Rows", "StateSlots", "CuSeqlens"],
            RoostTinyHybrid.PlanInputs.Layout.Select(p => p.Name).ToArray());
    }

    [Fact]
    public void GraphHasTheHybridLayerPattern()
    {
        string[] linear = ["RmsNorm", "Gemm", "CausalConv1d", "Gemm", "Gemm", "Gemm", "GatedDeltaRuleGates", "GatedDeltaRule", "GatedRmsNorm", "Gemm", "Residual", "RmsNorm", "Gemm", "SiluAndMul", "Gemm", "Residual"];
        string[] attention = ["RmsNorm", "Gemm", "Gemm", "Gemm", "Gemm", "QkNorm", "Rope", "CacheWrite", "PagedAttention", "SigmoidMul", "Gemm", "Residual", "RmsNorm", "Gemm", "SiluAndMul", "Gemm", "Residual"];
        var expected = new List<string> { "Embedding" };
        for (int l = 0; l < C.Layers; l++)
        {
            expected.AddRange(C.LayerType(l) == TinyLayerType.LinearAttention ? linear : attention);
        }

        expected.AddRange(["RmsNorm", "LmHead"]);
        var g = RoostTinyHybrid.BuildGraph();
        Assert.Equal(expected, g.Nodes.Select(n => n.Op).ToArray());
        Assert.Equal(6, g.Nodes.Count(n => n.Op == "GatedDeltaRule"));
        Assert.Equal(6, g.Nodes.Count(n => n.Op == "GatedDeltaRuleGates"));
        Assert.Equal(6, g.Nodes.Count(n => n.Op == "GatedRmsNorm"));
        Assert.Equal(2, g.Nodes.Count(n => n.Op == "PagedAttention"));
        // Full attention: an unbound WindowLeft is 0, a one-token window on CpuOps (R-022); -1 disables the window.
        Assert.All(g.Nodes.Where(n => n.Op == "PagedAttention"), n => Assert.Equal("-1", n.Attributes["arg.WindowLeft"]));
        Assert.Equal(2, g.Nodes.Count(n => n.Op == "SigmoidMul"));

        // The one fused Gemm of a linear layer produces all conv_dim channels, and the convolution runs over all of them.
        var qkv = g.Values.Single(v => v.Name == "l0.qkv");
        Assert.Equal($"tokens,{C.LinearConvDim}", string.Join(",", qkv.Shape.Select(d => d.ToString())));
        var conv = g.Nodes.First(n => n.Op == "CausalConv1d");
        Assert.Equal(C.LinearConvDim.ToString(CultureInfo.InvariantCulture), conv.Attributes["arg.Channels"].Trim('(', ')'));

        // Partial rotary embedding: a quarter of the head, as the checkpoint's partial_rotary_factor 0.25.
        Assert.Equal((C.HeadDim / 4).ToString(CultureInfo.InvariantCulture), g.Nodes.First(n => n.Op == "Rope").Attributes["arg.RotaryDim"].Trim('(', ')'));
        Assert.Equal("BF16", g.Nodes.First(n => n.Op == "PagedAttention").Attributes["arg.KvDType"]);
        Assert.Equal(GraphDType.F32, g.Values.Single(v => v.Name == "logits").DType);
    }

    // GatedDeltaRule reads q, k and v as three row-strided column views of the fused convolution output: Q at column 0,
    // K at k_heads*k_dim and V at 2*k_heads*k_dim of each conv_dim-wide token row (contract version 2).
    [Fact]
    public void GatedDeltaRuleReadsQKAndVAsViewsOfTheConvolutionOutput()
    {
        var g = RoostTinyHybrid.BuildGraph();
        int key = C.LinearKeyHeads * C.LinearKeyDim;
        int convDim = C.LinearConvDim;
        var deltas = g.Nodes.Where(n => n.Op == "GatedDeltaRule").ToArray();
        Assert.Equal(LinearLayers.Length, deltas.Length);
        foreach (var n in deltas)
        {
            string[] fields = n.Attributes["in"].Split(',');
            GraphValue Bound(string field) => g.Values[n.Inputs[Array.IndexOf(fields, field)]];
            string Arg(string name) => n.Attributes["arg." + name].Trim('(', ')');

            // All three views are of the one convolution output.
            var conv = Bound("Q");
            Assert.Equal(conv.Id, Bound("K").Id);
            Assert.Equal(conv.Id, Bound("V").Id);
            Assert.EndsWith(".qkvc", conv.Name, StringComparison.Ordinal);
            Assert.Equal(convDim, conv.Shape[1].Size);
            Assert.Equal($"0:{key}", n.Attributes["view.Q"]);
            Assert.Equal($"{key}:{2 * key}", n.Attributes["view.K"]);
            Assert.Equal($"{2 * key}:{convDim}", n.Attributes["view.V"]);
            Assert.Equal(convDim.ToString(CultureInfo.InvariantCulture), Arg("QStride"));
            Assert.Equal(convDim.ToString(CultureInfo.InvariantCulture), Arg("KStride"));
            Assert.Equal(convDim.ToString(CultureInfo.InvariantCulture), Arg("VStride"));

            // G (float32) and Beta come from the gates op, not the raw projections.
            var gv = Bound("G");
            var beta = Bound("Beta");
            Assert.EndsWith(".g", gv.Name, StringComparison.Ordinal);
            Assert.Equal(GraphDType.F32, gv.DType);
            Assert.EndsWith(".beta", beta.Name, StringComparison.Ordinal);
            Assert.Equal(GraphDType.BF16, beta.DType);
        }
    }

    [Fact]
    public void ForwardCallsEveryNodeInOrderWithStateAndPlanWired()
    {
        var weights = Enumerable.Range(0, RoostTinyHybrid.WeightSlots.Length).Select(i => new DeviceSpan { Address = WeightBase + (i * 0x1000), Length = 1, Dtype = DType.BF16 }).ToArray();
        var state = Enumerable.Range(0, RoostTinyHybrid.StateSlots.Length).Select(i => new DeviceSpan { Address = StateBase + (i * 0x1000), Length = 1, Dtype = RoostTinyHybrid.StateSlots[i].Dtype }).ToArray();
        var arena = new DeviceSpan { Address = ArenaBase, Length = (ulong)Layout.ArenaBytes, Dtype = DType.U8 };
        var plan = RoostTinyHybrid.PlanInputs.Bind(new DeviceSpan { Address = PlanBase, Length = (ulong)Layout.PlanBytes, Dtype = DType.U8 }, Layout, 7, 2, 2, 5);
        var ctx = new RoostTinyHybrid.StepContext(Layout, arena, weights, state, plan, 7, 2, 2, 5);

        RecordingOps.Reset();
        RoostTinyHybrid.Forward<RecordingOps>(in ctx);

        var graph = RoostTinyHybrid.BuildGraph();
        Assert.Equal(graph.Nodes.Select(n => Enum.Parse<OpId>(n.Op)).ToArray(), RecordingOps.Ops.Take(RecordingOps.Count).ToArray());
        Assert.Equal(plan.Ids.Address, RecordingOps.FirstAddress[0]);

        // Each layer's ops read that layer's slots, found by name.
        int StateAddress(string name) => (int)(StateBase + (Array.FindIndex(RoostTinyHybrid.StateSlots, s => s.Name == name) * 0x1000));
        var caches = Enumerable.Range(0, RecordingOps.Count).Where(i => RecordingOps.Ops[i] == OpId.CacheWrite).ToArray();
        var attns = Enumerable.Range(0, RecordingOps.Count).Where(i => RecordingOps.Ops[i] == OpId.PagedAttention).ToArray();
        var convs = Enumerable.Range(0, RecordingOps.Count).Where(i => RecordingOps.Ops[i] == OpId.CausalConv1d).ToArray();
        var deltas = Enumerable.Range(0, RecordingOps.Count).Where(i => RecordingOps.Ops[i] == OpId.GatedDeltaRule).ToArray();
        Assert.Equal(2, caches.Length);
        Assert.Equal(2, attns.Length);
        for (int a = 0; a < 2; a++)
        {
            Assert.Equal(StateAddress($"attn.l{AttentionLayers[a]}.k"), (int)RecordingOps.SecondAddress[caches[a]]);
            Assert.Equal(StateAddress($"attn.l{AttentionLayers[a]}.k"), (int)RecordingOps.SecondAddress[attns[a]]);
        }

        Assert.Equal(LinearLayers.Length, convs.Length);
        Assert.Equal(LinearLayers.Length, deltas.Length);
        for (int i = 0; i < LinearLayers.Length; i++)
        {
            Assert.Equal(StateAddress($"linear.l{LinearLayers[i]}.conv"), (int)RecordingOps.SecondAddress[convs[i]]);
            Assert.Equal(StateAddress($"linear.l{LinearLayers[i]}.state"), (int)RecordingOps.SecondAddress[deltas[i]]);
        }
    }

    [Fact]
    public void ForwardAllocatesNothingOnceWarm()
    {
        var weights = new DeviceSpan[RoostTinyHybrid.WeightSlots.Length];
        var state = new DeviceSpan[RoostTinyHybrid.StateSlots.Length];
        var arena = new DeviceSpan { Address = ArenaBase, Length = (ulong)Layout.ArenaBytes, Dtype = DType.U8 };
        var plan = RoostTinyHybrid.PlanInputs.Bind(new DeviceSpan { Address = PlanBase, Length = (ulong)Layout.PlanBytes, Dtype = DType.U8 }, Layout, 7, 2, 2, 5);
        void Step()
        {
            RecordingOps.Reset();
            var ctx = new RoostTinyHybrid.StepContext(Layout, arena, weights, state, plan, 7, 2, 2, 5);
            RoostTinyHybrid.Forward<RecordingOps>(in ctx);
        }

        AllocationProbe.AssertSteady(Step);
    }

    [Fact]
    public void CheckBackendNamesTheReservedOpsABackendLacks()
    {
        RoostTinyHybrid.CheckBackend<CpuOps>();
        RoostTinyHybrid.CheckBackend<RecordingOps>();
        // StubCpuOps implements neither GatedDeltaRule nor CausalConv1d, which CpuOps does.
        var e = Assert.Throws<NotSupportedException>(RoostTinyHybrid.CheckBackend<StubCpuOps>);
        Assert.Contains("CausalConv1d", e.Message, StringComparison.Ordinal);
        Assert.Contains("GatedDeltaRule", e.Message, StringComparison.Ordinal);
        Assert.Contains("roost-tiny-hybrid", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheWeightGeneratorFillsEverySlotAndTheSeedIsTheModels()
    {
        foreach (var slot in RoostTinyHybrid.WeightSlots)
        {
            double[] v = TinyWeights.Generate(C.Seed, slot.Name, slot.Shape, slot.Dtype);
            Assert.Equal(slot.Elements, v.Length);
        }

        Assert.NotEqual(TinyConfig.Dense.Seed, C.Seed);
        Assert.Equal(TinyConfig.All.Length, TinyConfig.All.Select(c => c.Seed).Distinct().Count());
    }
}

using System.Globalization;
using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.ModelCompiler.Tests;
using Starlings.Roost.Ops;
using Starlings.Roost.State;

namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>
/// roost-tiny-hybrid (Qwen3.5-shaped) structure: weight and state slots, the fused linear-attention projection, plan inputs,
/// graph pattern and call order. The numeric half is waiting for the C3-qkv item (views), see <see cref="NumericRunWaitsForViews"/>.
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
            Assert.DoesNotContain($"l{l}.q_proj", slots.Keys);
            Assert.DoesNotContain($"l{l}.lin_q_proj", slots.Keys);
        }

        foreach (int l in AttentionLayers)
        {
            Assert.DoesNotContain($"l{l}.in_proj_qkv", slots.Keys);
            Assert.Equal($"{C.Heads * C.HeadDim},{C.Hidden}", slots[$"l{l}.q_proj"].Shape);
            Assert.Equal($"{C.KvHeads * C.HeadDim},{C.Hidden}", slots[$"l{l}.k_proj"].Shape);
        }
    }

    [Fact]
    public void WeightSlotsAreBf16AndTheHeadIsTiedToTheEmbedding()
    {
        var names = RoostTinyHybrid.WeightSlots.Select(s => s.Name).ToArray();
        Assert.Equal(1 + (LinearLayers.Length * 9) + (AttentionLayers.Length * 10) + 1, names.Length);
        Assert.Equal("embed", names[0]);
        Assert.Equal("final_norm", names[^1]);
        Assert.DoesNotContain("lm_head", names);
        Assert.True(C.TiedHead);
        Assert.All(RoostTinyHybrid.WeightSlots, s => Assert.Equal(DType.BF16, s.Dtype));
        Assert.Equal(
            ["l0.input_norm", "l0.in_proj_qkv", "l0.conv", "l0.in_proj_a", "l0.in_proj_b", "l0.out_proj", "l0.post_norm", "l0.gate_up", "l0.down"],
            names.Skip(1).Take(9).ToArray());

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
        string[] linear = ["RmsNorm", "Gemm", "CausalConv1d", "Gemm", "Gemm", "GatedDeltaRule", "Gemm", "Residual", "RmsNorm", "Gemm", "SiluAndMul", "Gemm", "Residual"];
        string[] attention = ["RmsNorm", "Gemm", "Gemm", "Gemm", "QkNorm", "Rope", "CacheWrite", "PagedAttention", "Gemm", "Residual", "RmsNorm", "Gemm", "SiluAndMul", "Gemm", "Residual"];
        var expected = new List<string> { "Embedding" };
        for (int l = 0; l < C.Layers; l++)
        {
            expected.AddRange(C.LayerType(l) == TinyLayerType.LinearAttention ? linear : attention);
        }

        expected.AddRange(["RmsNorm", "LmHead"]);
        var g = RoostTinyHybrid.BuildGraph();
        Assert.Equal(expected, g.Nodes.Select(n => n.Op).ToArray());
        Assert.Equal(6, g.Nodes.Count(n => n.Op == "GatedDeltaRule"));
        Assert.Equal(2, g.Nodes.Count(n => n.Op == "PagedAttention"));

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

    // The placeholder, held in place by a test. GatedDeltaRule needs q, k and v as three contiguous buffers; the fused
    // convolution output has them side by side in each token row, which C3 cannot slice (no views). Until the C3-qkv item
    // adds them, the definition binds the whole convolution output to all three arguments, so Forward is structurally right
    // and numerically wrong. This test pins that state: when the definition starts reading slices, it fails, and the
    // numeric test below must be switched on in the same change.
    [Fact]
    public void GatedDeltaRuleReadsThePlaceholderUntilC3ViewsLand()
    {
        var g = RoostTinyHybrid.BuildGraph();
        foreach (var n in g.Nodes.Where(n => n.Op == "GatedDeltaRule"))
        {
            string[] fields = n.Attributes["in"].Split(',');
            GraphValue Bound(string field) => g.Values[n.Inputs[Array.IndexOf(fields, field)]];

            long Arg(string name) => long.Parse(n.Attributes["arg." + name].Trim('(', ')'), CultureInfo.InvariantCulture);
            long Cols(GraphValue v) => v.Shape[1].Size;

            var q = Bound("Q");
            Assert.Equal(q.Id, Bound("K").Id);
            Assert.Equal(q.Id, Bound("V").Id);
            Assert.Equal(C.LinearConvDim, Cols(q));
            Assert.NotEqual(Arg("KHeads") * Arg("KDim"), Cols(q));
            Assert.NotEqual(Arg("VHeads") * Arg("VDim"), Cols(q));

            // G and Beta are real separate projections already.
            Assert.Equal(Arg("VHeads"), Cols(Bound("G")));
            Assert.Equal(Arg("VHeads"), Cols(Bound("Beta")));
            Assert.NotEqual(Bound("G").Id, Bound("Beta").Id);
        }
    }

    [Fact(Skip = "Waiting for the C3-qkv item: GatedDeltaRule must read q, k and v as views of the fused convolution output (K at element offset k_heads*k_dim and V at 2*k_heads*k_dim of each token row). Until then roost-tiny-hybrid's Forward is numerically meaningless; switch this on, compare against a double transcription as RoostTinyTests does, and delete GatedDeltaRuleReadsThePlaceholderUntilC3ViewsLand.")]
    public void NumericRunWaitsForViews()
    {
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

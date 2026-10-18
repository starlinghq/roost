using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.ModelCompiler.Tests.Models;
using Starlings.Roost.State;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>
/// The Qwen3.5-shaped tiny hybrid (three recurrent layers, one attention layer, two state groups): descriptors, state
/// and weight slot order, the plan inputs the attention and recurrent ops add, the call order and wiring of the
/// generated forward, and the load-time check for reserved ops.
/// </summary>
public class HybridModelTests
{
    private const nint WeightBase = 0x1000_0000;
    private const nint StateBase = 0x2000_0000;
    private const nint ArenaBase = 0x4000_0000;
    private const nint PlanBase = 0x6000_0000;

    private static readonly HybridModel.ForwardLayout Layout = new(maxTokens: 4096, maxSequences: 16, maxKvBlocks: 512);

    [Fact]
    public void StateGroupsAreOneFullAndOneRecurrent()
    {
        var groups = HybridModel.CreateStateGroups();
        Assert.Equal(2, groups.Length);

        var attn = groups[0];
        Assert.Equal(0, attn.Index);
        Assert.Equal("attn", attn.Name);
        Assert.Equal(StateGroupKind.Full, attn.Kind);
        Assert.Equal([new LayerKvShape(3, 2, 16)], attn.Layers.ToArray());
        Assert.Equal(16, attn.BlockSize);
        Assert.Equal(DType.BF16, attn.DType);
        // K and V, 16 tokens x 2 heads x 16 dims x 2 bytes each, one layer.
        Assert.Equal(2L * 16 * 2 * 16 * 2, attn.BytesPerBlock);
        Assert.Equal(0, attn.Window);
        Assert.True(attn.RecurrentStateShape.IsEmpty);

        var lin = groups[1];
        Assert.Equal(1, lin.Index);
        Assert.Equal(StateGroupKind.Recurrent, lin.Kind);
        Assert.True(lin.Layers.IsEmpty);
        Assert.Equal(1, lin.BlockSize);
        Assert.Equal(DType.F32, lin.DType);
        Assert.Equal([4L, 16, 16], lin.RecurrentStateShape.ToArray());
        Assert.Equal(8, lin.SnapshotSlots);
        // Per layer: the float32 delta-rule state [4,16,16] plus the bf16 convolution state [64,3]; three layers; the conv state is part of the slot.
        Assert.Equal(3L * ((4 * 16 * 16 * 4) + (64 * 3 * 2)), lin.BytesPerBlock);
    }

    [Fact]
    public void StateSlotsAreOrderedByGroupThenLayerThenRole()
    {
        Assert.Equal(
            ["attn.l3.k", "attn.l3.v", "linear.l0.state", "linear.l0.conv", "linear.l1.state", "linear.l1.conv", "linear.l2.state", "linear.l2.conv"],
            HybridModel.StateSlots.Select(s => s.Name).ToArray());
        Assert.Equal([0, 0, 1, 1, 1, 1, 1, 1], HybridModel.StateSlots.Select(s => s.Group).ToArray());
        Assert.Equal([3, 3, 0, 0, 1, 1, 2, 2], HybridModel.StateSlots.Select(s => s.Layer).ToArray());
        Assert.Equal(16L * 2 * 16, HybridModel.StateSlots[0].ElementsPerBlock);
        Assert.Equal(4L * 16 * 16, HybridModel.StateSlots[2].ElementsPerBlock);
        Assert.Equal(64L * 3, HybridModel.StateSlots[3].ElementsPerBlock);
        Assert.Equal(DType.F32, HybridModel.StateSlots[2].Dtype);
        Assert.Equal(DType.BF16, HybridModel.StateSlots[3].Dtype);
    }

    [Fact]
    public void WeightSlotsInterleaveLayersLayerByLayer()
    {
        var names = HybridModel.WeightSlots.Select(s => s.Name).ToArray();
        Assert.Equal(1 + (3 * 10) + 9 + 2, names.Length);
        Assert.Equal("embed", names[0]);
        Assert.Equal(
            ["l0.norm", "l0.q_proj", "l0.k_proj", "l0.lin_v_proj", "l0.lin_conv", "l0.lin_gb", "l0.lin_out", "l0.mlp_norm", "l0.gate_up", "l0.down"],
            names.Skip(1).Take(10).ToArray());
        Assert.Equal(
            ["l3.norm", "l3.q_proj", "l3.k_proj", "l3.v_proj", "l3.o_proj", "l3.qk_norm", "l3.mlp_norm", "l3.gate_up", "l3.down"],
            names.Skip(31).Take(9).ToArray());
        Assert.Equal(["final_norm", "lm_head"], names.TakeLast(2).ToArray());
        Assert.All(HybridModel.WeightSlots, s => Assert.Equal(DType.BF16, s.Dtype));
    }

    [Fact]
    public void PlanInputsGrowWithTheAttentionAndRecurrentOps()
    {
        Assert.Equal(
            ["Ids", "Positions", "SlotMapping", "QIndptr", "KvIndptr", "KvIndices", "KvLastPageLength", "Rows", "StateSlots", "CuSeqlens"],
            HybridModel.PlanInputs.Layout.Select(p => p.Name).ToArray());
        Assert.Equal(DType.I64, HybridModel.PlanInputs.Layout.Single(p => p.Name == "SlotMapping").Dtype);
        Assert.Equal("seqs+1", HybridModel.PlanInputs.Layout.Single(p => p.Name == "QIndptr").Count);
        Assert.Equal("kv_blocks", HybridModel.PlanInputs.Layout.Single(p => p.Name == "KvIndices").Count);

        var plan = HybridModel.PlanInputs.Bind(new DeviceSpan { Address = PlanBase, Length = (ulong)Layout.PlanBytes, Dtype = DType.U8 }, Layout, 100, 3, 3, 40);
        Assert.Equal(PlanBase, plan.Ids.Address);
        Assert.Equal(4UL, plan.QIndptr.Length);
        Assert.Equal(40UL, plan.KvIndices.Length);
        Assert.Equal(100UL, plan.SlotMapping.Length);
        Assert.Equal(DType.I64, plan.SlotMapping.Dtype);
        // Each input starts on a 64-byte boundary and none overlap.
        var spans = new[] { plan.Ids, plan.Positions, plan.SlotMapping, plan.QIndptr, plan.KvIndptr, plan.KvIndices, plan.KvLastPageLength, plan.Rows, plan.StateSlots, plan.CuSeqlens };
        for (int i = 0; i < spans.Length; i++)
        {
            Assert.Equal(0, (spans[i].Address - PlanBase) % 64);
            if (i > 0)
            {
                Assert.True(spans[i].Address >= spans[i - 1].Address + (nint)(spans[i - 1].Length * (ulong)Native.Size(spans[i - 1].Dtype)));
            }
        }

        Assert.True(spans[^1].Address - PlanBase < Layout.PlanBytes);
    }

    [Fact]
    public void GraphHasTheHybridLayerPattern()
    {
        var g = HybridModel.BuildGraph();
        string[] linear = ["RmsNorm", "Gemm", "Gemm", "Gemm", "CausalConv1d", "Gemm", "GatedDeltaRule", "Gemm", "Residual", "RmsNorm", "Gemm", "SiluAndMul", "Gemm", "Residual"];
        string[] attention = ["RmsNorm", "Gemm", "Gemm", "Gemm", "QkNorm", "Rope", "CacheWrite", "PagedAttention", "Gemm", "Residual", "RmsNorm", "Gemm", "SiluAndMul", "Gemm", "Residual"];
        var expected = new List<string> { "Embedding" };
        for (int l = 0; l < 3; l++)
        {
            expected.AddRange(linear);
        }

        expected.AddRange(attention);
        expected.AddRange(["RmsNorm", "LmHead"]);
        Assert.Equal(expected, g.Nodes.Select(n => n.Op).ToArray());

        // In-place ops: the output is a new value of the same shape, listed as both an input and an output of the field.
        var rope = g.Nodes.Single(n => n.Op == "Rope");
        Assert.Equal("Q,K,Positions", rope.Attributes["in"]);
        Assert.Equal("Q,K", rope.Attributes["out"]);
        Assert.Equal(["l3.qr", "l3.kr"], rope.Outputs.Select(o => g.Values[o].Name).ToArray());
        Assert.Equal("Default", rope.Attributes["arg.Kind"]);

        // The KV dtype of the bound state slot fills the unbound KvDType field.
        Assert.Equal("BF16", g.Nodes.Single(n => n.Op == "PagedAttention").Attributes["arg.KvDType"]);
        var kv = g.Values.Single(v => v.Name == "attn.l3.k");
        Assert.Equal(ValueKind.State, kv.Kind);
        Assert.Equal(GraphLayout.PagedNhd, kv.Layout);
        Assert.Equal("blocks", kv.Shape[0].Symbol);
        Assert.Equal("blocks,16,2,16", string.Join(",", kv.Shape.Select(d => d.ToString())));
        Assert.Equal(ValueKind.State, g.Values.Single(v => v.Name == "linear.l2.state").Kind);
        Assert.Equal("slots,4,16,16", string.Join(",", g.Values.Single(v => v.Name == "linear.l2.state").Shape.Select(d => d.ToString())));
    }

    [Fact]
    public void ForwardCallsEveryNodeInOrderWithStateAndPlanWired()
    {
        var weights = Enumerable.Range(0, HybridModel.WeightSlots.Length).Select(i => new DeviceSpan { Address = WeightBase + (i * 0x1000), Length = 1, Dtype = DType.BF16 }).ToArray();
        var state = Enumerable.Range(0, HybridModel.StateSlots.Length).Select(i => new DeviceSpan { Address = StateBase + (i * 0x1000), Length = 1, Dtype = HybridModel.StateSlots[i].Dtype }).ToArray();
        var arena = new DeviceSpan { Address = ArenaBase, Length = (ulong)Layout.ArenaBytes, Dtype = DType.U8 };
        var plan = HybridModel.PlanInputs.Bind(new DeviceSpan { Address = PlanBase, Length = (ulong)Layout.PlanBytes, Dtype = DType.U8 }, Layout, 7, 2, 2, 5);
        var ctx = new HybridModel.StepContext(Layout, arena, weights, state, plan, 7, 2, 2, 5);

        RecordingOps.Reset();
        HybridModel.Forward<RecordingOps>(in ctx);

        var graph = HybridModel.BuildGraph();
        var expected = graph.Nodes.Select(n => Enum.Parse<OpId>(n.Op)).ToArray();
        Assert.Equal(expected, RecordingOps.Ops.Take(RecordingOps.Count).ToArray());

        // Embedding reads the plan's token ids.
        Assert.Equal(plan.Ids.Address, RecordingOps.FirstAddress[0]);
        // The attention layer's cache write and attention read the first state slot (attn.l3.k); every first
        // address of an activation lies in the arena.
        int cache = Array.IndexOf(RecordingOps.Ops, OpId.CacheWrite);
        int attn = Array.IndexOf(RecordingOps.Ops, OpId.PagedAttention);
        Assert.Equal(StateBase, RecordingOps.SecondAddress[cache]);
        Assert.Equal(StateBase, RecordingOps.SecondAddress[attn]);
        // Layer l's recurrent ops read that layer's slots: state index 2 + 2 * l is the state, 3 + 2 * l the conv state.
        var convs = Enumerable.Range(0, RecordingOps.Count).Where(i => RecordingOps.Ops[i] == OpId.CausalConv1d).ToArray();
        var deltas = Enumerable.Range(0, RecordingOps.Count).Where(i => RecordingOps.Ops[i] == OpId.GatedDeltaRule).ToArray();
        Assert.Equal(3, convs.Length);
        Assert.Equal(3, deltas.Length);
        for (int l = 0; l < 3; l++)
        {
            Assert.Equal(StateBase + ((3 + (2 * l)) * 0x1000), RecordingOps.SecondAddress[convs[l]]);
            Assert.Equal(StateBase + ((2 + (2 * l)) * 0x1000), RecordingOps.SecondAddress[deltas[l]]);
        }

        // Activations live in the arena.
        int firstGemm = Array.IndexOf(RecordingOps.Ops, OpId.Gemm);
        Assert.InRange(RecordingOps.FirstAddress[firstGemm], ArenaBase, ArenaBase + (nint)Layout.ArenaBytes);
    }

    [Fact]
    public void ForwardAllocatesNothingOnceWarm()
    {
        var weights = new DeviceSpan[HybridModel.WeightSlots.Length];
        var state = new DeviceSpan[HybridModel.StateSlots.Length];
        var arena = new DeviceSpan { Address = ArenaBase, Length = (ulong)Layout.ArenaBytes, Dtype = DType.U8 };
        var plan = HybridModel.PlanInputs.Bind(new DeviceSpan { Address = PlanBase, Length = (ulong)Layout.PlanBytes, Dtype = DType.U8 }, Layout, 7, 2, 2, 5);
        void Step()
        {
            RecordingOps.Reset();
            var ctx = new HybridModel.StepContext(Layout, arena, weights, state, plan, 7, 2, 2, 5);
            HybridModel.Forward<RecordingOps>(in ctx);
        }

        Assert.Equal(0, AllocationProbe.Rounds(Step).Min());
    }

    [Fact]
    public void ArenaReusesBuffersAcrossLayers()
    {
        var g = HybridModel.BuildGraph();
        long naive = g.Values
            .Where(v => v.Kind is ValueKind.Intermediate or ValueKind.Output && v.Shape[0].Symbol == "tokens")
            .Sum(v => v.Shape.Skip(1).Aggregate(1L, (a, d) => a * d.Size) * 2);
        long planned = HybridModel.ForwardLayout.PerTokenBytes;
        Assert.True(planned * 4 < naive, $"{planned} B/token planned vs {naive} B/token without reuse");
    }

    [Fact]
    public void CheckBackendNamesTheReservedOpsABackendLacks()
    {
        HybridModel.CheckBackend<RecordingOps>();
        var e = Assert.Throws<NotSupportedException>(HybridModel.CheckBackend<StubCpuOps>);
        Assert.Contains("CausalConv1d", e.Message, StringComparison.Ordinal);
        Assert.Contains("GatedDeltaRule", e.Message, StringComparison.Ordinal);
        Assert.Contains("tiny-hybrid", e.Message, StringComparison.Ordinal);
        string[] used = ["Embedding", "Gemm", "LmHead", "PagedAttention", "QkNorm", "Residual", "RmsNorm", "Rope", "SiluAndMul", "CacheWrite", "CausalConv1d", "GatedDeltaRule"];
        Assert.Equal(used.Order().ToArray(), HybridModel.UsedOps.Order().ToArray());
    }
}

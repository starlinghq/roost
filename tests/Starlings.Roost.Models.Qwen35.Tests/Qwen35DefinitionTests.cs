using System.Reflection;
using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.Models.Qwen35;
using Starlings.Roost.Ops;
using Starlings.Roost.State;

namespace Starlings.Roost.Models.Qwen35.Tests;

/// <summary>The Qwen3.5-0.8B definition (R-035): layer pattern, the two state groups, the graph, the generated forward's checks.</summary>
public class Qwen35DefinitionTests
{
    private static readonly int[] AttentionLayers = [3, 7, 11, 15, 19, 23];

    private static int LayerOf(OpNode n) => int.Parse(n.Attributes["layer"], System.Globalization.CultureInfo.InvariantCulture);

    [Fact]
    public void SixFullAttentionAndEighteenRecurrentLayers()
    {
        var g = Qwen35Model.BuildGraph();
        Assert.Equal(6, g.Nodes.Count(n => n.Op == "PagedAttention"));
        Assert.Equal(18, g.Nodes.Count(n => n.Op == "GatedDeltaRule"));
        Assert.Equal(18, g.Nodes.Count(n => n.Op == "CausalConv1d"));
        Assert.Equal(AttentionLayers, g.Nodes.Where(n => n.Op == "PagedAttention").Select(n => LayerOf(n)).ToArray());
        Assert.Equal(
            Enumerable.Range(0, 24).Except(AttentionLayers).ToArray(),
            g.Nodes.Where(n => n.Op == "GatedDeltaRule").Select(n => LayerOf(n)).ToArray());
        Assert.Equal(24, g.Nodes.Count(n => n.Op == "SiluAndMul"));
        Assert.Equal(49, g.Nodes.Count(n => n.Op == "RmsNorm"));
        Assert.Equal("Embedding", g.Nodes[0].Op);
        Assert.Equal("LmHead", g.Nodes[^1].Op);
    }

    [Fact]
    public void TheLayerPatternIsTheCheckpointsLayerTypes()
    {
        string? dir = Environment.GetEnvironmentVariable("ROOST_QWEN35_CHECKPOINT");
        if (dir is null || !File.Exists(Path.Combine(dir, "config.json")))
        {
            // Without the checkpoint the pattern is the interval of 4 that config.json states (full_attention_interval 4).
            Assert.Equal(AttentionLayers, Enumerable.Range(0, 24).Where(i => i % 4 == 3).ToArray());
            return;
        }

        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "config.json")));
        var text = doc.RootElement.GetProperty("text_config");
        var types = text.GetProperty("layer_types").EnumerateArray().Select(e => e.GetString()).ToArray();
        Assert.Equal(AttentionLayers, Enumerable.Range(0, types.Length).Where(i => types[i] == "full_attention").ToArray());
        Assert.Equal(1024, text.GetProperty("hidden_size").GetInt32());
        Assert.Equal(3584, text.GetProperty("intermediate_size").GetInt32());
        Assert.Equal(8, text.GetProperty("num_attention_heads").GetInt32());
        Assert.Equal(2, text.GetProperty("num_key_value_heads").GetInt32());
        Assert.Equal(256, text.GetProperty("head_dim").GetInt32());
        Assert.Equal(16, text.GetProperty("linear_num_key_heads").GetInt32());
        Assert.Equal(16, text.GetProperty("linear_num_value_heads").GetInt32());
        Assert.Equal(128, text.GetProperty("linear_key_head_dim").GetInt32());
        Assert.Equal(128, text.GetProperty("linear_value_head_dim").GetInt32());
        Assert.Equal(4, text.GetProperty("linear_conv_kernel_dim").GetInt32());
        Assert.Equal(248320, text.GetProperty("vocab_size").GetInt32());
        Assert.Equal(0.25, text.GetProperty("rope_parameters").GetProperty("partial_rotary_factor").GetDouble());
        Assert.Equal(10000000, text.GetProperty("rope_parameters").GetProperty("rope_theta").GetDouble());
    }

    [Fact]
    public void TwoStateGroupsOneFullAndOneRecurrent()
    {
        var groups = Qwen35Model.CreateStateGroups();
        Assert.Equal(2, groups.Length);

        var attn = groups[0];
        Assert.Equal("attn", attn.Name);
        Assert.Equal(StateGroupKind.Full, attn.Kind);
        Assert.Equal(AttentionLayers.Select(l => new LayerKvShape(l, 2, 256)).ToArray(), attn.Layers.ToArray());
        Assert.Equal(16, attn.BlockSize);
        Assert.Equal(DType.BF16, attn.DType);
        // K and V of 16 tokens x 2 heads x 256 dims x 2 bytes, six layers.
        Assert.Equal(6L * 2 * 16 * 2 * 256 * 2, attn.BytesPerBlock);

        var lin = groups[1];
        Assert.Equal("linear", lin.Name);
        Assert.Equal(StateGroupKind.Recurrent, lin.Kind);
        Assert.Equal([16L, 128, 128], lin.RecurrentStateShape.ToArray());
        Assert.Equal(DType.F32, lin.DType);
        // Eighteen layers of the float32 delta-rule state [16,128,128] plus the bfloat16 conv state [6144,3].
        Assert.Equal(18L * ((16 * 128 * 128 * 4) + (6144 * 3 * 2)), lin.BytesPerBlock);
    }

    [Fact]
    public void StateSlotsAreOrderedByGroupThenLayerThenRole()
    {
        var names = Qwen35Model.StateSlots.Select(s => s.Name).ToArray();
        Assert.Equal(12 + 36, names.Length);
        Assert.Equal(["attn.l3.k", "attn.l3.v", "attn.l7.k"], names.Take(3).ToArray());
        Assert.Equal(["linear.l0.state", "linear.l0.conv", "linear.l1.state"], names.Skip(12).Take(3).ToArray());
        var conv = Qwen35Model.StateSlots.First(s => s.Name == "linear.l0.conv");
        Assert.Equal(6144L * 3, conv.ElementsPerBlock);
        Assert.Equal(DType.BF16, conv.Dtype);
        var state = Qwen35Model.StateSlots.First(s => s.Name == "linear.l0.state");
        Assert.Equal(16L * 128 * 128, state.ElementsPerBlock);
        Assert.Equal(DType.F32, state.Dtype);
    }

    [Fact]
    public void FusedProjectionIsConvolvedThenSplitIntoStridedViews()
    {
        var g = Qwen35Model.BuildGraph();
        var delta = g.Nodes.First(n => n.Op == "GatedDeltaRule");
        Assert.Equal("0:2048", delta.Attributes["view.Q"]);
        Assert.Equal("2048:4096", delta.Attributes["view.K"]);
        Assert.Equal("4096:6144", delta.Attributes["view.V"]);
        Assert.Equal("(6144)", delta.Attributes["arg.QStride"]);
        Assert.Equal("(6144)", delta.Attributes["arg.KStride"]);
        Assert.Equal("(6144)", delta.Attributes["arg.VStride"]);
        string[] fields = delta.Attributes["in"].Split(',');
        var parent = g.Values[delta.Inputs[Array.IndexOf(fields, "Q")]];
        Assert.Equal(delta.Inputs[Array.IndexOf(fields, "K")], delta.Inputs[Array.IndexOf(fields, "Q")]);
        Assert.Equal(delta.Inputs[Array.IndexOf(fields, "V")], delta.Inputs[Array.IndexOf(fields, "Q")]);
        Assert.Equal("l0.c", parent.Name);
        Assert.Equal(6144, parent.Shape[1].Size);
        // The convolution reads the fused projection's output and the delta rule reads the convolution's: no op between them copies.
        Assert.Equal("l0.conv", g.Nodes.Single(n => n.Outputs.Contains(parent.Id)).Attributes["node"]);
        Assert.Equal("CausalConv1d", g.Nodes.Single(n => n.Outputs.Contains(parent.Id)).Op);
        // Value heads equal key heads here, so the grouping factor is 1 (the generator supports more value heads than key heads).
        Assert.Equal("(16)", delta.Attributes["arg.KHeads"]);
        Assert.Equal("(16)", delta.Attributes["arg.VHeads"]);
        Assert.Equal("0.08838834764831845", delta.Attributes["arg.Scale"]);
    }

    [Fact]
    public void AttentionLayersCarryTheQwen35Specifics()
    {
        var g = Qwen35Model.BuildGraph();
        var rope = g.Nodes.First(n => n.Op == "Rope");
        Assert.Equal("(64)", rope.Attributes["arg.RotaryDim"]);
        Assert.Equal("10000000", rope.Attributes["arg.Theta"]);
        Assert.Equal("Default", rope.Attributes["arg.Kind"]);
        var attn = g.Nodes.First(n => n.Op == "PagedAttention");
        Assert.Equal("(256)", attn.Attributes["arg.HeadDim"]);
        Assert.Equal("0.0625", attn.Attributes["arg.Scale"]);
        Assert.Equal("PagedNhd", attn.Attributes["arg.KvLayout"]);
        Assert.Equal("BF16", attn.Attributes["arg.KvDType"]);
        // Full attention: an unbound WindowLeft is 0, a window of one token (R-022 found the forward attending to the current
        // token only); -1 disables the window.
        Assert.All(g.Nodes.Where(n => n.Op == "PagedAttention"), n => Assert.Equal("-1", n.Attributes["arg.WindowLeft"]));
        var q = g.Values.Single(v => v.Name == "l3.q");
        Assert.Equal(2048, q.Shape[1].Size);
        // The head is tied: LmHead reads the embedding slot.
        var head = g.Nodes[^1];
        // LmHead writes float32 logits (C3); the value is declared F32 so that Output_logits and the arena agree.
        Assert.Equal(GraphDType.F32, g.Values[head.Outputs[0]].DType);
        Assert.Equal("embed", g.Values[head.Inputs[Array.IndexOf(head.Attributes["in"].Split(','), "Weight")]].Name);
    }

    [Fact]
    public void WeightSlotsFollowLayersAndHoldTheChosenDtypes()
    {
        var slots = Qwen35Model.WeightSlots;
        Assert.Equal(1 + (6 * 11) + (18 * 13) + 1, slots.Length);
        Assert.Equal("embed", slots[0].Name);
        Assert.Equal("final_norm", slots[^1].Name);
        Assert.Equal(248320L * 1024, slots[0].Elements);
        Assert.DoesNotContain(slots, s => s.Name.Contains("lm_head", StringComparison.Ordinal));
        Assert.Equal(
            ["l0.input_norm", "l0.in_proj_qkv", "l0.conv", "l0.in_proj_z", "l0.in_proj_a", "l0.in_proj_b", "l0.A_log", "l0.dt_bias", "l0.linear_norm", "l0.out_proj", "l0.post_norm", "l0.gate_up", "l0.down"],
            slots.Skip(1).Take(13).Select(s => s.Name).ToArray());
        Assert.Equal(
            ["l3.input_norm", "l3.q_proj", "l3.q_gate_proj", "l3.k_proj", "l3.v_proj", "l3.q_norm", "l3.k_norm", "l3.o_proj", "l3.post_norm", "l3.gate_up", "l3.down"],
            slots.Where(s => s.Layer == 3).Select(s => s.Name).ToArray());
        Assert.Equal(DType.F32, slots.First(s => s.Name == "l0.A_log").Dtype);
        Assert.Equal(DType.F32, slots.First(s => s.Name == "l0.linear_norm").Dtype);
        Assert.Equal(DType.BF16, slots.First(s => s.Name == "l0.dt_bias").Dtype);
        Assert.Equal(DType.BF16, slots.First(s => s.Name == "l0.in_proj_qkv").Dtype);
        Assert.Equal(6144L * 4, slots.First(s => s.Name == "l0.conv").Elements);
        // About 0.75 B parameters, as the checkpoint's text model.
        Assert.InRange(slots.Sum(s => s.Elements), 750_000_000L, 800_000_000L);
    }

    [Fact]
    public void PlanInputsAreThoseTheAttentionAndRecurrentOpsRead()
    {
        Assert.Equal(
            ["Ids", "Positions", "SlotMapping", "QIndptr", "KvIndptr", "KvIndices", "KvLastPageLength", "Rows", "StateSlots", "CuSeqlens"],
            Qwen35Model.PlanInputs.Layout.Select(p => p.Name).ToArray());
    }

    [Fact]
    public void CpuOpsRunsEveryOpTheDefinitionUses()
    {
        Qwen35Model.CheckBackend<CpuOps>();
        string[] used =
        [
            "Embedding", "Gemm", "LmHead", "PagedAttention", "QkNorm", "Residual", "RmsNorm", "Rope", "SiluAndMul", "CacheWrite",
            "CausalConv1d", "GatedDeltaRule", "GatedDeltaRuleGates", "GatedRmsNorm", "SigmoidMul",
        ];
        Assert.Equal(used.Order().ToArray(), Qwen35Model.UsedOps.Order().ToArray());
    }

    [Fact]
    public void TheGeneratedCodeIsInThisAssemblyAndNeedsNoGenerator()
    {
        // R-020's "model packages ship the generated code": the model is compiled into the package's assembly, which refers to the
        // contract assemblies it calls and not to the generator.
        var assembly = typeof(Qwen35Model).Assembly;
        Assert.NotNull(typeof(Qwen35Model).GetMethod("Forward", BindingFlags.Public | BindingFlags.Static));
        Assert.NotNull(typeof(Qwen35Model).GetMethod("BuildGraph", BindingFlags.Public | BindingFlags.Static));
        var references = assembly.GetReferencedAssemblies().Select(a => a.Name).ToArray();
        Assert.DoesNotContain(references, n => n!.Contains("ModelCompiler", StringComparison.Ordinal));
        Assert.Contains("Starlings.Roost.Ops", references);
        Assert.Contains("Starlings.Roost.Abstractions", references);
        Assert.DoesNotContain(references, n => n!.Contains("OnnxRuntime", StringComparison.Ordinal));
    }

    [Fact]
    public void TheGatingOpsAreInTheGraphAndTheirInputsAreWired()
    {
        var g = Qwen35Model.BuildGraph();
        Assert.True(Qwen35Model.NumericallyComplete);
        Assert.Equal(18, g.Nodes.Count(n => n.Op == "GatedDeltaRuleGates"));
        Assert.Equal(18, g.Nodes.Count(n => n.Op == "GatedRmsNorm"));
        Assert.Equal(6, g.Nodes.Count(n => n.Op == "SigmoidMul"));

        string In(OpNode n, string field) => g.Values[n.Inputs[Array.IndexOf(n.Attributes["in"].Split(','), field)]].Name;
        var gates = g.Nodes.First(n => n.Op == "GatedDeltaRuleGates");
        Assert.Equal(["l0.a", "l0.b", "l0.A_log", "l0.dt_bias"], new[] { In(gates, "A"), In(gates, "B"), In(gates, "ALog"), In(gates, "DtBias") });
        var gv = g.Values.Single(v => v.Name == "l0.g");
        Assert.Equal(GraphDType.F32, gv.DType);
        Assert.Equal(GraphDType.BF16, g.Values.Single(v => v.Name == "l0.beta").DType);
        var delta = g.Nodes.First(n => n.Op == "GatedDeltaRule");
        Assert.Equal("l0.g", In(delta, "G"));
        Assert.Equal("l0.beta", In(delta, "Beta"));

        var norm = g.Nodes.First(n => n.Op == "GatedRmsNorm");
        Assert.Equal(["l0.o", "l0.z", "l0.linear_norm"], new[] { In(norm, "Input"), In(norm, "Gate"), In(norm, "Weight") });
        Assert.Equal("(128)", norm.Attributes["arg.Dim"]);
        var outProj = g.Nodes.Single(n => n.Attributes["node"] == "l0.out_proj");
        Assert.Equal("l0.on", In(outProj, "A"));

        var sm = g.Nodes.First(n => n.Op == "SigmoidMul");
        Assert.Equal(["l3.ao", "l3.gate"], new[] { In(sm, "Input"), In(sm, "Gate") });
        var oProj = g.Nodes.Single(n => n.Attributes["node"] == "l3.o_proj");
        Assert.Equal("l3.aog", In(oProj, "A"));
    }
}

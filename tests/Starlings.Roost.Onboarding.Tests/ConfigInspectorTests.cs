using System.Text;
using System.Text.Json.Nodes;
using Starlings.Roost.Abi;
using Starlings.Roost.Models.Qwen35;
using Starlings.Roost.Onboarding;
using Starlings.Roost.Ops;
using Starlings.Roost.State;

namespace Starlings.Roost.Onboarding.Tests;

// The configs under TestData are written by hand from the public config.json fields of Qwen3-0.6B, Qwen3.5-0.8B (text_config,
// with a shortened vision_config), Qwen3-30B-A3B, Gemma 3 1B and DeepSeek-V3. Expected op statuses read CpuOps.Supports so the
// tests follow the reference as reserved ops land; only the composition logic is fixed here.
public sealed class ConfigInspectorTests
{
    internal static string DataPath(string name) => Path.Combine(AppContext.BaseDirectory, "TestData", name);

    private static InspectReport Inspect(string name, InspectOptions? options = null) => ConfigInspector.InspectFile(DataPath(name), options);

    private static InspectReport InspectJson(JsonNode config) => ConfigInspector.Inspect(Encoding.UTF8.GetBytes(config.ToJsonString()), "inline.json");

    private static JsonNode Load(string name) => JsonNode.Parse(File.ReadAllText(DataPath(name)))!;

    private static OpRequirement Op(InspectReport r, string name) => Assert.Single(r.Ops, o => o.Op == name);

    private static OpRequirementStatus Reference(OpId op) => CpuOps.Supports(op) ? OpRequirementStatus.Available : OpRequirementStatus.Declared;

    [Fact]
    public void Qwen3DenseComposesFromM0OpsWithOneFullAttentionGroup()
    {
        var r = Inspect("qwen3-0.6b.config.json");

        Assert.Equal(CompositionVerdict.Composes, r.Verdict);
        Assert.Equal("qwen3", r.ModelType);
        Assert.Equal(["Qwen3ForCausalLM"], r.Architectures);
        Assert.Equal(ContractVersion.Current, r.ContractVersion);
        Assert.Equal(new ModelDims { Hidden = 1024, Layers = 28, Heads = 16, KvHeads = 8, HeadDim = 128, Intermediate = 3072, Vocab = 151936, DType = "BF16", TiedEmbeddings = true }, r.Dims);
        Assert.Equal(Enumerable.Range(0, 28), r.Layers.FullAttention);
        Assert.Empty(r.Layers.SlidingAttention);
        Assert.Empty(r.Layers.LinearAttention);
        Assert.Empty(r.Layers.Moe);
        Assert.Null(r.Moe);

        string[] ops = ["Embedding", "RmsNorm", "Residual", "LmHead", "Argmax", "Sample", "Gemm", "QkNorm", "Rope", "CacheWrite", "PagedAttention", "SiluAndMul"];
        Assert.Equal(ops.Order(StringComparer.Ordinal), r.Ops.Select(o => o.Op).Order(StringComparer.Ordinal));
        Assert.All(r.Ops, o => Assert.Equal(OpRequirementStatus.Available, o.Status));
        Assert.All(r.Ops, o => Assert.Equal("M0", o.Phase));
        Assert.Empty(r.MissingOps);
        Assert.Empty(r.PendingOps);
        Assert.Empty(r.Warnings);

        var g = Assert.Single(r.StateGroups);
        Assert.Equal("attn", g.Name);
        Assert.Equal(nameof(StateGroupKind.Full), g.Kind);
        Assert.Equal(Enumerable.Range(0, 28), g.Layers);
        Assert.Equal((8, 128, 16), (g.KvHeads, g.HeadSize, g.BlockSize));
        // K and V, 16 tokens x 8 heads x 128 dims x 2 bytes, 28 layers.
        Assert.Equal(28L * 2 * 16 * 8 * 128 * 2, g.BytesPerBlock);
        Assert.False(g.Optional);
    }

    [Fact]
    public void Qwen3InspectReportIsSnakeCaseJsonThatRoundTrips()
    {
        var r = Inspect("qwen3-0.6b.config.json");
        string json = ConfigInspector.ToJson(r);

        Assert.Contains("\"plugin\": \"roost-v1\"", json, StringComparison.Ordinal);
        Assert.Contains("\"verdict\": \"composes\"", json, StringComparison.Ordinal);
        Assert.Contains("\"state_groups\"", json, StringComparison.Ordinal);
        Assert.Contains("\"bytes_per_block\"", json, StringComparison.Ordinal);
        Assert.Contains("\"status\": \"available\"", json, StringComparison.Ordinal);
        Assert.Contains("\"dtype\": \"BF16\"", json, StringComparison.Ordinal);
        Assert.DoesNotContain("d_type", json, StringComparison.Ordinal);
        var back = ConfigInspector.FromJson(json);
        Assert.Equal(json, ConfigInspector.ToJson(back));
        Assert.Matches("^[0-9a-f]{64}$", r.ConfigSha256);
    }

    [Fact]
    public void Qwen35HybridMapsLinearAndGatedAttentionAndMatchesTheDefinitionsStateGroups()
    {
        var r = Inspect("qwen3.5-0.8b.config.json", new InspectOptions { SnapshotSlots = 8 });

        Assert.Equal("qwen3_5_text", r.ModelType);
        int[] full = [3, 7, 11, 15, 19, 23];
        Assert.Equal(full, r.Layers.FullAttention);
        Assert.Equal(Enumerable.Range(0, 24).Except(full), r.Layers.LinearAttention);
        Assert.Equal(1024, r.Dims.Hidden);
        Assert.Equal(256, r.Dims.HeadDim);

        foreach (var op in (OpId[])[OpId.CausalConv1d, OpId.GatedDeltaRuleGates, OpId.GatedDeltaRule, OpId.GatedRmsNorm, OpId.RecurrentStateCopy, OpId.RecurrentStateSnapshot])
        {
            var req = Op(r, op.ToString());
            Assert.Equal(["linear_attention"], req.UsedBy);
            Assert.Equal(Reference(op), req.Status);
        }

        Assert.Equal(["attention"], Op(r, "SigmoidMul").UsedBy);
        Assert.Equal(["attention"], Op(r, "QkNorm").UsedBy);
        Assert.Equal(Reference(OpId.SigmoidMul), Op(r, "SigmoidMul").Status);
        Assert.DoesNotContain(r.Ops, o => o.Op == "RopeScaled");
        Assert.Contains(r.Notes, n => n.Contains("RotaryDim = 64 of 256", StringComparison.Ordinal));
        Assert.Contains(r.Notes, n => n.Contains("mrope_section", StringComparison.Ordinal));
        Assert.Contains(r.Notes, n => n.Contains("vision_config is not inspected", StringComparison.Ordinal));
        Assert.Contains(r.Notes, n => n.Contains("Multi-token-prediction layers (1)", StringComparison.Ordinal));
        Assert.Contains(r.LoadTransforms, t => t.StartsWith("norms: RMS norm weights are stored zero-centred", StringComparison.Ordinal));
        Assert.Empty(r.Warnings);
        bool allReference = r.Ops.All(o => o.Status == OpRequirementStatus.Available);
        Assert.Equal(allReference ? CompositionVerdict.Composes : CompositionVerdict.ComposesPendingReference, r.Verdict);

        // The generated Qwen3.5-0.8B definition declares the same two groups; inspect must reproduce them from config.json alone.
        var expected = Qwen35Model.CreateStateGroups();
        Assert.Equal(expected.Length, r.StateGroups.Count);
        for (int i = 0; i < expected.Length; i++)
        {
            var e = expected[i];
            var a = r.StateGroups[i];
            Assert.Equal(e.Index, a.Index);
            Assert.Equal(e.Name, a.Name);
            Assert.Equal(e.Kind.ToString(), a.Kind);
            Assert.Equal(e.BlockSize, a.BlockSize);
            Assert.Equal(e.DType.ToString(), a.DType);
            Assert.Equal(e.BytesPerBlock, a.BytesPerBlock);
            Assert.Equal(e.RecurrentStateShape.ToArray(), a.RecurrentStateShape);
            Assert.Equal(e.SnapshotSlots, a.SnapshotSlots);
            if (!e.Layers.IsEmpty)
            {
                Assert.Equal(e.Layers.Select(l => l.Layer), a.Layers);
                Assert.All(e.Layers, l => Assert.Equal((l.KvHeads, l.HeadSize), (a.KvHeads, a.HeadSize)));
            }
        }

        Assert.Equal([6144L, 3], r.StateGroups[1].ConvStateShape);
        Assert.Equal(r.Layers.LinearAttention, r.StateGroups[1].Layers);
    }

    [Fact]
    public void Qwen3MoeMapsRoutingToMoeOpsAndAnOptionalExpertGroup()
    {
        var r = Inspect("qwen3-30b-a3b.config.json");

        Assert.Equal(new MoeShape { Experts = 128, TopK = 8, ExpertIntermediate = 768, SharedExperts = 0, Scoring = "softmax", Renormalize = true }, r.Moe);
        Assert.Equal(Enumerable.Range(0, 48), r.Layers.Moe);
        Assert.Empty(r.Layers.DenseMlp);
        foreach (var op in (OpId[])[OpId.MoeRouteTopK, OpId.MoePermute, OpId.GroupedGemm, OpId.MoeUnpermute])
        {
            var req = Op(r, op.ToString());
            Assert.Equal(["moe"], req.UsedBy);
            Assert.Empty(req.Gaps);
            Assert.Equal(Reference(op), req.Status);
        }

        Assert.Equal(["moe"], Op(r, "SiluAndMul").UsedBy);
        Assert.Contains(r.Notes, n => n.Contains("Renormalize = true", StringComparison.Ordinal));
        Assert.Empty(r.MissingOps);
        Assert.NotEqual(CompositionVerdict.MissingOps, r.Verdict);

        Assert.Equal(["attn", "experts"], r.StateGroups.Select(g => g.Name));
        var attn = r.StateGroups[0];
        Assert.Equal(48L * 2 * 16 * 4 * 128 * 2, attn.BytesPerBlock);
        var experts = r.StateGroups[1];
        Assert.Equal(nameof(StateGroupKind.Expert), experts.Kind);
        Assert.True(experts.Optional);
        Assert.Equal(1, experts.BlockSize);
        // One routed expert's gate, up and down projections (3 x 2048 x 768 bf16) across all 48 MoE layers.
        Assert.Equal(48L * 3 * 2048 * 768 * 2, experts.BytesPerBlock);
    }

    [Fact]
    public void Gemma3SlidingWindowMapsToPagedAttentionWindowAndASlidingGroup()
    {
        var r = Inspect("gemma-3-1b.config.json");

        int[] global = [5, 11, 17, 23];
        Assert.Equal(global, r.Layers.FullAttention);
        Assert.Equal(Enumerable.Range(0, 26).Except(global), r.Layers.SlidingAttention);
        Assert.Equal(CompositionVerdict.Composes, r.Verdict);
        Assert.Equal(["attention", "sliding_attention"], Op(r, "PagedAttention").UsedBy);
        Assert.Equal(["attention", "sliding_attention"], Op(r, "QkNorm").UsedBy);
        Assert.Equal(OpRequirementStatus.Available, Op(r, "GeluAndMul").Status);
        Assert.DoesNotContain(r.Ops, o => o.Op is "SiluAndMul" or "SlidingWindowAttention" or "SigmoidMul");
        Assert.Contains(r.Notes, n => n.Contains("PagedAttentionArgs.WindowLeft = 511", StringComparison.Ordinal));
        Assert.Contains(r.Notes, n => n.Contains("rope_local_base_freq 10000", StringComparison.Ordinal));
        Assert.Contains(r.Notes, n => n.Contains("GeluAndMulArgs.Approximation = Tanh", StringComparison.Ordinal));
        Assert.Contains(r.Notes, n => n.Contains("query_pre_attn_scalar^-0.5 = 0.0625", StringComparison.Ordinal));
        Assert.Contains(r.LoadTransforms, t => t.StartsWith("embedding: the hidden state is scaled by sqrt(hidden_size)", StringComparison.Ordinal));
        Assert.Contains(r.LoadTransforms, t => t.StartsWith("norms:", StringComparison.Ordinal));

        Assert.Equal(["attn", "sliding"], r.StateGroups.Select(g => g.Name));
        var sliding = r.StateGroups[1];
        Assert.Equal(nameof(StateGroupKind.Sliding), sliding.Kind);
        Assert.Equal(512, sliding.Window);
        Assert.Equal(22L * 2 * 16 * 1 * 256 * 2, sliding.BytesPerBlock);
        Assert.Equal(0, r.StateGroups[0].Window);
    }

    [Fact]
    public void Gemma3LayerTypesGiveTheSameMapAsTheSlidingPattern()
    {
        var config = Load("gemma-3-1b.config.json");
        var types = new JsonArray();
        for (int i = 0; i < 26; i++)
        {
            types.Add((i + 1) % 6 == 0 ? "full_attention" : "sliding_attention");
        }

        config["layer_types"] = types;
        var withTypes = InspectJson(config);
        var withPattern = Inspect("gemma-3-1b.config.json");

        Assert.Equal(withPattern.Layers, withTypes.Layers, LayerMapEquals);
        Assert.Equal(withPattern.StateGroups, withTypes.StateGroups, (a, b) => ConfigInspectorJson(a) == ConfigInspectorJson(b));
    }

    [Fact]
    public void LogitSoftCappingIsAGapOnAttentionAndTheHead()
    {
        var config = Load("gemma-3-1b.config.json");
        config["attn_logit_softcapping"] = 50.0;
        config["final_logit_softcapping"] = 30.0;
        var r = InspectJson(config);

        Assert.Equal(CompositionVerdict.MissingOps, r.Verdict);
        Assert.Equal(OpRequirementStatus.Gap, Op(r, "PagedAttention").Status);
        Assert.Contains(Op(r, "PagedAttention").Gaps, g => g.Contains("attn_logit_softcapping 50", StringComparison.Ordinal));
        Assert.Equal(OpRequirementStatus.Gap, Op(r, "LmHead").Status);
        Assert.Equal(["LmHead", "PagedAttention"], r.MissingOps.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void DeepSeekMlaNeedsMlaAttentionAndReportsRoutingAndQuantisationGaps()
    {
        var r = Inspect("deepseek-v3.config.json");

        Assert.Equal(CompositionVerdict.MissingOps, r.Verdict);
        Assert.Equal(Enumerable.Range(0, 61), r.Layers.Mla);
        Assert.Empty(r.Layers.FullAttention);
        Assert.Equal([0, 1, 2], r.Layers.DenseMlp);
        Assert.Equal(Enumerable.Range(3, 58), r.Layers.Moe);
        Assert.Equal(new MoeShape { Experts = 256, TopK = 8, ExpertIntermediate = 2048, SharedExperts = 1, Scoring = "sigmoid", Renormalize = true }, r.Moe);

        var mla = Op(r, "MlaAttention");
        Assert.Equal("M4", mla.Phase);
        Assert.Equal(Reference(OpId.MlaAttention), mla.Status);
        Assert.Equal(["mla"], mla.UsedBy);
        Assert.Equal(["mla"], Op(r, "RopeScaled").UsedBy);
        Assert.DoesNotContain(r.Ops, o => o.Op is "PagedAttention" or "QkNorm");

        var route = Op(r, "MoeRouteTopK");
        Assert.Equal(OpRequirementStatus.Gap, route.Status);
        Assert.Equal(3, route.Gaps.Count);
        Assert.Contains(route.Gaps, g => g.Contains("scoring_func 'sigmoid'", StringComparison.Ordinal));
        Assert.Contains(route.Gaps, g => g.Contains("n_group 8, topk_group 4", StringComparison.Ordinal));
        Assert.Contains(route.Gaps, g => g.Contains("score correction bias", StringComparison.Ordinal));
        Assert.Equal(OpRequirementStatus.Gap, Op(r, "CacheWrite").Status);
        Assert.Equal(OpRequirementStatus.Gap, Op(r, "GemmFp8").Status);
        Assert.Contains(Op(r, "GemmFp8").Gaps, g => g.Contains("[128, 128]", StringComparison.Ordinal));
        Assert.Equal(["mlp", "moe", "shared_expert"], Op(r, "SiluAndMul").UsedBy);
        Assert.Contains("MoeRouteTopK", r.MissingOps);
        Assert.Contains(r.LoadTransforms, t => t.Contains("routed_scaling_factor 2.5", StringComparison.Ordinal));
        Assert.Contains(r.LoadTransforms, t => t.StartsWith("mla: the key and value up-projections", StringComparison.Ordinal));
        Assert.Contains(r.Notes, n => n.Contains("Multi-token-prediction layers (1)", StringComparison.Ordinal));
        if (!CpuOps.Supports(OpId.MlaAttention))
        {
            Assert.Contains("MlaAttention", r.PendingOps);
        }

        Assert.Equal(["mla", "experts"], r.StateGroups.Select(g => g.Name));
        var latent = r.StateGroups[0];
        Assert.Equal(nameof(StateGroupKind.MlaLatent), latent.Kind);
        Assert.Equal((1, 576), (latent.KvHeads, latent.HeadSize));
        // One latent row (512 + 64) per token, no separate V: 61 layers x 16 tokens x 576 x 2 bytes.
        Assert.Equal(61L * 16 * 576 * 2, latent.BytesPerBlock);
        var experts = r.StateGroups[1];
        Assert.Equal(nameof(DType.F8E4M3), experts.DType);
        Assert.Equal(58L * 3 * 7168 * 2048, experts.BytesPerBlock);
    }

    [Fact]
    public void KvDTypeAndBlockSizeOptionsScaleThePagedGroups()
    {
        var r = Inspect("qwen3-0.6b.config.json", new InspectOptions { BlockSize = 32, KvDType = DType.F8E4M3 });
        var g = Assert.Single(r.StateGroups);
        Assert.Equal(nameof(DType.F8E4M3), g.DType);
        Assert.Equal(32, g.BlockSize);
        Assert.Equal(28L * 2 * 32 * 8 * 128 * 1, g.BytesPerBlock);
    }

    [Fact]
    public void MistralStyleSlidingWindowAppliesToEveryLayer()
    {
        var config = Load("qwen3-0.6b.config.json");
        config["model_type"] = "mistral";
        config["sliding_window"] = 4096;
        config.AsObject().Remove("use_sliding_window");
        var r = InspectJson(config);

        Assert.Equal(Enumerable.Range(0, 28), r.Layers.SlidingAttention);
        Assert.DoesNotContain(r.Ops, o => o.Op == "QkNorm");
        var g = Assert.Single(r.StateGroups);
        Assert.Equal(4096, g.Window);
    }

    [Fact]
    public void Qwen3UseSlidingWindowAppliesFromMaxWindowLayers()
    {
        var config = Load("qwen3-0.6b.config.json");
        config["use_sliding_window"] = true;
        config["sliding_window"] = 4096;
        config["max_window_layers"] = 20;
        var r = InspectJson(config);

        Assert.Equal(Enumerable.Range(0, 20), r.Layers.FullAttention);
        Assert.Equal(Enumerable.Range(20, 8), r.Layers.SlidingAttention);
    }

    [Fact]
    public void UnknownFeaturesAreReportedAsMissingOps()
    {
        var config = Load("qwen3-0.6b.config.json");
        config["model_type"] = "newfamily";
        config["hidden_act"] = "relu";
        config["rope_scaling"] = new JsonObject { ["rope_type"] = "linear", ["factor"] = 4.0 };
        config.AsObject().Remove("rms_norm_eps");
        config["layer_norm_eps"] = 1e-5;
        var types = new JsonArray();
        for (int i = 0; i < 28; i++)
        {
            types.Add(i % 2 == 0 ? "mamba" : "full_attention");
        }

        config["layer_types"] = types;
        var r = InspectJson(config);

        Assert.Equal(CompositionVerdict.MissingOps, r.Verdict);
        Assert.Equal(OpRequirementStatus.Missing, Op(r, "LayerNorm").Status);
        Assert.Equal(OpRequirementStatus.Missing, Op(r, "GatedActivation(relu)").Status);
        Assert.Equal(OpRequirementStatus.Missing, Op(r, "Mixer(mamba)").Status);
        Assert.Null(Op(r, "Mixer(mamba)").Phase);
        Assert.Equal(OpRequirementStatus.Gap, Op(r, "Rope").Status);
        Assert.Equal(14, r.Layers.Unknown.Count);
        Assert.Equal("0:mamba", r.Layers.Unknown[0]);
        Assert.Contains(r.Warnings, w => w.Contains("model_type 'newfamily'", StringComparison.Ordinal));
        Assert.DoesNotContain(r.Ops, o => o.Op == "QkNorm");
        Assert.Equal(Enumerable.Range(0, 14).Select(i => (2 * i) + 1), Assert.Single(r.StateGroups).Layers);
    }

    [Fact]
    public void UnmappedQuantisationIsMissingAndAwqMapsToGemmW4A16()
    {
        var config = Load("qwen3-0.6b.config.json");
        config["quantization_config"] = new JsonObject { ["quant_method"] = "awq", ["bits"] = 4, ["group_size"] = 128 };
        Assert.Equal(OpRequirementStatus.Available, Op(InspectJson(config), "GemmW4A16").Status);

        config["quantization_config"] = new JsonObject { ["quant_method"] = "bitsandbytes", ["bits"] = 8 };
        var r = InspectJson(config);
        Assert.Equal(OpRequirementStatus.Missing, Op(r, "Quantization(bitsandbytes)").Status);
        Assert.Equal(CompositionVerdict.MissingOps, r.Verdict);
    }

    [Theory]
    [InlineData("hidden_size")]
    [InlineData("num_hidden_layers")]
    [InlineData("vocab_size")]
    public void AnAbsentRequiredFieldIsATypedErrorNamingIt(string field)
    {
        var config = Load("qwen3-0.6b.config.json");
        config.AsObject().Remove(field);
        var e = Assert.Throws<OnboardingConfigException>(() => InspectJson(config));
        Assert.Contains(field, e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MalformedConfigsAreTypedErrors()
    {
        Assert.Throws<OnboardingConfigException>(() => ConfigInspector.Inspect("[1, 2]"u8, "a.json"));
        Assert.Throws<OnboardingConfigException>(() => ConfigInspector.Inspect("{ not json"u8, "b.json"));

        var config = Load("qwen3-0.6b.config.json");
        config["hidden_size"] = "1024";
        Assert.Contains("not an integer", Assert.Throws<OnboardingConfigException>(() => InspectJson(config)).Message, StringComparison.Ordinal);

        config = Load("qwen3-0.6b.config.json");
        config["layer_types"] = new JsonArray("full_attention");
        Assert.Contains("layer_types has 1 entries for 28 layers", Assert.Throws<OnboardingConfigException>(() => InspectJson(config)).Message, StringComparison.Ordinal);

        config = Load("qwen3.5-0.8b.config.json");
        config["text_config"]!.AsObject().Remove("linear_num_value_heads");
        Assert.Contains("linear_num_value_heads", Assert.Throws<OnboardingConfigException>(() => InspectJson(config)).Message, StringComparison.Ordinal);
    }

    private static bool LayerMapEquals(LayerMap a, LayerMap b) =>
        a.FullAttention.SequenceEqual(b.FullAttention) && a.SlidingAttention.SequenceEqual(b.SlidingAttention)
        && a.LinearAttention.SequenceEqual(b.LinearAttention) && a.Mla.SequenceEqual(b.Mla) && a.Unknown.SequenceEqual(b.Unknown)
        && a.DenseMlp.SequenceEqual(b.DenseMlp) && a.Moe.SequenceEqual(b.Moe);

    private static string ConfigInspectorJson(StateGroupReport g) =>
        string.Join(",", g.Index, g.Name, g.Kind, string.Join(" ", g.Layers), g.KvHeads, g.HeadSize, g.BlockSize, g.DType, g.BytesPerBlock, g.Window);
}

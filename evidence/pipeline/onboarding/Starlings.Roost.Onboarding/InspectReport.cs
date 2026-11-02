using System.Text.Json.Serialization;

namespace Starlings.Roost.Onboarding;

/// <summary>Whether a model composes from the op vocabulary (C3), as <see cref="ConfigInspector"/> judges it from config.json.</summary>
public enum CompositionVerdict
{
    /// <summary>Every op the model needs is in C3, has a CPU reference, and can express what the config asks of it.</summary>
    [JsonStringEnumMemberName("composes")]
    Composes,

    /// <summary>Every op is in C3 and can express the config, but at least one is declared without a CPU reference yet.</summary>
    [JsonStringEnumMemberName("composes_pending_reference")]
    ComposesPendingReference,

    /// <summary>At least one op is not in C3, or a C3 op cannot express what the config asks of it.</summary>
    [JsonStringEnumMemberName("missing_ops")]
    MissingOps,
}

/// <summary>Status of one op the model needs.</summary>
public enum OpRequirementStatus
{
    /// <summary>In C3, implemented by <c>CpuOps</c>, and no gap. Whether a GPU kernel exists is a startup check (R-023), not inspect's.</summary>
    [JsonStringEnumMemberName("available")]
    Available,

    /// <summary>In C3 (reserved for a later phase) but <c>CpuOps</c> does not implement it yet.</summary>
    [JsonStringEnumMemberName("declared")]
    Declared,

    /// <summary>In C3, but its arguments cannot express what the config asks (see <see cref="OpRequirement.Gaps"/>).</summary>
    [JsonStringEnumMemberName("gap")]
    Gap,

    /// <summary>Not in C3 at all.</summary>
    [JsonStringEnumMemberName("missing")]
    Missing,
}

/// <summary>One op the model needs.</summary>
public sealed record OpRequirement
{
    /// <summary>Op name: a C3 <c>OpId</c> name, or a descriptive name for an op C3 lacks.</summary>
    public required string Op { get; init; }

    /// <summary>The C3 phase (<c>M0</c>, <c>M1</c>, …), or null for an op not in C3.</summary>
    public string? Phase { get; init; }

    /// <summary>The status.</summary>
    public required OpRequirementStatus Status { get; init; }

    /// <summary>Which parts of the model use it (attention, mlp, moe, linear_attention, …), in first-use order.</summary>
    public required IReadOnlyList<string> UsedBy { get; init; }

    /// <summary>What the config asks that the op's arguments cannot express; empty unless <see cref="Status"/> is gap or missing.</summary>
    public required IReadOnlyList<string> Gaps { get; init; }
}

/// <summary>The model's widths as read from config.json.</summary>
public sealed record ModelDims
{
    /// <summary><c>hidden_size</c>.</summary>
    public required int Hidden { get; init; }

    /// <summary><c>num_hidden_layers</c>.</summary>
    public required int Layers { get; init; }

    /// <summary><c>num_attention_heads</c>.</summary>
    public required int Heads { get; init; }

    /// <summary><c>num_key_value_heads</c>, defaulting to the query heads.</summary>
    public required int KvHeads { get; init; }

    /// <summary><c>head_dim</c>, defaulting to hidden / heads.</summary>
    public required int HeadDim { get; init; }

    /// <summary><c>intermediate_size</c> of the dense MLP; 0 when absent.</summary>
    public required int Intermediate { get; init; }

    /// <summary><c>vocab_size</c>.</summary>
    public required int Vocab { get; init; }

    /// <summary>Weight dtype from <c>dtype</c> or <c>torch_dtype</c> (BF16 when absent, with a warning).</summary>
    [JsonPropertyName("dtype")]
    public required string DType { get; init; }

    /// <summary><c>tie_word_embeddings</c> (true when absent, the transformers default).</summary>
    public required bool TiedEmbeddings { get; init; }
}

/// <summary>Which decoder layers have which token mixer and MLP.</summary>
public sealed record LayerMap
{
    /// <summary>Full (global) attention layers.</summary>
    public required IReadOnlyList<int> FullAttention { get; init; }

    /// <summary>Sliding-window attention layers.</summary>
    public required IReadOnlyList<int> SlidingAttention { get; init; }

    /// <summary>Linear-attention (Gated DeltaNet) layers.</summary>
    public required IReadOnlyList<int> LinearAttention { get; init; }

    /// <summary>Multi-head latent attention layers.</summary>
    public required IReadOnlyList<int> Mla { get; init; }

    /// <summary>Layers whose <c>layer_types</c> entry inspect does not know, as "index:type".</summary>
    public required IReadOnlyList<string> Unknown { get; init; }

    /// <summary>Layers with a dense MLP.</summary>
    public required IReadOnlyList<int> DenseMlp { get; init; }

    /// <summary>Layers with a mixture-of-experts MLP.</summary>
    public required IReadOnlyList<int> Moe { get; init; }
}

/// <summary>The MoE shape, when the model has experts.</summary>
public sealed record MoeShape
{
    /// <summary>Routed experts per MoE layer.</summary>
    public required int Experts { get; init; }

    /// <summary>Experts per token.</summary>
    public required int TopK { get; init; }

    /// <summary>Per-expert intermediate width.</summary>
    public required int ExpertIntermediate { get; init; }

    /// <summary>Shared experts per MoE layer (DeepSeek <c>n_shared_experts</c>, or 1 for a Qwen shared expert); 0 for none.</summary>
    public required int SharedExperts { get; init; }

    /// <summary>Router scoring: softmax or sigmoid.</summary>
    public required string Scoring { get; init; }

    /// <summary>Whether the top-k weights are renormalised (<c>norm_topk_prob</c>).</summary>
    public required bool Renormalize { get; init; }
}

/// <summary>A C4 state group the model needs, from a <c>StateGroupDescriptor</c> built by inspect.</summary>
public sealed record StateGroupReport
{
    /// <summary>Group index, as the step plan indexes groups.</summary>
    public required int Index { get; init; }

    /// <summary>Group name (<c>attn</c>, <c>sliding</c>, <c>linear</c>, <c>mla</c>, <c>experts</c>).</summary>
    public required string Name { get; init; }

    /// <summary>C4 kind name (<c>StateGroupKind</c>).</summary>
    public required string Kind { get; init; }

    /// <summary>Model layers in the group.</summary>
    public required IReadOnlyList<int> Layers { get; init; }

    /// <summary>KV heads per layer for paged kinds; 0 otherwise.</summary>
    public required int KvHeads { get; init; }

    /// <summary>Head size (latent width for MLA) per layer for paged kinds; 0 otherwise.</summary>
    public required int HeadSize { get; init; }

    /// <summary>Tokens per block; 1 for recurrent and expert groups.</summary>
    public required int BlockSize { get; init; }

    /// <summary>Stored element type (C1 <c>DType</c> name).</summary>
    [JsonPropertyName("dtype")]
    public required string DType { get; init; }

    /// <summary>Bytes per block across all layers of the group (C4 <c>BytesPerBlock</c>).</summary>
    public required long BytesPerBlock { get; init; }

    /// <summary>Window in tokens for a sliding group; 0 otherwise.</summary>
    public required int Window { get; init; }

    /// <summary>Per-layer recurrent state shape; empty otherwise.</summary>
    public required IReadOnlyList<long> RecurrentStateShape { get; init; }

    /// <summary>Per-layer convolution state shape held in the same recurrent slot; empty otherwise.</summary>
    public required IReadOnlyList<long> ConvStateShape { get; init; }

    /// <summary>Snapshot slots of the recurrent group (a profile choice, from the options).</summary>
    public required int SnapshotSlots { get; init; }

    /// <summary>True for a group a profile may choose not to allocate (expert paging, bet X6).</summary>
    public required bool Optional { get; init; }
}

/// <summary>What <see cref="ConfigInspector"/> found: typed, and written as JSON by <c>roost-onboard inspect</c>.</summary>
public sealed record InspectReport
{
    /// <summary>The onboarding plugin id.</summary>
    public string Plugin { get; init; } = OnboardingPlugin.Id;

    /// <summary>Version of this report's shape.</summary>
    public int ReportVersion { get; init; } = 1;

    /// <summary>The C1–C9 contract version the op and state-group mapping was made against.</summary>
    public required int ContractVersion { get; init; }

    /// <summary>The config file name as given.</summary>
    public required string Source { get; init; }

    /// <summary>SHA-256 of the config bytes, lowercase hex.</summary>
    public required string ConfigSha256 { get; init; }

    /// <summary><c>model_type</c> of the inspected (text) config; null when absent.</summary>
    public string? ModelType { get; init; }

    /// <summary><c>architectures</c>.</summary>
    public required IReadOnlyList<string> Architectures { get; init; }

    /// <summary>The verdict.</summary>
    public required CompositionVerdict Verdict { get; init; }

    /// <summary>Widths.</summary>
    public required ModelDims Dims { get; init; }

    /// <summary>Layer map.</summary>
    public required LayerMap Layers { get; init; }

    /// <summary>MoE shape; null for a dense model.</summary>
    public MoeShape? Moe { get; init; }

    /// <summary>Every op the model needs, in first-use order.</summary>
    public required IReadOnlyList<OpRequirement> Ops { get; init; }

    /// <summary>Names of ops with status missing or gap.</summary>
    public required IReadOnlyList<string> MissingOps { get; init; }

    /// <summary>Names of ops with status declared.</summary>
    public required IReadOnlyList<string> PendingOps { get; init; }

    /// <summary>C4 state groups.</summary>
    public required IReadOnlyList<StateGroupReport> StateGroups { get; init; }

    /// <summary>Weight transforms a loader must apply so the model maps onto C3 without a new op (zero-centred norms, folded scales).</summary>
    public required IReadOnlyList<string> LoadTransforms { get; init; }

    /// <summary>Facts about the mapping a reader should know (what was not inspected, equivalences used).</summary>
    public required IReadOnlyList<string> Notes { get; init; }

    /// <summary>Assumptions inspect had to make because config.json did not say.</summary>
    public required IReadOnlyList<string> Warnings { get; init; }
}

/// <summary>The roost-v1 onboarding plugin's identity.</summary>
public static class OnboardingPlugin
{
    /// <summary>Plugin id, also the recipe schema's <c>plugin</c> value.</summary>
    public const string Id = "roost-v1";
}

/// <summary>Source-generated JSON for the report (AOT-safe, no reflection serialisation).</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    UseStringEnumConverter = true,
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.Never)]
[JsonSerializable(typeof(InspectReport))]
internal sealed partial class OnboardingJsonContext : JsonSerializerContext;

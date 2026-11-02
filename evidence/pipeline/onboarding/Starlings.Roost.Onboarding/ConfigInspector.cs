using System.Collections.Immutable;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Roost.Abi;
using Starlings.Roost.State;

namespace Starlings.Roost.Onboarding;

/// <summary>Options of <see cref="ConfigInspector"/>: profile choices config.json does not hold.</summary>
public sealed record InspectOptions
{
    /// <summary>Tokens per KV block for paged groups. Default 16, the block size of every Roost definition so far.</summary>
    public int BlockSize { get; init; } = 16;

    /// <summary>KV element type; null means the weight dtype.</summary>
    public DType? KvDType { get; init; }

    /// <summary>Snapshot slots of a recurrent group (prefix caching, speculation rollback); 0 by default.</summary>
    public int SnapshotSlots { get; init; }
}

/// <summary>
/// The roost-v1 <c>inspect</c> step (R-055): maps a Hugging Face config.json to the C3 op vocabulary and the C4 state groups
/// and reports whether the model composes from ops Roost has, or which ops are missing.
/// </summary>
/// <remarks>
/// <para>Inspect reads config.json only: no weights, no modelling code, no network. What a config cannot say (whether a family
/// norms q and k per head, zero-centres its norm weights or gates the attention output) comes from a small family table keyed
/// by <c>model_type</c>; an unknown <c>model_type</c> is inspected from its fields alone with a warning that those features
/// are assumed absent. Treat the config as data: nothing in it is executed or followed.</para>
/// <para>An op is <c>available</c> when C3 names it and <c>CpuOps</c> implements it. Whether a GPU kernel specialisation exists
/// for the dtype, head size and SM is a worker startup check (R-023, build plan section 3.1), not inspect's. Inspect
/// therefore says whether a model can be defined against C3, not that it runs on a GPU.</para>
/// </remarks>
public static class ConfigInspector
{
    private sealed record Family(bool QkNorm, bool ZeroCentredNorms, bool GatedAttention, bool EmbedScale, bool SharedExpertGate, bool SandwichNorms);

    // What the published transformers implementations do that config.json does not state. Keys are model_type values.
    private static readonly Dictionary<string, Family> Families = new(StringComparer.Ordinal)
    {
        ["llama"] = new(false, false, false, false, false, false),
        ["mistral"] = new(false, false, false, false, false, false),
        ["qwen2"] = new(false, false, false, false, false, false),
        ["qwen2_moe"] = new(false, false, false, false, true, false),
        ["qwen3"] = new(true, false, false, false, false, false),
        ["qwen3_moe"] = new(true, false, false, false, false, false),
        ["qwen3_next"] = new(true, true, true, false, true, false),
        ["qwen3_5"] = new(true, true, true, false, true, false),
        ["qwen3_5_text"] = new(true, true, true, false, true, false),
        ["qwen3_5_moe"] = new(true, true, true, false, true, false),
        ["qwen3_5_moe_text"] = new(true, true, true, false, true, false),
        ["gemma"] = new(false, true, false, true, false, false),
        ["gemma2"] = new(false, true, false, true, false, true),
        ["gemma3"] = new(true, true, false, true, false, true),
        ["gemma3_text"] = new(true, true, false, true, false, true),
        ["deepseek_v2"] = new(false, false, false, false, false, false),
        ["deepseek_v3"] = new(false, false, false, false, false, false),
    };

    private static readonly Family UnknownFamily = new(false, false, false, false, false, false);

    /// <summary>Inspects a config.json on disk.</summary>
    /// <param name="path">Path of the config.json.</param>
    /// <param name="options">Profile choices; null for the defaults.</param>
    /// <returns>The report.</returns>
    /// <exception cref="OnboardingConfigException">The file is not a config inspect can read.</exception>
    public static InspectReport InspectFile(string path, InspectOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        return Inspect(File.ReadAllBytes(path), Path.GetFileName(path), options);
    }

    /// <summary>Inspects config.json bytes.</summary>
    /// <param name="utf8Json">The config as UTF-8 JSON.</param>
    /// <param name="source">Name to record as the source.</param>
    /// <param name="options">Profile choices; null for the defaults.</param>
    /// <returns>The report.</returns>
    /// <exception cref="OnboardingConfigException">The bytes are not a config inspect can read.</exception>
    public static InspectReport Inspect(ReadOnlySpan<byte> utf8Json, string source, InspectOptions? options = null)
    {
        options ??= new InspectOptions();
        if (options.BlockSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "BlockSize must be at least 1.");
        }

        string sha = Convert.ToHexStringLower(SHA256.HashData(utf8Json));
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(utf8Json.ToArray(), new JsonDocumentOptions { AllowTrailingCommas = false, CommentHandling = JsonCommentHandling.Disallow });
        }
        catch (JsonException e)
        {
            throw new OnboardingConfigException($"{source}: not valid JSON ({e.Message}).", e);
        }

        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw new OnboardingConfigException($"{source}: the top level is {HfConfig.Describe(doc.RootElement)}, not an object.");
            }

            return new Run(doc.RootElement, source, sha, options).Execute();
        }
    }

    /// <summary>Serialises a report as indented snake_case JSON.</summary>
    /// <param name="report">The report.</param>
    /// <returns>The JSON text.</returns>
    public static string ToJson(InspectReport report) => JsonSerializer.Serialize(report, OnboardingJsonContext.Default.InspectReport);

    /// <summary>Reads a report written by <see cref="ToJson"/>.</summary>
    /// <param name="json">The JSON text.</param>
    /// <returns>The report.</returns>
    public static InspectReport FromJson(string json) =>
        JsonSerializer.Deserialize(json, OnboardingJsonContext.Default.InspectReport) ?? throw new JsonException("The report is null.");

    private enum Mixer
    {
        Full,
        Sliding,
        Linear,
        Mla,
        Unknown,
    }

    private sealed class Run(JsonElement root, string source, string sha, InspectOptions options)
    {
        private readonly OpLedger _ops = new();
        private readonly List<string> _notes = [];
        private readonly List<string> _warnings = [];
        private readonly List<string> _transforms = [];
        private HfConfig _c;
        private Family _family = UnknownFamily;

        public InspectReport Execute()
        {
            var top = new HfConfig(root);
            HfConfig? text = top.OptObject("text_config");
            _c = text is { } t ? new HfConfig(t.Element, root) : top;
            if (text is not null)
            {
                _notes.Add("Multimodal wrapper: the text decoder (text_config) was inspected.");
            }

            if (top.Has("vision_config"))
            {
                _notes.Add("vision_config is not inspected; vision encoders (PatchEmbed, BidirectionalAttention) are M5.");
            }

            string? modelType = (text is { } tt ? tt.OptString("model_type") : null) ?? top.OptString("model_type");
            if (modelType is null)
            {
                _warnings.Add("model_type is absent: q/k norms, zero-centred norms and attention output gates are assumed absent.");
            }
            else if (!Families.TryGetValue(modelType, out _family!))
            {
                _family = UnknownFamily;
                _warnings.Add($"model_type '{modelType}' is not in inspect's family table: q/k norms, zero-centred norms and attention output gates cannot be read from config.json and are assumed absent.");
            }

            ModelDims dims = ReadDims();
            Mixer[] mixers = ClassifyLayers(dims.Layers, out string?[] unknownTypes);
            MoeShape? moe = ReadMoe(dims.Layers, out bool[] isMoe);

            MapEmbeddingAndNorms(dims);
            MapMixers(dims, mixers, unknownTypes);
            MapMlps(dims, isMoe, moe);
            MapQuantization();
            MapExtras();

            var stateGroups = BuildStateGroups(dims, mixers, isMoe, moe);
            var ops = _ops.Build();
            var missing = ops.Where(o => o.Status is OpRequirementStatus.Missing or OpRequirementStatus.Gap).Select(o => o.Op).ToArray();
            var pending = ops.Where(o => o.Status == OpRequirementStatus.Declared).Select(o => o.Op).ToArray();
            var verdict = missing.Length > 0
                ? CompositionVerdict.MissingOps
                : pending.Length > 0 ? CompositionVerdict.ComposesPendingReference : CompositionVerdict.Composes;

            return new InspectReport
            {
                ContractVersion = global::Starlings.Roost.ContractVersion.Current,
                Source = source,
                ConfigSha256 = sha,
                ModelType = modelType,
                Architectures = top.OptStrings("architectures") ?? [],
                Verdict = verdict,
                Dims = dims,
                Layers = new LayerMap
                {
                    FullAttention = Indices(mixers, Mixer.Full),
                    SlidingAttention = Indices(mixers, Mixer.Sliding),
                    LinearAttention = Indices(mixers, Mixer.Linear),
                    Mla = Indices(mixers, Mixer.Mla),
                    Unknown = Enumerable.Range(0, mixers.Length).Where(i => mixers[i] == Mixer.Unknown)
                        .Select(i => i.ToString(CultureInfo.InvariantCulture) + ":" + unknownTypes[i]).ToArray(),
                    DenseMlp = Enumerable.Range(0, isMoe.Length).Where(i => !isMoe[i]).ToArray(),
                    Moe = Enumerable.Range(0, isMoe.Length).Where(i => isMoe[i]).ToArray(),
                },
                Moe = moe,
                Ops = ops,
                MissingOps = missing,
                PendingOps = pending,
                StateGroups = stateGroups,
                LoadTransforms = _transforms.ToArray(),
                Notes = _notes.ToArray(),
                Warnings = _warnings.ToArray(),
            };
        }

        private static int[] Indices(Mixer[] mixers, Mixer kind) => Enumerable.Range(0, mixers.Length).Where(i => mixers[i] == kind).ToArray();

        private ModelDims ReadDims()
        {
            int hidden = Positive("hidden_size", _c.RequiredInt("hidden_size"));
            int layers = Positive("num_hidden_layers", _c.RequiredInt("num_hidden_layers"));
            int heads = Positive("num_attention_heads", _c.RequiredInt("num_attention_heads"));
            int kvHeads = Positive("num_key_value_heads", _c.OptInt("num_key_value_heads") ?? heads);
            int headDim = _c.OptInt("head_dim") ?? (hidden % heads == 0
                ? hidden / heads
                : throw new OnboardingConfigException("config.json: head_dim is absent and hidden_size is not a multiple of num_attention_heads."));
            if (heads % kvHeads != 0)
            {
                throw new OnboardingConfigException($"config.json: num_attention_heads {heads} is not a multiple of num_key_value_heads {kvHeads}.");
            }

            string? dtypeName = _c.OptString("dtype") ?? _c.OptString("torch_dtype");
            DType dtype = dtypeName switch
            {
                "bfloat16" => DType.BF16,
                "float16" => DType.F16,
                "float32" => DType.F32,
                null => DType.BF16,
                _ => throw new OnboardingConfigException($"config.json: dtype '{dtypeName}' is not bfloat16, float16 or float32."),
            };
            if (dtypeName is null)
            {
                _warnings.Add("dtype and torch_dtype are absent: weights are assumed bfloat16.");
            }

            return new ModelDims
            {
                Hidden = hidden,
                Layers = layers,
                Heads = heads,
                KvHeads = kvHeads,
                HeadDim = Positive("head_dim", headDim),
                Intermediate = _c.OptInt("intermediate_size") ?? 0,
                Vocab = Positive("vocab_size", _c.RequiredInt("vocab_size")),
                DType = dtype.ToString(),
                TiedEmbeddings = _c.OptBool("tie_word_embeddings") ?? true,
            };
        }

        private static int Positive(string name, int value) =>
            value > 0 ? value : throw new OnboardingConfigException($"config.json: {name} is {value}; it must be positive.");

        private Mixer[] ClassifyLayers(int layers, out string?[] unknownTypes)
        {
            var mixers = new Mixer[layers];
            unknownTypes = new string?[layers];
            IReadOnlyList<string>? types = _c.OptStrings("layer_types");
            int? window = _c.OptInt("sliding_window");
            if (types is not null)
            {
                if (types.Count != layers)
                {
                    throw new OnboardingConfigException($"config.json: layer_types has {types.Count} entries for {layers} layers.");
                }

                for (int i = 0; i < layers; i++)
                {
                    mixers[i] = types[i] switch
                    {
                        "full_attention" => Mixer.Full,
                        "sliding_attention" => Mixer.Sliding,
                        "linear_attention" => Mixer.Linear,
                        _ => Mixer.Unknown,
                    };
                    if (mixers[i] == Mixer.Unknown)
                    {
                        unknownTypes[i] = types[i];
                    }
                }
            }
            else if (_c.Has("linear_num_value_heads") && _c.OptInt("full_attention_interval") is int interval and > 0)
            {
                // Qwen3-Next and Qwen3.5 without layer_types: full attention at i % interval == interval - 1.
                for (int i = 0; i < layers; i++)
                {
                    mixers[i] = i % interval == interval - 1 ? Mixer.Full : Mixer.Linear;
                }

                _notes.Add($"Layer types derived from full_attention_interval {interval}.");
            }
            else if (window is not null && _c.OptInt("sliding_window_pattern") is int pattern and > 0)
            {
                // Gemma 3 without layer_types: every pattern-th layer is global, the rest sliding.
                for (int i = 0; i < layers; i++)
                {
                    mixers[i] = (i + 1) % pattern == 0 ? Mixer.Full : Mixer.Sliding;
                }

                _notes.Add($"Layer types derived from sliding_window_pattern {pattern}.");
            }
            else if (window is not null && _c.OptBool("use_sliding_window") is bool useSliding)
            {
                // Qwen2 and Qwen3: with use_sliding_window, layers at or above max_window_layers use the window.
                int maxWindowLayers = _c.OptInt("max_window_layers") ?? layers;
                for (int i = 0; i < layers; i++)
                {
                    mixers[i] = useSliding && i >= maxWindowLayers ? Mixer.Sliding : Mixer.Full;
                }
            }
            else if (window is not null)
            {
                // Mistral: a sliding_window with no switch applies to every layer.
                Array.Fill(mixers, Mixer.Sliding);
                _notes.Add("sliding_window is set without use_sliding_window or a pattern: every layer is taken as sliding (Mistral's rule).");
            }

            if (_c.Has("kv_lora_rank"))
            {
                for (int i = 0; i < layers; i++)
                {
                    if (mixers[i] == Mixer.Full)
                    {
                        mixers[i] = Mixer.Mla;
                    }
                    else if (mixers[i] == Mixer.Sliding)
                    {
                        throw new OnboardingConfigException("config.json: kv_lora_rank (MLA) with sliding-window layers is not a combination inspect knows.");
                    }
                }
            }

            if (mixers.Any(m => m == Mixer.Sliding) && window is null)
            {
                throw new OnboardingConfigException("config.json: sliding_attention layers without a sliding_window.");
            }

            return mixers;
        }

        private MoeShape? ReadMoe(int layers, out bool[] isMoe)
        {
            isMoe = new bool[layers];
            int experts = _c.OptInt("num_experts") ?? _c.OptInt("num_local_experts") ?? _c.OptInt("n_routed_experts") ?? 0;
            if (experts <= 0)
            {
                return null;
            }

            int topK = _c.OptInt("num_experts_per_tok")
                ?? throw new OnboardingConfigException("config.json: experts without num_experts_per_tok.");
            if (topK < 1 || topK > experts)
            {
                throw new OnboardingConfigException($"config.json: num_experts_per_tok {topK} is not between 1 and the {experts} experts.");
            }

            if (_c.OptInt("first_k_dense_replace") is int firstDense)
            {
                // DeepSeek: dense MLPs below first_k_dense_replace, MoE at every moe_layer_freq-th layer above.
                int freq = Math.Max(1, _c.OptInt("moe_layer_freq") ?? 1);
                for (int i = 0; i < layers; i++)
                {
                    isMoe[i] = i >= firstDense && i % freq == 0;
                }
            }
            else
            {
                // Qwen MoE: MoE where (i + 1) % decoder_sparse_step == 0 and i is not in mlp_only_layers; Mixtral: every layer.
                int step = Math.Max(1, _c.OptInt("decoder_sparse_step") ?? 1);
                var dense = new HashSet<int>(_c.OptInts("mlp_only_layers") ?? []);
                for (int i = 0; i < layers; i++)
                {
                    isMoe[i] = !dense.Contains(i) && (i + 1) % step == 0;
                }
            }

            int shared = _c.OptInt("n_shared_experts") ?? ((_c.OptInt("shared_expert_intermediate_size") ?? 0) > 0 ? 1 : 0);
            return new MoeShape
            {
                Experts = experts,
                TopK = topK,
                ExpertIntermediate = _c.OptInt("moe_intermediate_size") ?? _c.OptInt("intermediate_size")
                    ?? throw new OnboardingConfigException("config.json: experts without moe_intermediate_size or intermediate_size."),
                SharedExperts = shared,
                Scoring = _c.OptString("scoring_func") ?? "softmax",
                Renormalize = _c.OptBool("norm_topk_prob") ?? false,
            };
        }

        private void MapEmbeddingAndNorms(ModelDims dims)
        {
            _ops.Need(OpId.Embedding, "embedding");
            if (_family.EmbedScale)
            {
                double scale = Math.Sqrt(dims.Hidden);
                _transforms.Add($"embedding: the hidden state is scaled by sqrt(hidden_size) = {HfConfig.Number(scale)}, which EmbeddingArgs has no field for; Embedding reads a copy of the table scaled by it at load (rounded to {dims.DType}, as transformers rounds the normaliser) and LmHead the unscaled table, costing vocab x hidden extra weight bytes when the embeddings are tied.");
            }

            bool layerNorm = !_c.Has("rms_norm_eps") && (_c.Has("layer_norm_eps") || _c.Has("layer_norm_epsilon"));
            if (layerNorm)
            {
                _ops.Missing("LayerNorm", "norm", "mean-subtracting LayerNorm (layer_norm_eps): C3 has RmsNorm only.");
            }
            else
            {
                _ops.Need(OpId.RmsNorm, "norm");
                if (!_c.Has("rms_norm_eps"))
                {
                    _warnings.Add("rms_norm_eps is absent; transformers' default for the family applies.");
                }
            }

            if (_family.ZeroCentredNorms)
            {
                _transforms.Add("norms: RMS norm weights are stored zero-centred and applied as 1 + w; the slots hold 1 + w rounded to the weight dtype (the Qwen3.5-0.8B definition's rule), so parity of those norms is to weight rounding.");
            }

            if (_family.SandwichNorms)
            {
                _notes.Add("Pre- and post-norms around attention and the MLP (sandwich norms) are extra RmsNorm calls.");
            }

            _ops.Need(OpId.Residual, "residual");
            _ops.Need(OpId.LmHead, "lm_head");
            if (_c.OptDouble("final_logit_softcapping") is double fcap)
            {
                _ops.Gap(OpId.LmHead, "lm_head", $"final logit soft-capping (final_logit_softcapping {HfConfig.Number(fcap)}): LmHeadArgs has no soft cap.");
            }

            _ops.Need(OpId.Argmax, "sampler");
            _ops.Need(OpId.Sample, "sampler");
        }

        private void MapMixers(ModelDims dims, Mixer[] mixers, string?[] unknownTypes)
        {
            if (mixers.Contains(Mixer.Full))
            {
                MapAttention(dims, "attention", null);
            }

            if (mixers.Contains(Mixer.Sliding))
            {
                MapAttention(dims, "sliding_attention", _c.OptInt("sliding_window"));
            }

            if (mixers.Contains(Mixer.Mla))
            {
                MapMla();
            }

            if (mixers.Contains(Mixer.Linear))
            {
                MapLinearAttention();
            }

            foreach (string type in unknownTypes.Where(t => t is not null).Distinct()!)
            {
                _ops.Missing("Mixer(" + type + ")", "layers", $"layer_types entry '{type}' is not full_attention, sliding_attention or linear_attention; inspect cannot map it.");
            }

            if (_c.OptBool("attention_bias") == true)
            {
                _notes.Add("attention_bias: q, k, v (and o) projection biases go through GemmArgs.Bias.");
            }
        }

        private void MapAttention(ModelDims dims, string usedBy, int? window)
        {
            _ops.Need(OpId.Gemm, usedBy);
            if (_family.QkNorm)
            {
                _ops.Need(OpId.QkNorm, usedBy);
            }

            MapRope(dims.HeadDim, usedBy);
            _ops.Need(OpId.CacheWrite, usedBy);
            _ops.Need(OpId.PagedAttention, usedBy);
            if (window is int w)
            {
                _notes.Add($"Sliding window {w}: PagedAttentionArgs.WindowLeft = {w - 1} over a Sliding state group that releases blocks leaving the window.");
            }

            if (_c.OptDouble("query_pre_attn_scalar") is double qs && qs > 0)
            {
                _notes.Add($"Attention scale query_pre_attn_scalar^-0.5 = {HfConfig.Number(1 / Math.Sqrt(qs))} (PagedAttentionArgs.Scale).");
            }

            bool gated = _c.OptBool("attn_output_gate") ?? _family.GatedAttention;
            if (gated)
            {
                _ops.Need(OpId.SigmoidMul, usedBy);
            }

            if (_c.OptDouble("attn_logit_softcapping") is double cap)
            {
                _ops.Gap(OpId.PagedAttention, usedBy, $"attention logit soft-capping (attn_logit_softcapping {HfConfig.Number(cap)}): PagedAttentionArgs has no soft cap.");
            }
        }

        private void MapRope(int rotaryHeadDim, string usedBy)
        {
            HfConfig? rp = _c.OptObject("rope_parameters") ?? _c.OptObject("rope_scaling");
            if (rp is { } p && (p.Has("full_attention") || p.Has("sliding_attention")))
            {
                // transformers 5 per-layer-type rope (Gemma 3): one parameter set per attention kind.
                string key = usedBy == "sliding_attention" ? "sliding_attention" : "full_attention";
                rp = p.OptObject(key);
            }

            string type = rp?.OptString("rope_type") ?? rp?.OptString("type") ?? "default";
            double? theta = rp?.OptDouble("rope_theta") ?? _c.OptDouble("rope_theta");
            if (usedBy == "sliding_attention" && _c.OptDouble("rope_local_base_freq") is double local)
            {
                theta = local;
                _notes.Add($"Sliding layers rotate with rope_local_base_freq {HfConfig.Number(local)}; Rope takes its theta per call.");
            }

            if (theta is null)
            {
                _warnings.Add($"{usedBy}: rope_theta is absent; transformers' default (10000) is assumed.");
            }

            switch (type)
            {
                case "default":
                    _ops.Need(OpId.Rope, usedBy);
                    break;
                case "llama3":
                    _ops.Need(OpId.Rope, usedBy);
                    _notes.Add("llama3 rope scaling: RopeArgs.Kind = Llama3.");
                    break;
                case "yarn" or "longrope" or "dynamic":
                    _ops.Need(OpId.RopeScaled, usedBy);
                    _notes.Add($"{type} rope scaling: RopeScaled with a precomputed cos/sin cache (attention scaling factors folded into the cache).");
                    break;
                case "linear":
                    _ops.Gap(OpId.Rope, usedBy, $"linear rope position scaling (factor {HfConfig.Number(rp?.OptDouble("factor") ?? 1)}): RopeArgs has the default and llama3 kinds, RopeScaledArgs YaRN, longrope and dynamic NTK.");
                    break;
                case "mrope":
                    _ops.Need(OpId.Rope, usedBy);
                    break;
                default:
                    _ops.Missing("Rope(" + type + ")", usedBy, $"rope type '{type}' is not mapped to a C3 rope.");
                    break;
            }

            if (rp is { } r && r.Has("mrope_section"))
            {
                _notes.Add("Multimodal rope (mrope_section): for a text-only prompt every section sees the same position, so it is the plain rotary embedding (the Qwen3.5-0.8B definition's reading).");
            }

            double? partial = rp?.OptDouble("partial_rotary_factor") ?? _c.OptDouble("partial_rotary_factor");
            if (partial is double f && f is > 0 and < 1)
            {
                double dim = rotaryHeadDim * f;
                if (dim != Math.Floor(dim) || (int)dim % 2 != 0)
                {
                    _ops.Gap(OpId.Rope, usedBy, $"partial_rotary_factor {HfConfig.Number(f)} of head_dim {rotaryHeadDim} is not an even whole number of dimensions.");
                }
                else
                {
                    _notes.Add($"{usedBy}: partial rotary embedding, RopeArgs.RotaryDim = {(int)dim} of {rotaryHeadDim}.");
                }
            }
        }

        private void MapMla()
        {
            int kvLora = _c.RequiredInt("kv_lora_rank");
            int rope = _c.RequiredInt("qk_rope_head_dim");
            _c.RequiredInt("qk_nope_head_dim");
            _c.RequiredInt("v_head_dim");
            _ops.Need(OpId.Gemm, "mla");
            _ops.Need(OpId.RmsNorm, "mla");
            MapRope(rope, "mla");
            if (_c.OptBool("rope_interleave") == true)
            {
                _notes.Add("mla: rope_interleave, RopeArgs.Interleaved = true on the rope slice.");
            }

            _ops.Gap(OpId.CacheWrite, "mla", $"writing the MLA latent row (kv_lora_rank {kvLora} + qk_rope_head_dim {rope} per token, no separate V): CacheWriteArgs writes paged K and V of KvHeads x HeadDim, and no C3 op is specified for the latent cache MlaAttention reads.");
            _ops.Need(OpId.MlaAttention, "mla");
            if (_c.Has("q_lora_rank"))
            {
                _notes.Add("mla: low-rank query (q_lora_rank) is two Gemm calls with an RmsNorm between.");
            }

            _transforms.Add("mla: the key and value up-projections (kv_b_proj) are absorbed into the query and output projections at load for latent attention.");
        }

        private void MapLinearAttention()
        {
            foreach (string f in (string[])["linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim"])
            {
                Positive(f, _c.RequiredInt(f));
            }

            const string U = "linear_attention";
            _ops.Need(OpId.Gemm, U);
            _ops.Need(OpId.CausalConv1d, U);
            _ops.Need(OpId.GatedDeltaRuleGates, U);
            _ops.Need(OpId.GatedDeltaRule, U);
            _ops.Need(OpId.GatedRmsNorm, U);
            _ops.Need(OpId.RecurrentStateCopy, U);
            _ops.Need(OpId.RecurrentStateSnapshot, U);
        }

        private void MapMlps(ModelDims dims, bool[] isMoe, MoeShape? moe)
        {
            if (isMoe.Any(m => !m))
            {
                if (dims.Intermediate <= 0)
                {
                    throw new OnboardingConfigException("config.json: dense MLP layers without intermediate_size.");
                }

                _ops.Need(OpId.Gemm, "mlp");
                NeedActivation("mlp");
            }

            if (moe is null)
            {
                return;
            }

            const string U = "moe";
            _ops.Need(OpId.Gemm, U);
            _ops.Need(OpId.MoeRouteTopK, U);
            if (moe.Scoring != "softmax")
            {
                _ops.Gap(OpId.MoeRouteTopK, U, $"router scoring_func '{moe.Scoring}': MoeRouteTopK computes softmax weights only.");
            }

            string? topkMethod = _c.OptString("topk_method");
            int groups = _c.OptInt("n_group") ?? 1;
            if (groups > 1 || topkMethod is "group_limited_greedy" or "noaux_tc")
            {
                _ops.Gap(OpId.MoeRouteTopK, U, $"group-limited expert choice (topk_method '{topkMethod ?? "greedy"}', n_group {groups}, topk_group {_c.OptInt("topk_group") ?? groups}): MoeRouteTopKArgs chooses over all experts.");
            }

            if (topkMethod == "noaux_tc")
            {
                _ops.Gap(OpId.MoeRouteTopK, U, "per-expert score correction bias (noaux_tc): MoeRouteTopKArgs has no bias.");
            }

            if (_c.OptDouble("routed_scaling_factor") is double rsf && rsf != 1)
            {
                _transforms.Add($"moe: routed_scaling_factor {HfConfig.Number(rsf)} multiplies every routed expert's output, so it is folded into the routed experts' down projections at load.");
            }

            if (moe.Renormalize)
            {
                _notes.Add("norm_topk_prob: MoeRouteTopKArgs.Renormalize = true.");
            }

            _ops.Need(OpId.MoePermute, U);
            _ops.Need(OpId.GroupedGemm, U);
            NeedActivation(U);
            _ops.Need(OpId.MoeUnpermute, U);
            if (moe.SharedExperts > 0)
            {
                _ops.Need(OpId.Gemm, "shared_expert");
                NeedActivation("shared_expert");
                if (_family.SharedExpertGate)
                {
                    _ops.Gap(OpId.SigmoidMul, "shared_expert", "shared expert gate sigmoid(shared_expert_gate(x)) is one value per token; SigmoidMulArgs multiplies by a gate of the same [Rows, Dim] shape.");
                }
            }
        }

        private void NeedActivation(string usedBy)
        {
            string? act = _c.OptString("hidden_activation") ?? _c.OptString("hidden_act");
            if (act is null)
            {
                _warnings.Add("hidden_act is absent: SiLU is assumed.");
                act = "silu";
            }

            switch (act)
            {
                case "silu" or "swish":
                    _ops.Need(OpId.SiluAndMul, usedBy);
                    break;
                case "gelu_pytorch_tanh" or "gelu_new":
                    _ops.Need(OpId.GeluAndMul, usedBy);
                    _notes.Add($"{act}: GeluAndMulArgs.Approximation = Tanh.");
                    break;
                case "gelu":
                    _ops.Need(OpId.GeluAndMul, usedBy);
                    break;
                default:
                    _ops.Missing("GatedActivation(" + act + ")", usedBy, $"activation '{act}' has no gated C3 op (SiluAndMul, GeluAndMul).");
                    break;
            }
        }

        private void MapQuantization()
        {
            if (_c.OptObject("quantization_config") is not { } q)
            {
                return;
            }

            string method = q.OptString("quant_method") ?? "unknown";
            switch (method)
            {
                case "fp8":
                    _ops.Need(OpId.GemmFp8, "quantization");
                    if (q.OptInts("weight_block_size") is { Count: > 0 } block)
                    {
                        _ops.Gap(OpId.GemmFp8, "quantization", $"block-wise FP8 weight scales [{string.Join(", ", block)}]: GemmFp8Args has activation and weight scale spans but no scale block shape.");
                    }

                    break;
                case "awq" or "gptq" when (q.OptInt("bits") ?? 4) == 4:
                    _ops.Need(OpId.GemmW4A16, "quantization");
                    break;
                default:
                    _ops.Missing("Quantization(" + method + ")", "quantization", $"quantization_config quant_method '{method}' (bits {q.OptInt("bits")?.ToString(CultureInfo.InvariantCulture) ?? "unspecified"}) is not mapped to a C3 GEMM.");
                    break;
            }
        }

        private void MapExtras()
        {
            int mtp = _c.OptInt("mtp_num_hidden_layers") ?? _c.OptInt("num_nextn_predict_layers") ?? 0;
            if (mtp > 0)
            {
                _notes.Add($"Multi-token-prediction layers ({mtp}) are not inspected; they serve speculative decoding only.");
            }
        }

        private List<StateGroupReport> BuildStateGroups(ModelDims dims, Mixer[] mixers, bool[] isMoe, MoeShape? moe)
        {
            var reports = new List<StateGroupReport>();
            DType weight = Enum.Parse<DType>(dims.DType);
            DType kv = options.KvDType ?? weight;
            int bs = options.BlockSize;

            void Add(StateGroupDescriptor d, IReadOnlyList<int> layers, IReadOnlyList<long> conv, bool optional) => reports.Add(new StateGroupReport
            {
                Index = d.Index,
                Name = d.Name,
                Kind = d.Kind.ToString(),
                Layers = layers,
                KvHeads = d.Layers.IsEmpty ? 0 : d.Layers[0].KvHeads,
                HeadSize = d.Layers.IsEmpty ? 0 : d.Layers[0].HeadSize,
                BlockSize = d.BlockSize,
                DType = d.DType.ToString(),
                BytesPerBlock = d.BytesPerBlock,
                Window = d.Window,
                RecurrentStateShape = d.RecurrentStateShape.ToArray(),
                ConvStateShape = conv,
                SnapshotSlots = d.SnapshotSlots,
                Optional = optional,
            });

            void Paged(string name, StateGroupKind kind, int[] layers, int kvHeads, int headSize, int roles, int window)
            {
                if (layers.Length == 0)
                {
                    return;
                }

                var d = new StateGroupDescriptor
                {
                    Index = reports.Count,
                    Name = name,
                    Kind = kind,
                    Layers = [.. layers.Select(l => new LayerKvShape(l, kvHeads, headSize))],
                    BlockSize = bs,
                    DType = kv,
                    BytesPerBlock = layers.Length * (long)roles * bs * kvHeads * headSize * Bytes(kv),
                    Window = window,
                };
                Add(d, layers, [], false);
            }

            Paged("attn", StateGroupKind.Full, Indices(mixers, Mixer.Full), dims.KvHeads, dims.HeadDim, 2, 0);
            Paged("sliding", StateGroupKind.Sliding, Indices(mixers, Mixer.Sliding), dims.KvHeads, dims.HeadDim, 2, _c.OptInt("sliding_window") ?? 0);
            int[] mla = Indices(mixers, Mixer.Mla);
            if (mla.Length > 0)
            {
                // One latent row per token and layer, keys and values together: kv_lora_rank + qk_rope_head_dim.
                Paged("mla", StateGroupKind.MlaLatent, mla, 1, _c.RequiredInt("kv_lora_rank") + _c.RequiredInt("qk_rope_head_dim"), 1, 0);
            }

            int[] linear = Indices(mixers, Mixer.Linear);
            if (linear.Length > 0)
            {
                // As the model compiler lays out a recurrent slot: the float32 delta-rule state [v_heads, k_dim, v_dim] plus the
                // convolution state [conv_dim, width - 1] in the weight dtype, per layer.
                long kHeads = _c.RequiredInt("linear_num_key_heads");
                long vHeads = _c.RequiredInt("linear_num_value_heads");
                long kDim = _c.RequiredInt("linear_key_head_dim");
                long vDim = _c.RequiredInt("linear_value_head_dim");
                long width = _c.RequiredInt("linear_conv_kernel_dim");
                long convDim = (2 * kHeads * kDim) + (vHeads * vDim);
                long[] state = [vHeads, kDim, vDim];
                long[] conv = [convDim, width - 1];
                long perLayer = (vHeads * kDim * vDim * Bytes(DType.F32)) + (convDim * (width - 1) * Bytes(weight));
                var d = new StateGroupDescriptor
                {
                    Index = reports.Count,
                    Name = "linear",
                    Kind = StateGroupKind.Recurrent,
                    Layers = [],
                    BlockSize = 1,
                    DType = DType.F32,
                    BytesPerBlock = linear.Length * perLayer,
                    RecurrentStateShape = [.. state],
                    SnapshotSlots = options.SnapshotSlots,
                };
                Add(d, linear, conv, false);
            }

            int[] moeLayers = Enumerable.Range(0, isMoe.Length).Where(i => isMoe[i]).ToArray();
            if (moe is not null && moeLayers.Length > 0)
            {
                // Expert paging (bet X6) is a placement choice: one block is one routed expert's gate, up and down weights
                // across every MoE layer, in the weight dtype (FP8 checkpoints: one byte, scales not counted).
                bool fp8 = _c.OptObject("quantization_config")?.OptString("quant_method") == "fp8";
                long perLayer = 3L * dims.Hidden * moe.ExpertIntermediate * (fp8 ? 1 : Bytes(weight));
                var d = new StateGroupDescriptor
                {
                    Index = reports.Count,
                    Name = "experts",
                    Kind = StateGroupKind.Expert,
                    Layers = [.. moeLayers.Select(l => new LayerKvShape(l, 0, 0))],
                    BlockSize = 1,
                    DType = fp8 ? DType.F8E4M3 : weight,
                    BytesPerBlock = moeLayers.Length * perLayer,
                };
                Add(d, moeLayers, [], true);
            }

            return reports;
        }

        private static int Bytes(DType t) => t switch
        {
            DType.F32 => 4,
            DType.F16 or DType.BF16 => 2,
            DType.F8E4M3 or DType.F8E5M2 or DType.I8 => 1,
            DType.F64 => 8,
            _ => throw new OnboardingConfigException($"No byte size for {t}."),
        };
    }
}

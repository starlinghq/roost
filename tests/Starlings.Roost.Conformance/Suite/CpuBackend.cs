using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Conformance;

/// <summary>
/// <c>CpuOps</c> (R-018) behind the backend interface. It is the reference of the randomized cases and, run as a candidate
/// itself, what the CI half checks the suite with. <see cref="Perturb"/> lets a test damage a result to prove the suite notices.
/// </summary>
public sealed class CpuBackend : IKernelBackend
{
    /// <summary>The variant id every op has in this backend.</summary>
    public const string VariantId = "cpu-reference";

    /// <summary>The M0 attention-side ops (R-017), which the independent oracle also implements.</summary>
    public static readonly OpId[] M0Ops =
    [
        OpId.RmsNorm, OpId.QkNorm, OpId.Rope, OpId.PagedAttention, OpId.SiluAndMul, OpId.Gemm, OpId.Embedding, OpId.LmHead,
        OpId.Residual, OpId.CacheWrite, OpId.CacheCopy,
    ];

    /// <summary>The hybrid ops of Qwen3.5 (R-034, contract version 3), checked against <c>CpuOps</c> only.</summary>
    public static readonly OpId[] HybridOps =
    [
        OpId.GatedDeltaRule, OpId.CausalConv1d, OpId.RecurrentStateCopy, OpId.RecurrentStateSnapshot, OpId.GatedDeltaRuleGates,
        OpId.GatedRmsNorm, OpId.SigmoidMul,
    ];

    /// <summary>The ops this backend lists as variants: every op that has cases.</summary>
    public static readonly OpId[] CoveredOps = [.. M0Ops, .. HybridOps];

    private readonly List<VariantInfo> _variants =
    [
        .. CoveredOps.Select(op => new VariantInfo(op, VariantId, VariantTraits.Deterministic | VariantTraits.BatchInvariant, default)),
    ];

    /// <summary>Test hook: called with the case and the result before it is returned.</summary>
    public Action<OpCase, ExecResult>? Perturb { get; set; }

    /// <summary>Number of calls made, by any variant.</summary>
    public int Calls { get; private set; }

    /// <inheritdoc/>
    public string Name => "CpuOps";

    /// <inheritdoc/>
    public int Sm => 0;

    /// <inheritdoc/>
    public IReadOnlyList<VariantInfo> Variants => _variants;

    /// <inheritdoc/>
    public int? ProbeShortStruct(string variant, OpCase c) => null;

    /// <inheritdoc/>
    public ExecResult Execute(string variant, OpCase c)
    {
        Calls++;
        var b = c.CloneBuffers();
        try
        {
            Run(c, b);
            var r = new ExecResult();
            foreach (string name in c.Outputs)
            {
                r.Outputs[name] = b[name];
                b.Remove(name);
            }

            Perturb?.Invoke(c, r);
            return r;
        }
        finally
        {
            foreach (HostBuf x in b.Values)
            {
                x.Dispose();
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
    }

    /// <summary>The q (0), k (1) or v (2) column view of the fused [tokens, 3 * row] GatedDeltaRule input, spanning exactly its reach.</summary>
    /// <param name="fused">The fused buffer.</param>
    /// <param name="part">0, 1 or 2.</param>
    /// <param name="c">The case.</param>
    /// <returns>The view.</returns>
    public static DeviceSpan GdnView(HostBuf fused, int part, OpCase c) => FusedView(fused.Span, part, c);

    /// <summary>The same view over any base span (host or device).</summary>
    /// <param name="whole">The fused buffer's span.</param>
    /// <param name="part">0, 1 or 2.</param>
    /// <param name="c">The case.</param>
    /// <returns>The view.</returns>
    public static DeviceSpan FusedView(DeviceSpan whole, int part, OpCase c)
    {
        long row = c.I("seq_row_width"), tokens = c.I("tokens");
        long length = tokens == 0 ? 0 : ((tokens - 1) * 3 * row) + row;
        return new DeviceSpan { Address = whole.Address + (nint)(part * row * Elem.Size(whole.Dtype)), Length = (ulong)length, Dtype = whole.Dtype };
    }

    /// <summary>The a and b inputs of GatedDeltaRuleGates: separate buffers, or column views of the fused buffer <c>ab</c>.</summary>
    /// <param name="b">The buffers.</param>
    /// <param name="c">The case.</param>
    /// <returns>a and b.</returns>
    public static (DeviceSpan A, DeviceSpan B) GateViews(Dictionary<string, HostBuf> b, OpCase c) =>
        c.I("stride") == 0 ? (b["a"].Span, b["b"].Span) : GateViews(b["ab"].Span, c);

    /// <summary>The column views of a fused a and b buffer, over any base span.</summary>
    /// <param name="ab">The fused span.</param>
    /// <param name="c">The case.</param>
    /// <returns>a and b.</returns>
    public static (DeviceSpan A, DeviceSpan B) GateViews(DeviceSpan ab, OpCase c)
    {
        int es = Elem.Size(ab.Dtype);
        int aCol = c.I("a_col"), bCol = c.I("b_col");
        return (
            new DeviceSpan { Address = ab.Address + (aCol * es), Length = ab.Length - (ulong)aCol, Dtype = ab.Dtype },
            new DeviceSpan { Address = ab.Address + (bCol * es), Length = ab.Length - (ulong)bCol, Dtype = ab.Dtype });
    }

    private static void Run(OpCase c, Dictionary<string, HostBuf> b)
    {
        DType dt = c.Dtype;
        switch (c.Op)
        {
            case OpId.RmsNorm:
                CpuOps.RmsNorm(new RmsNormArgs { Input = b["x"].Span, Weight = b["w"].Span, Output = b["out"].Span, Rows = c.I("rows"), Dim = c.I("dim"), Eps = c.F("eps"), DType = dt });
                break;
            case OpId.QkNorm:
                CpuOps.QkNorm(new QkNormArgs { Q = b["q"].Span, K = b["k"].Span, QWeight = b["qw"].Span, KWeight = b["kw"].Span, Tokens = c.I("tokens"), QHeads = c.I("q_heads"), KvHeads = c.I("kv_heads"), HeadDim = c.I("head_dim"), Eps = c.F("eps"), DType = dt });
                break;
            case OpId.Rope:
                CpuOps.Rope(new RopeArgs
                {
                    Q = b["q"].Span,
                    K = b["k"].Span,
                    Positions = b["pos"].Span,
                    Tokens = c.I("tokens"),
                    QHeads = c.I("q_heads"),
                    KvHeads = c.I("kv_heads"),
                    HeadDim = c.I("head_dim"),
                    RotaryDim = c.I("rotary_dim"),
                    Kind = (RopeKind)c.I("kind"),
                    Interleaved = c.I("interleaved") != 0,
                    Theta = c.F("theta"),
                    Llama3Factor = c.F("llama3_factor"),
                    Llama3LowFreqFactor = c.F("llama3_low_freq_factor"),
                    Llama3HighFreqFactor = c.F("llama3_high_freq_factor"),
                    Llama3OriginalMaxPosition = c.I("llama3_original_max_position"),
                    DType = dt,
                });
                break;
            case OpId.PagedAttention:
                CpuOps.PagedAttention(new PagedAttentionArgs
                {
                    Q = b["q"].Span,
                    Output = b["out"].Span,
                    KCache = b["k_cache"].Span,
                    VCache = b["v_cache"].Span,
                    QIndptr = b["q_indptr"].Span,
                    KvIndptr = b["kv_indptr"].Span,
                    KvIndices = b["kv_indices"].Span,
                    KvLastPageLength = b["last_page_len"].Span,
                    Mode = (AttentionMode)c.I("mode"),
                    Batch = c.I("batch"),
                    Tokens = c.I("tokens"),
                    QHeads = c.I("q_heads"),
                    KvHeads = c.I("kv_heads"),
                    HeadDim = c.I("head_dim"),
                    BlockSize = c.I("page_size"),
                    KvLayout = (TensorLayout)c.I("kv_layout"),
                    Scale = 0f,
                    SoftCap = 0f,
                    WindowLeft = -1,
                    Causal = c.I("causal") != 0,
                    KvDType = dt,
                    DType = dt,
                });
                break;
            case OpId.SiluAndMul:
                CpuOps.SiluAndMul(new SiluAndMulArgs { Input = b["x"].Span, Output = b["out"].Span, Rows = c.I("rows"), Dim = c.I("dim"), DType = dt });
                break;
            case OpId.Gemm:
                CpuOps.Gemm(new GemmArgs
                {
                    A = b["a"].Span,
                    B = b["b"].Span,
                    C = c.I("use_c") != 0 ? b["c"].Span : default,
                    D = b["d"].Span,
                    Bias = c.I("use_bias") != 0 ? b["bias"].Span : default,
                    M = c.I("m"),
                    N = c.I("n"),
                    K = c.I("k"),
                    TransposeB = c.I("trans_b") != 0,
                    Alpha = c.F("alpha"),
                    Beta = c.F("beta"),
                    ShapeBucket = c.I("bucket"),
                    DType = dt,
                    WeightDType = dt,
                });
                break;
            case OpId.Embedding:
                CpuOps.Embedding(new EmbeddingArgs { Ids = b["ids"].Span, Table = b["table"].Span, Output = b["out"].Span, Tokens = c.I("tokens"), Dim = c.I("dim"), Vocab = c.I("vocab"), DType = dt });
                break;
            case OpId.LmHead:
                CpuOps.LmHead(new LmHeadArgs { Hidden = b["hidden"].Span, Rows = b["rows"].Span, Weight = b["weight"].Span, Logits = b["logits"].Span, Selected = c.I("selected"), Dim = c.I("dim"), Vocab = c.I("vocab"), VocabPadded = c.I("vocab"), SoftCap = c.F("soft_cap"), DType = dt });
                break;
            case OpId.Residual:
                CpuOps.Residual(new ResidualArgs { A = b["a"].Span, B = b["b"].Span, Output = b["out"].Span, Elements = c.I("elements"), DType = dt });
                break;
            case OpId.CacheWrite:
                CpuOps.CacheWrite(new CacheWriteArgs { K = b["k"].Span, V = b["v"].Span, KCache = b["k_cache"].Span, VCache = b["v_cache"].Span, SlotMapping = b["slots"].Span, Tokens = c.I("tokens"), KvHeads = c.I("kv_heads"), HeadDim = c.I("head_dim"), BlockSize = c.I("page_size"), KvLayout = (TensorLayout)c.I("kv_layout"), KScale = 1f, VScale = 1f, KvDType = dt });
                break;
            case OpId.CacheCopy:
                {
                    HostBuf src = b["src"];
                    HostBuf dst = c.I("same_pool") != 0 ? src : b["dst"];
                    CpuOps.CacheCopy(new CacheCopyArgs { Source = src.Span, Destination = dst.Span, Pairs = b["pairs"].Span, BlockBytes = c.I("block_bytes"), Count = c.I("count") });
                    if (c.I("same_pool") != 0)
                    {
                        // The output is named "dst"; with one pool it is the pool itself.
                        b["dst"] = src.Clone();
                    }

                    break;
                }

            case OpId.GatedDeltaRule:
                {
                    bool strided = c.I("strided") != 0;
                    var a = new GatedDeltaRuleArgs
                    {
                        G = b["g"].Span,
                        Beta = b["beta"].Span,
                        Output = b["out"].Span,
                        State = b["state"].Span,
                        StateSlots = b["slots"].Span,
                        CuSeqlens = b["cu"].Span,
                        Batch = c.I("batch"),
                        Tokens = c.I("tokens"),
                        KHeads = c.I("q_heads"),
                        VHeads = c.I("kv_heads"),
                        KDim = c.I("head_dim"),
                        VDim = c.I("head_dim"),
                        UseInitialState = c.I("init") != 0,
                        L2NormQk = true,
                        Scale = 0,
                        DType = dt,
                    };
                    if (strided)
                    {
                        a.Q = GdnView(b["qkv"], 0, c);
                        a.K = GdnView(b["qkv"], 1, c);
                        a.V = GdnView(b["qkv"], 2, c);
                        a.QStride = a.KStride = a.VStride = 3 * c.I("seq_row_width");
                    }
                    else
                    {
                        a.Q = b["q"].Span;
                        a.K = b["k"].Span;
                        a.V = b["v"].Span;
                    }

                    CpuOps.GatedDeltaRule(a);
                    break;
                }

            case OpId.CausalConv1d:
                CpuOps.CausalConv1d(new CausalConv1dArgs
                {
                    Input = b["x"].Span,
                    Weight = b["w"].Span,
                    Bias = c.I("use_bias") != 0 ? b["bias"].Span : default,
                    Output = b["out"].Span,
                    ConvState = b["conv_state"].Span,
                    StateSlots = b["slots"].Span,
                    CuSeqlens = b["cu"].Span,
                    Batch = c.I("batch"),
                    Tokens = c.I("tokens"),
                    Channels = c.I("channels"),
                    Width = c.I("width"),
                    Silu = c.I("silu") != 0,
                    DType = dt,
                });
                break;
            case OpId.RecurrentStateCopy:
                CpuOps.RecurrentStateCopy(new RecurrentStateCopyArgs { Pool = b["pool"].Span, Pairs = b["pairs"].Span, SlotBytes = c.I("slot_bytes"), Count = c.I("count") });
                break;
            case OpId.RecurrentStateSnapshot:
                CpuOps.RecurrentStateSnapshot(new RecurrentStateSnapshotArgs { Pool = b["pool"].Span, Snapshots = b["snapshots"].Span, Pairs = b["pairs"].Span, SlotBytes = c.I("slot_bytes"), Count = c.I("count"), Restore = c.I("restore") != 0 });
                break;
            case OpId.GatedDeltaRuleGates:
                {
                    (DeviceSpan sa, DeviceSpan sb) = GateViews(b, c);
                    CpuOps.GatedDeltaRuleGates(new GatedDeltaRuleGatesArgs
                    {
                        A = sa,
                        B = sb,
                        ALog = b["a_log"].Span,
                        DtBias = b["dt_bias"].Span,
                        G = b["g"].Span,
                        Beta = b["beta"].Span,
                        Tokens = c.I("tokens"),
                        VHeads = 16,
                        AStride = c.I("stride"),
                        BStride = c.I("stride"),
                        DType = dt,
                    });
                    break;
                }

            case OpId.GatedRmsNorm:
                {
                    bool inPlace = c.I("in_place") != 0;
                    CpuOps.GatedRmsNorm(new GatedRmsNormArgs { Input = b["x"].Span, Gate = b["gate"].Span, Weight = b["w"].Span, Output = inPlace ? b["x"].Span : b["out"].Span, Rows = c.I("rows"), Dim = c.I("dim"), Eps = c.F("eps"), DType = dt });
                    if (inPlace)
                    {
                        b["x"].Raw().CopyTo(b["out"].Raw());
                    }

                    break;
                }

            case OpId.SigmoidMul:
                {
                    bool inPlace = c.I("in_place") != 0;
                    CpuOps.SigmoidMul(new SigmoidMulArgs { Input = b["x"].Span, Gate = b["gate"].Span, Output = inPlace ? b["x"].Span : b["out"].Span, Rows = c.I("rows"), Dim = c.I("dim"), DType = dt });
                    if (inPlace)
                    {
                        b["x"].Raw().CopyTo(b["out"].Raw());
                    }

                    break;
                }

            default:
                throw new NotSupportedException($"No CPU case runner for {c.Op}.");
        }
    }
}

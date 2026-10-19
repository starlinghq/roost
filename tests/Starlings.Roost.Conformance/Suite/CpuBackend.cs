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

    /// <summary>The ops this backend lists as variants: the M0 ops that have cases.</summary>
    public static readonly OpId[] CoveredOps =
    [
        OpId.RmsNorm, OpId.QkNorm, OpId.Rope, OpId.PagedAttention, OpId.SiluAndMul, OpId.Gemm, OpId.Embedding, OpId.LmHead,
        OpId.Residual, OpId.CacheWrite, OpId.CacheCopy,
    ];

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

            default:
                throw new NotSupportedException($"No CPU case runner for {c.Op}.");
        }
    }
}

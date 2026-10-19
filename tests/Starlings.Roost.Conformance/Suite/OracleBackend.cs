using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>
/// A second, independent implementation of the M0 ops, written from the op definitions in the C3 docs and the C1 header with
/// plain loops over <see cref="HostBuf.Get"/> and <see cref="HostBuf.Set"/>. It shares no code with <c>CpuOps</c>. The CI half
/// runs the suite with this backend as the candidate, so the reference every GPU result is judged by is itself checked on every
/// push, whether or not a goldens file is at hand.
/// </summary>
public sealed class OracleBackend : IKernelBackend
{
    /// <summary>The variant id.</summary>
    public const string VariantId = "oracle";

    private readonly List<VariantInfo> _variants =
    [
        .. CpuBackend.CoveredOps.Select(op => new VariantInfo(op, VariantId, VariantTraits.Deterministic | VariantTraits.BatchInvariant, default)),
    ];

    /// <inheritdoc/>
    public string Name => "Oracle";

    /// <inheritdoc/>
    public int Sm => 0;

    /// <inheritdoc/>
    public IReadOnlyList<VariantInfo> Variants => _variants;

    /// <inheritdoc/>
    public int? ProbeShortStruct(string variant, OpCase c) => null;

    /// <inheritdoc/>
    public void Dispose()
    {
    }

    /// <inheritdoc/>
    public ExecResult Execute(string variant, OpCase c)
    {
        var b = c.CloneBuffers();
        try
        {
            switch (c.Op)
            {
                case OpId.RmsNorm: RmsNorm(c, b); break;
                case OpId.QkNorm: QkNorm(c, b); break;
                case OpId.Rope: Rope(c, b); break;
                case OpId.PagedAttention: Attention(c, b); break;
                case OpId.SiluAndMul: Silu(c, b); break;
                case OpId.Gemm: Gemm(c, b); break;
                case OpId.Embedding: Embedding(c, b); break;
                case OpId.LmHead: LmHead(c, b); break;
                case OpId.Residual: Residual(c, b); break;
                case OpId.CacheWrite: CacheWrite(c, b); break;
                case OpId.CacheCopy: CacheCopy(c, b); break;
                default: throw new NotSupportedException(c.Op.ToString());
            }

            var r = new ExecResult();
            foreach (string name in c.Outputs)
            {
                r.Outputs[name] = b[name];
                b.Remove(name);
            }

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

    private static void NormRow(HostBuf x, long xOff, HostBuf w, HostBuf y, long yOff, int n, double eps)
    {
        double ss = 0;
        for (int i = 0; i < n; i++)
        {
            double v = x.Get(xOff + i);
            ss += v * v;
        }

        double inv = 1.0 / Math.Sqrt((ss / n) + eps);
        for (int i = 0; i < n; i++)
        {
            y.Set(yOff + i, x.Get(xOff + i) * inv * w.Get(i));
        }
    }

    private static void RmsNorm(OpCase c, Dictionary<string, HostBuf> b)
    {
        int rows = c.I("rows"), dim = c.I("dim");
        for (int r = 0; r < rows; r++)
        {
            NormRow(b["x"], (long)r * dim, b["w"], b["out"], (long)r * dim, dim, c.Num["eps"]);
        }
    }

    private static void QkNorm(OpCase c, Dictionary<string, HostBuf> b)
    {
        int hd = c.I("head_dim");
        for (int t = 0; t < c.I("tokens"); t++)
        {
            for (int h = 0; h < c.I("q_heads"); h++)
            {
                long off = (((long)t * c.I("q_heads")) + h) * hd;
                NormRow(b["q"], off, b["qw"], b["q"], off, hd, c.Num["eps"]);
            }

            for (int h = 0; h < c.I("kv_heads"); h++)
            {
                long off = (((long)t * c.I("kv_heads")) + h) * hd;
                NormRow(b["k"], off, b["kw"], b["k"], off, hd, c.Num["eps"]);
            }
        }
    }

    private static double InvFreq(OpCase c, int i, int rot)
    {
        double f = 1.0 / Math.Pow(c.Num["theta"], 2.0 * i / rot);
        if (c.I("kind") == (int)RopeKind.Default)
        {
            return f;
        }

        double orig = c.Num["llama3_original_max_position"], lowF = c.Num["llama3_low_freq_factor"], highF = c.Num["llama3_high_freq_factor"], factor = c.Num["llama3_factor"];
        double wavelength = 2 * Math.PI / f;
        if (wavelength < orig / highF)
        {
            return f;
        }

        if (wavelength > orig / lowF)
        {
            return f / factor;
        }

        double smooth = ((orig / wavelength) - lowF) / (highF - lowF);
        return (((1 - smooth) * f) / factor) + (smooth * f);
    }

    private static void Rope(OpCase c, Dictionary<string, HostBuf> b)
    {
        int hd = c.I("head_dim");
        int rot = c.I("rotary_dim") == 0 ? hd : c.I("rotary_dim");
        bool inter = c.I("interleaved") != 0;
        foreach ((string name, string heads) in new[] { ("q", "q_heads"), ("k", "kv_heads") })
        {
            HostBuf x = b[name];
            for (int t = 0; t < c.I("tokens"); t++)
            {
                double pos = b["pos"].Get(t);
                for (int h = 0; h < c.I(heads); h++)
                {
                    long off = (((long)t * c.I(heads)) + h) * hd;
                    for (int i = 0; i < rot / 2; i++)
                    {
                        double angle = pos * InvFreq(c, i, rot);
                        int p0 = inter ? 2 * i : i, p1 = inter ? (2 * i) + 1 : i + (rot / 2);
                        double a = x.Get(off + p0), d = x.Get(off + p1);
                        x.Set(off + p0, (a * Math.Cos(angle)) - (d * Math.Sin(angle)));
                        x.Set(off + p1, (d * Math.Cos(angle)) + (a * Math.Sin(angle)));
                    }
                }
            }
        }
    }

    private static long CacheIndex(TensorLayout layout, int ps, int kvHeads, int hd, long page, int slot, int head, int d)
        => layout == TensorLayout.PagedNhd
            ? ((((page * ps) + slot) * kvHeads) + head) * hd + d
            : ((((page * kvHeads) + head) * ps) + slot) * hd + d;

    private static void Attention(OpCase c, Dictionary<string, HostBuf> b)
    {
        int qh = c.I("q_heads"), kh = c.I("kv_heads"), hd = c.I("head_dim"), ps = c.I("page_size");
        var layout = (TensorLayout)c.I("kv_layout");
        bool causal = c.I("causal") != 0;
        double scale = 1.0 / Math.Sqrt(hd);
        int group = qh / kh;
        HostBuf qi = b["q_indptr"], ki = b["kv_indptr"], idx = b["kv_indices"], last = b["last_page_len"];
        for (int s = 0; s < c.I("batch"); s++)
        {
            int q0 = (int)qi.Get(s), q1 = (int)qi.Get(s + 1);
            int p0 = (int)ki.Get(s), p1 = (int)ki.Get(s + 1);
            int n = q1 - q0;
            int kvLen = ((p1 - p0 - 1) * ps) + (int)last.Get(s);
            for (int i = 0; i < n; i++)
            {
                int visible = (causal || c.I("mode") == (int)AttentionMode.Decode) ? kvLen - n + i + 1 : kvLen;
                if (c.I("mode") == (int)AttentionMode.Decode)
                {
                    visible = kvLen;
                }

                for (int h = 0; h < qh; h++)
                {
                    int g = h / group;
                    var score = new double[visible];
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < visible; j++)
                    {
                        long page = (long)idx.Get(p0 + (j / ps));
                        double dot = 0;
                        for (int d = 0; d < hd; d++)
                        {
                            dot += b["q"].Get(((((long)q0 + i) * qh) + h) * hd + d) * b["k_cache"].Get(CacheIndex(layout, ps, kh, hd, page, j % ps, g, d));
                        }

                        score[j] = dot * scale;
                        max = Math.Max(max, score[j]);
                    }

                    double sum = 0;
                    for (int j = 0; j < visible; j++)
                    {
                        score[j] = Math.Exp(score[j] - max);
                        sum += score[j];
                    }

                    for (int d = 0; d < hd; d++)
                    {
                        double acc = 0;
                        for (int j = 0; j < visible; j++)
                        {
                            long page = (long)idx.Get(p0 + (j / ps));
                            acc += score[j] * b["v_cache"].Get(CacheIndex(layout, ps, kh, hd, page, j % ps, g, d));
                        }

                        b["out"].Set(((((long)q0 + i) * qh) + h) * hd + d, acc / sum);
                    }
                }
            }
        }
    }

    private static void Silu(OpCase c, Dictionary<string, HostBuf> b)
    {
        int rows = c.I("rows"), dim = c.I("dim");
        for (int r = 0; r < rows; r++)
        {
            for (int d = 0; d < dim; d++)
            {
                double g = b["x"].Get((long)r * 2 * dim + d), u = b["x"].Get((long)r * 2 * dim + dim + d);
                b["out"].Set((long)r * dim + d, g / (1.0 + Math.Exp(-g)) * u);
            }
        }
    }

    private static void Gemm(OpCase c, Dictionary<string, HostBuf> b)
    {
        int m = c.I("m"), n = c.I("n"), k = c.I("k");
        bool tb = c.I("trans_b") != 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int p = 0; p < k; p++)
                {
                    acc += b["a"].Get((long)i * k + p) * b["b"].Get(tb ? ((long)j * k) + p : ((long)p * n) + j);
                }

                double v = c.Num["alpha"] * acc;
                if (c.I("use_c") != 0)
                {
                    v += c.Num["beta"] * b["c"].Get((long)i * n + j);
                }

                if (c.I("use_bias") != 0)
                {
                    v += b["bias"].Get(j);
                }

                b["d"].Set((long)i * n + j, v);
            }
        }
    }

    private static void Embedding(OpCase c, Dictionary<string, HostBuf> b)
    {
        int dim = c.I("dim");
        int es = Elem.Size(c.Dtype);
        for (int t = 0; t < c.I("tokens"); t++)
        {
            int id = (int)b["ids"].Get(t);
            b["table"].Raw().Slice(id * dim * es, dim * es).CopyTo(b["out"].Raw().Slice(t * dim * es, dim * es));
        }
    }

    private static void LmHead(OpCase c, Dictionary<string, HostBuf> b)
    {
        int dim = c.I("dim"), vocab = c.I("vocab");
        double cap = c.Num["soft_cap"];
        for (int s = 0; s < c.I("selected"); s++)
        {
            int row = (int)b["rows"].Get(s);
            for (int v = 0; v < vocab; v++)
            {
                double acc = 0;
                for (int d = 0; d < dim; d++)
                {
                    acc += b["hidden"].Get((long)row * dim + d) * b["weight"].Get((long)v * dim + d);
                }

                // The float32 output is formed from the double sum once; a cap is applied in double before that rounding.
                b["logits"].Set((long)s * vocab + v, cap > 0 ? cap * Math.Tanh(acc / cap) : acc);
            }
        }
    }

    private static void Residual(OpCase c, Dictionary<string, HostBuf> b)
    {
        for (long i = 0; i < c.I("elements"); i++)
        {
            b["out"].Set(i, b["a"].Get(i) + b["b"].Get(i));
        }
    }

    private static void CacheWrite(OpCase c, Dictionary<string, HostBuf> b)
    {
        int kh = c.I("kv_heads"), hd = c.I("head_dim"), ps = c.I("page_size");
        var layout = (TensorLayout)c.I("kv_layout");
        int es = Elem.Size(c.Dtype);
        for (int t = 0; t < c.I("tokens"); t++)
        {
            long slot = (long)b["slots"].Get(t);
            if (slot < 0)
            {
                continue;
            }

            for (int h = 0; h < kh; h++)
            {
                long dst = CacheIndex(layout, ps, kh, hd, slot / ps, (int)(slot % ps), h, 0);
                long src = (((long)t * kh) + h) * hd;
                b["k"].Raw().Slice((int)(src * es), hd * es).CopyTo(b["k_cache"].Raw().Slice((int)(dst * es), hd * es));
                b["v"].Raw().Slice((int)(src * es), hd * es).CopyTo(b["v_cache"].Raw().Slice((int)(dst * es), hd * es));
            }
        }
    }

    private static void CacheCopy(OpCase c, Dictionary<string, HostBuf> b)
    {
        int bb = c.I("block_bytes");
        HostBuf src = b["src"];
        HostBuf dst = c.I("same_pool") != 0 ? src : b["dst"];
        // Pairs are independent by construction, so the order in which they are applied does not matter.
        var snapshot = src.Clone();
        for (int p = 0; p < c.I("count"); p++)
        {
            int s = (int)b["pairs"].Get(2 * p), d = (int)b["pairs"].Get((2 * p) + 1);
            snapshot.Raw().Slice(s * bb, bb).CopyTo(dst.Raw().Slice(d * bb, bb));
        }

        snapshot.Dispose();
        if (c.I("same_pool") != 0)
        {
            b["dst"] = src.Clone();
        }
    }
}

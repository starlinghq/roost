using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>How many cases to generate.</summary>
/// <param name="Name">Profile name recorded in results.</param>
/// <param name="Large">Include the long-context attention batches and the full-size projections (slow on the CPU reference).</param>
/// <param name="RandomPerOp">Randomized cases per op and dtype, drawn from the run's base seed.</param>
public sealed record CaseProfile(string Name, bool Large, int RandomPerOp)
{
    /// <summary>Every fixed shape of the M0 model and 12 random cases per op and dtype.</summary>
    public static CaseProfile Full { get; } = new("full", true, 12);

    /// <summary>What CI runs against the CPU reference: small fixed shapes, the long batches left out, 4 random cases.</summary>
    public static CaseProfile Ci { get; } = new("ci", false, 4);

    /// <summary>What each compute-sanitizer tool runs per variant: every fixed shape except the slowest, 3 random cases.</summary>
    public static CaseProfile Sanitizer { get; } = new("sanitizer", false, 3);

    /// <summary>Looks a profile up by name.</summary>
    /// <param name="name">"full", "ci" or "sanitizer".</param>
    /// <returns>The profile.</returns>
    public static CaseProfile Named(string name) => name switch
    {
        "full" => Full,
        "ci" => Ci,
        "sanitizer" => Sanitizer,
        _ => throw new ArgumentException($"Unknown profile '{name}' (full, ci, sanitizer).", nameof(name)),
    };
}

/// <summary>
/// Generates the cases of every M0 op: fixed shapes of the Qwen3.5-0.8B full-attention layer (hidden 1024, 8 query heads over
/// 2 KV heads, head_dim 256, 64 rotary dimensions, theta 1e7, MLP 3584, rms_norm_eps 1e-6, config.json of
/// Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17) and randomized shapes and values. Every case carries the
/// seed it was drawn from, so a failure reproduces from its id and the run's base seed.
/// </summary>
public static class CaseCatalog
{
    private const float Eps = 1e-6f;
    private const int QHeads = 8, KvHeads = 2, HeadDim = 256;
    private static readonly DType[] Half = [DType.BF16, DType.F16];

    /// <summary>The ops this catalogue has cases for.</summary>
    public static IReadOnlyList<OpId> Ops => CpuBackend.CoveredOps;

    /// <summary>Tolerance class name of an elementwise op for an activation type.</summary>
    /// <param name="t">The type.</param>
    /// <returns>The class name.</returns>
    public static string Elementwise(DType t) => t switch
    {
        DType.BF16 => "elementwise-bf16",
        DType.F16 => "elementwise-f16",
        _ => "elementwise-f32",
    };

    /// <summary>Generates the cases of one op.</summary>
    /// <param name="op">The op.</param>
    /// <param name="baseSeed">The run's base seed; random cases derive theirs from it.</param>
    /// <param name="p">The profile.</param>
    /// <returns>The cases; the caller disposes them.</returns>
    public static IEnumerable<OpCase> Generate(OpId op, ulong baseSeed, CaseProfile p) => op switch
    {
        OpId.RmsNorm => RmsNormCases(baseSeed, p),
        OpId.QkNorm => QkNormCases(baseSeed, p),
        OpId.Rope => RopeCases(baseSeed, p),
        OpId.PagedAttention => AttentionCases(baseSeed, p),
        OpId.SiluAndMul => SiluCases(baseSeed, p),
        OpId.Gemm => GemmCases(baseSeed, p),
        OpId.Embedding => EmbeddingCases(baseSeed, p),
        OpId.LmHead => LmHeadCases(baseSeed, p),
        OpId.Residual => ResidualCases(baseSeed, p),
        OpId.CacheWrite => CacheWriteCases(baseSeed, p),
        OpId.CacheCopy => CacheCopyCases(baseSeed, p),
        _ => throw new NotSupportedException($"No cases for {op}."),
    };

    private static ulong SeedFor(OpId op, string id, ulong baseSeed) => Rng.SeedOf($"{op}/{id}", baseSeed);

    private static string Dim(params int[] v) => string.Join('x', v);

    // ---- RmsNorm ------------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> RmsNormCases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in Half)
        {
            foreach (int rows in new[] { 1, 7, 64, 257 })
            {
                yield return RmsNorm("fixed", dt, rows, 1024, 0, 0, baseSeed);
            }

            yield return RmsNorm("fixed-strided", dt, 5, 1024, 1280, 1152, baseSeed);
            yield return RmsNorm("fixed", dt, 3, 3584, 0, 0, baseSeed);
            var rng = new Rng(SeedFor(OpId.RmsNorm, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                int dim = 8 * rng.Int(8, 520);
                int rows = rng.Int(1, 300);
                bool strided = rng.Int(0, 3) == 0;
                yield return RmsNorm($"random-{i}", dt, rows, dim, strided ? dim + (8 * rng.Int(1, 40)) : 0, strided ? dim + (8 * rng.Int(1, 40)) : 0, baseSeed);
            }
        }
    }

    private static OpCase RmsNorm(string family, DType dt, int rows, int dim, int inStride, int outStride, ulong baseSeed)
    {
        string id = $"{family}-{dt}-{Dim(rows, dim)}{(inStride > 0 ? $"-s{inStride}-{outStride}" : string.Empty)}";
        ulong seed = SeedFor(OpId.RmsNorm, id, baseSeed);
        var rng = new Rng(seed);
        var x = new HostBuf(dt, (long)rows * dim).Randn(rng, 2.0);
        var w = new HostBuf(dt, dim).Randn(rng, 0.2, 1.0);
        return RmsNormFrom(id, dt, rows, dim, inStride, outStride, seed, x, w, singleRowOf: true);
    }

    private static OpCase RmsNormFrom(string id, DType dt, int rows, int dim, int inStride, int outStride, ulong seed, HostBuf x, HostBuf w, bool singleRowOf)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.RmsNorm,
            Dtype = dt,
            ToleranceClass = Elementwise(dt),
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = rows,
            SingleRow = singleRowOf
                ? r => RmsNormFrom(id + $"-row{r}", dt, 1, dim, 0, 0, seed, CopyRows(x, r, 1, dim), w.Clone(), false)
                : null,
        };
        c.Num["rows"] = rows;
        c.Num["dim"] = dim;
        c.Num["eps"] = Eps;
        c.Num["in_stride"] = inStride;
        c.Num["out_stride"] = outStride;
        c.Buffers["x"] = x;
        c.Buffers["w"] = w;
        c.Buffers["out"] = new HostBuf(dt, (long)rows * dim);
        c.Outputs.Add("out");
        c.RowWidth["out"] = dim;
        return c;
    }

    private static HostBuf CopyRows(HostBuf b, int row, int rows, int width)
    {
        var o = new HostBuf(b.Dtype, (long)rows * width);
        int es = Elem.Size(b.Dtype);
        b.Raw().Slice(row * width * es, rows * width * es).CopyTo(o.Raw());
        return o;
    }

    // ---- QkNorm -------------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> QkNormCases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in Half)
        {
            foreach (int tokens in new[] { 1, 5, 33 })
            {
                yield return QkNorm("fixed", dt, tokens, 0, 0, baseSeed);
            }

            yield return QkNorm("fixed-strided", dt, 9, 2560, 704, baseSeed);
            var rng = new Rng(SeedFor(OpId.QkNorm, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                bool strided = rng.Int(0, 3) == 0;
                yield return QkNorm($"random-{i}", dt, rng.Int(1, 70), strided ? (QHeads * HeadDim) + (8 * rng.Int(1, 30)) : 0, strided ? (KvHeads * HeadDim) + (8 * rng.Int(1, 30)) : 0, baseSeed);
            }
        }
    }

    private static OpCase QkNorm(string family, DType dt, int tokens, int qStride, int kStride, ulong baseSeed)
    {
        string id = $"{family}-{dt}-t{tokens}{(qStride > 0 ? $"-s{qStride}-{kStride}" : string.Empty)}";
        ulong seed = SeedFor(OpId.QkNorm, id, baseSeed);
        var rng = new Rng(seed);
        var q = new HostBuf(dt, (long)tokens * QHeads * HeadDim).Randn(rng, 1.5);
        var k = new HostBuf(dt, (long)tokens * KvHeads * HeadDim).Randn(rng, 1.5);
        var qw = new HostBuf(dt, HeadDim).Randn(rng, 0.2, 1.0);
        var kw = new HostBuf(dt, HeadDim).Randn(rng, 0.2, 1.0);
        return QkNormFrom(id, dt, tokens, qStride, kStride, seed, q, k, qw, kw, true);
    }

    private static OpCase QkNormFrom(string id, DType dt, int tokens, int qStride, int kStride, ulong seed, HostBuf q, HostBuf k, HostBuf qw, HostBuf kw, bool withRows)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.QkNorm,
            Dtype = dt,
            ToleranceClass = Elementwise(dt),
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = tokens,
            SingleRow = withRows
                ? t => QkNormFrom(id + $"-row{t}", dt, 1, 0, 0, seed, CopyRows(q, t, 1, QHeads * HeadDim), CopyRows(k, t, 1, KvHeads * HeadDim), qw.Clone(), kw.Clone(), false)
                : null,
        };
        c.Num["tokens"] = tokens;
        c.Num["q_heads"] = QHeads;
        c.Num["kv_heads"] = KvHeads;
        c.Num["head_dim"] = HeadDim;
        c.Num["eps"] = Eps;
        c.Num["q_stride"] = qStride;
        c.Num["k_stride"] = kStride;
        c.Buffers["q"] = q;
        c.Buffers["k"] = k;
        c.Buffers["qw"] = qw;
        c.Buffers["kw"] = kw;
        c.Outputs.AddRange(["q", "k"]);
        c.RowWidth["q"] = QHeads * HeadDim;
        c.RowWidth["k"] = KvHeads * HeadDim;
        return c;
    }

    // ---- Rope ---------------------------------------------------------------------------------------------------

    private sealed record RopeFamily(string Name, int QHeads, int KvHeads, int HeadDim, int Rotary, RopeKind Kind, bool Interleaved, float Theta, int MaxPos);

    private static readonly RopeFamily[] RopeFamilies =
    [
        new("qwen35", 8, 2, 256, 64, RopeKind.Default, false, 1e7f, 4096),
        new("qwen35-longpos", 8, 2, 256, 64, RopeKind.Default, false, 1e7f, 262144),
        new("full-rotary", 8, 2, 256, 0, RopeKind.Default, false, 1e6f, 4096),
        new("interleaved", 8, 2, 256, 64, RopeKind.Default, true, 1e7f, 4096),
        new("hd64", 8, 2, 64, 0, RopeKind.Default, false, 1e4f, 4096),
        new("hd128", 16, 8, 128, 0, RopeKind.Default, false, 1e6f, 4096),
        new("llama3-hd128", 8, 2, 128, 0, RopeKind.Llama3, false, 5e5f, 8192),
        new("llama3-hd256-rot64", 8, 2, 256, 64, RopeKind.Llama3, false, 5e5f, 8192),
        new("huge-positions", 8, 2, 256, 64, RopeKind.Default, false, 1e7f, int.MaxValue),
        new("llama3-longpos", 8, 2, 128, 0, RopeKind.Llama3, false, 5e5f, 131072),
    ];

    private static IEnumerable<OpCase> RopeCases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in Half)
        {
            foreach (RopeFamily f in RopeFamilies)
            {
                yield return Rope("fixed", dt, f, f.Name == "qwen35" ? 17 : 33, baseSeed);
            }

            yield return Rope("fixed", dt, RopeFamilies[0], 1, baseSeed);
            yield return Rope("fixed", dt, RopeFamilies[0], 129, baseSeed);
            var rng = new Rng(SeedFor(OpId.Rope, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                yield return Rope($"random-{i}", dt, rng.Pick(RopeFamilies), rng.Int(1, 140), baseSeed);
            }
        }
    }

    private static OpCase Rope(string family, DType dt, RopeFamily f, int tokens, ulong baseSeed)
    {
        string id = $"{family}-{f.Name}-{dt}-t{tokens}";
        ulong seed = SeedFor(OpId.Rope, id, baseSeed);
        var rng = new Rng(seed);
        var q = new HostBuf(dt, (long)tokens * f.QHeads * f.HeadDim).Randn(rng);
        var k = new HostBuf(dt, (long)tokens * f.KvHeads * f.HeadDim).Randn(rng);
        var pos = new HostBuf(DType.I32, tokens);
        for (int i = 0; i < tokens; i++)
        {
            pos.As<int>()[i] = rng.Int(0, f.MaxPos);
        }

        return RopeFrom(id, dt, f, tokens, seed, q, k, pos, true);
    }

    private static OpCase RopeFrom(string id, DType dt, RopeFamily f, int tokens, ulong seed, HostBuf q, HostBuf k, HostBuf pos, bool withRows)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.Rope,
            Dtype = dt,
            ToleranceClass = Elementwise(dt),
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = tokens,
            SingleRow = withRows
                ? t => RopeFrom(id + $"-row{t}", dt, f, 1, seed, CopyRows(q, t, 1, f.QHeads * f.HeadDim), CopyRows(k, t, 1, f.KvHeads * f.HeadDim), CopyRows(pos, t, 1, 1), false)
                : null,
        };
        c.Num["tokens"] = tokens;
        c.Num["q_heads"] = f.QHeads;
        c.Num["kv_heads"] = f.KvHeads;
        c.Num["head_dim"] = f.HeadDim;
        c.Num["rotary_dim"] = f.Rotary;
        c.Num["kind"] = (double)f.Kind;
        c.Num["interleaved"] = f.Interleaved ? 1 : 0;
        c.Num["theta"] = f.Theta;
        c.Num["llama3_factor"] = 8;
        c.Num["llama3_low_freq_factor"] = 1;
        c.Num["llama3_high_freq_factor"] = 4;
        c.Num["llama3_original_max_position"] = 8192;
        c.Buffers["q"] = q;
        c.Buffers["k"] = k;
        c.Buffers["pos"] = pos;
        c.Outputs.AddRange(["q", "k"]);
        c.RowWidth["q"] = f.QHeads * f.HeadDim;
        c.RowWidth["k"] = f.KvHeads * f.HeadDim;
        return c;
    }

    // ---- PagedAttention -----------------------------------------------------------------------------------------

    private sealed record Seq(int QLen, int KvLen);

    private static IEnumerable<OpCase> AttentionCases(ulong baseSeed, CaseProfile p)
    {
        foreach ((TensorLayout layout, int ps) in new[] { (TensorLayout.PagedNhd, 16), (TensorLayout.PagedHnd, 16), (TensorLayout.PagedNhd, 32), (TensorLayout.PagedHnd, 64) })
        {
            yield return Attention($"decode-mixed4-{layout}-ps{ps}", AttentionMode.Decode, Decode(5, 37, 130, 16), ps, layout, true, baseSeed);
        }

        yield return Attention("decode-single-token-history", AttentionMode.Decode, Decode(1), 16, TensorLayout.PagedNhd, true, baseSeed);
        yield return Attention("decode-page-multiples", AttentionMode.Decode, Decode(16, 32, 48, 64), 16, TensorLayout.PagedNhd, true, baseSeed);
        yield return Attention("prefill-one-token", AttentionMode.Prefill, [new(1, 1)], 16, TensorLayout.PagedNhd, true, baseSeed);
        yield return Attention("prefill-fresh-100", AttentionMode.Prefill, [new(100, 100)], 16, TensorLayout.PagedNhd, true, baseSeed);
        yield return Attention("prefill-fresh-100-hnd", AttentionMode.Prefill, [new(100, 100)], 16, TensorLayout.PagedHnd, true, baseSeed);
        yield return Attention("prefill-chunk-64-of-300", AttentionMode.Prefill, [new(64, 300)], 16, TensorLayout.PagedNhd, true, baseSeed);
        yield return Attention("prefill-chunk-32-of-200-hnd-ps32", AttentionMode.Prefill, [new(32, 200)], 32, TensorLayout.PagedHnd, true, baseSeed);
        yield return Attention("prefill-multi-seq", AttentionMode.Prefill, [new(17, 17), new(64, 300), new(1, 50), new(60, 60)], 16, TensorLayout.PagedNhd, true, baseSeed);
        yield return Attention("prefill-non-causal-32-of-200", AttentionMode.Prefill, [new(32, 200)], 16, TensorLayout.PagedNhd, false, baseSeed);
        yield return Attention("prefill-partial-last-page", AttentionMode.Prefill, [new(33, 33), new(5, 21)], 16, TensorLayout.PagedNhd, true, baseSeed);
        if (p.Large)
        {
            yield return Attention("decode-single-6000-split-kv", AttentionMode.Decode, Decode(6000), 16, TensorLayout.PagedNhd, true, baseSeed);
            yield return Attention("decode-mixed8", AttentionMode.Decode, Decode(900, 17, 1, 2048, 333, 64, 1024, 777), 16, TensorLayout.PagedNhd, true, baseSeed);
            yield return Attention("decode-mixed8-hnd-ps64", AttentionMode.Decode, Decode(900, 17, 1, 2048, 333, 64, 1024, 777), 64, TensorLayout.PagedHnd, true, baseSeed);
            yield return Attention("prefill-fresh-700-ps64", AttentionMode.Prefill, [new(700, 700)], 64, TensorLayout.PagedNhd, true, baseSeed);
            yield return Attention("prefill-fresh-1024", AttentionMode.Prefill, [new(1024, 1024)], 16, TensorLayout.PagedNhd, true, baseSeed);
            yield return Attention("prefill-chunk-128-of-1000", AttentionMode.Prefill, [new(128, 1000)], 16, TensorLayout.PagedNhd, true, baseSeed);
        }

        var rng = new Rng(SeedFor(OpId.PagedAttention, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            int ps = rng.Pick(16, 32, 64);
            TensorLayout layout = rng.Pick(TensorLayout.PagedNhd, TensorLayout.PagedHnd);
            int batch = rng.Int(1, 6);
            int cap = p.Large ? 700 : 150;
            if (rng.Int(0, 2) == 0)
            {
                yield return Attention($"random-{i}-decode", AttentionMode.Decode, Enumerable.Range(0, batch).Select(_ => new Seq(1, rng.Int(1, cap))).ToArray(), ps, layout, true, baseSeed);
            }
            else
            {
                var seqs = Enumerable.Range(0, batch).Select(_ =>
                {
                    int kv = rng.Int(1, cap);
                    return new Seq(rng.Int(1, kv + 1), kv);
                }).ToArray();
                yield return Attention($"random-{i}-prefill", AttentionMode.Prefill, seqs, ps, layout, rng.Int(0, 4) != 0, baseSeed);
            }
        }
    }

    /// <summary>
    /// One sequence of a multi-sequence attention case as a call of its own: its query rows, its page list and last page length,
    /// over the same K/V pool, so the same bytes of cache are read. Used for the batch-composition check of prefill.
    /// </summary>
    /// <param name="c">A PagedAttention case.</param>
    /// <param name="s">Sequence index.</param>
    /// <returns>The single-sequence case.</returns>
    public static OpCase AttentionSequence(OpCase c, int s)
    {
        int[] qIndptr = c.Buffers["q_indptr"].As<int>().ToArray();
        int[] kvIndptr = c.Buffers["kv_indptr"].As<int>().ToArray();
        int[] pages = c.Buffers["kv_indices"].As<int>().Slice(kvIndptr[s], kvIndptr[s + 1] - kvIndptr[s]).ToArray();
        int last = c.Buffers["last_page_len"].As<int>()[s];
        int tokens = qIndptr[s + 1] - qIndptr[s];
        int width = QHeads * HeadDim;
        var one = new OpCase
        {
            Id = c.Id + $"-seq{s}",
            Op = OpId.PagedAttention,
            Dtype = c.Dtype,
            ToleranceClass = c.ToleranceClass,
            Source = c.Source,
            Seed = c.Seed,
            ScaleBuffer = c.ScaleBuffer,
        };
        foreach (KeyValuePair<string, double> kv in c.Num)
        {
            one.Num[kv.Key] = kv.Value;
        }

        one.Num["batch"] = 1;
        one.Num["tokens"] = tokens;
        one.Buffers["q"] = CopyRows(c.Buffers["q"], qIndptr[s], tokens, width);
        one.Buffers["k_cache"] = c.Buffers["k_cache"].Clone();
        one.Buffers["v_cache"] = c.Buffers["v_cache"].Clone();
        one.Buffers["q_indptr"] = HostBuf.Ints(0, tokens);
        one.Buffers["kv_indptr"] = HostBuf.Ints(0, pages.Length);
        one.Buffers["kv_indices"] = HostBuf.Ints(pages);
        one.Buffers["last_page_len"] = HostBuf.Ints(last);
        one.Buffers["out"] = new HostBuf(c.Dtype, (long)tokens * width);
        one.Outputs.Add("out");
        return one;
    }

    private static Seq[] Decode(params int[] kvLens) => kvLens.Select(l => new Seq(1, l)).ToArray();

    private static OpCase Attention(string label, AttentionMode mode, Seq[] seqs, int ps, TensorLayout layout, bool causal, ulong baseSeed)
    {
        string id = $"{label}-q{string.Join('.', seqs.Take(4).Select(s => s.QLen))}-kv{string.Join('.', seqs.Take(4).Select(s => s.KvLen))}{(seqs.Length > 4 ? "+" : string.Empty)}-{(causal ? "causal" : "full")}-{layout}-ps{ps}";
        ulong seed = SeedFor(OpId.PagedAttention, id, baseSeed);
        var rng = new Rng(seed);
        int batch = seqs.Length;
        int tokens = seqs.Sum(s => s.QLen);
        int[] pagesPer = seqs.Select(s => (s.KvLen + ps - 1) / ps).ToArray();
        int total = pagesPer.Sum();
        int poolPages = total + 3;
        int[] perm = Enumerable.Range(0, poolPages).ToArray();
        for (int i = perm.Length - 1; i > 0; i--)
        {
            int j = rng.Int(0, i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }

        var qIndptr = new int[batch + 1];
        var kvIndptr = new int[batch + 1];
        var last = new int[batch];
        for (int b = 0; b < batch; b++)
        {
            qIndptr[b + 1] = qIndptr[b] + seqs[b].QLen;
            kvIndptr[b + 1] = kvIndptr[b] + pagesPer[b];
            last[b] = seqs[b].KvLen - ((pagesPer[b] - 1) * ps);
        }

        long poolElems = (long)poolPages * ps * KvHeads * HeadDim;
        var c = new OpCase
        {
            Id = id,
            Op = OpId.PagedAttention,
            Dtype = DType.BF16,
            ToleranceClass = "attention-bf16",
            Source = label.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            ScaleBuffer = "v_cache",
        };
        c.Num["mode"] = (double)mode;
        c.Num["batch"] = batch;
        c.Num["tokens"] = tokens;
        c.Num["q_heads"] = QHeads;
        c.Num["kv_heads"] = KvHeads;
        c.Num["head_dim"] = HeadDim;
        c.Num["page_size"] = ps;
        c.Num["kv_layout"] = (double)layout;
        c.Num["causal"] = causal ? 1 : 0;
        c.Buffers["q"] = new HostBuf(DType.BF16, (long)tokens * QHeads * HeadDim).Randn(rng);
        c.Buffers["k_cache"] = new HostBuf(DType.BF16, poolElems).Randn(rng);
        c.Buffers["v_cache"] = new HostBuf(DType.BF16, poolElems).Randn(rng);
        c.Buffers["q_indptr"] = HostBuf.Ints(qIndptr);
        c.Buffers["kv_indptr"] = HostBuf.Ints(kvIndptr);
        c.Buffers["kv_indices"] = HostBuf.Ints(perm.Take(total).ToArray());
        c.Buffers["last_page_len"] = HostBuf.Ints(last);
        c.Buffers["out"] = new HostBuf(DType.BF16, (long)tokens * QHeads * HeadDim);
        c.Outputs.Add("out");
        return c;
    }

    // ---- SiluAndMul ---------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> SiluCases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in Half)
        {
            foreach ((int rows, int dim) in new[] { (1, 3584), (8, 3584), (33, 3584), (3, 1000), (3, 1001) })
            {
                yield return Silu("fixed", dt, rows, dim, baseSeed);
            }

            var rng = new Rng(SeedFor(OpId.SiluAndMul, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                yield return Silu($"random-{i}", dt, rng.Int(1, 200), rng.Int(0, 3) == 0 ? rng.Int(1, 2000) : 8 * rng.Int(1, 500), baseSeed);
            }
        }
    }

    private static OpCase Silu(string family, DType dt, int rows, int dim, ulong baseSeed)
    {
        string id = $"{family}-{dt}-{Dim(rows, dim)}";
        ulong seed = SeedFor(OpId.SiluAndMul, id, baseSeed);
        var x = new HostBuf(dt, (long)rows * 2 * dim).Randn(new Rng(seed), 2.0);
        return SiluFrom(id, dt, rows, dim, seed, x, true);
    }

    private static OpCase SiluFrom(string id, DType dt, int rows, int dim, ulong seed, HostBuf x, bool withRows)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.SiluAndMul,
            Dtype = dt,
            ToleranceClass = Elementwise(dt),
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = rows,
            SingleRow = withRows ? r => SiluFrom(id + $"-row{r}", dt, 1, dim, seed, CopyRows(x, r, 1, 2 * dim), false) : null,
        };
        c.Num["rows"] = rows;
        c.Num["dim"] = dim;
        c.Buffers["x"] = x;
        c.Buffers["out"] = new HostBuf(dt, (long)rows * dim);
        c.Outputs.Add("out");
        c.RowWidth["out"] = dim;
        return c;
    }

    // ---- Gemm ---------------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> GemmCases(ulong baseSeed, CaseProfile p)
    {
        // Projections of a full-attention layer (hidden 1024, q_proj 2 x 8 x 256 with the output gate, kv 2 x 256, MLP 3584).
        (string Name, int N, int K)[] shapes = p.Large
            ? [("q_proj", 4096, 1024), ("k_proj", 512, 1024), ("o_proj", 1024, 2048), ("gate_up", 7168, 1024), ("down", 1024, 3584)]
            : [("k_proj", 256, 512), ("o_proj", 512, 1024)];
        foreach (DType dt in Half)
        {
            foreach ((string name, int n, int k) in shapes)
            {
                foreach (int m in p.Large ? new[] { 1, 3, 17, 64 } : new[] { 1, 3, 17 })
                {
                    yield return Gemm($"fixed-{name}", dt, m, n, k, true, false, 1f, 0f, 0, baseSeed);
                }
            }

            (int n1, int k1) = p.Large ? (1024, 1792) : (256, 512);
            (int n2, int k2) = p.Large ? (1024, 2048) : (256, 512);
            yield return Gemm("fixed-nontransposed", dt, 17, n1, k1, false, false, 1f, 0f, 0, baseSeed);
            yield return Gemm("fixed-bias", dt, 17, n2, k2, true, true, 1f, 0f, 0, baseSeed);
            yield return Gemm("fixed-alpha-beta-c", dt, 17, n2, k2, true, false, 0.5f, 0.25f, 0, baseSeed);
            yield return Gemm("fixed-alpha-beta-c-bias-bucket64", dt, 5, p.Large ? 512 : 128, p.Large ? 1024 : 256, false, true, 2f, 1f, 64, baseSeed);
            yield return Gemm("fixed-small-k8", dt, 4, 64, 8, true, false, 1f, 0f, 0, baseSeed);

            // The epilogue matrix: every combination of alpha, beta * C and bias, on one shape whose partial sums reach several
            // units, so that an intermediate rounded to the 16-bit output type before the epilogue is finished shows.
            foreach (int bucket in new[] { 0, 49 })
            {
                foreach ((float alpha, float beta, bool bias) in new[] { (0.5f, 0f, true), (1f, 1f, true), (0.5f, 1f, false), (0.5f, 1f, true) })
                {
                    yield return Gemm("fixed-epilogue", dt, 39, 256, 200, true, bias, alpha, beta, bucket, baseSeed);
                }
            }
            var rng = new Rng(SeedFor(OpId.Gemm, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                int m = rng.Int(1, 70);
                int bucket = rng.Int(0, 2) == 0 ? 0 : m + rng.Int(0, 40);
                yield return Gemm($"random-{i}", dt, m, 8 * rng.Int(1, p.Large ? 130 : 40), 8 * rng.Int(1, p.Large ? 200 : 60), rng.Int(0, 3) != 0, rng.Int(0, 2) == 0, rng.Pick(1f, 0.5f, 2f), rng.Pick(0f, 0f, 0.25f, 1f), bucket, baseSeed);
            }
        }
    }

    private static OpCase Gemm(string family, DType dt, int m, int n, int k, bool transB, bool bias, float alpha, float beta, int bucket, ulong baseSeed)
    {
        string id = $"{family}-{dt}-{Dim(m, n, k)}-tb{(transB ? 1 : 0)}-bias{(bias ? 1 : 0)}-ab{alpha}.{beta}-bucket{bucket}";
        ulong seed = SeedFor(OpId.Gemm, id, baseSeed);
        var rng = new Rng(seed);
        var a = new HostBuf(dt, (long)m * k).Randn(rng);
        var b = new HostBuf(dt, (long)n * k).Randn(rng, 1.0 / Math.Sqrt(k));
        var bi = new HostBuf(dt, n).Randn(rng, 0.5);
        var cc = new HostBuf(dt, (long)m * n).Randn(rng);
        return GemmFrom(id, dt, m, n, k, transB, bias, alpha, beta, bucket, seed, a, b, bi, cc, true);
    }

    private static OpCase GemmFrom(string id, DType dt, int m, int n, int k, bool transB, bool bias, float alpha, float beta, int bucket, ulong seed, HostBuf a, HostBuf b, HostBuf bi, HostBuf cc, bool withRows)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.Gemm,
            Dtype = dt,
            ToleranceClass = dt == DType.BF16 ? "gemm-bf16" : "gemm-f16",
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = m,
            // A row of A (and C) alone, planned for the same bucket: pinned algorithms must give the batch result bit for bit.
            SingleRow = withRows
                ? r => GemmFrom(id + $"-row{r}", dt, 1, n, k, transB, bias, alpha, beta, bucket == 0 ? m : bucket, seed, CopyRows(a, r, 1, k), b.Clone(), bi.Clone(), CopyRows(cc, r, 1, n), false)
                : null,
        };
        c.Num["m"] = m;
        c.Num["n"] = n;
        c.Num["k"] = k;
        c.Num["trans_b"] = transB ? 1 : 0;
        c.Num["alpha"] = alpha;
        c.Num["beta"] = beta;
        c.Num["bucket"] = bucket;
        c.Num["use_bias"] = bias ? 1 : 0;
        c.Num["use_c"] = beta != 0f ? 1 : 0;
        c.Buffers["a"] = a;
        c.Buffers["b"] = b;
        c.Buffers["bias"] = bi;
        c.Buffers["c"] = cc;
        c.Buffers["d"] = new HostBuf(dt, (long)m * n);
        c.Outputs.Add("d");
        c.RowWidth["d"] = n;
        return c;
    }

    // ---- Embedding ----------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> EmbeddingCases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in new[] { DType.BF16, DType.F16, DType.F32 })
        {
            int vocab = p.Large ? 4096 : 512;
            yield return Embedding("fixed", dt, vocab, 1024, 1, baseSeed);
            yield return Embedding("fixed", dt, vocab, 1024, 300, baseSeed);
            if (dt == DType.BF16)
            {
                yield return Embedding("fixed", dt, vocab, 1001, 17, baseSeed);
                yield return Embedding("fixed", dt, vocab, 1002, 17, baseSeed);
            }

            var rng = new Rng(SeedFor(OpId.Embedding, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                yield return Embedding($"random-{i}", dt, rng.Int(2, 3000), rng.Int(0, 2) == 0 ? rng.Int(1, 1500) : 8 * rng.Int(1, 200), rng.Int(1, 200), baseSeed);
            }
        }
    }

    private static OpCase Embedding(string family, DType dt, int vocab, int dim, int tokens, ulong baseSeed)
    {
        string id = $"{family}-{dt}-v{vocab}-d{dim}-t{tokens}";
        ulong seed = SeedFor(OpId.Embedding, id, baseSeed);
        var rng = new Rng(seed);
        var table = new HostBuf(dt, (long)vocab * dim).Randn(rng);
        var ids = new HostBuf(DType.I32, tokens);
        for (int i = 0; i < tokens; i++)
        {
            ids.As<int>()[i] = i % 7 == 0 ? vocab - 1 : rng.Int(0, vocab);
        }

        return EmbeddingFrom(id, dt, vocab, dim, tokens, seed, table, ids, true);
    }

    private static OpCase EmbeddingFrom(string id, DType dt, int vocab, int dim, int tokens, ulong seed, HostBuf table, HostBuf ids, bool withRows)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.Embedding,
            Dtype = dt,
            ToleranceClass = "exact",
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = tokens,
            SingleRow = withRows ? t => EmbeddingFrom(id + $"-row{t}", dt, vocab, dim, 1, seed, table.Clone(), CopyRows(ids, t, 1, 1), false) : null,
        };
        c.Num["tokens"] = tokens;
        c.Num["dim"] = dim;
        c.Num["vocab"] = vocab;
        c.Buffers["ids"] = ids;
        c.Buffers["table"] = table;
        c.Buffers["out"] = new HostBuf(dt, (long)tokens * dim);
        c.Outputs.Add("out");
        c.RowWidth["out"] = dim;
        return c;
    }

    // ---- LmHead -------------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> LmHeadCases(ulong baseSeed, CaseProfile p)
    {
        yield return LmHead("fixed", DType.BF16, 4096, 1024, 4096, 1, 0f, baseSeed);
        yield return LmHead("fixed", DType.BF16, 4096, 1024, 4096, 3, 0f, baseSeed);
        yield return LmHead("fixed", DType.BF16, 4000, 1024, 4096, 5, 30f, baseSeed);
        yield return LmHead("fixed", DType.F16, 4096, 1024, 4128, 2, 0f, baseSeed);
        var rng = new Rng(SeedFor(OpId.LmHead, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            int vocab = 8 * rng.Int(8, 600);
            yield return LmHead($"random-{i}", rng.Pick(DType.BF16, DType.F16), vocab, 8 * rng.Int(8, 160), vocab + (8 * rng.Int(0, 4)), rng.Int(1, 9), rng.Pick(0f, 0f, 30f), baseSeed);
        }
    }

    private static OpCase LmHead(string family, DType dt, int vocab, int dim, int padded, int selected, float softCap, ulong baseSeed)
    {
        string id = $"{family}-{dt}-v{vocab}-d{dim}-pad{padded}-sel{selected}-cap{softCap}";
        ulong seed = SeedFor(OpId.LmHead, id, baseSeed);
        var rng = new Rng(seed);
        int tokens = selected + 3;
        var weight = new HostBuf(dt, (long)vocab * dim).Randn(rng, 0.05);
        var hidden = new HostBuf(dt, (long)tokens * dim).Randn(rng);
        var rows = new HostBuf(DType.I32, selected);
        for (int i = 0; i < selected; i++)
        {
            rows.As<int>()[i] = tokens - 1 - i;
        }

        return LmHeadFrom(id, dt, vocab, dim, padded, selected, softCap, tokens, seed, weight, hidden, rows, 0, true);
    }

    private static OpCase LmHeadFrom(string id, DType dt, int vocab, int dim, int padded, int selected, float softCap, int tokens, ulong seed, HostBuf weight, HostBuf hidden, HostBuf rows, int bucket, bool withRows)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.LmHead,
            Dtype = dt,
            ToleranceClass = "lm-head-f32",
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = selected,
            SingleRow = withRows
                ? r => LmHeadFrom(id + $"-row{r}", dt, vocab, dim, padded, 1, softCap, tokens, seed, weight.Clone(), hidden.Clone(), CopyRows(rows, r, 1, 1), selected, false)
                : null,
        };
        c.Num["selected"] = selected;
        c.Num["dim"] = dim;
        c.Num["vocab"] = vocab;
        c.Num["vocab_padded"] = padded;
        c.Num["soft_cap"] = softCap;
        c.Num["tokens"] = tokens;
        c.Num["bucket"] = bucket;
        c.Buffers["hidden"] = hidden;
        c.Buffers["rows"] = rows;
        c.Buffers["weight"] = weight;
        c.Buffers["logits"] = new HostBuf(DType.F32, (long)selected * vocab);
        c.Outputs.Add("logits");
        c.RowWidth["logits"] = vocab;
        return c;
    }

    // ---- Residual -----------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> ResidualCases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in new[] { DType.F32, DType.BF16, DType.F16 })
        {
            yield return Residual("fixed", dt, 1024, baseSeed);
            yield return Residual("fixed", dt, 7 * 1024, baseSeed);
            yield return Residual("fixed", dt, 1001, baseSeed);
            var rng = new Rng(SeedFor(OpId.Residual, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                yield return Residual($"random-{i}", dt, rng.Int(1, 100000), baseSeed);
            }
        }
    }

    private static OpCase Residual(string family, DType dt, int elements, ulong baseSeed)
    {
        string id = $"{family}-{dt}-n{elements}";
        ulong seed = SeedFor(OpId.Residual, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase
        {
            Id = id,
            Op = OpId.Residual,
            Dtype = dt,
            ToleranceClass = Elementwise(dt),
            Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
        };
        c.Num["elements"] = elements;
        c.Buffers["a"] = new HostBuf(dt, elements).Randn(rng);
        c.Buffers["b"] = new HostBuf(dt, elements).Randn(rng);
        c.Buffers["out"] = new HostBuf(dt, elements);
        c.Outputs.Add("out");
        return c;
    }

    // ---- CacheWrite ---------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> CacheWriteCases(ulong baseSeed, CaseProfile p)
    {
        foreach (TensorLayout layout in new[] { TensorLayout.PagedNhd, TensorLayout.PagedHnd })
        {
            foreach (int ps in new[] { 16, 32, 64 })
            {
                yield return CacheWrite("fixed", DType.BF16, layout, ps, 37, baseSeed);
            }
        }

        yield return CacheWrite("fixed", DType.F32, TensorLayout.PagedNhd, 16, 9, baseSeed);
        yield return CacheWrite("fixed", DType.F16, TensorLayout.PagedHnd, 32, 12, baseSeed);
        var rng = new Rng(SeedFor(OpId.CacheWrite, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            yield return CacheWrite($"random-{i}", rng.Pick(DType.BF16, DType.F16, DType.F32), rng.Pick(TensorLayout.PagedNhd, TensorLayout.PagedHnd), rng.Pick(16, 32, 64), rng.Int(1, 90), baseSeed);
        }
    }

    private static OpCase CacheWrite(string family, DType dt, TensorLayout layout, int ps, int tokens, ulong baseSeed)
    {
        string id = $"{family}-{dt}-{layout}-ps{ps}-t{tokens}";
        ulong seed = SeedFor(OpId.CacheWrite, id, baseSeed);
        var rng = new Rng(seed);
        const int kh = 2, hd = 256;
        int pages = 40;
        long poolElems = (long)pages * ps * kh * hd;
        var slots = new long[tokens];
        var used = new HashSet<long>();
        for (int t = 0; t < tokens; t++)
        {
            long s;
            do
            {
                s = rng.Int(0, pages * ps);
            }
            while (!used.Add(s));
            slots[t] = t % 11 == 5 ? -1 : s;
        }

        var c = new OpCase
        {
            Id = id,
            Op = OpId.CacheWrite,
            Dtype = dt,
            ToleranceClass = "exact",
            Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
        };
        c.Num["tokens"] = tokens;
        c.Num["kv_heads"] = kh;
        c.Num["head_dim"] = hd;
        c.Num["page_size"] = ps;
        c.Num["kv_layout"] = (double)layout;
        c.Buffers["k"] = new HostBuf(dt, (long)tokens * kh * hd).Randn(rng);
        c.Buffers["v"] = new HostBuf(dt, (long)tokens * kh * hd).Randn(rng);
        c.Buffers["k_cache"] = new HostBuf(dt, poolElems).FillBytes(0xAB);
        c.Buffers["v_cache"] = new HostBuf(dt, poolElems).FillBytes(0xAB);
        c.Buffers["slots"] = HostBuf.Longs(slots);
        c.Outputs.AddRange(["k_cache", "v_cache"]);
        return c;
    }

    // ---- CacheCopy ----------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> CacheCopyCases(ulong baseSeed, CaseProfile p)
    {
        yield return CacheCopy("fixed", 16384, true, 24, baseSeed);
        yield return CacheCopy("fixed", 16384, false, 24, baseSeed);
        yield return CacheCopy("fixed", 1001, true, 24, baseSeed);
        var rng = new Rng(SeedFor(OpId.CacheCopy, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            yield return CacheCopy($"random-{i}", rng.Pick(64, 256, 1001, 4096, 16384), rng.Int(0, 2) == 0, rng.Int(8, 40), baseSeed);
        }
    }

    private static OpCase CacheCopy(string family, int blockBytes, bool same, int blocks, ulong baseSeed)
    {
        string id = $"{family}-bb{blockBytes}-{(same ? "one-pool" : "two-pools")}-n{blocks}";
        ulong seed = SeedFor(OpId.CacheCopy, id, baseSeed);
        var rng = new Rng(seed);
        long bytes = (long)blocks * blockBytes;
        var src = new HostBuf(DType.U8, bytes);
        for (long i = 0; i < bytes; i++)
        {
            src.As<byte>()[(int)i] = (byte)rng.Int(0, 256);
        }

        // Independent pairs (the CPU reference copies in order, the kernel in parallel): distinct sources and distinct,
        // never-a-source destinations, plus one self pair.
        int count = Math.Min(blocks / 3, 8);
        var order = Enumerable.Range(0, blocks).OrderBy(_ => rng.Int(0, 1 << 20)).ToArray();
        var pairs = new List<int>();
        for (int i = 0; i < count; i++)
        {
            pairs.Add(order[i]);
            pairs.Add(order[count + i]);
        }

        pairs.Add(order[2 * count]);
        pairs.Add(order[2 * count]);
        var c = new OpCase
        {
            Id = id,
            Op = OpId.CacheCopy,
            Dtype = DType.U8,
            ToleranceClass = "exact",
            Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
        };
        c.Num["block_bytes"] = blockBytes;
        c.Num["count"] = count + 1;
        c.Num["same_pool"] = same ? 1 : 0;
        c.Num["blocks"] = blocks;
        c.Buffers["src"] = src;
        if (!same)
        {
            c.Buffers["dst"] = new HostBuf(DType.U8, bytes).FillBytes(0xAB);
        }

        c.Buffers["pairs"] = HostBuf.Ints([.. pairs]);
        c.Outputs.Add("dst");
        return c;
    }
}

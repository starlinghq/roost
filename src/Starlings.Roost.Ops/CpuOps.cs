using Starlings.Roost.Abi;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

/// <summary>
/// The CPU reference backend (R-018): every M0 op of <see cref="IRoostOps"/> plus GatedDeltaRule, CausalConv1d and the
/// recurrent-state copies, written as the plainest correct definition. Buffers are host addresses.
/// </summary>
/// <remarks>
/// <para>
/// Precision: each element is read as a double (bf16, fp16 and fp32 inputs are upcast exactly, integers are read
/// exactly), all arithmetic is in double, and each result is rounded once, to nearest even, into the output element
/// type of its span. Element types are taken from the DType fields of the arguments; a span's own Dtype is used
/// only where the arguments name no type for it (GatedDeltaRule G and Beta, CausalConv1d state).
/// </para>
/// <para>
/// Allocation: calls allocate no managed memory. Small scratch is on the stack; scratch that scales with a model
/// dimension (vocabulary, recurrent state) is native memory freed before the call returns. Calls do not synchronise
/// and are single-threaded. Spans are bounds-checked against their lengths and an out-of-range index or an
/// undersized span throws <see cref="ArgumentException"/> before any write that depends on it.
/// </para>
/// <para>Inputs and outputs of one call must not overlap unless the op says so (Residual A and Output, RmsNorm Input and Output, in-place QkNorm and Rope).</para>
/// </remarks>
public readonly partial struct CpuOps : IRoostOps
{
    private const int MaxHeadDim = 1024;

    /// <inheritdoc cref="IRoostOps.Supports(OpId)"/>
    /// <remarks>The M0 ops plus the four recurrent-hybrid ops of Qwen3.5-0.8B.</remarks>
    public static bool Supports(OpId op) =>
        OpCatalog.IsM0(op)
        || op is OpId.GatedDeltaRule or OpId.CausalConv1d or OpId.RecurrentStateCopy or OpId.RecurrentStateSnapshot;

    /// <inheritdoc/>
    /// <remarks>Per row: <c>mean = sum(x^2) / Dim</c>, <c>Output = x / sqrt(mean + Eps) * Weight</c>, in double. Input may alias Output.</remarks>
    public static void RmsNorm(in RmsNormArgs a)
    {
        long n = (long)a.Rows * a.Dim;
        Need(a.Input, n, nameof(a.Input));
        Need(a.Weight, a.Dim, nameof(a.Weight));
        Need(a.Output, n, nameof(a.Output));
        DType t = a.DType;
        for (int r = 0; r < a.Rows; r++)
        {
            long o = (long)r * a.Dim;
            NormRow(a.Input.Address, t, o, a.Weight.Address, t, a.Dim, a.Eps, a.Output.Address, t, o);
        }
    }

    /// <inheritdoc/>
    /// <remarks>RmsNorm over HeadDim of every query head with QWeight and every key head with KWeight, in place.</remarks>
    public static void QkNorm(in QkNormArgs a)
    {
        Need(a.Q, (long)a.Tokens * a.QHeads * a.HeadDim, nameof(a.Q));
        Need(a.K, (long)a.Tokens * a.KvHeads * a.HeadDim, nameof(a.K));
        Need(a.QWeight, a.HeadDim, nameof(a.QWeight));
        Need(a.KWeight, a.HeadDim, nameof(a.KWeight));
        DType t = a.DType;
        long qRows = (long)a.Tokens * a.QHeads;
        for (long r = 0; r < qRows; r++)
        {
            long o = r * a.HeadDim;
            NormRow(a.Q.Address, t, o, a.QWeight.Address, t, a.HeadDim, a.Eps, a.Q.Address, t, o);
        }

        long kRows = (long)a.Tokens * a.KvHeads;
        for (long r = 0; r < kRows; r++)
        {
            long o = r * a.HeadDim;
            NormRow(a.K.Address, t, o, a.KWeight.Address, t, a.HeadDim, a.Eps, a.K.Address, t, o);
        }
    }

    private static void NormRow(nint x, DType xt, long xo, nint w, DType wt, int dim, float eps, nint y, DType yt, long yo)
    {
        double ss = 0;
        for (int i = 0; i < dim; i++)
        {
            double v = Load(x, xt, xo + i);
            ss += v * v;
        }

        double inv = 1.0 / Math.Sqrt((ss / dim) + eps);
        for (int i = 0; i < dim; i++)
        {
            Store(y, yt, yo + i, Load(x, xt, xo + i) * inv * Load(w, wt, i));
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>Rotary dimensions are R = RotaryDim (0 means HeadDim), frequencies <c>f_i = Theta^(-2i/R)</c> for i in [0, R/2), angle <c>Positions[t] * f_i</c>.
    /// NeoX form pairs element i with i + R/2; the interleaved (GPT-J) form pairs 2i with 2i + 1. A pair (x1, x2) becomes
    /// <c>(x1 cos - x2 sin, x2 cos + x1 sin)</c>. Elements at and beyond R are unchanged.</para>
    /// <para>Llama 3 scaling: wavelength <c>2 pi / f_i</c>; below <c>OriginalMax / HighFreqFactor</c> the frequency is kept, above
    /// <c>OriginalMax / LowFreqFactor</c> it is divided by Llama3Factor, between them it blends linearly in <c>OriginalMax / wavelength</c>.</para>
    /// </remarks>
    public static void Rope(in RopeArgs a)
    {
        int hd = a.HeadDim;
        int rot = a.RotaryDim == 0 ? hd : a.RotaryDim;
        if (rot < 0 || rot > hd || (rot & 1) != 0)
        {
            throw new ArgumentException("RotaryDim must be even and at most HeadDim.", nameof(a));
        }

        Need(a.Q, (long)a.Tokens * a.QHeads * hd, nameof(a.Q));
        Need(a.K, (long)a.Tokens * a.KvHeads * hd, nameof(a.K));
        Need(a.Positions, a.Tokens, nameof(a.Positions));
        DType t = a.DType;
        int half = rot / 2;
        for (int tok = 0; tok < a.Tokens; tok++)
        {
            double pos = LoadInt(a.Positions.Address, a.Positions.Dtype, tok);
            for (int i = 0; i < half; i++)
            {
                double f = RopeFrequency(a, i, rot);
                double ang = pos * f;
                double c = Math.Cos(ang);
                double s = Math.Sin(ang);
                int ia = a.Interleaved ? 2 * i : i;
                int ib = a.Interleaved ? (2 * i) + 1 : i + half;
                RotatePair(a.Q.Address, t, (long)tok * a.QHeads * hd, a.QHeads, hd, ia, ib, c, s);
                RotatePair(a.K.Address, t, (long)tok * a.KvHeads * hd, a.KvHeads, hd, ia, ib, c, s);
            }
        }
    }

    private static double RopeFrequency(in RopeArgs a, int i, int rot)
    {
        double f = Math.Pow(a.Theta, -2.0 * i / rot);
        if (a.Kind != RopeKind.Llama3)
        {
            return f;
        }

        double wavelen = 2.0 * Math.PI / f;
        double low = a.Llama3OriginalMaxPosition / (double)a.Llama3LowFreqFactor;
        double high = a.Llama3OriginalMaxPosition / (double)a.Llama3HighFreqFactor;
        if (wavelen < high)
        {
            return f;
        }

        if (wavelen > low)
        {
            return f / a.Llama3Factor;
        }

        double smooth = ((a.Llama3OriginalMaxPosition / wavelen) - a.Llama3LowFreqFactor) / (a.Llama3HighFreqFactor - a.Llama3LowFreqFactor);
        return ((1.0 - smooth) * f / a.Llama3Factor) + (smooth * f);
    }

    private static void RotatePair(nint p, DType t, long baseIdx, int heads, int hd, int ia, int ib, double c, double s)
    {
        for (int h = 0; h < heads; h++)
        {
            long o = baseIdx + ((long)h * hd);
            double x1 = Load(p, t, o + ia);
            double x2 = Load(p, t, o + ib);
            Store(p, t, o + ia, (x1 * c) - (x2 * s));
            Store(p, t, o + ib, (x2 * c) + (x1 * s));
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>Sequence b has <c>QIndptr[b+1] - QIndptr[b]</c> query tokens (n) and <c>kvLen = (pages - 1) * BlockSize + KvLastPageLength[b]</c>
    /// keys, pages being <c>KvIndptr[b+1] - KvIndptr[b]</c>. Query i sits at key position <c>kvLen - n + i</c> (the last n keys).
    /// Key j is visible when j &lt;= that position (only if Causal; a decode query sees every key) and, when WindowLeft is not
    /// negative, j &gt;= position - WindowLeft.</para>
    /// <para>Score = <c>q . k * Scale</c> (Scale 0 means 1/sqrt(HeadDim)); if SoftCap is not 0 the score becomes <c>SoftCap * tanh(score / SoftCap)</c>.
    /// Softmax and the weighted sum of V are in double with an online maximum; query head h reads KV head <c>h / (QHeads / KvHeads)</c>.
    /// A query that sees no key outputs zeros. Pools are indexed as PagedNhd or PagedHnd and read in KvDType. HeadDim is at most 1024.</para>
    /// </remarks>
    public static void PagedAttention(in PagedAttentionArgs a)
    {
        int hd = a.HeadDim;
        if (hd < 1 || hd > MaxHeadDim)
        {
            throw new ArgumentException($"HeadDim must be in [1, {MaxHeadDim}].", nameof(a));
        }

        if (a.KvHeads < 1 || a.QHeads % a.KvHeads != 0)
        {
            throw new ArgumentException("QHeads must be a multiple of KvHeads.", nameof(a));
        }

        if (a.BlockSize < 1)
        {
            throw new ArgumentException("BlockSize must be positive.", nameof(a));
        }

        bool hnd = a.KvLayout switch
        {
            TensorLayout.PagedHnd => true,
            TensorLayout.PagedNhd => false,
            _ => throw new ArgumentException("KvLayout must be PagedNhd or PagedHnd.", nameof(a)),
        };

        Need(a.Q, (long)a.Tokens * a.QHeads * hd, nameof(a.Q));
        Need(a.Output, (long)a.Tokens * a.QHeads * hd, nameof(a.Output));
        Need(a.QIndptr, a.Batch + 1L, nameof(a.QIndptr));
        Need(a.KvIndptr, a.Batch + 1L, nameof(a.KvIndptr));
        Need(a.KvLastPageLength, a.Batch, nameof(a.KvLastPageLength));
        double scale = a.Scale == 0f ? 1.0 / Math.Sqrt(hd) : a.Scale;
        int group = a.QHeads / a.KvHeads;
        DType qt = a.DType;
        DType kt = a.KvDType;
        Span<double> acc = stackalloc double[MaxHeadDim];
        for (int b = 0; b < a.Batch; b++)
        {
            long q0 = LoadInt(a.QIndptr.Address, a.QIndptr.Dtype, b);
            long q1 = LoadInt(a.QIndptr.Address, a.QIndptr.Dtype, b + 1L);
            long p0 = LoadInt(a.KvIndptr.Address, a.KvIndptr.Dtype, b);
            long p1 = LoadInt(a.KvIndptr.Address, a.KvIndptr.Dtype, b + 1L);
            long pages = p1 - p0;
            long n = q1 - q0;
            if (n < 0 || pages < 0 || q1 > a.Tokens)
            {
                throw new ArgumentException($"Sequence {b} has an invalid QIndptr or KvIndptr range.", nameof(a));
            }

            long last = LoadInt(a.KvLastPageLength.Address, a.KvLastPageLength.Dtype, b);
            long kvLen = pages == 0 ? 0 : ((pages - 1) * a.BlockSize) + last;
            if (kvLen < n && a.Causal)
            {
                throw new ArgumentException($"Sequence {b} has {n} query tokens but only {kvLen} keys.", nameof(a));
            }

            Need(a.KvIndices, p1, nameof(a.KvIndices));
            for (long i = 0; i < n; i++)
            {
                long qpos = kvLen - n + i;
                long hi = a.Causal ? qpos : kvLen - 1;
                long lo = a.WindowLeft >= 0 ? Math.Max(0, qpos - a.WindowLeft) : 0;
                for (int h = 0; h < a.QHeads; h++)
                {
                    int kvh = h / group;
                    long qo = (((q0 + i) * a.QHeads) + h) * hd;
                    double m = double.NegativeInfinity;
                    double denom = 0;
                    acc[..hd].Clear();
                    for (long j = lo; j <= hi; j++)
                    {
                        long block = LoadInt(a.KvIndices.Address, a.KvIndices.Dtype, p0 + (j / a.BlockSize));
                        int off = (int)(j % a.BlockSize);
                        long row = hnd
                            ? ((((block * a.KvHeads) + kvh) * a.BlockSize) + off) * hd
                            : ((((block * a.BlockSize) + off) * a.KvHeads) + kvh) * hd;
                        NeedRange(a.KCache, row, hd, nameof(a.KCache));
                        NeedRange(a.VCache, row, hd, nameof(a.VCache));
                        double s = 0;
                        for (int d = 0; d < hd; d++)
                        {
                            s += Load(a.Q.Address, qt, qo + d) * Load(a.KCache.Address, kt, row + d);
                        }

                        s *= scale;
                        if (a.SoftCap != 0f)
                        {
                            s = a.SoftCap * Math.Tanh(s / a.SoftCap);
                        }

                        double newM = Math.Max(m, s);
                        double corr = double.IsNegativeInfinity(m) ? 0.0 : Math.Exp(m - newM);
                        double w = Math.Exp(s - newM);
                        denom = (denom * corr) + w;
                        for (int d = 0; d < hd; d++)
                        {
                            acc[d] = (acc[d] * corr) + (w * Load(a.VCache.Address, kt, row + d));
                        }

                        m = newM;
                    }

                    for (int d = 0; d < hd; d++)
                    {
                        Store(a.Output.Address, qt, qo + d, denom == 0 ? 0.0 : acc[d] / denom);
                    }
                }
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>Input row is <c>[gate | up]</c>, each Dim wide: <c>Output = gate / (1 + exp(-gate)) * up</c>.</remarks>
    public static void SiluAndMul(in SiluAndMulArgs a)
    {
        Need(a.Input, 2L * a.Rows * a.Dim, nameof(a.Input));
        Need(a.Output, (long)a.Rows * a.Dim, nameof(a.Output));
        for (long r = 0; r < a.Rows; r++)
        {
            for (int i = 0; i < a.Dim; i++)
            {
                double g = Load(a.Input.Address, a.DType, (r * 2 * a.Dim) + i);
                double u = Load(a.Input.Address, a.DType, (r * 2 * a.Dim) + a.Dim + i);
                Store(a.Output.Address, a.DType, (r * a.Dim) + i, Silu(g) * u);
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>Exact: <c>0.5 x erfc(-x / sqrt 2)</c>. Tanh: <c>0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))</c>. Then times the up half.</remarks>
    public static void GeluAndMul(in GeluAndMulArgs a)
    {
        Need(a.Input, 2L * a.Rows * a.Dim, nameof(a.Input));
        Need(a.Output, (long)a.Rows * a.Dim, nameof(a.Output));
        for (long r = 0; r < a.Rows; r++)
        {
            for (int i = 0; i < a.Dim; i++)
            {
                double g = Load(a.Input.Address, a.DType, (r * 2 * a.Dim) + i);
                double u = Load(a.Input.Address, a.DType, (r * 2 * a.Dim) + a.Dim + i);
                double act = a.Approximation == GeluApproximation.Tanh
                    ? 0.5 * g * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (g + (0.044715 * g * g * g))))
                    : 0.5 * g * Erfc(-g / Math.Sqrt(2.0));
                Store(a.Output.Address, a.DType, (r * a.Dim) + i, act * u);
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <c>D = Alpha * sum_k A[m,k] B[..] + Beta * C[m,n] + Bias[n]</c> in double. A, C, Bias and D are DType; B is WeightDType.
    /// C is read only when Beta is not 0 (then it must be present); Bias is added when its address is not 0.
    /// </remarks>
    public static void Gemm(in GemmArgs a)
    {
        Need(a.A, a.M * a.K, nameof(a.A));
        Need(a.B, a.K * a.N, nameof(a.B));
        Need(a.D, a.M * a.N, nameof(a.D));
        bool useC = a.Beta != 0f;
        if (useC)
        {
            Need(a.C, a.M * a.N, nameof(a.C));
        }

        bool bias = a.Bias.Address != 0;
        if (bias)
        {
            Need(a.Bias, a.N, nameof(a.Bias));
        }

        for (long m = 0; m < a.M; m++)
        {
            for (long n = 0; n < a.N; n++)
            {
                double s = 0;
                for (long k = 0; k < a.K; k++)
                {
                    long bi = a.TransposeB ? (n * a.K) + k : (k * a.N) + n;
                    s += Load(a.A.Address, a.DType, (m * a.K) + k) * Load(a.B.Address, a.WeightDType, bi);
                }

                double r = a.Alpha * s;
                if (useC)
                {
                    r += a.Beta * Load(a.C.Address, a.DType, (m * a.N) + n);
                }

                if (bias)
                {
                    r += Load(a.Bias.Address, a.DType, n);
                }

                Store(a.D.Address, a.DType, (m * a.N) + n, r);
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>Bit-exact row copy; an id outside [0, Vocab) throws.</remarks>
    public static void Embedding(in EmbeddingArgs a)
    {
        Need(a.Ids, a.Tokens, nameof(a.Ids));
        Need(a.Table, (long)a.Vocab * a.Dim, nameof(a.Table));
        Need(a.Output, (long)a.Tokens * a.Dim, nameof(a.Output));
        int es = SizeOf(a.DType);
        long rowBytes = (long)a.Dim * es;
        for (int i = 0; i < a.Tokens; i++)
        {
            long id = LoadInt(a.Ids.Address, a.Ids.Dtype, i);
            if (id < 0 || id >= a.Vocab)
            {
                throw new ArgumentOutOfRangeException(nameof(a), $"Token id {id} at position {i} is outside the vocabulary of {a.Vocab}.");
            }

            CopyBytes(a.Table.Address + (nint)(id * rowBytes), a.Output.Address + (nint)(i * rowBytes), rowBytes);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <c>Logits[s, v] = sum_d Hidden[Rows[s], d] * Weight[v, d]</c> for v in [0, Vocab), in double, stored as float32. With SoftCap not 0 the logit
    /// becomes <c>SoftCap * tanh(logit / SoftCap)</c>. Columns Vocab to VocabPadded are left untouched.
    /// </remarks>
    public static unsafe void LmHead(in LmHeadArgs a)
    {
        if (a.VocabPadded < a.Vocab)
        {
            throw new ArgumentException("VocabPadded must be at least Vocab.", nameof(a));
        }

        Need(a.Rows, a.Selected, nameof(a.Rows));
        Need(a.Weight, (long)a.Vocab * a.Dim, nameof(a.Weight));
        Need(a.Logits, (long)a.Selected * a.VocabPadded, nameof(a.Logits));
        for (int s = 0; s < a.Selected; s++)
        {
            long row = LoadInt(a.Rows.Address, a.Rows.Dtype, s);
            NeedRange(a.Hidden, row * a.Dim, a.Dim, nameof(a.Hidden));
            for (int v = 0; v < a.Vocab; v++)
            {
                double acc = 0;
                for (int d = 0; d < a.Dim; d++)
                {
                    acc += Load(a.Hidden.Address, a.DType, (row * a.Dim) + d) * Load(a.Weight.Address, a.DType, ((long)v * a.Dim) + d);
                }

                if (a.SoftCap != 0f)
                {
                    acc = a.SoftCap * Math.Tanh(acc / a.SoftCap);
                }

                ((float*)a.Logits.Address)[((long)s * a.VocabPadded) + v] = (float)acc;
            }
        }
    }

    /// <inheritdoc/>
    public static void Residual(in ResidualArgs a)
    {
        Need(a.A, a.Elements, nameof(a.A));
        Need(a.B, a.Elements, nameof(a.B));
        Need(a.Output, a.Elements, nameof(a.Output));
        for (long i = 0; i < a.Elements; i++)
        {
            Store(a.Output.Address, a.DType, i, Load(a.A.Address, a.DType, i) + Load(a.B.Address, a.DType, i));
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Token t goes to <c>slot = SlotMapping[t]</c> (block = slot / BlockSize, offset = slot % BlockSize); a negative slot skips the token.
    /// When KvDType differs from DType the value is divided by KScale (K) or VScale (V) and rounded into KvDType; when they match the copy is exact.
    /// </remarks>
    public static void CacheWrite(in CacheWriteArgs a)
    {
        bool hnd = a.KvLayout switch
        {
            TensorLayout.PagedHnd => true,
            TensorLayout.PagedNhd => false,
            _ => throw new ArgumentException("KvLayout must be PagedNhd or PagedHnd.", nameof(a)),
        };

        int hd = a.HeadDim;
        long perToken = (long)a.KvHeads * hd;
        Need(a.K, a.Tokens * perToken, nameof(a.K));
        Need(a.V, a.Tokens * perToken, nameof(a.V));
        Need(a.SlotMapping, a.Tokens, nameof(a.SlotMapping));
        // The activation type is not an argument of CacheWrite: the K and V spans carry it.
        DType src = a.K.Dtype;
        DType srcV = a.V.Dtype;
        for (int t = 0; t < a.Tokens; t++)
        {
            long slot = LoadInt(a.SlotMapping.Address, a.SlotMapping.Dtype, t);
            if (slot < 0)
            {
                continue;
            }

            long block = slot / a.BlockSize;
            int off = (int)(slot % a.BlockSize);
            for (int h = 0; h < a.KvHeads; h++)
            {
                long dst = hnd
                    ? ((((block * a.KvHeads) + h) * a.BlockSize) + off) * hd
                    : ((((block * a.BlockSize) + off) * a.KvHeads) + h) * hd;
                NeedRange(a.KCache, dst, hd, nameof(a.KCache));
                NeedRange(a.VCache, dst, hd, nameof(a.VCache));
                long s0 = (t * perToken) + ((long)h * hd);
                CopyConvert(a.K.Address, src, s0, a.KCache.Address, a.KvDType, dst, hd, a.KScale);
                CopyConvert(a.V.Address, srcV, s0, a.VCache.Address, a.KvDType, dst, hd, a.VScale);
            }
        }
    }

    private static void CopyConvert(nint from, DType ft, long fo, nint to, DType tt, long to0, int count, float scale)
    {
        if (ft == tt)
        {
            int es = SizeOf(ft);
            CopyBytes(from + (nint)(fo * es), to + (nint)(to0 * es), (long)count * es);
            return;
        }

        for (int d = 0; d < count; d++)
        {
            Store(to, tt, to0 + d, Load(from, ft, fo + d) / scale);
        }
    }

    /// <inheritdoc/>
    /// <remarks>Whole-block byte copies in pair order, each as a memmove (so Source may equal Destination); a pair with equal blocks is a no-op.</remarks>
    public static void CacheCopy(in CacheCopyArgs a)
    {
        Need(a.Pairs, 2L * a.Count, nameof(a.Pairs));
        CopyPairs(a.Source, a.Destination, a.Pairs, a.Count, a.BlockBytes, nameof(a));
    }

    private static void CopyPairs(in DeviceSpan source, in DeviceSpan destination, in DeviceSpan pairs, int count, long unitBytes, string what, bool swap = false)
    {
        long srcBytes = ByteLength(source);
        long dstBytes = ByteLength(destination);
        bool same = source.Address == destination.Address;
        for (int i = 0; i < count; i++)
        {
            long s = LoadInt(pairs.Address, pairs.Dtype, (2L * i) + (swap ? 1 : 0));
            long d = LoadInt(pairs.Address, pairs.Dtype, (2L * i) + (swap ? 0 : 1));
            if (s < 0 || d < 0 || (s + 1) * unitBytes > srcBytes || (d + 1) * unitBytes > dstBytes)
            {
                throw new ArgumentOutOfRangeException(what, $"Pair {i} ({s}, {d}) is outside the pool.");
            }

            if (same && s == d)
            {
                continue;
            }

            CopyBytes(source.Address + (nint)(s * unitBytes), destination.Address + (nint)(d * unitBytes), unitBytes);
        }
    }

    /// <inheritdoc/>
    /// <remarks>Over <c>Logits[r, 0..Vocab)</c> (row stride Stride, 0 meaning Vocab). Ties go to the lowest index; NaN is never chosen unless the whole row is NaN, which gives index 0.</remarks>
    public static unsafe void Argmax(in ArgmaxArgs a)
    {
        int stride = a.Stride == 0 ? a.Vocab : a.Stride;
        Need(a.Logits, ((a.Rows - 1L) * stride) + a.Vocab, nameof(a.Logits));
        Need(a.Ids, a.Rows, nameof(a.Ids));
        for (int r = 0; r < a.Rows; r++)
        {
            float* row = (float*)a.Logits.Address + ((long)r * stride);
            int best = 0;
            float bv = float.NaN;
            for (int v = 0; v < a.Vocab; v++)
            {
                float x = row[v];
                if (!float.IsNaN(x) && (float.IsNaN(bv) || x > bv))
                {
                    bv = x;
                    best = v;
                }
            }

            ((int*)a.Ids.Address)[r] = best;
        }
    }

    private static unsafe void CopyBytes(nint from, nint to, long bytes)
    {
        if (bytes > 0)
        {
            Buffer.MemoryCopy((void*)from, (void*)to, bytes, bytes);
        }
    }

    // Validation helpers. Exceptions allocate, but only on the error path.
    private static void Need(in DeviceSpan s, long elements, string name)
    {
        if (elements <= 0)
        {
            return;
        }

        if (s.Address == 0 || s.Length < (ulong)elements)
        {
            throw new ArgumentException($"{name} needs {elements} elements (address {(s.Address == 0 ? "null" : "set")}, length {s.Length}).");
        }
    }

    private static void NeedRange(in DeviceSpan s, long start, long count, string name)
    {
        if (start < 0 || s.Address == 0 || (ulong)(start + count) > s.Length)
        {
            throw new ArgumentException($"{name} range [{start}, {start + count}) is outside its {s.Length} elements.");
        }
    }
}

using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

public readonly partial struct CpuOps
{
    /// <inheritdoc/>
    /// <remarks>
    /// <para>Per token t, in double: the Experts logits are read as DType; the TopK experts are chosen in descending logit order, a tie
    /// going to the lower expert index (so the order is total and does not depend on the backend's reduction); a NaN logit ranks below
    /// every number. Choosing by logit rather than by probability is the same order wherever the probabilities differ, and keeps two logits
    /// that differ by one ulp apart when their probabilities would round equal.</para>
    /// <para>Weights are the softmax over all Experts, <c>p_e = exp(l_e - max) / sum_j exp(l_j - max)</c>, of the chosen experts; with
    /// Renormalize they become <c>p_k / sum_k p_k</c> (Qwen3 MoE's <c>norm_topk_prob</c>). Each is rounded once into Weights' element type
    /// (float32 when the span names none). ExpertIds[t, k] is the k-th chosen expert as int32, Weights[t, k] its weight.</para>
    /// <para>bf16, fp16, fp32 and fp64 logits; FP8 and INT4 are not activation types here.</para>
    /// </remarks>
    public static unsafe void MoeRouteTopK(in MoeRouteTopKArgs a)
    {
        int e = a.Experts;
        int k = a.TopK;
        if (e < 1 || k < 1 || k > e || a.Tokens < 0)
        {
            throw new ArgumentException($"MoeRouteTopK needs 1 <= TopK <= Experts and Tokens >= 0 (Tokens {a.Tokens}, Experts {e}, TopK {k}).", nameof(a));
        }

        ActivationType(a.DType, nameof(MoeRouteTopK));
        long tk = (long)a.Tokens * k;
        Need(a.Logits, (long)a.Tokens * e, nameof(a.Logits));
        Need(a.Weights, tk, nameof(a.Weights));
        Need(a.ExpertIds, tk, nameof(a.ExpertIds));
        Int32Span(a.ExpertIds, nameof(a.ExpertIds));
        DType wt = SpanType(a.Weights, DType.F32);
        int* ids = (int*)a.ExpertIds.Address;
        for (long t = 0; t < a.Tokens; t++)
        {
            long row = t * e;
            double max = double.NegativeInfinity;
            for (int j = 0; j < e; j++)
            {
                max = Math.Max(max, Load(a.Logits.Address, a.DType, row + j));
            }

            double denom = 0;
            for (int j = 0; j < e; j++)
            {
                denom += Math.Exp(Load(a.Logits.Address, a.DType, row + j) - max);
            }

            // K passes, each taking the best expert ranked after the previous pick: no scratch, and the order is total.
            int prev = -1;
            double prevValue = double.PositiveInfinity;
            double chosen = 0;
            for (int s = 0; s < k; s++)
            {
                int best = -1;
                double bestValue = 0;
                for (int j = 0; j < e; j++)
                {
                    double v = Load(a.Logits.Address, a.DType, row + j);
                    if (prev >= 0 && !RanksBefore(prevValue, prev, v, j))
                    {
                        continue;
                    }

                    if (best < 0 || RanksBefore(v, j, bestValue, best))
                    {
                        best = j;
                        bestValue = v;
                    }
                }

                ids[(t * k) + s] = best;
                chosen += Math.Exp(bestValue - max) / denom;
                prev = best;
                prevValue = bestValue;
            }

            for (int s = 0; s < k; s++)
            {
                double p = Math.Exp(Load(a.Logits.Address, a.DType, row + ids[(t * k) + s]) - max) / denom;
                Store(a.Weights.Address, wt, (t * k) + s, a.Renormalize ? p / chosen : p);
            }
        }
    }

    // Whether (v1, i1) comes before (v2, i2) in routing order: larger value first, NaN last, equal values by lower index.
    private static bool RanksBefore(double v1, int i1, double v2, int i2)
    {
        bool n1 = double.IsNaN(v1);
        bool n2 = double.IsNaN(v2);
        if (n1 != n2)
        {
            return n2;
        }

        if (!n1 && v1 != v2)
        {
            return v1 > v2;
        }

        return i1 < i2;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>Stable counting sort of the Tokens * TopK (token, slot) pairs by expert. ExpertCounts[e] is the number of pairs routed to e and
    /// ExpertOffsets its exclusive prefix sum (ExpertOffsets[Experts] = Tokens * TopK). Pairs are placed in ascending flat index
    /// <c>t * TopK + k</c> within each expert, so the order does not depend on the backend (the S6 GPU scatter uses atomics and does not
    /// promise it). Permutation[t * TopK + k] is the permuted row of pair (t, k), the form the weighted combine reads; Output row
    /// Permutation[t * TopK + k] is a bit-exact copy of Input row t. An expert with no pairs has count 0 and an empty range.</para>
    /// <para>An expert id outside [0, Experts) throws before anything is written. The int32 buffers are written as int32.</para>
    /// </remarks>
    public static unsafe void MoePermute(in MoePermuteArgs a)
    {
        int e = a.Experts;
        if (e < 1 || a.TopK < 1 || a.Tokens < 0 || a.Dim < 0)
        {
            throw new ArgumentException($"MoePermute needs Experts >= 1, TopK >= 1 and non-negative Tokens and Dim (Tokens {a.Tokens}, Dim {a.Dim}, Experts {e}, TopK {a.TopK}).", nameof(a));
        }

        long tk = (long)a.Tokens * a.TopK;
        Need(a.Input, (long)a.Tokens * a.Dim, nameof(a.Input));
        Need(a.ExpertIds, tk, nameof(a.ExpertIds));
        Need(a.Output, tk * a.Dim, nameof(a.Output));
        Need(a.ExpertCounts, e, nameof(a.ExpertCounts));
        Need(a.ExpertOffsets, e + 1L, nameof(a.ExpertOffsets));
        Need(a.Permutation, tk, nameof(a.Permutation));
        Int32Span(a.ExpertCounts, nameof(a.ExpertCounts));
        Int32Span(a.ExpertOffsets, nameof(a.ExpertOffsets));
        Int32Span(a.Permutation, nameof(a.Permutation));
        if (tk > int.MaxValue)
        {
            throw new ArgumentException($"MoePermute: Tokens * TopK ({tk}) does not fit the int32 offsets.", nameof(a));
        }

        for (long i = 0; i < tk; i++)
        {
            long id = LoadInt(a.ExpertIds.Address, SpanType(a.ExpertIds, DType.I32), i);
            if (id < 0 || id >= e)
            {
                throw new ArgumentOutOfRangeException(nameof(a), $"MoePermute: expert id {id} at pair {i} is outside [0, {e}).");
            }
        }

        int* counts = (int*)a.ExpertCounts.Address;
        int* offsets = (int*)a.ExpertOffsets.Address;
        int* perm = (int*)a.Permutation.Address;
        DType idt = SpanType(a.ExpertIds, DType.I32);
        new Span<int>(counts, e).Clear();
        for (long i = 0; i < tk; i++)
        {
            counts[LoadInt(a.ExpertIds.Address, idt, i)]++;
        }

        offsets[0] = 0;
        for (int j = 0; j < e; j++)
        {
            offsets[j + 1] = offsets[j] + counts[j];
        }

        // Second pass: counts serve as the per-expert cursor and end where they started.
        new Span<int>(counts, e).Clear();
        long rowBytes = (long)a.Dim * SizeOf(a.DType);
        for (long i = 0; i < tk; i++)
        {
            int x = (int)LoadInt(a.ExpertIds.Address, idt, i);
            int pos = offsets[x] + counts[x]++;
            perm[i] = pos;
            CopyBytes(a.Input.Address + (nint)(i / a.TopK * rowBytes), a.Output.Address + (nint)(pos * rowBytes), rowBytes);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <c>Output[t, d] = sum_k Weights[t, k] * Input[Permutation[t * TopK + k], d]</c>, summed in double in ascending k and rounded once into
    /// DType. Input and Output are DType; Weights are read as their span's element type (float32 when it names none). A permuted row outside
    /// [0, Tokens * TopK) throws before anything is written. Output must not overlap Input.
    /// </remarks>
    public static void MoeUnpermute(in MoeUnpermuteArgs a)
    {
        int k = a.TopK;
        if (k < 1 || a.Tokens < 0 || a.Dim < 0)
        {
            throw new ArgumentException($"MoeUnpermute needs TopK >= 1 and non-negative Tokens and Dim (Tokens {a.Tokens}, Dim {a.Dim}, TopK {k}).", nameof(a));
        }

        long tk = (long)a.Tokens * k;
        Need(a.Input, tk * a.Dim, nameof(a.Input));
        Need(a.Permutation, tk, nameof(a.Permutation));
        Need(a.Weights, tk, nameof(a.Weights));
        Need(a.Output, (long)a.Tokens * a.Dim, nameof(a.Output));
        DType pt = SpanType(a.Permutation, DType.I32);
        DType wt = SpanType(a.Weights, DType.F32);
        for (long i = 0; i < tk; i++)
        {
            long r = LoadInt(a.Permutation.Address, pt, i);
            if (r < 0 || r >= tk)
            {
                throw new ArgumentOutOfRangeException(nameof(a), $"MoeUnpermute: permuted row {r} of pair {i} is outside [0, {tk}).");
            }
        }

        for (long t = 0; t < a.Tokens; t++)
        {
            for (int d = 0; d < a.Dim; d++)
            {
                double acc = 0;
                for (int s = 0; s < k; s++)
                {
                    long i = (t * k) + s;
                    long r = LoadInt(a.Permutation.Address, pt, i);
                    acc += Load(a.Weights.Address, wt, i) * Load(a.Input.Address, a.DType, (r * a.Dim) + d);
                }

                Store(a.Output.Address, a.DType, (t * a.Dim) + d, acc);
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>For each expert g, rows r in [ExpertOffsets[g], ExpertOffsets[g + 1]): <c>D[r, n] = sum_k A[r, k] * B[g, n, k]</c> in double,
    /// rounded once into DType (B is stored like a TransposeB Gemm weight, one [N, K] matrix per expert). A and D are DType, B is
    /// WeightDType. An expert with an empty range is skipped; rows before ExpertOffsets[0] are untouched. Offsets must be non-negative and
    /// non-decreasing, and when MaxRows is not 0 the last may not exceed it; otherwise the call throws before writing.</para>
    /// <para>Weights are bf16, fp16, fp32 or fp64. FP8 and INT4 weights (R-077) throw <see cref="NotSupportedException"/>: they are
    /// <c>unsupported</c> on this backend.</para>
    /// </remarks>
    public static void GroupedGemm(in GroupedGemmArgs a)
    {
        int e = a.Experts;
        if (e < 1 || a.N < 0 || a.K < 0)
        {
            throw new ArgumentException($"GroupedGemm needs Experts >= 1 and non-negative N and K (Experts {e}, N {a.N}, K {a.K}).", nameof(a));
        }

        ActivationType(a.DType, nameof(GroupedGemm));
        MoeWeightType(a.WeightDType, nameof(GroupedGemm));
        Need(a.ExpertOffsets, e + 1L, nameof(a.ExpertOffsets));
        DType ot = SpanType(a.ExpertOffsets, DType.I32);
        long rows = 0;
        long previous = 0;
        for (int g = 0; g <= e; g++)
        {
            long o = LoadInt(a.ExpertOffsets.Address, ot, g);
            if (o < 0 || (g > 0 && o < previous))
            {
                throw new ArgumentException($"GroupedGemm: ExpertOffsets[{g}] = {o} is negative or below ExpertOffsets[{g - 1}] = {previous}.", nameof(a));
            }

            previous = o;
            rows = o;
        }

        if (a.MaxRows > 0 && rows > a.MaxRows)
        {
            throw new ArgumentException($"GroupedGemm: {rows} rows exceed MaxRows {a.MaxRows}.", nameof(a));
        }

        Need(a.A, rows * a.K, nameof(a.A));
        Need(a.D, rows * a.N, nameof(a.D));
        Need(a.B, e * a.N * a.K, nameof(a.B));
        for (int g = 0; g < e; g++)
        {
            long r0 = LoadInt(a.ExpertOffsets.Address, ot, g);
            long r1 = LoadInt(a.ExpertOffsets.Address, ot, g + 1);
            long wBase = g * a.N * a.K;
            for (long r = r0; r < r1; r++)
            {
                for (long n = 0; n < a.N; n++)
                {
                    double s = 0;
                    for (long kk = 0; kk < a.K; kk++)
                    {
                        s += Load(a.A.Address, a.DType, (r * a.K) + kk) * Load(a.B.Address, a.WeightDType, wBase + (n * a.K) + kk);
                    }

                    Store(a.D.Address, a.DType, (r * a.N) + n, s);
                }
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>Exactly the composition of the unfused ops, each rounding where it stores, on native scratch freed before the call returns:
    /// <see cref="MoeRouteTopK"/> on RouterLogits (read as their span's element type, DType when it names none; float32 weights), then
    /// <see cref="MoePermute"/> of Input, <see cref="GroupedGemm"/> with W13 ([Experts, 2 * Intermediate, Dim], gate rows then up rows),
    /// <see cref="SiluAndMul"/>, <see cref="GroupedGemm"/> with W2 ([Experts, Dim, Intermediate]), and <see cref="MoeUnpermute"/> into Output.
    /// The intermediates are DType, as in the S6 GPU pipeline (gemm1, SiLU-and-mul and gemm2 outputs rounded to bf16, combine from them).</para>
    /// <para>Input may alias Output: it is not read after the permute. FP8 and INT4 weights throw <see cref="NotSupportedException"/>.</para>
    /// </remarks>
    public static unsafe void FusedMoe(in FusedMoeArgs a)
    {
        int e = a.Experts;
        int k = a.TopK;
        if (e < 1 || k < 1 || k > e || a.Tokens < 0 || a.Dim < 0 || a.Intermediate < 0)
        {
            throw new ArgumentException($"FusedMoe needs 1 <= TopK <= Experts and non-negative Tokens, Dim and Intermediate (Tokens {a.Tokens}, Dim {a.Dim}, Intermediate {a.Intermediate}, Experts {e}, TopK {k}).", nameof(a));
        }

        ActivationType(a.DType, nameof(FusedMoe));
        MoeWeightType(a.WeightDType, nameof(FusedMoe));
        Need(a.Input, (long)a.Tokens * a.Dim, nameof(a.Input));
        Need(a.RouterLogits, (long)a.Tokens * e, nameof(a.RouterLogits));
        Need(a.W13, (long)e * 2 * a.Intermediate * a.Dim, nameof(a.W13));
        Need(a.W2, (long)e * a.Dim * a.Intermediate, nameof(a.W2));
        Need(a.Output, (long)a.Tokens * a.Dim, nameof(a.Output));
        if (a.Tokens == 0)
        {
            return;
        }

        long tk = (long)a.Tokens * k;
        int es = SizeOf(a.DType);
        long oWeights = 0;
        long oIds = Align(oWeights + (tk * sizeof(float)));
        long oCounts = Align(oIds + (tk * sizeof(int)));
        long oOffsets = Align(oCounts + (e * (long)sizeof(int)));
        long oPerm = Align(oOffsets + ((e + 1L) * sizeof(int)));
        long oXp = Align(oPerm + (tk * sizeof(int)));
        long oG1 = Align(oXp + (tk * a.Dim * es));
        long oAct = Align(oG1 + (tk * 2 * a.Intermediate * es));
        long oY = Align(oAct + (tk * a.Intermediate * es));
        long total = Align(oY + (tk * a.Dim * es));
        nint mem = (nint)NativeMemory.AlignedAlloc((nuint)total, 64);
        try
        {
            DeviceSpan weights = Scratch(mem, oWeights, tk, DType.F32);
            DeviceSpan ids = Scratch(mem, oIds, tk, DType.I32);
            DeviceSpan counts = Scratch(mem, oCounts, e, DType.I32);
            DeviceSpan offsets = Scratch(mem, oOffsets, e + 1L, DType.I32);
            DeviceSpan perm = Scratch(mem, oPerm, tk, DType.I32);
            DeviceSpan xp = Scratch(mem, oXp, tk * a.Dim, a.DType);
            DeviceSpan g1 = Scratch(mem, oG1, tk * 2 * a.Intermediate, a.DType);
            DeviceSpan act = Scratch(mem, oAct, tk * a.Intermediate, a.DType);
            DeviceSpan y = Scratch(mem, oY, tk * a.Dim, a.DType);
            MoeRouteTopK(new MoeRouteTopKArgs
            {
                Logits = a.RouterLogits, Weights = weights, ExpertIds = ids, Tokens = a.Tokens, Experts = e, TopK = k,
                Renormalize = a.Renormalize, DType = SpanType(a.RouterLogits, a.DType),
            });
            MoePermute(new MoePermuteArgs
            {
                Input = a.Input, ExpertIds = ids, Output = xp, ExpertCounts = counts, ExpertOffsets = offsets, Permutation = perm,
                Tokens = a.Tokens, Dim = a.Dim, Experts = e, TopK = k, DType = a.DType,
            });
            GroupedGemm(new GroupedGemmArgs
            {
                A = xp, B = a.W13, D = g1, ExpertOffsets = offsets, Experts = e, N = 2L * a.Intermediate, K = a.Dim, MaxRows = tk,
                DType = a.DType, WeightDType = a.WeightDType,
            });
            SiluAndMul(new SiluAndMulArgs { Input = g1, Output = act, Rows = (int)tk, Dim = a.Intermediate, DType = a.DType });
            GroupedGemm(new GroupedGemmArgs
            {
                A = act, B = a.W2, D = y, ExpertOffsets = offsets, Experts = e, N = a.Dim, K = a.Intermediate, MaxRows = tk,
                DType = a.DType, WeightDType = a.WeightDType,
            });
            MoeUnpermute(new MoeUnpermuteArgs
            {
                Input = y, Permutation = perm, Weights = weights, Output = a.Output, Tokens = a.Tokens, Dim = a.Dim, TopK = k, DType = a.DType,
            });
        }
        finally
        {
            NativeMemory.AlignedFree((void*)mem);
        }
    }

    private static long Align(long bytes) => (bytes + 63) & ~63L;

    private static DeviceSpan Scratch(nint mem, long offset, long elements, DType t) =>
        new() { Address = mem + (nint)offset, Length = (ulong)elements, Dtype = t };

    private static void ActivationType(DType t, string op)
    {
        if (t is not (DType.BF16 or DType.F16 or DType.F32 or DType.F64))
        {
            throw new NotSupportedException($"{op}: CpuOps takes bf16, fp16, fp32 or fp64 activations, got {t}.");
        }
    }

    private static void MoeWeightType(DType t, string op)
    {
        if (t is not (DType.BF16 or DType.F16 or DType.F32 or DType.F64))
        {
            throw new NotSupportedException($"{op}: CpuOps takes bf16, fp16, fp32 or fp64 expert weights, got {t}; FP8 and INT4 MoE weights are unsupported (R-077).");
        }
    }

    private static void Int32Span(in DeviceSpan s, string name)
    {
        if (s.Dtype is not (DType.Undefined or DType.I32))
        {
            throw new ArgumentException($"{name} is int32, got {s.Dtype}.");
        }
    }
}

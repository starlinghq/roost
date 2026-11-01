using Starlings.Roost.Abi;
using System.Runtime.InteropServices;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

public readonly partial struct CpuOps
{
    /// <inheritdoc/>
    /// <remarks>
    /// <para>Temperature, TopP, MinP and the three penalties are float32, TopK, TokenCounts, StopIds, StopRanges and the
    /// integer outputs int32, Seeds and Offsets uint64, Mask and PromptMask uint32, whatever the spans' Dtype says. Per row,
    /// in order:</para>
    /// <list type="number">
    /// <item>Penalties, when TokenCounts is set (vLLM's order): with <c>c_v</c> the token's count in the generated output and
    /// <c>seen_v</c> true when <c>c_v &gt; 0</c> or the token's PromptMask bit is 1, a seen logit x becomes
    /// <c>x / RepetitionPenalty</c> if it is above 0 and <c>x * RepetitionPenalty</c> otherwise; then
    /// <c>x - FrequencyPenalty * c_v</c>; then, when <c>c_v &gt; 0</c>, <c>x - PresencePenalty</c>.</item>
    /// <item>Tokens whose bit is 0 in Mask (bit v % 32 of word v / 32) are removed; with no Mask all tokens stay. NaN and
    /// -inf logits are removed too.</item>
    /// <item>Temperature 0 (or less) is greedy: the highest remaining logit, ties to the lowest index.</item>
    /// <item>Otherwise <c>p_v = softmax((logit_v - max) / Temperature)</c> over the remaining tokens, ordered by probability
    /// descending, ties by lower index.</item>
    /// <item>TopK &gt; 0 keeps the first TopK; the kept probabilities are renormalised.</item>
    /// <item>TopP &lt; 1 keeps the shortest prefix whose renormalised mass reaches TopP (the token that crosses it is kept).</item>
    /// <item>MinP &gt; 0 drops tokens whose probability is below MinP times the largest.</item>
    /// <item>The draw is <c>u = Philox4x32-10(counter = (Offsets[r] low, high, 0, 0), key = (Seeds[r] low, high))</c>, first
    /// word, top 24 bits times 2^-24; the chosen token is the first in the order above whose cumulative kept mass exceeds
    /// <c>u * total kept mass</c> (the last kept token if rounding leaves none). The draw depends on the row's own seed and
    /// offset only, so the same seed and offset give the same token in any batch and at any row index.</item>
    /// </list>
    /// <para>Logprobs (when the span is set) is the log-softmax of the raw logits (before penalties, mask and temperature;
    /// NaN logits take no part) at the chosen token. TopLogprobIds and TopLogprobValues (when TopLogprobs &gt; 0) are the TopLogprobs highest raw
    /// logits, ties to the lower index, and their log-softmax; a row with fewer non-NaN logits pads with id -1 and -inf.
    /// StopFlags (when set) is 1 when the chosen id is among <c>StopIds[start .. start + count)</c> for the row's
    /// <c>StopRanges[r] = (start, count)</c>, else 0. A row with no allowed token gets id -1, logprob -inf and stop flag 0.
    /// Seeds and Offsets are not advanced; the caller owns the counters.</para>
    /// <para><b>Math.</b> <see cref="SampleMath.Reference"/> computes the penalties, softmax, sums and logprobs in double.
    /// <see cref="SampleMath.Fp32"/> is the bit-exact definition for a device sampler: every value is a float32; the
    /// penalties are separate divides, multiplies and subtracts; <c>exp</c> and <c>log</c> are the operation sequences of
    /// <c>Fp32Math</c>; the softmax denominator and the logprob normaliser are its 32-lane row sum (lane <c>v % 32</c>,
    /// increasing v, then a halving tree); the probabilities are <c>e_v / sum</c>; the top-p total, the top-p running mass,
    /// the kept mass and the draw's running mass are sequential float32 sums in the sorted order; the top-p bar is
    /// <c>TopP * total</c>, the min-p floor <c>MinP * p_max</c> and the draw target <c>u * mass</c>, each one float32
    /// multiply; a logprob is <c>(x - max) - log(sum)</c>.</para>
    /// </remarks>
    public static unsafe void Sample(in SampleArgs a)
    {
        long stride = a.Stride == 0 ? a.Vocab : a.Stride;
        if (stride < a.Vocab)
        {
            throw new ArgumentException($"Stride {a.Stride} is smaller than Vocab {a.Vocab}.", nameof(a));
        }

        if (a.Math is not (SampleMath.Reference or SampleMath.Fp32))
        {
            throw new ArgumentException($"Math {a.Math} is not a sampling math mode.", nameof(a));
        }

        if (a.TopLogprobs < 0 || a.TopLogprobs > a.Vocab)
        {
            throw new ArgumentException($"TopLogprobs {a.TopLogprobs} is outside [0, {a.Vocab}].", nameof(a));
        }

        Need(a.Logits, ((a.Rows - 1L) * stride) + a.Vocab, nameof(a.Logits));
        Need(a.Temperature, a.Rows, nameof(a.Temperature));
        Need(a.TopK, a.Rows, nameof(a.TopK));
        Need(a.TopP, a.Rows, nameof(a.TopP));
        Need(a.MinP, a.Rows, nameof(a.MinP));
        Need(a.Seeds, a.Rows, nameof(a.Seeds));
        Need(a.Offsets, a.Rows, nameof(a.Offsets));
        Need(a.Ids, a.Rows, nameof(a.Ids));
        long words = (a.Vocab + 31) / 32;
        bool mask = a.Mask.Address != 0;
        if (mask)
        {
            Need(a.Mask, a.Rows * words, nameof(a.Mask));
        }

        bool logprobs = a.Logprobs.Address != 0;
        if (logprobs)
        {
            Need(a.Logprobs, a.Rows, nameof(a.Logprobs));
        }

        bool penalties = a.TokenCounts.Address != 0;
        bool prompt = penalties && a.PromptMask.Address != 0;
        if (penalties)
        {
            Need(a.TokenCounts, (long)a.Rows * a.Vocab, nameof(a.TokenCounts));
            Need(a.RepetitionPenalty, a.Rows, nameof(a.RepetitionPenalty));
            Need(a.PresencePenalty, a.Rows, nameof(a.PresencePenalty));
            Need(a.FrequencyPenalty, a.Rows, nameof(a.FrequencyPenalty));
            if (prompt)
            {
                Need(a.PromptMask, a.Rows * words, nameof(a.PromptMask));
            }
        }

        bool stops = a.StopFlags.Address != 0;
        if (stops)
        {
            Need(a.StopFlags, a.Rows, nameof(a.StopFlags));
            Need(a.StopRanges, a.Rows * 2L, nameof(a.StopRanges));
            for (int r = 0; r < a.Rows; r++)
            {
                int start = ((int*)a.StopRanges.Address)[2 * r];
                int count = ((int*)a.StopRanges.Address)[(2 * r) + 1];
                if (count < 0 || (count > 0 && (start < 0 || a.StopIds.Address == 0 || (ulong)((long)start + count) > a.StopIds.Length)))
                {
                    throw new ArgumentException($"StopRanges row {r} ({start}, {count}) is outside StopIds (length {a.StopIds.Length}).", nameof(a));
                }
            }
        }

        int top = a.TopLogprobs;
        if (top > 0)
        {
            Need(a.TopLogprobIds, (long)a.Rows * top, nameof(a.TopLogprobIds));
            Need(a.TopLogprobValues, (long)a.Rows * top, nameof(a.TopLogprobValues));
        }

        bool fp32 = a.Math == SampleMath.Fp32;

        // Native scratch, freed before returning: the penalised row and the probabilities in double (under Fp32 every value
        // is a float32 widened exactly) and the order.
        nuint n = (nuint)Math.Max(1, a.Vocab);
        double* z = (double*)NativeMemory.Alloc(n, sizeof(double));
        double* prob = (double*)NativeMemory.Alloc(n, sizeof(double));
        int* order = (int*)NativeMemory.Alloc(n, sizeof(int));
        try
        {
            for (int r = 0; r < a.Rows; r++)
            {
                float* logit = (float*)a.Logits.Address + ((long)r * stride);
                uint* bits = mask ? (uint*)a.Mask.Address + (r * words) : null;
                if (penalties)
                {
                    int* counts = (int*)a.TokenCounts.Address + ((long)r * a.Vocab);
                    uint* seen = prompt ? (uint*)a.PromptMask.Address + (r * words) : null;
                    float rep = (float)Load(a.RepetitionPenalty.Address, DType.F32, r);
                    float pres = (float)Load(a.PresencePenalty.Address, DType.F32, r);
                    float freq = (float)Load(a.FrequencyPenalty.Address, DType.F32, r);
                    Penalise(logit, a.Vocab, counts, seen, rep, pres, freq, fp32, z);
                }
                else
                {
                    for (int v = 0; v < a.Vocab; v++)
                    {
                        z[v] = logit[v];
                    }
                }

                float temp = (float)Load(a.Temperature.Address, DType.F32, r);
                int topK = (int)LoadInt(a.TopK.Address, DType.I32, r);
                double topP = Load(a.TopP.Address, DType.F32, r);
                double minP = Load(a.MinP.Address, DType.F32, r);
                ulong seed = (ulong)LoadInt(a.Seeds.Address, DType.U64, r);
                ulong offset = (ulong)LoadInt(a.Offsets.Address, DType.U64, r);
                int id = fp32
                    ? SampleRowF32(z, a.Vocab, bits, temp, topK, (float)topP, (float)minP, seed, offset, prob, order)
                    : SampleRow(z, a.Vocab, bits, temp, topK, topP, minP, seed, offset, prob, order);
                ((int*)a.Ids.Address)[r] = id;
                if (logprobs || top > 0)
                {
                    var (max, lse) = fp32 ? LogNormF32(logit, a.Vocab) : LogNorm(logit, a.Vocab);
                    if (logprobs)
                    {
                        ((float*)a.Logprobs.Address)[r] = id < 0 ? float.NegativeInfinity : LogprobAt(logit[id], max, lse, fp32);
                    }

                    if (top > 0)
                    {
                        TopLogprobs(logit, a.Vocab, top, max, lse, fp32, (int*)a.TopLogprobIds.Address + ((long)r * top), (float*)a.TopLogprobValues.Address + ((long)r * top));
                    }
                }

                if (stops)
                {
                    int start = ((int*)a.StopRanges.Address)[2 * r];
                    int count = ((int*)a.StopRanges.Address)[(2 * r) + 1];
                    bool hit = id >= 0 && count > 0 && new ReadOnlySpan<int>((int*)a.StopIds.Address + start, count).Contains(id);
                    ((int*)a.StopFlags.Address)[r] = hit ? 1 : 0;
                }
            }
        }
        finally
        {
            NativeMemory.Free(z);
            NativeMemory.Free(prob);
            NativeMemory.Free(order);
        }
    }

    private static unsafe void Penalise(float* logit, int vocab, int* counts, uint* prompt, float rep, float pres, float freq, bool fp32, double* z)
    {
        for (int v = 0; v < vocab; v++)
        {
            int c = counts[v];
            bool seen = c > 0 || (prompt != null && ((prompt[v >> 5] >> (v & 31)) & 1u) != 0);
            if (fp32)
            {
                float x = logit[v];
                if (seen)
                {
                    x = x > 0f ? x / rep : x * rep;
                }

                float cost = freq * c;
                x -= cost;
                if (c > 0)
                {
                    x -= pres;
                }

                z[v] = x;
            }
            else
            {
                double x = logit[v];
                if (seen)
                {
                    x = x > 0 ? x / rep : x * rep;
                }

                x -= (double)freq * c;
                if (c > 0)
                {
                    x -= pres;
                }

                z[v] = x;
            }
        }
    }

    // Collects the allowed tokens (mask bit set, not NaN, not -inf) into order, in increasing index, with their max.
    private static unsafe int Allowed(double* z, int vocab, uint* bits, int* order, out double max)
    {
        int n = 0;
        max = double.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            if (bits != null && ((bits[v >> 5] >> (v & 31)) & 1u) == 0)
            {
                continue;
            }

            double x = z[v];
            if (double.IsNaN(x) || double.IsNegativeInfinity(x))
            {
                continue;
            }

            order[n++] = v;
            if (x > max)
            {
                max = x;
            }
        }

        return n;
    }

    private static unsafe int Greedy(double* z, int* order, int n)
    {
        int best = order[0];
        for (int i = 1; i < n; i++)
        {
            if (z[order[i]] > z[best])
            {
                best = order[i];
            }
        }

        return best;
    }

    private static unsafe int SampleRow(double* z, int vocab, uint* bits, float temp, int topK, double topP, double minP, ulong seed, ulong offset, double* prob, int* order)
    {
        int n = Allowed(z, vocab, bits, order, out double max);
        if (n == 0)
        {
            return -1;
        }

        if (temp <= 0f)
        {
            return Greedy(z, order, n);
        }

        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            int v = order[i];
            double p = Math.Exp((z[v] - max) / temp);
            prob[v] = p;
            sum += p;
        }

        for (int i = 0; i < n; i++)
        {
            prob[order[i]] /= sum;
        }

        HeapSort(order, n, prob);
        int keep = n;
        if (topK > 0 && topK < keep)
        {
            keep = topK;
        }

        if (topP < 1.0)
        {
            double total = 0;
            for (int i = 0; i < keep; i++)
            {
                total += prob[order[i]];
            }

            double cum = 0;
            int cut = keep;
            for (int i = 0; i < keep; i++)
            {
                cum += prob[order[i]];
                if (cum >= topP * total)
                {
                    cut = i + 1;
                    break;
                }
            }

            keep = cut;
        }

        if (minP > 0.0)
        {
            double floor = minP * prob[order[0]];
            int cut = keep;
            for (int i = 0; i < keep; i++)
            {
                if (prob[order[i]] < floor)
                {
                    cut = i;
                    break;
                }
            }

            keep = Math.Max(1, cut);
        }

        double mass = 0;
        for (int i = 0; i < keep; i++)
        {
            mass += prob[order[i]];
        }

        double target = CpuPhilox.Uniform(seed, offset) * mass;
        double run = 0;
        for (int i = 0; i < keep; i++)
        {
            run += prob[order[i]];
            if (run > target)
            {
                return order[i];
            }
        }

        return order[keep - 1];
    }

    // The float32 definition (SampleMath.Fp32). z holds float32 values; prob receives float32 values widened exactly, so the
    // shared heapsort orders them as float32 comparisons would.
    private static unsafe int SampleRowF32(double* z, int vocab, uint* bits, float temp, int topK, float topP, float minP, ulong seed, ulong offset, double* prob, int* order)
    {
        int n = Allowed(z, vocab, bits, order, out double maxWide);
        if (n == 0)
        {
            return -1;
        }

        if (temp <= 0f)
        {
            return Greedy(z, order, n);
        }

        float max = (float)maxWide;
        Span<float> lanes = stackalloc float[32];
        lanes.Clear();
        for (int i = 0; i < n; i++)
        {
            int v = order[i];
            float e = Fp32Math.Exp(((float)z[v] - max) / temp);
            prob[v] = e;
            Fp32Math.AddToLane(lanes, v, e);
        }

        float sum = Fp32Math.ReduceLanes(lanes);
        for (int i = 0; i < n; i++)
        {
            int v = order[i];
            prob[v] = (float)prob[v] / sum;
        }

        HeapSort(order, n, prob);
        int keep = n;
        if (topK > 0 && topK < keep)
        {
            keep = topK;
        }

        if (topP < 1f)
        {
            float total = 0f;
            for (int i = 0; i < keep; i++)
            {
                total += (float)prob[order[i]];
            }

            float bar = topP * total;
            float cum = 0f;
            int cut = keep;
            for (int i = 0; i < keep; i++)
            {
                cum += (float)prob[order[i]];
                if (cum >= bar)
                {
                    cut = i + 1;
                    break;
                }
            }

            keep = cut;
        }

        if (minP > 0f)
        {
            float floor = minP * (float)prob[order[0]];
            int cut = keep;
            for (int i = 0; i < keep; i++)
            {
                if ((float)prob[order[i]] < floor)
                {
                    cut = i;
                    break;
                }
            }

            keep = Math.Max(1, cut);
        }

        float mass = 0f;
        for (int i = 0; i < keep; i++)
        {
            mass += (float)prob[order[i]];
        }

        float target = (float)CpuPhilox.Uniform(seed, offset) * mass;
        float run = 0f;
        for (int i = 0; i < keep; i++)
        {
            run += (float)prob[order[i]];
            if (run > target)
            {
                return order[i];
            }
        }

        return order[keep - 1];
    }

    // In-place heapsort of idx[0..n) by probability descending, ties by lower index. No allocation, deterministic.
    private static unsafe void HeapSort(int* idx, int n, double* prob)
    {
        for (int i = (n / 2) - 1; i >= 0; i--)
        {
            SiftDown(idx, i, n, prob);
        }

        for (int end = n - 1; end > 0; end--)
        {
            (idx[0], idx[end]) = (idx[end], idx[0]);
            SiftDown(idx, 0, end, prob);
        }
    }

    // "Greater" in heap terms means later in the final order, so the heap root is the last element.
    private static unsafe bool After(int a, int b, double* prob) => prob[a] < prob[b] || (prob[a] == prob[b] && a > b);

    private static unsafe void SiftDown(int* idx, int root, int n, double* prob)
    {
        while (true)
        {
            int child = (2 * root) + 1;
            if (child >= n)
            {
                return;
            }

            if (child + 1 < n && After(idx[child + 1], idx[child], prob))
            {
                child++;
            }

            if (!After(idx[child], idx[root], prob))
            {
                return;
            }

            (idx[root], idx[child]) = (idx[child], idx[root]);
            root = child;
        }
    }

    // max and log(sum exp(x - max)) of the raw row over non-NaN logits, in double.
    private static unsafe (double Max, double Lse) LogNorm(float* logit, int vocab)
    {
        double max = double.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            if (logit[v] > max)
            {
                max = logit[v];
            }
        }

        double sum = 0;
        for (int v = 0; v < vocab; v++)
        {
            if (!float.IsNaN(logit[v]))
            {
                sum += Math.Exp(logit[v] - max);
            }
        }

        return (max, Math.Log(sum));
    }

    // The float32 normaliser: the max over non-NaN logits, the 32-lane sum of Fp32Math.Exp(x - max) (NaN adds nothing) and
    // Fp32Math.Log of it, each widened exactly.
    private static unsafe (double Max, double Lse) LogNormF32(float* logit, int vocab)
    {
        float max = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            if (logit[v] > max)
            {
                max = logit[v];
            }
        }

        Span<float> lanes = stackalloc float[32];
        lanes.Clear();
        for (int v = 0; v < vocab; v++)
        {
            float x = logit[v];
            if (!float.IsNaN(x))
            {
                Fp32Math.AddToLane(lanes, v, Fp32Math.Exp(x - max));
            }
        }

        return (max, Fp32Math.Log(Fp32Math.ReduceLanes(lanes)));
    }

    private static float LogprobAt(float x, double max, double lse, bool fp32) =>
        fp32 ? (x - (float)max) - (float)lse : (float)(x - max - lse);

    // The `top` highest raw logits, ties to the lower index, NaN skipped: insertion into a sorted list of at most `top`.
    private static unsafe void TopLogprobs(float* logit, int vocab, int top, double max, double lse, bool fp32, int* ids, float* values)
    {
        int filled = 0;
        for (int v = 0; v < vocab; v++)
        {
            float x = logit[v];
            if (float.IsNaN(x) || (filled == top && !(x > logit[ids[top - 1]])))
            {
                continue;
            }

            int at = filled < top ? filled++ : top - 1;
            while (at > 0 && x > logit[ids[at - 1]])
            {
                ids[at] = ids[at - 1];
                at--;
            }

            ids[at] = v;
        }

        for (int i = 0; i < top; i++)
        {
            if (i < filled)
            {
                values[i] = LogprobAt(logit[ids[i]], max, lse, fp32);
            }
            else
            {
                ids[i] = -1;
                values[i] = float.NegativeInfinity;
            }
        }
    }
}

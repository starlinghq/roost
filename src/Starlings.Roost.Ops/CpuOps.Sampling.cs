using Starlings.Roost.Abi;
using System.Runtime.InteropServices;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

public readonly partial struct CpuOps
{
    /// <inheritdoc/>
    /// <remarks>
    /// <para>Temperature, TopP and MinP are float32, TopK int32, Seeds and Offsets uint64, Mask uint32, whatever the spans' Dtype says. Per row, in order:</para>
    /// <list type="number">
    /// <item>Tokens whose bit is 0 in Mask (bit v % 32 of word v / 32) are removed; with no Mask all tokens stay.</item>
    /// <item>Temperature 0 (or less) is greedy: the highest remaining logit, ties to the lowest index.</item>
    /// <item>Otherwise <c>p_v = softmax(logit_v / Temperature)</c> over the remaining tokens in double, ordered by probability descending, ties by lower index.</item>
    /// <item>TopK &gt; 0 keeps the first TopK; the kept probabilities are renormalised.</item>
    /// <item>TopP &lt; 1 keeps the shortest prefix whose renormalised mass reaches TopP (the token that crosses it is kept).</item>
    /// <item>MinP &gt; 0 drops tokens whose probability is below MinP times the largest.</item>
    /// <item>The draw is <c>u = Philox4x32-10(counter = (Offsets[r] low, high, 0, 0), key = (Seeds[r] low, high))</c>, first word, top 24 bits times 2^-24;
    /// the chosen token is the first in the order above whose cumulative kept mass exceeds <c>u * total kept mass</c> (the last kept token if rounding leaves none).</item>
    /// </list>
    /// <para>Logprobs (when the span is set) is the log-softmax of the raw logits (before mask and temperature) at the chosen token, as float32.
    /// A row with no allowed token gets id -1. Seeds and Offsets are not advanced; the caller owns the counters.</para>
    /// </remarks>
    public static unsafe void Sample(in SampleArgs a)
    {
        long stride = a.Stride == 0 ? a.Vocab : a.Stride;
        if (stride < a.Vocab)
        {
            throw new ArgumentException($"Stride {a.Stride} is smaller than Vocab {a.Vocab}.", nameof(a));
        }

        Need(a.Logits, ((a.Rows - 1L) * stride) + a.Vocab, nameof(a.Logits));
        Need(a.Temperature, a.Rows, nameof(a.Temperature));
        Need(a.TopK, a.Rows, nameof(a.TopK));
        Need(a.TopP, a.Rows, nameof(a.TopP));
        Need(a.MinP, a.Rows, nameof(a.MinP));
        Need(a.Seeds, a.Rows, nameof(a.Seeds));
        Need(a.Offsets, a.Rows, nameof(a.Offsets));
        Need(a.Ids, a.Rows, nameof(a.Ids));
        bool mask = a.Mask.Address != 0;
        if (mask)
        {
            Need(a.Mask, a.Rows * (long)((a.Vocab + 31) / 32), nameof(a.Mask));
        }

        bool logprobs = a.Logprobs.Address != 0;
        if (logprobs)
        {
            Need(a.Logprobs, a.Rows, nameof(a.Logprobs));
        }

        long words = (a.Vocab + 31) / 32;
        // Native scratch: probabilities (double) and the order (int), freed before returning.
        double* prob = (double*)NativeMemory.Alloc((nuint)Math.Max(1, a.Vocab), sizeof(double));
        int* order = (int*)NativeMemory.Alloc((nuint)Math.Max(1, a.Vocab), sizeof(int));
        try
        {
            for (int r = 0; r < a.Rows; r++)
            {
                float* logit = (float*)a.Logits.Address + ((long)r * stride);
                uint* bits = mask ? (uint*)a.Mask.Address + (r * words) : null;
                float temp = (float)Load(a.Temperature.Address, DType.F32, r);
                int topK = (int)LoadInt(a.TopK.Address, DType.I32, r);
                double topP = Load(a.TopP.Address, DType.F32, r);
                double minP = Load(a.MinP.Address, DType.F32, r);
                ulong seed = (ulong)LoadInt(a.Seeds.Address, DType.U64, r);
                ulong offset = (ulong)LoadInt(a.Offsets.Address, DType.U64, r);
                int id = SampleRow(logit, a.Vocab, bits, temp, topK, topP, minP, seed, offset, prob, order);
                ((int*)a.Ids.Address)[r] = id;
                if (logprobs)
                {
                    ((float*)a.Logprobs.Address)[r] = id < 0 ? float.NegativeInfinity : (float)LogSoftmaxAt(logit, a.Vocab, id);
                }
            }
        }
        finally
        {
            NativeMemory.Free(prob);
            NativeMemory.Free(order);
        }
    }

    private static unsafe int SampleRow(float* logit, int vocab, uint* bits, float temp, int topK, double topP, double minP, ulong seed, ulong offset, double* prob, int* order)
    {
        int n = 0;
        double max = double.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            if (bits != null && ((bits[v >> 5] >> (v & 31)) & 1u) == 0)
            {
                continue;
            }

            double x = logit[v];
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

        if (n == 0)
        {
            return -1;
        }

        if (temp <= 0f)
        {
            int best = order[0];
            for (int i = 1; i < n; i++)
            {
                if (logit[order[i]] > logit[best])
                {
                    best = order[i];
                }
            }

            return best;
        }

        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            int v = order[i];
            double p = Math.Exp(((double)logit[v] - max) / temp);
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

    private static unsafe double LogSoftmaxAt(float* logit, int vocab, int id)
    {
        double max = double.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            max = Math.Max(max, logit[v]);
        }

        double sum = 0;
        for (int v = 0; v < vocab; v++)
        {
            sum += Math.Exp(logit[v] - max);
        }

        return logit[id] - max - Math.Log(sum);
    }
}

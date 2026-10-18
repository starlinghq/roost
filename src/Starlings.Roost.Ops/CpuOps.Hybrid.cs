using Starlings.Roost.Abi;
using System.Runtime.InteropServices;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

public readonly partial struct CpuOps
{
    private const int MaxConvWidth = 16;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>The plain step-by-step recurrence, the exact definition the chunked GPU kernels are compared against (it is the loop of flash-linear-attention's
    /// <c>fused_recurrent_gated_delta_rule_fwd_kernel</c> at 329b534245d3d7bb27cc03a99e87bdc51fb28d06). For sequence b (tokens CuSeqlens[b] to CuSeqlens[b+1]) and value head h,
    /// with key head <c>h / (VHeads / KHeads)</c> and state S of shape [KDim, VDim] taken from slot StateSlots[b]:</para>
    /// <code>
    /// q, k = q / sqrt(sum(q^2) + 1e-6), k / sqrt(sum(k^2) + 1e-6)   (only when L2NormQk)
    /// q    = q * Scale                                              (Scale 0 means 1/sqrt(KDim))
    /// S    = S * exp(g_t)
    /// d    = beta_t * (v_t - S^T k)
    /// S    = S + k (outer) d
    /// o_t  = S^T q
    /// </code>
    /// <para>S starts as the slot's contents when UseInitialState is set and as zero otherwise; it is carried in double for the whole call and written back
    /// to the slot as float32 once at the end (a sequence of zero tokens leaves its slot untouched). Q, K, V and Output are DType; G and Beta are read as their span's
    /// Dtype (DType when the span has none). Beta is used as given (any sigmoid is applied upstream), G is the log decay.</para>
    /// </remarks>
    public static unsafe void GatedDeltaRule(in GatedDeltaRuleArgs a)
    {
        if (a.KHeads < 1 || a.VHeads % a.KHeads != 0)
        {
            throw new ArgumentException("VHeads must be a multiple of KHeads.", nameof(a));
        }

        int kd = a.KDim;
        int vd = a.VDim;
        Need(a.Q, (long)a.Tokens * a.KHeads * kd, nameof(a.Q));
        Need(a.K, (long)a.Tokens * a.KHeads * kd, nameof(a.K));
        Need(a.V, (long)a.Tokens * a.VHeads * vd, nameof(a.V));
        Need(a.G, (long)a.Tokens * a.VHeads, nameof(a.G));
        Need(a.Beta, (long)a.Tokens * a.VHeads, nameof(a.Beta));
        Need(a.Output, (long)a.Tokens * a.VHeads * vd, nameof(a.Output));
        Need(a.StateSlots, a.Batch, nameof(a.StateSlots));
        Need(a.CuSeqlens, a.Batch + 1L, nameof(a.CuSeqlens));
        DType gt = a.G.Dtype == DType.Undefined ? a.DType : a.G.Dtype;
        DType bt = a.Beta.Dtype == DType.Undefined ? a.DType : a.Beta.Dtype;
        double scale = a.Scale == 0f ? 1.0 / Math.Sqrt(kd) : a.Scale;
        int group = a.VHeads / a.KHeads;
        long stateLen = (long)kd * vd;
        // Native scratch: S [kd, vd], q [kd], k [kd], d [vd].
        long scratch = stateLen + (2L * kd) + vd;
        double* mem = (double*)NativeMemory.Alloc((nuint)scratch, sizeof(double));
        try
        {
            double* s = mem;
            double* q = mem + stateLen;
            double* k = q + kd;
            double* d = k + kd;
            for (int b = 0; b < a.Batch; b++)
            {
                long t0 = LoadInt(a.CuSeqlens.Address, a.CuSeqlens.Dtype, b);
                long t1 = LoadInt(a.CuSeqlens.Address, a.CuSeqlens.Dtype, b + 1L);
                long slot = LoadInt(a.StateSlots.Address, a.StateSlots.Dtype, b);
                if (t0 < 0 || t1 < t0 || t1 > a.Tokens)
                {
                    throw new ArgumentException($"Sequence {b} has an invalid CuSeqlens range.", nameof(a));
                }

                if (t1 == t0)
                {
                    continue;
                }

                for (int h = 0; h < a.VHeads; h++)
                {
                    long so = ((slot * a.VHeads) + h) * stateLen;
                    NeedRange(a.State, so, stateLen, nameof(a.State));
                    float* state = (float*)a.State.Address + so;
                    for (long i = 0; i < stateLen; i++)
                    {
                        s[i] = a.UseInitialState ? state[i] : 0.0;
                    }

                    int kh = h / group;
                    for (long t = t0; t < t1; t++)
                    {
                        long qo = ((t * a.KHeads) + kh) * kd;
                        double qq = 0;
                        double kk = 0;
                        for (int i = 0; i < kd; i++)
                        {
                            q[i] = Load(a.Q.Address, a.DType, qo + i);
                            k[i] = Load(a.K.Address, a.DType, qo + i);
                            qq += q[i] * q[i];
                            kk += k[i] * k[i];
                        }

                        double qn = a.L2NormQk ? 1.0 / Math.Sqrt(qq + 1e-6) : 1.0;
                        double kn = a.L2NormQk ? 1.0 / Math.Sqrt(kk + 1e-6) : 1.0;
                        for (int i = 0; i < kd; i++)
                        {
                            q[i] *= qn * scale;
                            k[i] *= kn;
                        }

                        double decay = Math.Exp(Load(a.G.Address, gt, (t * a.VHeads) + h));
                        double beta = Load(a.Beta.Address, bt, (t * a.VHeads) + h);
                        long vo = ((t * a.VHeads) + h) * vd;
                        for (int j = 0; j < vd; j++)
                        {
                            double kv = 0;
                            for (int i = 0; i < kd; i++)
                            {
                                s[(i * vd) + j] *= decay;
                                kv += s[(i * vd) + j] * k[i];
                            }

                            d[j] = beta * (Load(a.V.Address, a.DType, vo + j) - kv);
                        }

                        for (int j = 0; j < vd; j++)
                        {
                            double o = 0;
                            for (int i = 0; i < kd; i++)
                            {
                                s[(i * vd) + j] += k[i] * d[j];
                                o += s[(i * vd) + j] * q[i];
                            }

                            Store(a.Output.Address, a.DType, vo + j, o);
                        }
                    }

                    for (long i = 0; i < stateLen; i++)
                    {
                        state[i] = (float)s[i];
                    }
                }
            }
        }
        finally
        {
            NativeMemory.Free(mem);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>Depthwise causal convolution, one filter per channel. With <c>x_{-(Width-1)..-1}</c> the slot's ConvState for the channel (oldest first) and x_0.. the sequence's inputs,
    /// <c>y_t = Bias + sum_{j=0}^{Width-1} Weight[c, j] * x_{t - (Width - 1) + j}</c>, so Weight[c, Width - 1] multiplies the current token. SiLU is applied last when Silu is set.
    /// Afterwards the slot holds the last Width - 1 inputs of the history followed by the sequence, oldest first.</para>
    /// <para>The state is always read: a fresh sequence needs a zeroed slot (the arguments carry no UseInitialState). Input and Output must not overlap. Width is at most 16.
    /// ConvState is read and written as its span's Dtype (DType when it has none); a sequence of zero tokens leaves its slot untouched.</para>
    /// </remarks>
    public static void CausalConv1d(in CausalConv1dArgs a)
    {
        int w = a.Width;
        if (w < 1 || w > MaxConvWidth)
        {
            throw new ArgumentException($"Width must be in [1, {MaxConvWidth}].", nameof(a));
        }

        int sl = w - 1;
        Need(a.Input, (long)a.Tokens * a.Channels, nameof(a.Input));
        Need(a.Weight, (long)a.Channels * w, nameof(a.Weight));
        Need(a.Output, (long)a.Tokens * a.Channels, nameof(a.Output));
        Need(a.StateSlots, a.Batch, nameof(a.StateSlots));
        Need(a.CuSeqlens, a.Batch + 1L, nameof(a.CuSeqlens));
        bool bias = a.Bias.Address != 0;
        if (bias)
        {
            Need(a.Bias, a.Channels, nameof(a.Bias));
        }

        DType st = a.ConvState.Dtype == DType.Undefined ? a.DType : a.ConvState.Dtype;
        Span<double> next = stackalloc double[MaxConvWidth];
        for (int b = 0; b < a.Batch; b++)
        {
            long t0 = LoadInt(a.CuSeqlens.Address, a.CuSeqlens.Dtype, b);
            long t1 = LoadInt(a.CuSeqlens.Address, a.CuSeqlens.Dtype, b + 1L);
            long slot = LoadInt(a.StateSlots.Address, a.StateSlots.Dtype, b);
            if (t0 < 0 || t1 < t0 || t1 > a.Tokens)
            {
                throw new ArgumentException($"Sequence {b} has an invalid CuSeqlens range.", nameof(a));
            }

            long n = t1 - t0;
            if (n == 0)
            {
                continue;
            }

            for (int c = 0; c < a.Channels; c++)
            {
                long so = ((slot * a.Channels) + c) * sl;
                NeedRange(a.ConvState, so, sl, nameof(a.ConvState));
                for (long t = 0; t < n; t++)
                {
                    double y = bias ? Load(a.Bias.Address, a.DType, c) : 0.0;
                    for (int j = 0; j < w; j++)
                    {
                        y += Load(a.Weight.Address, a.DType, ((long)c * w) + j) * ConvInput(a, st, t0, so, c, t + j, sl);
                    }

                    Store(a.Output.Address, a.DType, ((t0 + t) * a.Channels) + c, a.Silu ? Silu(y) : y);
                }

                for (int j = 0; j < sl; j++)
                {
                    next[j] = ConvInput(a, st, t0, so, c, n + j, sl);
                }

                for (int j = 0; j < sl; j++)
                {
                    Store(a.ConvState.Address, st, so + j, next[j]);
                }
            }
        }
    }

    // Element i of [state (Width - 1 entries), sequence inputs] for one channel.
    private static double ConvInput(in CausalConv1dArgs a, DType stateType, long t0, long stateOffset, int c, long i, int stateLen) =>
        i < stateLen
            ? Load(a.ConvState.Address, stateType, stateOffset + i)
            : Load(a.Input.Address, a.DType, ((t0 + i - stateLen) * a.Channels) + c);

    /// <inheritdoc/>
    /// <remarks>Copies slot Pairs[i,0] to slot Pairs[i,1] of Pool as SlotBytes bytes, in pair order (a later pair sees an earlier pair's write); a pair with equal slots is a no-op.</remarks>
    public static void RecurrentStateCopy(in RecurrentStateCopyArgs a)
    {
        Need(a.Pairs, 2L * a.Count, nameof(a.Pairs));
        CopyPairs(a.Pool, a.Pool, a.Pairs, a.Count, a.SlotBytes, nameof(a));
    }

    /// <inheritdoc/>
    /// <remarks>Pairs[i,0] is a Pool slot and Pairs[i,1] a Snapshots entry; Restore false copies slot to snapshot, true copies snapshot to slot. Byte copies in pair order.</remarks>
    public static void RecurrentStateSnapshot(in RecurrentStateSnapshotArgs a)
    {
        Need(a.Pairs, 2L * a.Count, nameof(a.Pairs));
        CopyPairs(a.Restore ? a.Snapshots : a.Pool, a.Restore ? a.Pool : a.Snapshots, a.Pairs, a.Count, a.SlotBytes, nameof(a), swap: a.Restore);
    }
}

using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

public sealed class HybridTests
{
    [Fact]
    public void GatedDeltaRuleHandComputedScalarCase()
    {
        // KDim = VDim = 1, no normalisation, scale 1, empty initial state.
        using var q = Mem.F64(1, 3);
        using var k = Mem.F64(1, 1);
        using var v = Mem.F64(2, 1);
        using var g = Mem.F64(0, Math.Log(2));
        using var beta = Mem.F64(0.5, 1);
        using var o = Mem.Zeros(DType.F64, 2);
        using var state = Mem.F32(123); // ignored: UseInitialState is false
        using var slots = Mem.I32(0);
        using var cu = Mem.I32(0, 2);
        CpuOps.GatedDeltaRule(Args(q, k, v, g, beta, o, state, slots, cu, 1, 2, 1, 1, 1, 1, 1f, false, false));
        // t0: S = 0, d = 0.5 * (2 - 0) = 1, S = 1, o = 1.   t1: S = 2, d = 1 * (1 - 2) = -1, S = 1, o = 1 * 3.
        Near.Eq([1.0, 3.0], o.ReadF64(), 1e-14);
        Assert.Equal([1f], state.ReadF32());
    }

    private static GatedDeltaRuleArgs Args(Mem q, Mem k, Mem v, Mem g, Mem beta, Mem o, Mem state, Mem slots, Mem cu, int batch, int tokens, int kh, int vh, int kd, int vd, float scale, bool useInit, bool l2) => new()
    {
        Q = q.Span, K = k.Span, V = v.Span, G = g.Span, Beta = beta.Span, Output = o.Span, State = state.Span, StateSlots = slots.Span, CuSeqlens = cu.Span,
        Batch = batch, Tokens = tokens, KHeads = kh, VHeads = vh, KDim = kd, VDim = vd, Scale = scale, UseInitialState = useInit, L2NormQk = l2, DType = DType.F64,
    };

    // Independent formulation: S' = (I - beta k k^T) (e^g S) + beta k v^T, as explicit matrices; o = S'^T q.
    private static (double[] Out, double[,] State) Matrix(double[][] q, double[][] k, double[][] v, double[] g, double[] beta, double[,] s0, double scale, bool l2)
    {
        int kd = q[0].Length;
        int vd = v[0].Length;
        var s = (double[,])s0.Clone();
        var outs = new List<double>();
        for (int t = 0; t < q.Length; t++)
        {
            double[] qt = q[t];
            double[] kt = k[t];
            if (l2)
            {
                qt = qt.Select(x => x / Math.Sqrt(qt.Sum(y => y * y) + 1e-6)).ToArray();
                kt = kt.Select(x => x / Math.Sqrt(k[t].Sum(y => y * y) + 1e-6)).ToArray();
            }

            var decayed = new double[kd, vd];
            for (int i = 0; i < kd; i++)
            {
                for (int j = 0; j < vd; j++)
                {
                    decayed[i, j] = Math.Exp(g[t]) * s[i, j];
                }
            }

            var next = new double[kd, vd];
            for (int i = 0; i < kd; i++)
            {
                for (int j = 0; j < vd; j++)
                {
                    double proj = 0;
                    for (int m = 0; m < kd; m++)
                    {
                        proj += ((i == m ? 1.0 : 0.0) - (beta[t] * kt[i] * kt[m])) * decayed[m, j];
                    }

                    next[i, j] = proj + (beta[t] * kt[i] * v[t][j]);
                }
            }

            s = next;
            for (int j = 0; j < vd; j++)
            {
                double acc = 0;
                for (int i = 0; i < kd; i++)
                {
                    acc += s[i, j] * qt[i] * scale;
                }

                outs.Add(acc);
            }
        }

        return (outs.ToArray(), s);
    }

    private static double[][] Rand(Random r, int n, int w) => Enumerable.Range(0, n).Select(_ => Enumerable.Range(0, w).Select(_ => (r.NextDouble() * 2) - 1).ToArray()).ToArray();

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GatedDeltaRuleMatchesTheMatrixFormulationWithGroupedHeadsAndSlots(bool l2)
    {
        const int kh = 2;
        const int vh = 4;
        const int kd = 3;
        const int vd = 2;
        var rng = new Random(77);
        int[] lens = [5, 3];
        int tokens = lens.Sum();
        var q = Rand(rng, tokens * kh, kd);
        var k = Rand(rng, tokens * kh, kd);
        var v = Rand(rng, tokens * vh, vd);
        double[] g = Enumerable.Range(0, tokens * vh).Select(_ => -rng.NextDouble()).ToArray();
        double[] beta = Enumerable.Range(0, tokens * vh).Select(_ => rng.NextDouble()).ToArray();
        // State pool of 4 slots; sequences use slots 3 and 1 with non-zero initial state; slots 0 and 2 must stay untouched.
        var pool = Enumerable.Range(0, 4 * vh * kd * vd).Select(_ => (float)((rng.NextDouble() * 2) - 1)).ToArray();
        float[] before = (float[])pool.Clone();
        using var qm = Mem.F64(q.SelectMany(x => x).ToArray());
        using var km = Mem.F64(k.SelectMany(x => x).ToArray());
        using var vm = Mem.F64(v.SelectMany(x => x).ToArray());
        using var gm = Mem.F64(g);
        using var bm = Mem.F64(beta);
        using var om = Mem.Zeros(DType.F64, tokens * vh * vd);
        using var st = Mem.F32(pool);
        using var slots = Mem.I32(3, 1);
        using var cu = Mem.I32(0, 5, 8);
        double scale = (float)0.7;
        CpuOps.GatedDeltaRule(Args(qm, km, vm, gm, bm, om, st, slots, cu, 2, tokens, kh, vh, kd, vd, (float)scale, true, l2));
        double[] got = om.ReadF64();
        float[] after = st.ReadF32();
        int t0 = 0;
        for (int b = 0; b < 2; b++)
        {
            int slot = slots.ReadI32()[b];
            for (int h = 0; h < vh; h++)
            {
                var s0 = new double[kd, vd];
                for (int i = 0; i < kd; i++)
                {
                    for (int j = 0; j < vd; j++)
                    {
                        s0[i, j] = before[(((slot * vh) + h) * kd * vd) + (i * vd) + j];
                    }
                }

                int khd = h / (vh / kh);
                var (exp, sEnd) = Matrix(
                    Enumerable.Range(t0, lens[b]).Select(t => q[(t * kh) + khd]).ToArray(),
                    Enumerable.Range(t0, lens[b]).Select(t => k[(t * kh) + khd]).ToArray(),
                    Enumerable.Range(t0, lens[b]).Select(t => v[(t * vh) + h]).ToArray(),
                    Enumerable.Range(t0, lens[b]).Select(t => g[(t * vh) + h]).ToArray(),
                    Enumerable.Range(t0, lens[b]).Select(t => beta[(t * vh) + h]).ToArray(),
                    s0, scale, l2);
                for (int t = 0; t < lens[b]; t++)
                {
                    for (int j = 0; j < vd; j++)
                    {
                        Assert.Equal(exp[(t * vd) + j], got[((((t0 + t) * vh) + h) * vd) + j], 1e-12);
                    }
                }

                for (int i = 0; i < kd; i++)
                {
                    for (int j = 0; j < vd; j++)
                    {
                        Assert.Equal((float)sEnd[i, j], after[(((slot * vh) + h) * kd * vd) + (i * vd) + j]);
                    }
                }
            }

            t0 += lens[b];
        }

        foreach (int untouched in new[] { 0, 2 })
        {
            int n = vh * kd * vd;
            Assert.Equal(before.Skip(untouched * n).Take(n), after.Skip(untouched * n).Take(n));
        }
    }

    [Fact]
    public void GatedDeltaRuleInTwoCallsEqualsOneCallUpToTheFloatStateRounding()
    {
        const int kd = 4;
        const int vd = 3;
        const int n = 9;
        var rng = new Random(11);
        var q = Rand(rng, n, kd);
        var k = Rand(rng, n, kd);
        var v = Rand(rng, n, vd);
        double[] g = Enumerable.Range(0, n).Select(_ => -rng.NextDouble()).ToArray();
        double[] beta = Enumerable.Range(0, n).Select(_ => rng.NextDouble()).ToArray();

        double[] Flat(double[][] a, int from, int count) => a.Skip(from).Take(count).SelectMany(x => x).ToArray();

        double[] RunRange(int from, int count, Mem state, bool init)
        {
            using var qm = Mem.F64(Flat(q, from, count));
            using var km = Mem.F64(Flat(k, from, count));
            using var vm = Mem.F64(Flat(v, from, count));
            using var gm = Mem.F64(g.Skip(from).Take(count).ToArray());
            using var bm = Mem.F64(beta.Skip(from).Take(count).ToArray());
            using var om = Mem.Zeros(DType.F64, count * vd);
            using var slots = Mem.I32(0);
            using var cu = Mem.I32(0, count);
            CpuOps.GatedDeltaRule(Args(qm, km, vm, gm, bm, om, state, slots, cu, 1, count, 1, 1, kd, vd, 0.5f, init, true));
            return om.ReadF64();
        }

        using var whole = Mem.Zeros(DType.F32, kd * vd);
        double[] all = RunRange(0, n, whole, false);
        using var split = Mem.Zeros(DType.F32, kd * vd);
        double[] first = RunRange(0, 4, split, false);
        double[] second = RunRange(4, 5, split, true);
        Near.Eq(all, [.. first, .. second], 1e-6);
        Near.Eq(whole.ReadF32().Select(x => (double)x).ToArray(), split.ReadF32().Select(x => (double)x).ToArray(), 1e-6);
        // Token-by-token decode through the same slot agrees too.
        using var steps = Mem.Zeros(DType.F32, kd * vd);
        var decode = new List<double>();
        for (int t = 0; t < n; t++)
        {
            decode.AddRange(RunRange(t, 1, steps, t > 0));
        }

        Near.Eq(all, decode.ToArray(), 1e-5);
    }

    [Fact]
    public void GatedDeltaRuleScaleZeroMeansInverseRootOfKDim()
    {
        double[] Run(float scale)
        {
            using var q = Mem.F64(1, 2, 3, 4);
            using var k = Mem.F64(0.5, 0.5, 0.5, 0.5);
            using var v = Mem.F64(1);
            using var g = Mem.F64(0);
            using var b = Mem.F64(1);
            using var o = Mem.Zeros(DType.F64, 1);
            using var st = Mem.Zeros(DType.F32, 4);
            using var slots = Mem.I32(0);
            using var cu = Mem.I32(0, 1);
            CpuOps.GatedDeltaRule(Args(q, k, v, g, b, o, st, slots, cu, 1, 1, 1, 1, 4, 1, scale, false, false));
            return o.ReadF64();
        }

        Assert.Equal(Run(0.5f)[0], Run(0f)[0], 1e-14);
    }

    private static CausalConv1dArgs ConvArgs(Mem x, Mem w, Mem? bias, Mem y, Mem state, Mem slots, Mem cu, int batch, int tokens, int ch, int width, bool silu) => new()
    {
        Input = x.Span, Weight = w.Span, Bias = bias?.Span ?? default, Output = y.Span, ConvState = state.Span, StateSlots = slots.Span, CuSeqlens = cu.Span,
        Batch = batch, Tokens = tokens, Channels = ch, Width = width, Silu = silu, DType = DType.F64,
    };

    [Fact]
    public void CausalConv1dHandComputedWithAsymmetricKernelAndState()
    {
        // Width 3, one channel; state (oldest first) [1, 2]; weights [1, 10, 100]; inputs 3, 4.
        using var x = Mem.F64(3, 4);
        using var w = Mem.F64(1, 10, 100);
        using var y = Mem.Zeros(DType.F64, 2);
        using var st = Mem.F64(1, 2);
        using var slots = Mem.I32(0);
        using var cu = Mem.I32(0, 2);
        CpuOps.CausalConv1d(ConvArgs(x, w, null, y, st, slots, cu, 1, 2, 1, 3, false));
        Assert.Equal([(1 * 1) + (10 * 2) + (100 * 3), (1 * 2) + (10 * 3) + (100 * 4)], y.ReadF64().Select(d => (int)d));
        Assert.Equal([3.0, 4.0], st.ReadF64());
    }

    [Fact]
    public void CausalConv1dShortSequenceShiftsTheStateAndBiasAndSiluApply()
    {
        using var x = Mem.F64(3);
        using var w = Mem.F64(1, 10, 100);
        using var bias = Mem.F64(-5);
        using var y = Mem.Zeros(DType.F64, 1);
        using var st = Mem.F64(1, 2);
        using var slots = Mem.I32(0);
        using var cu = Mem.I32(0, 1);
        CpuOps.CausalConv1d(ConvArgs(x, w, bias, y, st, slots, cu, 1, 1, 1, 3, true));
        double pre = 1 + 20 + 300 - 5;
        Assert.Equal(pre / (1 + Math.Exp(-pre)), y.ReadF64()[0], 1e-10);
        Assert.Equal([2.0, 3.0], st.ReadF64());
    }

    [Fact]
    public void CausalConv1dOneCallEqualsTokenByTokenCallsAndSlotsAreIndependent()
    {
        const int ch = 3;
        const int width = 4;
        const int n = 7;
        var rng = new Random(3);
        double[] xs = Enumerable.Range(0, n * ch).Select(_ => (rng.NextDouble() * 2) - 1).ToArray();
        double[] ws = Enumerable.Range(0, ch * width).Select(_ => (rng.NextDouble() * 2) - 1).ToArray();
        double[] init = Enumerable.Range(0, 2 * ch * (width - 1)).Select(_ => (rng.NextDouble() * 2) - 1).ToArray();
        using var w = Mem.F64(ws);
        using var x = Mem.F64(xs);
        using var y = Mem.Zeros(DType.F64, n * ch);
        using var st = Mem.F64(init);
        using var slots = Mem.I32(1);
        using var cu = Mem.I32(0, n);
        CpuOps.CausalConv1d(ConvArgs(x, w, null, y, st, slots, cu, 1, n, ch, width, true));

        using var st2 = Mem.F64(init);
        var steps = new List<double>();
        for (int t = 0; t < n; t++)
        {
            using var xt = Mem.F64(xs.Skip(t * ch).Take(ch).ToArray());
            using var yt = Mem.Zeros(DType.F64, ch);
            using var cu1 = Mem.I32(0, 1);
            CpuOps.CausalConv1d(ConvArgs(xt, w, null, yt, st2, slots, cu1, 1, 1, ch, width, true));
            steps.AddRange(yt.ReadF64());
        }

        Near.Eq(y.ReadF64(), steps.ToArray(), 0);
        Assert.Equal(st.ReadF64(), st2.ReadF64());
        // Slot 0 is untouched.
        Assert.Equal(init.Take(ch * (width - 1)), st.ReadF64().Take(ch * (width - 1)));
        // The final state is the last width - 1 inputs per channel, oldest first.
        for (int c = 0; c < ch; c++)
        {
            for (int j = 0; j < width - 1; j++)
            {
                Assert.Equal(xs[((n - (width - 1) + j) * ch) + c], st.ReadF64()[(ch * (width - 1)) + (c * (width - 1)) + j]);
            }
        }
    }

    [Fact]
    public void CausalConv1dMatchesAnIndependentConvolutionOverTwoPackedSequences()
    {
        const int ch = 2;
        const int width = 3;
        int[] lens = [4, 2];
        var rng = new Random(9);
        double[] xs = Enumerable.Range(0, 6 * ch).Select(_ => (rng.NextDouble() * 2) - 1).ToArray();
        double[] ws = Enumerable.Range(0, ch * width).Select(_ => (rng.NextDouble() * 2) - 1).ToArray();
        using var x = Mem.F64(xs);
        using var w = Mem.F64(ws);
        using var y = Mem.Zeros(DType.F64, 6 * ch);
        using var st = Mem.Zeros(DType.F64, 2 * ch * (width - 1));
        using var slots = Mem.I32(0, 1);
        using var cu = Mem.I32(0, 4, 6);
        CpuOps.CausalConv1d(ConvArgs(x, w, null, y, st, slots, cu, 2, 6, ch, width, false));
        double[] got = y.ReadF64();
        int start = 0;
        for (int b = 0; b < 2; b++)
        {
            for (int t = 0; t < lens[b]; t++)
            {
                for (int c = 0; c < ch; c++)
                {
                    double acc = 0;
                    for (int j = 0; j < width; j++)
                    {
                        int src = t - (width - 1) + j;
                        acc += src < 0 ? 0 : ws[(c * width) + j] * xs[((start + src) * ch) + c];
                    }

                    Assert.Equal(acc, got[((start + t) * ch) + c], 1e-13);
                }
            }

            start += lens[b];
        }
    }

    [Fact]
    public void RecurrentStateCopyAndSnapshotAreBitExactByteCopies()
    {
        float[] src = [1.5f, -0f, float.NaN, 4f, 5f, 6f];
        using var pool = Mem.F32(src);
        using var pairs = Mem.I32(0, 2, 1, 0);
        // Slot = 2 floats (8 bytes). Slot 2 <- slot 0, then slot 0 <- slot 1.
        CpuOps.RecurrentStateCopy(new RecurrentStateCopyArgs { Pool = pool.Span, Pairs = pairs.Span, SlotBytes = 8, Count = 2 });
        float[] after = pool.ReadF32();
        Assert.Equal(BitConverter.SingleToInt32Bits(float.NaN), BitConverter.SingleToInt32Bits(after[2]));
        Assert.Equal([float.NaN, 4f], after[0..2].Select(x => x).ToArray(), new NanEqual());
        Assert.Equal([1.5f, -0f], after[4..6]);
        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(after[5]));

        using var snaps = Mem.Zeros(DType.F32, 4);
        using var p2 = Mem.I32(2, 1);
        CpuOps.RecurrentStateSnapshot(new RecurrentStateSnapshotArgs { Pool = pool.Span, Snapshots = snaps.Span, Pairs = p2.Span, SlotBytes = 8, Count = 1, Restore = false });
        Assert.Equal([0f, 0f, 1.5f, -0f], snaps.ReadF32());
        using var pool2 = Mem.Zeros(DType.F32, 6);
        CpuOps.RecurrentStateSnapshot(new RecurrentStateSnapshotArgs { Pool = pool2.Span, Snapshots = snaps.Span, Pairs = p2.Span, SlotBytes = 8, Count = 1, Restore = true });
        Assert.Equal([0f, 0f, 0f, 0f, 1.5f, -0f], pool2.ReadF32());
        using var oob = Mem.I32(3, 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuOps.RecurrentStateCopy(new RecurrentStateCopyArgs { Pool = pool.Span, Pairs = oob.Span, SlotBytes = 8, Count = 1 }));
    }

    private sealed class NanEqual : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => BitConverter.SingleToInt32Bits(x) == BitConverter.SingleToInt32Bits(y);

        public int GetHashCode(float obj) => BitConverter.SingleToInt32Bits(obj);
    }

    [Fact]
    public void Qwen35SmokeShapesRun()
    {
        // Qwen3.5-0.8B linear layer: 16 key and 16 value heads of width 128; conv width 4; two tokens, one sequence.
        const int heads = 16;
        const int dim = 128;
        const int tokens = 2;
        var rng = new Random(1);
        float[] Rand(int n) => Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2) - 1)).ToArray();
        using var q = Mem.F32(Rand(tokens * heads * dim));
        using var k = Mem.F32(Rand(tokens * heads * dim));
        using var v = Mem.F32(Rand(tokens * heads * dim));
        using var g = Mem.F32(Enumerable.Repeat(-0.1f, tokens * heads).ToArray());
        using var b = Mem.F32(Enumerable.Repeat(0.5f, tokens * heads).ToArray());
        using var o = Mem.Zeros(DType.F32, tokens * heads * dim);
        using var st = Mem.Zeros(DType.F32, heads * dim * dim);
        using var slots = Mem.I32(0);
        using var cu = Mem.I32(0, tokens);
        var a = Args(q, k, v, g, b, o, st, slots, cu, 1, tokens, heads, heads, dim, dim, 0f, false, true);
        a.DType = DType.F32;
        CpuOps.GatedDeltaRule(a);
        Assert.All(o.ReadF32(), x => Assert.True(float.IsFinite(x)));
        Assert.Contains(o.ReadF32(), x => x != 0f);

        const int ch = 3 * heads * dim;
        using var cx = Mem.F32(Rand(tokens * ch));
        using var cw = Mem.F32(Rand(ch * 4));
        using var cy = Mem.Zeros(DType.F32, tokens * ch);
        using var cs = Mem.Zeros(DType.F32, ch * 3);
        var c = ConvArgs(cx, cw, null, cy, cs, slots, cu, 1, tokens, ch, 4, true);
        c.DType = DType.F32;
        CpuOps.CausalConv1d(c);
        Assert.All(cy.ReadF32(), x => Assert.True(float.IsFinite(x)));
    }
}

using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>The MoE ops of R-080 on CpuOps: hand-computed cases, the edge cases of routing and permutation, and FusedMoe against an independent per-token transcription.</summary>
public sealed class MoeTests
{
    private static float Bf16(double v) => BitConverter.Int32BitsToSingle(CpuElements.ToBFloat16(v) << 16);

    private static (int[] Ids, float[] Weights) Route(Mem logits, int tokens, int experts, int topK, bool renormalize, DType dtype)
    {
        using var w = Mem.Zeros(DType.F32, tokens * topK);
        using var ids = Mem.Zeros(DType.I32, tokens * topK);
        CpuOps.MoeRouteTopK(new MoeRouteTopKArgs
        {
            Logits = logits.Span, Weights = w.Span, ExpertIds = ids.Span, Tokens = tokens, Experts = experts, TopK = topK, Renormalize = renormalize, DType = dtype,
        });
        return (ids.ReadI32(), w.ReadF32());
    }

    [Fact]
    public void RouteTopKTakesTheLargestLogitsWithTiesToTheLowerExpert()
    {
        using var logits = Mem.F32(1, 3, 2, 3, /* token 1 */ 0, 0, 0, 0);
        (int[] ids, float[] w) = Route(logits, 2, 4, 2, renormalize: false, DType.F32);
        Assert.Equal([1, 3, 0, 1], ids);
        double denom = Math.Exp(-2) + 1 + Math.Exp(-1) + 1;
        Assert.Equal((float)(1 / denom), w[0]);
        Assert.Equal((float)(1 / denom), w[1]);
        Assert.Equal(0.25f, w[2]);
        Assert.Equal(0.25f, w[3]);

        (_, float[] rn) = Route(logits, 2, 4, 2, renormalize: true, DType.F32);
        Assert.Equal([0.5f, 0.5f, 0.5f, 0.5f], rn);
    }

    [Fact]
    public void RouteTopKSeparatesANearTieOfOneBf16Ulp()
    {
        // 1.0 and its bf16 successor 1.0078125: the larger wins although it has the higher index, and the order is by logit.
        using var logits = Mem.Bf16(1.0f, -2f, 1.0078125f, 0.5f);
        (int[] ids, float[] w) = Route(logits, 1, 4, 3, renormalize: true, DType.BF16);
        Assert.Equal([2, 0, 3], ids);
        Assert.True(w[0] > w[1] && w[1] > w[2]);
        Assert.Equal(1.0, w.Sum(), 6);
    }

    [Fact]
    public void RouteTopKRanksNaNBelowEveryNumber()
    {
        using var logits = Mem.F32(float.NaN, float.NegativeInfinity, -5);
        (int[] ids, _) = Route(logits, 1, 3, 3, renormalize: false, DType.F32);
        Assert.Equal([2, 1, 0], ids);
    }

    [Fact]
    public void RouteTopKUpcastsBf16AndRoundsTheWeightsOnceToFloat32()
    {
        float[] l = [0.375f, -1.25f, 2.5f, 0.0625f, 1.5f];
        using var logits = Mem.Bf16(l);
        (int[] ids, float[] w) = Route(logits, 1, 5, 2, renormalize: true, DType.BF16);
        Assert.Equal([2, 4], ids);
        double a = Math.Exp(2.5), b = Math.Exp(1.5);
        Assert.Equal((float)(a / (a + b)), w[0]);
        Assert.Equal((float)(b / (a + b)), w[1]);
    }

    [Fact]
    public void RouteTopKRejectsATopKAboveTheExpertCount()
    {
        using var logits = Mem.F32(1, 2);
        Assert.Throws<ArgumentException>(() => Route(logits, 1, 2, 3, renormalize: true, DType.F32));
    }

    [Fact]
    public void PermuteIsAStableCountingSortWithEmptyExperts()
    {
        // 3 tokens of width 2, top-2, 4 experts; expert 2 gets nothing.
        using var x = Mem.F32(10, 11, 20, 21, 30, 31);
        using var ids = Mem.I32(3, 0, 0, 3, 1, 0);
        using var xp = Mem.Zeros(DType.F32, 12);
        using var counts = Mem.Zeros(DType.I32, 4);
        using var offsets = Mem.Zeros(DType.I32, 5);
        using var perm = Mem.Zeros(DType.I32, 6);
        CpuOps.MoePermute(new MoePermuteArgs
        {
            Input = x.Span, ExpertIds = ids.Span, Output = xp.Span, ExpertCounts = counts.Span, ExpertOffsets = offsets.Span, Permutation = perm.Span,
            Tokens = 3, Dim = 2, Experts = 4, TopK = 2, DType = DType.F32,
        });
        Assert.Equal([3, 1, 0, 2], counts.ReadI32());
        Assert.Equal([0, 3, 4, 4, 6], offsets.ReadI32());
        // Pairs in flat order: (0,0)->3, (0,1)->0, (1,0)->0, (1,1)->3, (2,0)->1, (2,1)->0. Expert 0 takes pairs 1, 2, 5 in that order.
        Assert.Equal([4, 0, 1, 5, 3, 2], perm.ReadI32());
        Assert.Equal([10, 11, 20, 21, 30, 31, 30, 31, 10, 11, 20, 21], xp.ReadF32());
    }

    [Fact]
    public void PermuteRejectsAnExpertIdOutOfRangeBeforeWriting()
    {
        using var x = Mem.F32(1, 2);
        using var ids = Mem.I32(0, 4);
        using var xp = Mem.F32(-1, -1, -1, -1);
        using var counts = Mem.I32(7, 7, 7, 7);
        using var offsets = Mem.Zeros(DType.I32, 5);
        using var perm = Mem.Zeros(DType.I32, 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuOps.MoePermute(new MoePermuteArgs
        {
            Input = x.Span, ExpertIds = ids.Span, Output = xp.Span, ExpertCounts = counts.Span, ExpertOffsets = offsets.Span, Permutation = perm.Span,
            Tokens = 1, Dim = 2, Experts = 4, TopK = 2, DType = DType.F32,
        }));
        Assert.Equal([7, 7, 7, 7], counts.ReadI32());
        Assert.Equal([-1f, -1, -1, -1], xp.ReadF32());
    }

    [Fact]
    public void UnpermuteIsTheWeightedSumOfEachTokensRows()
    {
        // Rows of the permuted buffer, width 2; token 0 takes rows 2 and 0, token 1 rows 1 and 3.
        using var y = Mem.F64(1, 2, 10, 20, 100, 200, 1000, 2000);
        using var perm = Mem.I32(2, 0, 1, 3);
        using var w = Mem.F32(0.25f, 0.75f, 0.5f, 0.5f);
        using var o = Mem.Zeros(DType.F64, 4);
        CpuOps.MoeUnpermute(new MoeUnpermuteArgs { Input = y.Span, Permutation = perm.Span, Weights = w.Span, Output = o.Span, Tokens = 2, Dim = 2, TopK = 2, DType = DType.F64 });
        Assert.Equal([25.75, 51.5, 505, 1010], o.ReadF64());
    }

    [Fact]
    public void UnpermuteAfterPermuteWithUnitWeightsRestoresTheInput()
    {
        float[] v = [.. Enumerable.Range(0, 5 * 3).Select(i => (float)Math.Round(Math.Sin(i) * 64) / 64)];
        using var x = Mem.Bf16(v);
        using var ids = Mem.I32(2, 0, 2, 1, 0);
        using var xp = Mem.Zeros(DType.BF16, 15);
        using var counts = Mem.Zeros(DType.I32, 3);
        using var offsets = Mem.Zeros(DType.I32, 4);
        using var perm = Mem.Zeros(DType.I32, 5);
        CpuOps.MoePermute(new MoePermuteArgs
        {
            Input = x.Span, ExpertIds = ids.Span, Output = xp.Span, ExpertCounts = counts.Span, ExpertOffsets = offsets.Span, Permutation = perm.Span,
            Tokens = 5, Dim = 3, Experts = 3, TopK = 1, DType = DType.BF16,
        });
        using var w = Mem.F32(1, 1, 1, 1, 1);
        using var back = Mem.Zeros(DType.BF16, 15);
        CpuOps.MoeUnpermute(new MoeUnpermuteArgs { Input = xp.Span, Permutation = perm.Span, Weights = w.Span, Output = back.Span, Tokens = 5, Dim = 3, TopK = 1, DType = DType.BF16 });
        Assert.Equal(x.ReadU16(), back.ReadU16());
    }

    [Fact]
    public void GroupedGemmEqualsOneTransposedGemmPerExpertAndSkipsEmptyExperts()
    {
        const int e = 3, n = 4, k = 5, rows = 6;
        var rng = new Random(3);
        float[] av = [.. Enumerable.Range(0, rows * k).Select(_ => Bf16((rng.NextDouble() * 2) - 1))];
        float[] bv = [.. Enumerable.Range(0, e * n * k).Select(_ => Bf16((rng.NextDouble() * 2) - 1))];
        using var a = Mem.Bf16(av);
        using var b = Mem.Bf16(bv);
        using var offsets = Mem.I32(0, 0, 4, 6); // expert 0 empty
        using var d = Mem.Zeros(DType.BF16, rows * n);
        CpuOps.GroupedGemm(new GroupedGemmArgs { A = a.Span, B = b.Span, D = d.Span, ExpertOffsets = offsets.Span, Experts = e, N = n, K = k, MaxRows = rows, DType = DType.BF16, WeightDType = DType.BF16 });

        ushort[] got = d.ReadU16();
        int[] off = [0, 0, 4, 6];
        for (int g = 1; g < e; g++)
        {
            int m = off[g + 1] - off[g];
            using var ag = Mem.Bf16(av[(off[g] * k)..(off[g + 1] * k)]);
            using var bg = Mem.Bf16(bv[(g * n * k)..((g + 1) * n * k)]);
            using var dg = Mem.Zeros(DType.BF16, m * n);
            CpuOps.Gemm(new GemmArgs { A = ag.Span, B = bg.Span, D = dg.Span, M = m, N = n, K = k, TransposeB = true, Alpha = 1f, DType = DType.BF16, WeightDType = DType.BF16 });
            Assert.Equal(dg.ReadU16(), got[(off[g] * n)..(off[g + 1] * n)]);
        }
    }

    [Theory]
    [InlineData(DType.F8E4M3)]
    [InlineData(DType.F8E5M2)]
    [InlineData(DType.I4)]
    [InlineData(DType.U4)]
    public void Fp8AndInt4ExpertWeightsAreUnsupported(DType weight)
    {
        using var a = Mem.Zeros(DType.BF16, 4);
        using var b = Mem.Zeros(DType.U8, 16);
        using var d = Mem.Zeros(DType.BF16, 4);
        using var offsets = Mem.I32(0, 1);
        using var logits = Mem.Zeros(DType.BF16, 1);
        Assert.Throws<NotSupportedException>(() => CpuOps.GroupedGemm(new GroupedGemmArgs { A = a.Span, B = b.Span, D = d.Span, ExpertOffsets = offsets.Span, Experts = 1, N = 2, K = 2, DType = DType.BF16, WeightDType = weight }));
        Assert.Throws<NotSupportedException>(() => CpuOps.FusedMoe(new FusedMoeArgs
        {
            Input = a.Span, RouterLogits = logits.Span, W13 = b.Span, W2 = b.Span, Output = d.Span, Tokens = 1, Dim = 2, Intermediate = 1, Experts = 1, TopK = 1, DType = DType.BF16, WeightDType = weight,
        }));
    }

    [Fact]
    public void GroupedGemmRejectsDecreasingOffsets()
    {
        using var a = Mem.Zeros(DType.F32, 8);
        using var b = Mem.Zeros(DType.F32, 8);
        using var d = Mem.Zeros(DType.F32, 8);
        using var offsets = Mem.I32(0, 3, 2);
        Assert.Throws<ArgumentException>(() => CpuOps.GroupedGemm(new GroupedGemmArgs { A = a.Span, B = b.Span, D = d.Span, ExpertOffsets = offsets.Span, Experts = 2, N = 2, K = 2, DType = DType.F32, WeightDType = DType.F32 }));
    }

    /// <summary>
    /// An independent transcription of a Qwen3 MoE block per token, written from the definition: softmax over the router logits, the
    /// top-k by logit (ties to the lower expert), renormalised weights in float32, and per chosen expert gate/up, SiLU times up and down,
    /// each rounded to bf16 where the pipeline stores, then the weighted sum rounded once.
    /// </summary>
    private static float[] ReferenceMoe(float[] x, float[] logits, float[] w13, float[] w2, int tokens, int dim, int inter, int experts, int topK)
    {
        var o = new float[tokens * dim];
        for (int t = 0; t < tokens; t++)
        {
            double[] l = [.. logits[(t * experts)..((t + 1) * experts)].Select(v => (double)v)];
            int[] chosen = [.. Enumerable.Range(0, experts).OrderByDescending(j => l[j]).ThenBy(j => j).Take(topK)];
            double max = l.Max();
            double denom = l.Sum(v => Math.Exp(v - max));
            double[] p = [.. chosen.Select(j => Math.Exp(l[j] - max) / denom)];
            float[] w = [.. p.Select(v => (float)(v / p.Sum()))];
            var acc = new double[dim];
            for (int s = 0; s < topK; s++)
            {
                int ex = chosen[s];
                var act = new float[inter];
                for (int i = 0; i < inter; i++)
                {
                    double g = 0, u = 0;
                    for (int d = 0; d < dim; d++)
                    {
                        g += x[(t * dim) + d] * (double)w13[(((ex * 2 * inter) + i) * dim) + d];
                        u += x[(t * dim) + d] * (double)w13[(((ex * 2 * inter) + inter + i) * dim) + d];
                    }

                    double gr = Bf16(g), ur = Bf16(u);
                    act[i] = Bf16(gr / (1 + Math.Exp(-gr)) * ur);
                }

                for (int d = 0; d < dim; d++)
                {
                    double y = 0;
                    for (int i = 0; i < inter; i++)
                    {
                        y += act[i] * (double)w2[(((ex * dim) + d) * inter) + i];
                    }

                    acc[d] += w[s] * (double)Bf16(y);
                }
            }

            for (int d = 0; d < dim; d++)
            {
                o[(t * dim) + d] = Bf16(acc[d]);
            }
        }

        return o;
    }

    private static float[] RunFused(float[] x, float[] logits, float[] w13, float[] w2, int tokens, int dim, int inter, int experts, int topK)
    {
        using var mx = Mem.Bf16(x);
        using var ml = Mem.Bf16(logits);
        using var m13 = Mem.Bf16(w13);
        using var m2 = Mem.Bf16(w2);
        using var o = Mem.Zeros(DType.BF16, tokens * dim);
        CpuOps.FusedMoe(new FusedMoeArgs
        {
            Input = mx.Span, RouterLogits = ml.Span, W13 = m13.Span, W2 = m2.Span, Output = o.Span, Tokens = tokens, Dim = dim, Intermediate = inter,
            Experts = experts, TopK = topK, Renormalize = true, DType = DType.BF16, WeightDType = DType.BF16,
        });
        return o.ReadBf16();
    }

    private static float[] Draw(Random rng, int n, double scale) => [.. Enumerable.Range(0, n).Select(_ => Bf16(((rng.NextDouble() * 2) - 1) * scale))];

    [Theory]
    [InlineData(1, 8, 2)]
    [InlineData(9, 8, 2)]
    [InlineData(5, 16, 4)]
    public void FusedMoeMatchesAnIndependentPerTokenTranscriptionBitForBit(int tokens, int experts, int topK)
    {
        const int dim = 24, inter = 12;
        var rng = new Random(tokens * 31 + experts);
        float[] x = Draw(rng, tokens * dim, 1.0);
        float[] logits = Draw(rng, tokens * experts, 2.0);
        float[] w13 = Draw(rng, experts * 2 * inter * dim, 1.5 / Math.Sqrt(dim));
        float[] w2 = Draw(rng, experts * dim * inter, 1.5 / Math.Sqrt(inter));
        float[] want = ReferenceMoe(x, logits, w13, w2, tokens, dim, inter, experts, topK);
        Assert.Equal(want, RunFused(x, logits, w13, w2, tokens, dim, inter, experts, topK));
    }

    [Fact]
    public void FusedMoeGivesIdenticalRowsForIdenticalTokensAndEqualsTheUnfusedComposition()
    {
        const int tokens = 6, dim = 16, inter = 8, experts = 8, topK = 2;
        var rng = new Random(11);
        float[] row = Draw(rng, dim, 1.0);
        float[] lrow = Draw(rng, experts, 2.0);
        float[] x = [.. Enumerable.Repeat(row, tokens).SelectMany(r => r)];
        float[] logits = [.. Enumerable.Repeat(lrow, tokens).SelectMany(r => r)];
        float[] w13 = Draw(rng, experts * 2 * inter * dim, 0.3);
        float[] w2 = Draw(rng, experts * dim * inter, 0.4);
        float[] fused = RunFused(x, logits, w13, w2, tokens, dim, inter, experts, topK);
        for (int t = 1; t < tokens; t++)
        {
            Assert.Equal(fused[..dim], fused[(t * dim)..((t + 1) * dim)]);
        }

        // The unfused ops called one by one give the same bytes.
        using var f = new Bag();
        var mx = f.Add(Mem.Bf16(x));
        var ml = f.Add(Mem.Bf16(logits));
        var m13 = f.Add(Mem.Bf16(w13));
        var m2 = f.Add(Mem.Bf16(w2));
        var tw = f.Add(Mem.Zeros(DType.F32, tokens * topK));
        var ids = f.Add(Mem.Zeros(DType.I32, tokens * topK));
        var counts = f.Add(Mem.Zeros(DType.I32, experts));
        var offsets = f.Add(Mem.Zeros(DType.I32, experts + 1));
        var perm = f.Add(Mem.Zeros(DType.I32, tokens * topK));
        var xp = f.Add(Mem.Zeros(DType.BF16, tokens * topK * dim));
        var g1 = f.Add(Mem.Zeros(DType.BF16, tokens * topK * 2 * inter));
        var act = f.Add(Mem.Zeros(DType.BF16, tokens * topK * inter));
        var y = f.Add(Mem.Zeros(DType.BF16, tokens * topK * dim));
        var o = f.Add(Mem.Zeros(DType.BF16, tokens * dim));
        CpuOps.MoeRouteTopK(new MoeRouteTopKArgs { Logits = ml.Span, Weights = tw.Span, ExpertIds = ids.Span, Tokens = tokens, Experts = experts, TopK = topK, Renormalize = true, DType = DType.BF16 });
        CpuOps.MoePermute(new MoePermuteArgs { Input = mx.Span, ExpertIds = ids.Span, Output = xp.Span, ExpertCounts = counts.Span, ExpertOffsets = offsets.Span, Permutation = perm.Span, Tokens = tokens, Dim = dim, Experts = experts, TopK = topK, DType = DType.BF16 });
        CpuOps.GroupedGemm(new GroupedGemmArgs { A = xp.Span, B = m13.Span, D = g1.Span, ExpertOffsets = offsets.Span, Experts = experts, N = 2 * inter, K = dim, MaxRows = tokens * topK, DType = DType.BF16, WeightDType = DType.BF16 });
        CpuOps.SiluAndMul(new SiluAndMulArgs { Input = g1.Span, Output = act.Span, Rows = tokens * topK, Dim = inter, DType = DType.BF16 });
        CpuOps.GroupedGemm(new GroupedGemmArgs { A = act.Span, B = m2.Span, D = y.Span, ExpertOffsets = offsets.Span, Experts = experts, N = dim, K = inter, MaxRows = tokens * topK, DType = DType.BF16, WeightDType = DType.BF16 });
        CpuOps.MoeUnpermute(new MoeUnpermuteArgs { Input = y.Span, Permutation = perm.Span, Weights = tw.Span, Output = o.Span, Tokens = tokens, Dim = dim, TopK = topK, DType = DType.BF16 });
        Assert.Equal(fused, o.ReadBf16());

        // All tokens identical: two experts take every row and the other six are empty.
        int[] c = counts.ReadI32();
        Assert.Equal(2, c.Count(n => n == tokens));
        Assert.Equal(experts - 2, c.Count(n => n == 0));
    }

    [Fact]
    public void FusedMoeAllowsInputToAliasOutputAndDoesNothingForZeroTokens()
    {
        const int tokens = 3, dim = 8, inter = 4, experts = 4, topK = 2;
        var rng = new Random(5);
        float[] x = Draw(rng, tokens * dim, 1.0);
        float[] logits = Draw(rng, tokens * experts, 1.0);
        float[] w13 = Draw(rng, experts * 2 * inter * dim, 0.5);
        float[] w2 = Draw(rng, experts * dim * inter, 0.5);
        float[] want = RunFused(x, logits, w13, w2, tokens, dim, inter, experts, topK);
        using var mx = Mem.Bf16(x);
        using var ml = Mem.Bf16(logits);
        using var m13 = Mem.Bf16(w13);
        using var m2 = Mem.Bf16(w2);
        var args = new FusedMoeArgs
        {
            Input = mx.Span, RouterLogits = ml.Span, W13 = m13.Span, W2 = m2.Span, Output = mx.Span, Tokens = tokens, Dim = dim, Intermediate = inter,
            Experts = experts, TopK = topK, Renormalize = true, DType = DType.BF16, WeightDType = DType.BF16,
        };
        CpuOps.FusedMoe(args);
        Assert.Equal(want, mx.ReadBf16());

        ushort[] before = mx.ReadU16();
        CpuOps.FusedMoe(args with { Tokens = 0 });
        Assert.Equal(before, mx.ReadU16());
    }

    [Fact]
    public void MoeCallsAllocateNoManagedMemory()
    {
        const int tokens = 4, dim = 8, inter = 4, experts = 4, topK = 2;
        var rng = new Random(2);
        using var f = new Bag();
        var x = f.Add(Mem.Bf16(Draw(rng, tokens * dim, 1)));
        var logits = f.Add(Mem.Bf16(Draw(rng, tokens * experts, 1)));
        var w13 = f.Add(Mem.Bf16(Draw(rng, experts * 2 * inter * dim, 0.5)));
        var w2 = f.Add(Mem.Bf16(Draw(rng, experts * dim * inter, 0.5)));
        var o = f.Add(Mem.Zeros(DType.BF16, tokens * dim));
        var tw = f.Add(Mem.Zeros(DType.F32, tokens * topK));
        var ids = f.Add(Mem.Zeros(DType.I32, tokens * topK));
        var counts = f.Add(Mem.Zeros(DType.I32, experts));
        var offsets = f.Add(Mem.Zeros(DType.I32, experts + 1));
        var perm = f.Add(Mem.Zeros(DType.I32, tokens * topK));
        var xp = f.Add(Mem.Zeros(DType.BF16, tokens * topK * dim));
        var g1 = f.Add(Mem.Zeros(DType.BF16, tokens * topK * 2 * inter));

        void Everything()
        {
            CpuOps.MoeRouteTopK(new MoeRouteTopKArgs { Logits = logits.Span, Weights = tw.Span, ExpertIds = ids.Span, Tokens = tokens, Experts = experts, TopK = topK, Renormalize = true, DType = DType.BF16 });
            CpuOps.MoePermute(new MoePermuteArgs { Input = x.Span, ExpertIds = ids.Span, Output = xp.Span, ExpertCounts = counts.Span, ExpertOffsets = offsets.Span, Permutation = perm.Span, Tokens = tokens, Dim = dim, Experts = experts, TopK = topK, DType = DType.BF16 });
            CpuOps.GroupedGemm(new GroupedGemmArgs { A = xp.Span, B = w13.Span, D = g1.Span, ExpertOffsets = offsets.Span, Experts = experts, N = 2 * inter, K = dim, DType = DType.BF16, WeightDType = DType.BF16 });
            CpuOps.MoeUnpermute(new MoeUnpermuteArgs { Input = xp.Span, Permutation = perm.Span, Weights = tw.Span, Output = o.Span, Tokens = tokens, Dim = dim, TopK = topK, DType = DType.BF16 });
            CpuOps.FusedMoe(new FusedMoeArgs { Input = x.Span, RouterLogits = logits.Span, W13 = w13.Span, W2 = w2.Span, Output = o.Span, Tokens = tokens, Dim = dim, Intermediate = inter, Experts = experts, TopK = topK, Renormalize = true, DType = DType.BF16, WeightDType = DType.BF16 });
        }

        Everything();
        Everything();
        long before = GC.GetAllocatedBytesForCurrentThread();
        Everything();
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }
}

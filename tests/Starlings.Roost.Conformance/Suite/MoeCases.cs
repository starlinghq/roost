using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>
/// The cases of the MoE ops (R-080): MoeRouteTopK, MoePermute, MoeUnpermute, GroupedGemm and FusedMoe, bf16 activations and bf16
/// expert weights. Widths are Qwen3-30B-A3B's as the S6 report (docs/m0/roost-s6-moe-kernels.md) read them: hidden 2048,
/// moe_intermediate_size 768, 128 experts, 8 per token, norm_topk_prob true. GroupedGemm and FusedMoe hold every expert's weights,
/// so their fixed Qwen3 cases keep the widths and use 16 of the 128 experts, and only in the large profile; CI runs scaled-down shapes.
/// Every op has the edge cases R-080 names: experts that receive no rows, all tokens identical (the same experts take every row), and
/// near-tie routing (two logits one bf16 ulp apart and an exact tie at the top-k boundary, which must go to the lower expert).
/// </summary>
/// <remarks>
/// No GPU variant of these ops exists yet (stream C, R-080's kernel half): on a GPU every case is reported <c>not_run</c>, and the CUDA
/// side of R-080 is <c>untested</c>. The cases borrow existing tolerance classes (<c>hybrid-f32</c> for the float32 routing weights,
/// <c>elementwise-bf16</c> for the combine, <c>gemm-bf16</c> for GroupedGemm and FusedMoe, <c>exact</c> for ids, counts, offsets,
/// permutations and permuted rows); none was sized against a GPU kernel.
/// </remarks>
public static partial class CaseCatalog
{
    private const int MoeHidden = 2048, MoeInter = 768, MoeExperts = 128, MoeTopK = 8;

    private static IEnumerable<OpCase> MoeCases(OpId op, ulong baseSeed, CaseProfile p) => op switch
    {
        OpId.MoeRouteTopK => RouteCases(baseSeed, p),
        OpId.MoePermute => PermuteCases(baseSeed, p),
        OpId.MoeUnpermute => UnpermuteCases(baseSeed, p),
        OpId.GroupedGemm => GroupedGemmCases(baseSeed, p),
        OpId.FusedMoe => FusedMoeCases(baseSeed, p),
        _ => throw new NotSupportedException($"No cases for {op}."),
    };

    private static string MoeSource(string family) => family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed";

    /// <summary>Router logits [tokens, experts] in bf16: normal ("random"), one row repeated ("identical") or with near-ties ("near-tie").</summary>
    private static HostBuf Logits(string kind, int tokens, int experts, int topK, Rng rng)
    {
        var l = new HostBuf(DType.BF16, (long)tokens * experts);
        if (kind == "identical")
        {
            using var row = new HostBuf(DType.BF16, experts).Randn(rng, 1.5);
            for (int t = 0; t < tokens; t++)
            {
                row.Raw().CopyTo(l.Raw()[(t * experts * 2)..]);
            }

            return l;
        }

        l.Randn(rng, 1.5);
        if (kind != "near-tie")
        {
            return l;
        }

        // Per token, in a seeded expert order: the first two at 5 and 5 + 2^-5 (one bf16 ulp apart), the next topK - 3 above 3, then two at
        // exactly 2.5, which straddle the top-k boundary, and every other expert below 0. Which of each pair has the lower index is random.
        for (int t = 0; t < tokens; t++)
        {
            int[] order = Shuffled(rng, experts);
            long row = (long)t * experts;
            for (int j = 0; j < experts; j++)
            {
                double v = j switch
                {
                    0 => 5.0,
                    1 => 5.03125,
                    _ when j < topK - 1 => 3.0 + (0.25 * (topK - j)),
                    _ when j <= topK => 2.5,
                    _ => -Math.Abs(rng.Normal()) - 0.5,
                };
                l.Set(row + order[j], v);
            }
        }

        return l;
    }

    private static int[] Shuffled(Rng rng, int n)
    {
        int[] all = Enumerable.Range(0, n).ToArray();
        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Int(0, i + 1);
            (all[i], all[j]) = (all[j], all[i]);
        }

        return all;
    }

    /// <summary>Expert ids [tokens, topK]: distinct per token, drawn from the first <paramref name="pool"/> experts (the rest stay empty); identical rows when asked.</summary>
    private static HostBuf ExpertIds(int tokens, int topK, int pool, bool identical, Rng rng)
    {
        var ids = new HostBuf(DType.I32, (long)tokens * topK);
        int[] first = Shuffled(rng, pool)[..topK];
        for (int t = 0; t < tokens; t++)
        {
            int[] chosen = identical ? first : Shuffled(rng, pool)[..topK];
            for (int k = 0; k < topK; k++)
            {
                ids.As<int>()[(t * topK) + k] = chosen[k];
            }
        }

        return ids;
    }

    // ---- MoeRouteTopK -------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> RouteCases(ulong baseSeed, CaseProfile p)
    {
        foreach (int tokens in p.Large ? new[] { 1, 16, 128, 512 } : new[] { 1, 16, 128 })
        {
            yield return Route("fixed", "random", tokens, MoeExperts, MoeTopK, true, baseSeed);
        }

        yield return Route("fixed", "identical", 64, MoeExperts, MoeTopK, true, baseSeed);
        yield return Route("fixed", "near-tie", 32, MoeExperts, MoeTopK, true, baseSeed);
        yield return Route("fixed", "near-tie", 32, 16, 4, false, baseSeed);
        yield return Route("fixed", "random", 16, MoeExperts, MoeTopK, false, baseSeed);
        var rng = new Rng(SeedFor(OpId.MoeRouteTopK, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            int experts = rng.Pick(4, 8, 16, 60, 64, 128);
            int topK = rng.Int(1, Math.Min(experts, 8) + 1);
            yield return Route($"random-{i}", rng.Pick("random", "near-tie"), rng.Int(1, 200), experts, Math.Max(topK, 3), rng.Int(0, 4) != 0, baseSeed);
        }
    }

    private static OpCase Route(string family, string kind, int tokens, int experts, int topK, bool renormalize, ulong baseSeed)
    {
        topK = Math.Min(topK, experts);
        string id = $"{family}-BF16-{kind}-{Dim(tokens, experts)}-top{topK}{(renormalize ? "-renorm" : string.Empty)}";
        ulong seed = SeedFor(OpId.MoeRouteTopK, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase { Id = id, Op = OpId.MoeRouteTopK, Dtype = DType.BF16, ToleranceClass = "hybrid-f32", Source = MoeSource(family), Seed = seed };
        c.Num["tokens"] = tokens;
        c.Num["experts"] = experts;
        c.Num["top_k"] = topK;
        c.Num["renormalize"] = renormalize ? 1 : 0;
        c.Buffers["logits"] = Logits(kind, tokens, experts, topK, rng);
        c.Buffers["weights"] = new HostBuf(DType.F32, (long)tokens * topK);
        c.Buffers["ids"] = new HostBuf(DType.I32, (long)tokens * topK);
        c.Outputs.Add("weights");
        c.Outputs.Add("ids");
        c.OutputClasses["ids"] = "exact";
        return c;
    }

    // ---- MoePermute ---------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> PermuteCases(ulong baseSeed, CaseProfile p)
    {
        foreach (int tokens in p.Large ? new[] { 1, 16, 128, 512 } : new[] { 1, 16, 128 })
        {
            yield return Permute("fixed", tokens, MoeHidden, MoeExperts, MoeTopK, MoeExperts, false, baseSeed);
        }

        // All tokens identical: 8 experts take every row and 120 get none. Then a pool of 12 of 64 experts.
        yield return Permute("fixed-identical", 128, MoeHidden, MoeExperts, MoeTopK, MoeExperts, true, baseSeed);
        yield return Permute("fixed-empty", 40, 256, 64, 4, 12, false, baseSeed);
        var rng = new Rng(SeedFor(OpId.MoePermute, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            int experts = rng.Pick(4, 8, 16, 64, 128);
            int topK = rng.Int(1, Math.Min(experts, 8) + 1);
            int pool = rng.Int(topK, experts + 1);
            yield return Permute($"random-{i}", rng.Int(1, 200), 8 * rng.Int(1, 64), experts, topK, pool, rng.Int(0, 5) == 0, baseSeed);
        }
    }

    private static OpCase Permute(string family, int tokens, int dim, int experts, int topK, int pool, bool identical, ulong baseSeed)
    {
        string id = $"{family}-BF16-{Dim(tokens, dim)}-e{experts}-top{topK}-pool{pool}{(identical ? "-identical" : string.Empty)}";
        ulong seed = SeedFor(OpId.MoePermute, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase { Id = id, Op = OpId.MoePermute, Dtype = DType.BF16, ToleranceClass = "exact", Source = MoeSource(family), Seed = seed };
        c.Num["tokens"] = tokens;
        c.Num["dim"] = dim;
        c.Num["experts"] = experts;
        c.Num["top_k"] = topK;
        c.Buffers["x"] = new HostBuf(DType.BF16, (long)tokens * dim).Randn(rng, 1.0);
        c.Buffers["ids"] = ExpertIds(tokens, topK, pool, identical, rng);
        c.Buffers["out"] = new HostBuf(DType.BF16, (long)tokens * topK * dim);
        c.Buffers["counts"] = new HostBuf(DType.I32, experts);
        c.Buffers["offsets"] = new HostBuf(DType.I32, experts + 1);
        c.Buffers["perm"] = new HostBuf(DType.I32, (long)tokens * topK);
        c.Outputs.AddRange(["out", "counts", "offsets", "perm"]);
        return c;
    }

    // ---- MoeUnpermute -------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> UnpermuteCases(ulong baseSeed, CaseProfile p)
    {
        foreach (int tokens in p.Large ? new[] { 1, 16, 128 } : new[] { 1, 16 })
        {
            yield return Unpermute("fixed", tokens, MoeHidden, MoeTopK, false, baseSeed);
        }

        yield return Unpermute("fixed-identical", 16, MoeHidden, MoeTopK, true, baseSeed);
        var rng = new Rng(SeedFor(OpId.MoeUnpermute, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            yield return Unpermute($"random-{i}", rng.Int(1, 100), 8 * rng.Int(1, 64), rng.Int(1, 9), false, baseSeed);
        }
    }

    private static OpCase Unpermute(string family, int tokens, int dim, int topK, bool identical, ulong baseSeed)
    {
        string id = $"{family}-BF16-{Dim(tokens, dim)}-top{topK}";
        ulong seed = SeedFor(OpId.MoeUnpermute, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase { Id = id, Op = OpId.MoeUnpermute, Dtype = DType.BF16, ToleranceClass = "elementwise-bf16", Source = MoeSource(family), Seed = seed };
        int pairs = tokens * topK;
        c.Num["tokens"] = tokens;
        c.Num["dim"] = dim;
        c.Num["top_k"] = topK;
        var y = new HostBuf(DType.BF16, (long)pairs * dim).Randn(rng, 1.0);
        if (identical)
        {
            // Every expert output row the same, as when identical tokens meet the same experts.
            for (int r = 1; r < pairs; r++)
            {
                y.Raw()[..(dim * 2)].CopyTo(y.Raw()[(r * dim * 2)..]);
            }
        }

        // Renormalised weights: positive, summing to one per token.
        var w = new HostBuf(DType.F32, pairs);
        for (int t = 0; t < tokens; t++)
        {
            double[] raw = [.. Enumerable.Range(0, topK).Select(_ => 0.05 + rng.Uniform())];
            double sum = raw.Sum();
            for (int k = 0; k < topK; k++)
            {
                w.Set((t * topK) + k, raw[k] / sum);
            }
        }

        c.Buffers["y"] = y;
        c.Buffers["perm"] = HostBuf.Ints(Shuffled(rng, pairs));
        c.Buffers["weights"] = w;
        c.Buffers["out"] = new HostBuf(DType.BF16, (long)tokens * dim);
        c.Outputs.Add("out");
        return c;
    }

    // ---- GroupedGemm --------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> GroupedGemmCases(ulong baseSeed, CaseProfile p)
    {
        if (p.Large)
        {
            // Qwen3-30B-A3B expert widths with 16 experts: gate+up (N 1536, K 2048) and down (N 2048, K 768).
            yield return GroupedGemm("fixed-gate-up", 16, 2 * MoeInter, MoeHidden, [5, 0, 1, 12, 0, 0, 3, 7, 2, 0, 9, 1, 0, 4, 8, 12], baseSeed);
            yield return GroupedGemm("fixed-down", 16, MoeHidden, MoeInter, [5, 0, 1, 12, 0, 0, 3, 7, 2, 0, 9, 1, 0, 4, 8, 12], baseSeed);
        }

        yield return GroupedGemm("fixed-empty", 8, 192, 256, [3, 0, 0, 7, 1, 0, 12, 1], baseSeed);
        yield return GroupedGemm("fixed-one-expert", 8, 128, 64, [0, 0, 0, 0, 24, 0, 0, 0], baseSeed);
        yield return GroupedGemm("fixed-all-empty", 4, 64, 64, [0, 0, 0, 0], baseSeed);
        var rng = new Rng(SeedFor(OpId.GroupedGemm, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            int experts = rng.Pick(2, 4, 8, 16);
            int[] counts = [.. Enumerable.Range(0, experts).Select(_ => rng.Int(0, 4) == 0 ? 0 : rng.Int(1, 20))];
            yield return GroupedGemm($"random-{i}", experts, 8 * rng.Int(1, 40), 8 * rng.Int(1, 40), counts, baseSeed);
        }
    }

    private static OpCase GroupedGemm(string family, int experts, int n, int k, int[] counts, ulong baseSeed)
    {
        int rows = counts.Sum();
        string id = $"{family}-BF16-e{experts}-{Dim(rows, n, k)}-{string.Join('.', counts)}";
        ulong seed = SeedFor(OpId.GroupedGemm, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase { Id = id, Op = OpId.GroupedGemm, Dtype = DType.BF16, ToleranceClass = "gemm-bf16", Source = MoeSource(family), Seed = seed };
        int[] offsets = new int[experts + 1];
        for (int g = 0; g < experts; g++)
        {
            offsets[g + 1] = offsets[g] + counts[g];
        }

        c.Num["experts"] = experts;
        c.Num["n"] = n;
        c.Num["k"] = k;
        c.Num["rows"] = rows;
        c.Buffers["a"] = new HostBuf(DType.BF16, (long)rows * k).Randn(rng, 1.0);
        c.Buffers["b"] = new HostBuf(DType.BF16, (long)experts * n * k).Randn(rng, 1.0 / Math.Sqrt(k));
        c.Buffers["offsets"] = HostBuf.Ints(offsets);
        c.Buffers["d"] = new HostBuf(DType.BF16, (long)rows * n);
        c.Outputs.Add("d");
        return c;
    }

    // ---- FusedMoe -----------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> FusedMoeCases(ulong baseSeed, CaseProfile p)
    {
        if (p.Large)
        {
            foreach (int tokens in new[] { 1, 16, 64 })
            {
                yield return FusedMoe("fixed-qwen3", "random", tokens, MoeHidden, MoeInter, 16, MoeTopK, baseSeed);
            }
        }

        foreach (int tokens in new[] { 1, 7, 33 })
        {
            yield return FusedMoe("fixed", "random", tokens, 128, 64, 16, 4, baseSeed);
        }

        yield return FusedMoe("fixed", "identical", 16, 128, 64, 16, 4, baseSeed);
        yield return FusedMoe("fixed", "near-tie", 12, 128, 64, 16, 4, baseSeed);
        yield return FusedMoe("fixed-empty", "random", 2, 64, 32, 32, 2, baseSeed);
        var rng = new Rng(SeedFor(OpId.FusedMoe, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            int experts = rng.Pick(4, 8, 16);
            yield return FusedMoe($"random-{i}", rng.Pick("random", "near-tie"), rng.Int(1, 24), 8 * rng.Int(2, 16), 8 * rng.Int(1, 10), experts, rng.Int(3, Math.Min(experts, 8) + 1), baseSeed);
        }
    }

    private static OpCase FusedMoe(string family, string kind, int tokens, int dim, int inter, int experts, int topK, ulong baseSeed)
    {
        string id = $"{family}-BF16-{kind}-{Dim(tokens, dim, inter)}-e{experts}-top{topK}";
        ulong seed = SeedFor(OpId.FusedMoe, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase { Id = id, Op = OpId.FusedMoe, Dtype = DType.BF16, ToleranceClass = "gemm-bf16", Source = MoeSource(family), Seed = seed };
        c.Num["tokens"] = tokens;
        c.Num["dim"] = dim;
        c.Num["inter"] = inter;
        c.Num["experts"] = experts;
        c.Num["top_k"] = topK;
        c.Num["renormalize"] = 1;
        var x = new HostBuf(DType.BF16, (long)tokens * dim).Randn(rng, 1.0);
        if (kind == "identical")
        {
            for (int t = 1; t < tokens; t++)
            {
                x.Raw()[..(dim * 2)].CopyTo(x.Raw()[(t * dim * 2)..]);
            }
        }

        c.Buffers["x"] = x;
        c.Buffers["logits"] = Logits(kind, tokens, experts, topK, rng);
        c.Buffers["w13"] = new HostBuf(DType.BF16, (long)experts * 2 * inter * dim).Randn(rng, 1.0 / Math.Sqrt(dim));
        c.Buffers["w2"] = new HostBuf(DType.BF16, (long)experts * dim * inter).Randn(rng, 1.0 / Math.Sqrt(inter));
        c.Buffers["out"] = new HostBuf(DType.BF16, (long)tokens * dim);
        c.Outputs.Add("out");
        return c;
    }
}

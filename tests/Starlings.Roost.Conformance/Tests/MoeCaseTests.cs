using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

/// <summary>The MoE cases (R-080): the edge cases are in the catalogue, they exercise what they name, and the suite notices a backend that routes or permutes differently.</summary>
public sealed class MoeCaseTests
{
    private static List<OpCase> Cases(OpId op) => [.. CaseCatalog.Generate(op, 7, CaseProfile.Ci)];

    [Fact]
    public void EveryMoeOpHasItsEdgeCasesInTheCiProfile()
    {
        var expected = new Dictionary<OpId, string[]>
        {
            [OpId.MoeRouteTopK] = ["-identical-", "-near-tie-", "-random-"],
            [OpId.MoePermute] = ["fixed-identical-", "fixed-empty-"],
            [OpId.MoeUnpermute] = ["fixed-identical-"],
            [OpId.GroupedGemm] = ["fixed-empty-", "fixed-one-expert-", "fixed-all-empty-"],
            [OpId.FusedMoe] = ["-identical-", "-near-tie-", "fixed-empty-"],
        };
        foreach ((OpId op, string[] marks) in expected)
        {
            List<OpCase> cases = Cases(op);
            try
            {
                Assert.All(marks, m => Assert.Contains(cases, c => c.Id.Contains(m, StringComparison.Ordinal)));
            }
            finally
            {
                cases.ForEach(c => c.Dispose());
            }
        }
    }

    [Fact]
    public void NearTieRoutingCasesHoldAnUlpPairAndAnExactTieAtTheBoundary()
    {
        List<OpCase> cases = Cases(OpId.MoeRouteTopK);
        try
        {
            OpCase c = cases.First(x => x.Id.Contains("-near-tie-", StringComparison.Ordinal));
            int experts = c.I("experts"), topK = c.I("top_k");
            using var cpu = new CpuBackend();
            using ExecResult r = cpu.Execute(CpuBackend.VariantId, c);
            for (int t = 0; t < c.I("tokens"); t++)
            {
                double[] row = [.. Enumerable.Range(0, experts).Select(j => c.Buffers["logits"].Get(((long)t * experts) + j))];
                Assert.Equal(1, row.Count(v => v == 5.0));
                Assert.Equal(1, row.Count(v => v == 5.03125));
                Assert.Equal(2, row.Count(v => v == 2.5));
                int[] ids = [.. Enumerable.Range(0, topK).Select(k => (int)r.Outputs["ids"].Get(((long)t * topK) + k))];
                Assert.Equal(Array.IndexOf(row, 5.03125), ids[0]);
                Assert.Equal(Array.IndexOf(row, 5.0), ids[1]);

                // The tie at 2.5 straddles the boundary: the lower expert is in, the higher is out.
                int low = Array.IndexOf(row, 2.5), high = Array.LastIndexOf(row, 2.5);
                Assert.Equal(low, ids[^1]);
                Assert.DoesNotContain(high, ids);
            }
        }
        finally
        {
            cases.ForEach(c => c.Dispose());
        }
    }

    [Fact]
    public void ARouterThatBreaksTiesTowardTheHigherExpertFails()
    {
        // Swap the last two ids of every row: the tie at the top-k boundary would then be resolved the other way round.
        using var cpu = new CpuBackend
        {
            Perturb = (c, r) =>
            {
                int k = c.I("top_k");
                Span<int> ids = r.Outputs["ids"].As<int>();
                for (int t = 0; k > 1 && t < c.I("tokens"); t++)
                {
                    (ids[(t * k) + k - 1], ids[(t * k) + k - 2]) = (ids[(t * k) + k - 2], ids[(t * k) + k - 1]);
                }
            },
        };
        RunOptions o = Fixtures.Options(variants: new HashSet<string> { CpuBackend.VariantId }, ops: new HashSet<OpId> { OpId.MoeRouteTopK }, determinism: false, invariance: false);
        ConformanceReport rep = Fixtures.RunOn(cpu, o);
        // Every routing case has a top-k of at least 3, so every one must fail.
        var ran = rep.Results.Where(x => x.Case != "(goldens)").ToList();
        Assert.NotEmpty(ran);
        Assert.All(ran, x => Assert.Equal("fail", x.Outcome));
    }

    [Fact]
    public void APermutationThatIsNotStableFails()
    {
        // Reverse the order of the rows inside each expert's range: counts and offsets stay right, the permutation and rows do not.
        using var cpu = new CpuBackend
        {
            Perturb = (c, r) =>
            {
                Span<int> off = r.Outputs["offsets"].As<int>();
                Span<int> perm = r.Outputs["perm"].As<int>();
                for (int i = 0; i < perm.Length; i++)
                {
                    int e = 0;
                    while (off[e + 1] <= perm[i])
                    {
                        e++;
                    }

                    perm[i] = off[e] + off[e + 1] - 1 - perm[i];
                }
            },
        };
        RunOptions o = Fixtures.Options(variants: new HashSet<string> { CpuBackend.VariantId }, ops: new HashSet<OpId> { OpId.MoePermute }, determinism: false, invariance: false);
        ConformanceReport rep = Fixtures.RunOn(cpu, o);
        Assert.Contains(rep.Results, x => x.Outcome == "fail");
    }

    [Fact]
    public void AFusedMoeWithAFourUlpErrorFailsAndAnUntouchedOnePasses()
    {
        using var cpu = new CpuBackend { Perturb = (c, r) => Fixtures.Map(r.Outputs["out"], (i, v) => i % 7 == 0 ? v + (Math.Abs(v) * 4 * Math.Pow(2, -8)) + 0.05 : v) };
        RunOptions o = Fixtures.Options(variants: new HashSet<string> { CpuBackend.VariantId }, ops: new HashSet<OpId> { OpId.FusedMoe }, determinism: false, invariance: false);
        Assert.Contains(Fixtures.RunOn(cpu, o).Results, x => x.Outcome == "fail");
        using var clean = new CpuBackend();
        var ran = Fixtures.RunOn(clean, o).Results.Where(x => x.Case != "(goldens)").ToList();
        Assert.NotEmpty(ran);
        Assert.All(ran, x => Assert.Equal("pass", x.Outcome));
    }
}

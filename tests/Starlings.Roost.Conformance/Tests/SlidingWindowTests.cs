using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

/// <summary>
/// The sliding-window PagedAttention cases (R-113): the CPU half (the independent oracle against <c>CpuOps</c>) runs them in CI;
/// the GPU half is ready but the FA2 variant refuses them, so they are reported <c>refused</c> there and never count as passes.
/// </summary>
public sealed class SlidingWindowTests
{
    // The registered rule, applied to the CPU variant so a test can stand in for the GPU one.
    private static readonly RefusalRule CpuWindowRule = RefusalRules.PagedAttentionSlidingWindow with { Variant = CpuBackend.VariantId };

    private static List<OpCase> Cases(CaseProfile p) => [.. CaseCatalog.Generate(OpId.PagedAttention, 20261019, p)];

    private static bool Windowed(OpCase c) => c.IOr("window_left", -1) != -1;

    private static RunOptions WindowOnly(string variant, IReadOnlyList<RefusalRule> refusals) => new()
    {
        Profile = CaseProfile.Ci,
        Tolerances = ToleranceTable.LoadProposal(),
        Variants = new HashSet<string> { variant },
        Ops = new HashSet<OpId> { OpId.PagedAttention },
        Determinism = false,
        Invariance = false,
        CaseFilter = "-w",
        Refusals = refusals,
    };

    [Fact]
    public void TheCatalogueHasWindowedCasesOfEveryKindAndLeavesTheEarlierCasesUnchanged()
    {
        var all = Cases(CaseProfile.Ci);
        try
        {
            var win = all.Where(Windowed).ToList();
            Assert.Contains(win, c => c.I("mode") == (int)AttentionMode.Decode);
            Assert.Contains(win, c => c.I("mode") == (int)AttentionMode.Prefill && c.I("batch") > 1);
            Assert.Contains(win, c => c.I("causal") == 0);
            Assert.Contains(win, c => c.I("window_left") == 1);
            Assert.Contains(win, c => c.I("window_left") >= 4096);
            Assert.Contains(win, c => c.Buffers["kv_indices"].As<int>().Contains(-1));
            Assert.Contains(win, c => c.Source == "random");
            Assert.All(win, c => Assert.Contains("-w", c.Id, StringComparison.Ordinal));
            // A case without a window carries no window_left at all, so its id, seed and bytes are those of before R-113.
            Assert.All(all.Where(c => !Windowed(c)), c => Assert.DoesNotContain("window_left", c.Num.Keys));
            Assert.Contains(all, c => c.Id == "prefill-multi-seq-q17.64.1.60-kv17.300.50.60-causal-PagedNhd-ps16");
        }
        finally
        {
            all.ForEach(c => c.Dispose());
        }
    }

    [Fact]
    public void ReleasedEntriesLieBeforeEveryQuerysWindowAndOnlyThose()
    {
        var all = Cases(CaseProfile.Full);
        try
        {
            int released = 0;
            foreach (OpCase c in all.Where(Windowed))
            {
                int[] qi = c.Buffers["q_indptr"].As<int>().ToArray(), ki = c.Buffers["kv_indptr"].As<int>().ToArray();
                int[] idx = c.Buffers["kv_indices"].As<int>().ToArray(), last = c.Buffers["last_page_len"].As<int>().ToArray();
                int ps = c.I("page_size"), w = c.I("window_left");
                bool releasing = c.Id.Contains("-released", StringComparison.Ordinal);
                for (int b = 0; b < c.I("batch"); b++)
                {
                    int kvLen = ((ki[b + 1] - ki[b] - 1) * ps) + last[b];
                    int firstKey = Math.Max(0, kvLen - (qi[b + 1] - qi[b]) - w);
                    for (int page = 0; page < ki[b + 1] - ki[b]; page++)
                    {
                        bool gone = idx[ki[b] + page] == -1;
                        bool readable = ((page + 1) * ps) - 1 >= firstKey;
                        Assert.False(gone && readable, $"{c.Id}: sequence {b} page {page} is released but read");
                        Assert.Equal(releasing && !readable, gone);
                        released += gone ? 1 : 0;
                    }
                }
            }

            Assert.True(released > 100, $"only {released} released pages");
        }
        finally
        {
            all.ForEach(c => c.Dispose());
        }
    }

    [Fact]
    public void TheWindowChangesTheResultOfEveryCaseWhoseWindowIsShorterThanItsContext()
    {
        // Guards against vacuous cases: dropping the window changes CpuOps' output unless the window covers every context.
        using var cpu = new CpuBackend();
        var all = Cases(CaseProfile.Ci);
        try
        {
            foreach (OpCase c in all.Where(c => Windowed(c) && !c.Id.Contains("-released", StringComparison.Ordinal)))
            {
                int[] ki = c.Buffers["kv_indptr"].As<int>().ToArray(), last = c.Buffers["last_page_len"].As<int>().ToArray();
                int longest = Enumerable.Range(0, c.I("batch")).Max(b => ((ki[b + 1] - ki[b] - 1) * c.I("page_size")) + last[b]);
                using ExecResult windowed = cpu.Execute(CpuBackend.VariantId, c);
                int w = c.I("window_left");
                c.Num.Remove("window_left");
                using ExecResult full = cpu.Execute(CpuBackend.VariantId, c);
                c.Num["window_left"] = w;
                Assert.Equal(0, windowed.Status);
                bool same = Compare.BytesEqual(windowed.Outputs["out"], full.Outputs["out"]);
                // The last key (position longest - 1) reads from position longest - 1 - w; nothing is hidden when that is 0 or less.
                if (w + 1 >= longest)
                {
                    Assert.True(same, c.Id);
                }
                else if (c.Id.StartsWith("window", StringComparison.Ordinal))
                {
                    Assert.False(same, c.Id);
                }
            }
        }
        finally
        {
            all.ForEach(c => c.Dispose());
        }
    }

    [Fact]
    public void TheIndependentOracleAgreesWithCpuOpsOnEveryWindowedCase()
    {
        using var oracle = new OracleBackend();
        ConformanceReport rep = ConformanceRunner.Run(oracle, WindowOnly(OracleBackend.VariantId, RefusalRules.Registered));
        var ran = rep.Results.Where(r => r.Case.Contains("-w", StringComparison.Ordinal)).ToList();
        Assert.True(ran.Count >= 13, $"{ran.Count} windowed cases ran");
        Assert.All(ran, r => Assert.True(r.Outcome == "pass", $"{r.Case}: {r.Outcome} {r.Detail}"));
    }

    [Fact]
    public void TheRuleSelectsExactlyTheWindowedAttentionCases()
    {
        var all = Cases(CaseProfile.Full);
        try
        {
            Assert.All(all, c => Assert.Equal(Windowed(c), RefusalRules.PagedAttentionSlidingWindow.Applies(c)));
            Assert.Equal("paged-attention-flashinfer-fa2", RefusalRules.PagedAttentionSlidingWindow.Variant);
            Assert.Equal(KernelStatus.Unsupported, RefusalRules.PagedAttentionSlidingWindow.Status);
        }
        finally
        {
            all.ForEach(c => c.Dispose());
        }

        // No other op's case and no case without a window is refused by it.
        foreach (OpCase c in CaseCatalog.Generate(OpId.Gemm, 1, CaseProfile.Ci))
        {
            using (c)
            {
                Assert.False(RefusalRules.PagedAttentionSlidingWindow.Applies(c));
            }
        }
    }

    [Fact]
    public void AVariantThatRefusesTheWindowReportsTheWindowedCasesRefusedNeverPassed()
    {
        // As the FA2 variant does today: UNSUPPORTED for a window, everything else served.
        using var cpu = new CpuBackend();
        using var fa2Like = new DelegatingBackend(cpu, replace: (v, c) => Windowed(c) ? new ExecResult { Status = (int)KernelStatus.Unsupported, Message = "window" } : null);
        ConformanceReport rep = ConformanceRunner.Run(fa2Like, WindowOnly(CpuBackend.VariantId, [CpuWindowRule]));
        Assert.True(rep.Green);
        var win = rep.Results.Where(r => r.Case.Contains("-w", StringComparison.Ordinal)).ToList();
        Assert.NotEmpty(win);
        Assert.All(win, r => Assert.Equal("refused", r.Outcome));
        Assert.All(win, r => Assert.Contains("R-113", r.Detail, StringComparison.Ordinal));
        Assert.DoesNotContain(rep.Results, r => r.Outcome == "pass");
    }

    [Fact]
    public void ServingAWindowWhileTheRuleIsRegisteredIsAFailure()
    {
        // A variant that starts serving windows must come with the rule removed, so its cases are compared; until then it fails.
        using var cpu = new CpuBackend();
        ConformanceReport rep = ConformanceRunner.Run(cpu, WindowOnly(CpuBackend.VariantId, [CpuWindowRule]));
        Assert.False(rep.Green);
        Assert.All(rep.Results.Where(r => r.Case.Contains("-w", StringComparison.Ordinal)), r => Assert.Equal("fail", r.Outcome));
    }
}

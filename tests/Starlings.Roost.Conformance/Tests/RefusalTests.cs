using System.Text.RegularExpressions;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

/// <summary>R-019 decisions taken from data: the Gemm epilogue refusal (C or bias), and the prefill batch-composition check.</summary>
public sealed partial class RefusalTests
{
    private static readonly RefusalRule CpuRule = new(CpuBackend.VariantId, RefusalRules.GemmEpilogueWithCOrBias.Applies, KernelStatus.Unsupported, "test");

    private static ConformanceReport RunGemm(IKernelBackend b)
    {
        RunOptions o = new()
        {
            Profile = CaseProfile.Ci,
            Tolerances = ToleranceTable.LoadProposal(),
            Variants = new HashSet<string> { CpuBackend.VariantId },
            Ops = new HashSet<OpId> { OpId.Gemm },
            Determinism = false,
            Invariance = false,
            Refusals = [CpuRule],
        };
        return ConformanceRunner.Run(b, o);
    }

    private static ExecResult Refuse(OpCase c, KernelStatus s) => new() { Status = (int)s, Message = "refused" };

    [Fact]
    public void TheRuleSelectsExactlyTheCasesWithACInputOrABias()
    {
        var cases = CaseCatalog.Generate(OpId.Gemm, 20261019, CaseProfile.Full).ToList();
        try
        {
            var hit = cases.Where(RefusalRules.GemmEpilogueWithCOrBias.Applies).ToList();
            Assert.NotEmpty(hit);
            Assert.All(hit, c => Assert.True(c.I("use_bias") == 1 || c.I("use_c") == 1));
            Assert.All(cases.Where(c => !hit.Contains(c)), c => Assert.Equal((0, 0), (c.I("use_bias"), c.I("use_c"))));
            // Both dtypes, each epilogue alone and together; the five cases that exceeded the class in the first SM 86 run are among them.
            Assert.Contains(hit, c => c.Dtype == DType.F16);
            Assert.Contains(hit, c => c.Dtype == DType.BF16 && c.I("use_bias") == 1 && c.I("use_c") == 0);
            Assert.Contains(hit, c => c.Dtype == DType.BF16 && c.I("use_bias") == 0 && c.I("use_c") == 1);
            Assert.Contains(hit, c => c.Id.StartsWith("fixed-epilogue-BF16-39x256x200-tb1-bias1-ab1.1-bucket49", StringComparison.Ordinal));
            Assert.Contains(hit, c => c.Id.StartsWith("fixed-epilogue-BF16-39x256x200-tb1-bias1-ab0.5.1-bucket49", StringComparison.Ordinal));
            // Alpha alone is served.
            Assert.Contains(cases.Except(hit), c => c.F("alpha") != 1f);
        }
        finally
        {
            cases.ForEach(c => c.Dispose());
        }
    }

    [Fact]
    public void ATypedRefusalOfTheCombinationIsCountedRefusedAndEverythingElseStillPasses()
    {
        using var cpu = new CpuBackend();
        using var wrapped = new DelegatingBackend(cpu, replace: (v, c) => CpuRule.Applies(c) ? Refuse(c, KernelStatus.Unsupported) : null);
        ConformanceReport rep = RunGemm(wrapped);
        Assert.True(rep.Green);
        Assert.Contains(rep.Results, x => x.Outcome == "refused");
        Assert.Contains(rep.Results, x => x.Outcome == "refused" && x.Dtype == DType.F16);
        Assert.Equal(0, rep.Count("fail") + rep.Count("error"));
        Assert.Contains(rep.Results, x => x.Outcome == "pass" && x.Dtype == DType.F16);
        Assert.Contains(rep.Results, x => x.Outcome == "pass" && x.Dtype == DType.BF16);
    }

    [Fact]
    public void ServingTheCombinationInsteadOfRefusingItIsAFailure()
    {
        using var cpu = new CpuBackend();
        ConformanceReport rep = RunGemm(cpu);
        Assert.False(rep.Green);
        var served = rep.Results.Where(x => x.Detail.Contains("served a combination", StringComparison.Ordinal)).ToList();
        Assert.NotEmpty(served);
        Assert.All(served, x => Assert.Equal("fail", x.Outcome));
        Assert.All(rep.Results.Except(served).Where(x => x.Outcome != "not_run"), x => Assert.Equal("pass", x.Outcome));
    }

    [Fact]
    public void ARefusalWithAnotherStatusIsAFailure()
    {
        using var cpu = new CpuBackend();
        using var wrapped = new DelegatingBackend(cpu, replace: (v, c) => CpuRule.Applies(c) ? Refuse(c, KernelStatus.InvalidArgument) : null);
        ConformanceReport rep = RunGemm(wrapped);
        Assert.False(rep.Green);
        Assert.Contains(rep.Results, x => x.Outcome == "fail" && x.Detail.Contains("instead of Unsupported", StringComparison.Ordinal));
    }

    [Fact]
    public void ARefusalOutsideTheRuleStaysAFailure()
    {
        using var cpu = new CpuBackend();
        using var wrapped = new DelegatingBackend(cpu, replace: (v, c) => c.Dtype == DType.F16 ? Refuse(c, KernelStatus.Unsupported) : null);
        ConformanceReport rep = RunGemm(wrapped);
        // f16 cases the rule covers (C or bias) are refused as registered; every other f16 case refused is a failure.
        var f16 = rep.Results.Where(x => x.Dtype == DType.F16 && x.Outcome is not ("not_run" or "refused")).ToList();
        Assert.NotEmpty(f16);
        Assert.All(f16, x => Assert.Equal("fail", x.Outcome));
        Assert.All(rep.Results.Where(x => x.Outcome == "refused"), x => Assert.Contains("as registered", x.Detail, StringComparison.Ordinal));
    }

    [Fact]
    public void TheRegisteredRulesAreOnlyTheGemmRule()
    {
        Assert.Equal(["gemm-cublaslt"], RefusalRules.Registered.Select(r => r.Variant));
    }

    [Fact]
    public void NoGeneratedModelForwardUsesGemmWithBetaOrBias()
    {
        // The evidence behind the refusal: the Gemm nodes of Qwen3.5-0.8B, roost-tiny and roost-tiny-hybrid pass A, B, D, M, N, K,
        // TransposeB, Alpha and WeightDType only. The nodes are the source of the generated forwards. If a model adds C, Beta or
        // Bias to a Gemm, this fails: then the kernel fix is no longer optional.
        string root = RepoRoot();
        var files = Directory.GetFiles(Path.Combine(root, "src"), "*.cs", SearchOption.AllDirectories)
            .Where(f => f.Contains("Starlings.Roost.Models.", StringComparison.Ordinal)).ToList();
        Assert.True(files.Count >= 3, "model definitions not found");
        int gemms = 0;
        foreach (string f in files)
        {
            foreach (Match m in GemmNode().Matches(File.ReadAllText(f)))
            {
                gemms++;
                string[] keys = [.. m.Groups[1].Value.Split(';', StringSplitOptions.TrimEntries).Select(kv => kv.Split('=')[0].Trim())];
                Assert.DoesNotContain(keys, k => k is "C" or "Beta" or "Bias");
                Assert.Contains("Alpha", keys);
            }
        }

        Assert.True(gemms >= 30, $"expected the Gemm nodes of three models, found {gemms}");
    }

    [Fact]
    public void ABatchCompositionBreakInPrefillFailsTheCase()
    {
        using var ok = new CpuBackend();
        ConformanceReport green = Prefill(ok);
        Assert.DoesNotContain(green.Results, x => x.Outcome is "fail" or "error");
        Assert.Contains(green.Results, x => x.Checks.Any(k => k.Name.StartsWith("prefill batch composition", StringComparison.Ordinal) && k.Ok));

        // A bit flipped in the last row of a multi-sequence call only: far inside the class, caught by the composition check alone.
        using var bad = new CpuBackend { Perturb = (c, r) => { if (c.IOr("batch", 1) > 1) { r.Outputs["out"].Raw()[^2] ^= 1; } } };
        ConformanceReport rep = Prefill(bad);
        Assert.Contains(rep.Results, x => x.Outcome == "fail" && x.Checks.Any(k => k.Name.StartsWith("prefill batch composition", StringComparison.Ordinal) && !k.Ok));
        Assert.All(rep.Results.Where(x => x.Outcome == "fail"), x => Assert.All(x.Outputs, o => Assert.Equal(0, o.Stat.Violations)));
    }

    private static ConformanceReport Prefill(IKernelBackend b)
        => ConformanceRunner.Run(b, Fixtures.Options(variants: new HashSet<string> { CpuBackend.VariantId }, ops: new HashSet<OpId> { OpId.PagedAttention }, determinism: false, invariance: false));

    private static string RepoRoot()
    {
        string? d = AppContext.BaseDirectory;
        while (d is not null && !File.Exists(Path.Combine(d, "Roost.slnx")))
        {
            d = Path.GetDirectoryName(d);
        }

        return d ?? throw new InvalidOperationException("Roost.slnx not found above " + AppContext.BaseDirectory);
    }

    [GeneratedRegex("""\[Node\("[^"]*",\s*"Gemm",\s*"([^"]*)"\s*[,)]""")]
    private static partial Regex GemmNode();
}

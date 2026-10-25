using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

/// <summary>The suite must notice a damaged backend: each test damages CpuOps in one way and expects the matching failure.</summary>
public sealed class MutationTests
{
    private static ConformanceReport Run(IKernelBackend candidate, params OpId[] ops) => RunWith(candidate, false, false, ops);

    private static ConformanceReport RunWith(IKernelBackend candidate, bool determinism, bool invariance, params OpId[] ops)
    {
        // The reference inside the runner is a separate CpuBackend, so damaging this one does not move the reference.
        RunOptions o = Fixtures.Options(variants: new HashSet<string> { CpuBackend.VariantId }, ops: ops.Length == 0 ? null : new HashSet<OpId>(ops), determinism: determinism, invariance: invariance);
        return Fixtures.RunOn(candidate, o);
    }

    [Fact]
    public void AnUntouchedCpuBackendIsGreen()
    {
        using var cpu = new CpuBackend();
        ConformanceReport r = Run(cpu);
        Assert.DoesNotContain(r.Results, x => x.Outcome is "fail" or "error");
    }

    [Fact]
    public void OneFlippedBitInAnExactOpFails()
    {
        using var cpu = new CpuBackend { Perturb = (c, r) => r.Outputs["out"].Raw()[5] ^= 1 };
        ConformanceReport rep = Run(cpu, OpId.Embedding);
        Assert.All(rep.Results.Where(x => x.Outcome != "not_run"), x => Assert.Equal("fail", x.Outcome));
        Assert.NotEmpty(rep.Results);
    }

    [Fact]
    public void ACacheWriteThatMissesOneByteFails()
    {
        using var cpu = new CpuBackend { Perturb = (c, r) => r.Outputs["k_cache"].Raw()[^1] ^= 0x80 };
        ConformanceReport rep = Run(cpu, OpId.CacheWrite);
        Assert.Contains(rep.Results, x => x.Outcome == "fail");
    }

    [Fact]
    public void AnErrorBeyondTheClassFailsAndOneInsideItPasses()
    {
        using var big = new CpuBackend { Perturb = (c, r) => Fixtures.Map(r.Outputs["out"], (_, v) => v * (1 + Math.Pow(2, -4))) };
        Assert.Contains(Run(big, OpId.RmsNorm).Results, x => x.Outcome == "fail");
        using var small = new CpuBackend { Perturb = (c, r) => Fixtures.Map(r.Outputs["out"], (_, v) => v * (1 + Math.Pow(2, -11))) };
        ConformanceReport ok = Run(small, OpId.RmsNorm);
        Assert.DoesNotContain(ok.Results, x => x.Outcome is "fail" or "error");
        Assert.All(ok.Results.Where(x => x.Outcome == "pass" && x.Dtype == DType.F16 && x.Case.Contains("1024", StringComparison.Ordinal)), x => Assert.InRange(x.Outputs[0].Stat.MaxUtilisation, 0.0001, 1.0));
    }

    [Fact]
    public void ANaNInOneElementFails()
    {
        using var cpu = new CpuBackend { Perturb = (c, r) => r.Outputs["d"].Set(0, double.NaN) };
        ConformanceReport rep = Run(cpu, OpId.Gemm);
        Assert.All(rep.Results.Where(x => x.Outcome != "not_run"), x => Assert.Equal("fail", x.Outcome));
    }

    [Fact]
    public void AGemmThatLosesAccumulationPrecisionFails()
    {
        // The R-017 finding: split-K partial sums kept in 16 bits show as several ulp of error. Four bf16 ulp is outside the class.
        using var cpu = new CpuBackend { Perturb = (c, r) => Fixtures.Map(r.Outputs["d"], (i, v) => i % 5 == 0 ? v + (Math.Abs(v) * 4 * Math.Pow(2, -8)) + 0.05 : v) };
        Assert.Contains(Run(cpu, OpId.Gemm).Results, x => x.Outcome == "fail");
    }

    [Fact]
    public void ADeterministicVariantThatChangesBetweenRunsFailsTheDeterminismCheck()
    {
        int n = 0;
        using var inner = new CpuBackend { Perturb = (c, r) => { if (++n % 2 == 0) { r.Outputs["out"].Raw()[0] ^= 1; } } };
        ConformanceReport rep = RunWith(inner, true, false, OpId.SiluAndMul);
        Assert.Contains(rep.Results, x => x.Determinism == "fail" && x.Outcome == "fail");
        // The error itself is far inside the class: only the determinism check caught it.
        Assert.All(rep.Results.Where(x => x.Determinism == "fail"), x => Assert.All(x.Outputs, o => Assert.Equal(0, o.Stat.Violations)));
    }

    [Fact]
    public void ABatchDependentVariantFailsTheInvarianceCheck()
    {
        // Rows of a multi-row call differ slightly from the same row alone; the difference is inside the tolerance class.
        using var cpu = new CpuBackend { Perturb = (c, r) => { if (c.Rows > 1) { r.Outputs["out"].Raw()[0] ^= 1; } } };
        ConformanceReport rep = RunWith(cpu, false, true, OpId.RmsNorm);
        Assert.Contains(rep.Results, x => x.Invariance == "fail");
        Assert.All(rep.Results.Where(x => x.Invariance == "fail"), x => Assert.All(x.Outputs, o => Assert.Equal(0, o.Stat.Violations)));
    }

    [Fact]
    public void DeterminismAndInvarianceAreOnlyDemandedOfVariantsThatClaimThem()
    {
        using var cpu = new CpuBackend { Perturb = (c, r) => { if (c.Rows > 1) { r.Outputs["out"].Raw()[0] ^= 1; } } };
        using var weak = new DelegatingBackend(cpu, flags: VariantTraits.None);
        ConformanceReport rep = RunWith(weak, true, true, OpId.RmsNorm);
        Assert.All(rep.Results.Where(x => x.Outcome == "pass"), x => Assert.Equal(("n/a", "n/a"), (x.Determinism, x.Invariance)));
    }

    [Fact]
    public void ARefusalInsideTheConstraintsIsAFailure()
    {
        using var cpu = new CpuBackend();
        using var refuses = new DelegatingBackend(cpu, replace: (v, c) => new ExecResult { Status = (int)KernelStatus.Unsupported, Message = "nope" });
        ConformanceReport rep = Run(refuses, OpId.Residual);
        Assert.All(rep.Results.Where(x => x.Outcome != "not_run"), x => Assert.Equal("fail", x.Outcome));
    }

    [Fact]
    public void ACaseOutsideTheRegisteredConstraintsIsNotRunAndNeverLaunched()
    {
        using var cpu = new CpuBackend();
        var narrow = cpu.Variants.Select(v => v.Op == OpId.RmsNorm ? v with { Constraints = new KernelConstraints { ActDtypeMask = Elem.Bit(DType.BF16) } } : v).ToList();
        using var wrapped = new RestrictedBackend(cpu, narrow);
        ConformanceReport rep = Run(wrapped, OpId.RmsNorm);
        Assert.All(rep.Results.Where(x => x.Dtype == DType.F16), x => Assert.Equal("not_run", x.Outcome));
        Assert.Contains(rep.Results, x => x.Dtype == DType.BF16 && x.Outcome == "pass");
        Assert.Equal(0, wrapped.F16Calls);
    }

    [Fact]
    public void AThrowingBackendIsAnErrorNotACrash()
    {
        using var cpu = new CpuBackend { Perturb = (c, r) => throw new InvalidOperationException("device lost") };
        ConformanceReport rep = Run(cpu, OpId.Residual);
        Assert.All(rep.Results.Where(x => x.Outcome != "not_run"), x => Assert.Equal("error", x.Outcome));
        Assert.False(rep.Green);
    }

    [Fact]
    public void ABackendCheckThatFailsFailsTheCase()
    {
        using var cpu = new CpuBackend();
        using var checks = new DelegatingBackend(cpu, after: (v, c, r) => { r.Checks.Add(new BackendCheck("output padding untouched", false, "byte 3 changed")); return r; });
        ConformanceReport rep = Run(checks, OpId.Residual);
        Assert.All(rep.Results.Where(x => x.Outcome != "not_run"), x => Assert.Equal("fail", x.Outcome));
    }

    [Fact]
    public void ARefusedShortStructIsAnAbiProbePassAndAnAcceptedOneIsAFail()
    {
        using var cpu = new CpuBackend();
        using var good = new ProbeBackend(cpu, (int)KernelStatus.AbiMismatch);
        Assert.All(Fixtures.RunOn(good, Fixtures.Options(variants: new HashSet<string> { CpuBackend.VariantId })).Variants, v => Assert.Equal("pass", v.AbiProbe));
        using var bad = new ProbeBackend(cpu, (int)KernelStatus.Ok);
        ConformanceReport rep = Fixtures.RunOn(bad, Fixtures.Options(variants: new HashSet<string> { CpuBackend.VariantId }));
        Assert.All(rep.Variants, v => Assert.Equal("fail", v.AbiProbe));
        Assert.False(rep.Green);
    }

    private sealed class ProbeBackend : IKernelBackend
    {
        private readonly IKernelBackend _inner;
        private readonly int _status;

        public ProbeBackend(IKernelBackend inner, int status)
        {
            _inner = inner;
            _status = status;
        }

        public string Name => "probe";

        public int Sm => 0;

        public IReadOnlyList<VariantInfo> Variants => _inner.Variants;

        public ExecResult Execute(string variant, OpCase c) => _inner.Execute(variant, c);

        public int? ProbeShortStruct(string variant, OpCase c) => _status;

        public void Dispose()
        {
        }
    }

    private sealed class RestrictedBackend : IKernelBackend
    {
        private readonly IKernelBackend _inner;

        public RestrictedBackend(IKernelBackend inner, IReadOnlyList<VariantInfo> variants)
        {
            _inner = inner;
            Variants = variants;
        }

        public int F16Calls { get; private set; }

        public string Name => "restricted";

        public int Sm => 0;

        public IReadOnlyList<VariantInfo> Variants { get; }

        public ExecResult Execute(string variant, OpCase c)
        {
            if (c.Op == OpId.RmsNorm && c.Dtype == DType.F16)
            {
                F16Calls++;
            }

            return _inner.Execute(variant, c);
        }

        public int? ProbeShortStruct(string variant, OpCase c) => null;

        public void Dispose()
        {
        }
    }
}

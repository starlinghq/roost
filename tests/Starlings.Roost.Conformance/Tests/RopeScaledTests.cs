using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Conformance.Tests;

/// <summary>The CPU half of RopeScaled (R-098): its cases, the oracle against CpuOps, and the report for a backend without a variant.</summary>
public sealed class RopeScaledTests
{
    private static readonly HashSet<OpId> Only = [OpId.RopeScaled];

    [Fact]
    public void OracleAgreesWithCpuOpsOnEveryRopeScaledCase()
    {
        using var oracle = new OracleBackend();
        ConformanceReport r = Fixtures.RunOn(oracle, Fixtures.Options(ops: Only));
        var ran = r.Results.Where(x => x.Op == OpId.RopeScaled && x.Source is "fixed" or "random").ToList();
        Assert.NotEmpty(ran);
        Assert.All(ran, x => Assert.Equal("pass", x.Outcome));
        Assert.True(r.Green);
    }

    [Fact]
    public void TheCasesCoverEveryScalingKindAndBothHalfTypes()
    {
        var cases = CaseCatalog.Generate(OpId.RopeScaled, 3, CaseProfile.Ci).ToList();
        try
        {
            Assert.Equal([1, 2, 3], cases.Select(c => c.I("scaling_kind")).Distinct().Order());
            Assert.Equal([DType.F16, DType.BF16], cases.Select(c => c.Dtype).Distinct().Order());
            Assert.Contains(cases, c => c.I("rotary_dim") == 0);
            Assert.Contains(cases, c => c.I("rotary_dim") != 0 && c.I("rotary_dim") < c.I("head_dim"));
        }
        finally
        {
            cases.ForEach(c => c.Dispose());
        }
    }

    [Fact]
    public void AnUnscaledCacheFailsTheScaledCases()
    {
        // Discrimination: a backend that rotates by the unscaled frequencies is reported as failing.
        using var cpu = new CpuBackend();
        using var wrong = new DelegatingBackend(cpu, replace: (v, c) =>
        {
            using OpCase plain = Unscaled(c);
            return cpu.Execute(v, plain);
        });
        ConformanceReport r = Fixtures.RunOn(wrong, Fixtures.Options(ops: Only, determinism: false, invariance: false));
        Assert.Contains(r.Results, x => x.Op == OpId.RopeScaled && x.Outcome == "fail");
    }

    [Fact]
    public void ABackendWithoutAVariantIsReportedAsUnsupportedNotOmitted()
    {
        using var cpu = new CpuBackend();
        using var m0 = new WithoutOp(cpu, OpId.RopeScaled);
        ConformanceReport r = Fixtures.RunOn(m0, Fixtures.Options(ops: Only));
        CaseResult row = Assert.Single(r.Results);
        Assert.Equal(OpId.RopeScaled, row.Op);
        Assert.Equal("not_run", row.Outcome);
        Assert.Contains("unsupported", row.Detail, StringComparison.Ordinal);
        Assert.Contains("untested", row.Detail, StringComparison.Ordinal);
    }

    private static OpCase Unscaled(OpCase c)
    {
        var u = new OpCase { Id = c.Id, Op = c.Op, Dtype = c.Dtype, ToleranceClass = c.ToleranceClass, Source = c.Source, Seed = c.Seed };
        foreach ((string name, double v) in c.Num)
        {
            u.Num[name] = v;
        }

        foreach ((string name, HostBuf b) in c.Buffers)
        {
            u.Buffers[name] = b.Clone();
        }

        u.Outputs.AddRange(c.Outputs);
        int hd = c.I("head_dim");
        int rot = c.I("rotary_dim") == 0 ? hd : c.I("rotary_dim");
        HostBuf cache = u.Buffers["cos_sin_cache"];
        // The families' bases: 1e4 for the 128-wide heads, 1e7 for the Qwen3.5 shape.
        RopeScaling.WriteCosSinCache(RopeScaling.Default(rot, hd == 128 ? 1e4 : 1e7), cache.Span, (int)(cache.Count / rot));
        return u;
    }

    private sealed class WithoutOp(IKernelBackend inner, OpId op) : IKernelBackend
    {
        public string Name => "Without" + op;

        public int Sm => inner.Sm;

        public IReadOnlyList<VariantInfo> Variants { get; } = [.. inner.Variants.Where(v => v.Op != op)];

        public ExecResult Execute(string variant, OpCase c) => inner.Execute(variant, c);

        public int? ProbeShortStruct(string variant, OpCase c) => inner.ProbeShortStruct(variant, c);

        public void Dispose() => inner.Dispose();
    }
}

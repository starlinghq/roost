using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

/// <summary>The CI half: CpuOps, the reference of every GPU comparison, checked against an independent implementation.</summary>
public sealed class OracleTests
{
    // One run serves the tests that read it: the Ci profile against the oracle takes tens of seconds.
    private static readonly Lazy<ConformanceReport> Report = new(() =>
    {
        using var oracle = new OracleBackend();
        return Fixtures.RunOn(oracle);
    });

    [Fact]
    public void IndependentOracleAgreesWithCpuOpsOnEveryCiCase()
    {
        ConformanceReport r = Report.Value;
        var bad = r.Results.Where(x => x.Outcome is not ("pass" or "not_run")).Select(x => $"{x.Op} {x.Case}: {x.Outcome} {x.Detail} {string.Join(',', x.Outputs.Select(o => $"{o.Name}:{o.Stat.MaxUtilisation:F3}/{o.Stat.Violations}"))}").ToList();
        Assert.Empty(bad);
        Assert.True(r.Count("pass") > 150, $"only {r.Count("pass")} cases ran");
        Assert.All(CaseCatalog.Ops, op => Assert.Contains(r.Results, x => x.Op == op && x.Outcome == "pass"));
    }

    [Fact]
    public void ExactOpsAreBitForBitAndFloatOpsStayInsideTheirClasses()
    {
        ConformanceReport r = Report.Value;
        foreach (CaseResult x in r.Results.Where(x => x.Class == "exact" && x.Outcome == "pass"))
        {
            Assert.All(x.Outputs, o => Assert.Equal(0, o.Stat.Violations));
        }

        foreach (CaseResult x in r.Results.Where(x => x.Class != "exact" && x.Outcome == "pass"))
        {
            Assert.All(x.Outputs, o => Assert.True(o.Stat.MaxUtilisation <= 1.0, $"{x.Case}: {o.Stat.MaxUtilisation}"));
        }
    }

    [Fact]
    public void CpuOpsDoesNotMutateTheCaseItIsGiven()
    {
        using var cpu = new CpuBackend();
        foreach (OpId op in CaseCatalog.Ops)
        {
            foreach (OpCase c in CaseCatalog.Generate(op, 21, CaseProfile.Ci).Take(3))
            {
                using (c)
                {
                    var before = c.Buffers.ToDictionary(kv => kv.Key, kv => kv.Value.Clone());
                    using ExecResult r = cpu.Execute(CpuBackend.VariantId, c);
                    foreach ((string name, HostBuf b) in before)
                    {
                        Assert.True(Compare.BytesEqual(b, c.Buffers[name]), $"{c.Id}: {name} changed");
                        b.Dispose();
                    }
                }
            }
        }
    }
}

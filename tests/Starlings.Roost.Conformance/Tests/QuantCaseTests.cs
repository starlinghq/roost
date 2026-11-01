using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

/// <summary>
/// The quantisation cases (R-077, R-043). No independent oracle and no GPU variant exists for them yet, so this checks the
/// catalogue itself: every case runs on CpuOps through the runner, repeats bit for bit, and feeds only valid codes.
/// </summary>
public sealed class QuantCaseTests
{
    [Fact]
    public void EveryQuantCaseRunsOnCpuOpsThroughTheRunner()
    {
        using var cpu = new CpuBackend();
        ConformanceReport r = Fixtures.RunOn(cpu, Fixtures.Options(ops: new HashSet<OpId>(CpuBackend.QuantOps)));
        Assert.All(CpuBackend.QuantOps, op => Assert.Contains(r.Results, x => x.Op == op && x.Outcome == "pass"));
        Assert.All(r.Results.Where(x => x.Case != "(goldens)"), x => Assert.True(x.Outcome == "pass", $"{x.Op} {x.Case}: {x.Outcome} {x.Detail}"));
        Assert.True(r.Results.Count(x => x.Outcome == "pass") >= 96, $"only {r.Count("pass")} cases passed");
    }

    [Fact]
    public void QuantisedInputsHoldNoNaNOrInfinityCodes()
    {
        foreach (OpId op in new[] { OpId.Dequantize, OpId.GemmFp8 })
        {
            foreach (OpCase c in CaseCatalog.Generate(op, 5, CaseProfile.Ci))
            {
                using (c)
                {
                    foreach (HostBuf b in c.Buffers.Values.Where(b => b.Dtype is DType.F8E4M3 or DType.F8E5M2))
                    {
                        Assert.All(b.Raw().ToArray(), code => Assert.False(
                            b.Dtype == DType.F8E4M3 ? (code & 0x7F) == 0x7F : (code & 0x7C) == 0x7C,
                            $"{c.Id}: code 0x{code:X2}"));
                    }
                }
            }
        }
    }
}

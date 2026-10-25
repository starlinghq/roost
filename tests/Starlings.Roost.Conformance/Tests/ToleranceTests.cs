using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

public sealed class ToleranceTests
{
    [Fact]
    public void ProposalIsLoadedAndMarkedAsProposal()
    {
        ToleranceTable t = ToleranceTable.LoadProposal();
        Assert.Equal("proposal", t.Status);
        Assert.Equal(64, t.Sha256.Length);
        Assert.True(t["exact"].Exact);
    }

    [Fact]
    public void ProposalHoldsTheValuesR017Measured()
    {
        ToleranceTable t = ToleranceTable.LoadProposal();
        Assert.Equal((Math.Pow(2, -6), Math.Pow(2, -12)), (t["elementwise-bf16"].Rtol, t["elementwise-bf16"].AtolRel));
        Assert.Equal((Math.Pow(2, -9), Math.Pow(2, -15)), (t["elementwise-f16"].Rtol, t["elementwise-f16"].AtolRel));
        Assert.Equal((Math.Pow(2, -6), Math.Pow(2, -9)), (t["gemm-bf16"].Rtol, t["gemm-bf16"].AtolRel));
        Assert.Equal((Math.Pow(2, -9), Math.Pow(2, -12)), (t["gemm-f16"].Rtol, t["gemm-f16"].AtolRel));
        Assert.Equal((Math.Pow(2, -12), Math.Pow(2, -13)), (t["lm-head-f32"].Rtol, t["lm-head-f32"].AtolRel));
        Assert.Equal((Math.Pow(2, -7), Math.Pow(2, -8)), (t["attention-bf16"].Rtol, t["attention-bf16"].AtolRel));
        Assert.Equal(ToleranceClass.MaxAbsV, t["attention-bf16"].Scale);
    }

    [Fact]
    public void EveryClassACaseNamesIsInTheFile()
    {
        ToleranceTable t = ToleranceTable.LoadProposal();
        foreach (OpId op in CaseCatalog.Ops)
        {
            foreach (OpCase c in CaseCatalog.Generate(op, 1, CaseProfile.Ci))
            {
                using (c)
                {
                    Assert.True(t.Classes.ContainsKey(c.ToleranceClass), $"{c.Id}: no class {c.ToleranceClass}");
                    Assert.All(c.OutputClasses.Values, k => Assert.True(t.Classes.ContainsKey(k), $"{c.Id}: no class {k}"));
                }
            }
        }
    }

    [Fact]
    public void ExactClassComparesBytesAndNeverAcceptsAnUlp()
    {
        ToleranceTable t = ToleranceTable.LoadProposal();
        using var a = new HostBuf(DType.BF16, 8).Randn(new Rng(1));
        using var b = a.Clone();
        Assert.True(Compare.Close(a, b, t["exact"]).Pass);
        b.As<ushort>()[3] ^= 1;
        CompareStat s = Compare.Close(a, b, t["exact"]);
        Assert.False(s.Pass);
        Assert.Equal(1, s.Violations);
    }

    [Fact]
    public void ExactClassSeesDifferentNaNPatterns()
    {
        using var a = new HostBuf(DType.F32, 2);
        using var b = new HostBuf(DType.F32, 2);
        a.As<uint>()[0] = 0x7fc00000;
        b.As<uint>()[0] = 0x7fc00001;
        Assert.False(Compare.Bits(a, b).Pass);
    }

    [Fact]
    public void FloatClassAllowsOneRelativeStepAndRejectsTwoHundredPercentOfIt()
    {
        ToleranceClass cls = ToleranceTable.LoadProposal()["elementwise-bf16"];
        using var want = new HostBuf(DType.F32, 4);
        for (int i = 0; i < 4; i++)
        {
            want.Set(i, 1.0);
        }

        using var inside = want.Clone();
        inside.Set(0, 1.0 + (cls.Rtol * 0.9));
        CompareStat ok = Compare.Close(inside, want, cls);
        Assert.True(ok.Pass);
        Assert.InRange(ok.MaxUtilisation, 0.5, 1.0);
        using var outside = want.Clone();
        outside.Set(0, 1.0 + (cls.Rtol * 2) + (cls.AtolRel * 2));
        Assert.False(Compare.Close(outside, want, cls).Pass);
    }

    [Fact]
    public void NonFiniteCandidateIsNeverAPass()
    {
        ToleranceClass cls = ToleranceTable.LoadProposal()["elementwise-f16"];
        using var want = new HostBuf(DType.F32, 3).Randn(new Rng(2));
        using var got = want.Clone();
        got.Set(1, double.NaN);
        CompareStat s = Compare.Close(got, want, cls);
        Assert.False(s.Pass);
        Assert.Equal(1, s.NonFinite);
    }

    [Fact]
    public void MaxAbsVClassScalesByThePoolNotTheOutput()
    {
        ToleranceClass cls = ToleranceTable.LoadProposal()["attention-bf16"];
        using var want = new HostBuf(DType.F32, 2);
        using var got = new HostBuf(DType.F32, 2);
        got.Set(0, 0.0039); // 2^-8 of a pool whose largest value is 1 is 0.0039
        Assert.True(Compare.Close(got, want, cls, maxAbsV: 1.0).Pass);
        Assert.False(Compare.Close(got, want, cls, maxAbsV: 0.1).Pass);
    }

    [Fact]
    public void AFileWithoutTheClassIsAnError()
    {
        string path = Path.Combine(Path.GetTempPath(), "roost-tol-" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(path, """{"status":"approved","classes":{"exact":{"exact":true}}}""");
        try
        {
            ToleranceTable t = ToleranceTable.Load(path);
            Assert.Equal("approved", t.Status);
            Assert.Throws<KeyNotFoundException>(() => t["gemm-bf16"]);
        }
        finally
        {
            File.Delete(path);
        }
    }
}

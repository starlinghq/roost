using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels.Tests;

public class VariantSelectorTests
{
    private static readonly KernelRequirement Gemm86 = new() { Op = OpId.Gemm, Sm = 86, Activation = DType.BF16, Tokens = 200 };

    private static VariantCandidate Candidate(string id, char code = 'c', OpId op = OpId.Gemm) =>
        new(op, id, VariantCandidate.DigestOf(op, id, new string(code, 64)));

    private static QualifiedVariant Pass(VariantCandidate c, int sm = 86, string? bucket = null) =>
        new(c.Op, c.VariantId, c.Digest, sm, bucket, "evidence/test");

    [Fact]
    public void ShapeBucketsRoundBatchAndTokensUpToAPowerOfTwoAndOrderFieldsCanonically()
    {
        var a = ShapeBucket.For(Gemm86);
        var b = ShapeBucket.For(Gemm86 with { Tokens = 256 });
        var c = ShapeBucket.For(Gemm86 with { Tokens = 257 });
        Assert.Equal("act=BF16;tokens=256", a.Key);
        Assert.Equal(a, b);
        Assert.Equal("act=BF16;tokens=512", c.Key);
        Assert.Equal("*", ShapeBucket.For(new KernelRequirement { Op = OpId.Residual, Sm = 86 }).Key);
        Assert.Equal(
            "head_dim=128;page_size=16;batch=8;flags=1",
            ShapeBucket.For(new KernelRequirement
            {
                Op = OpId.PagedAttention,
                Sm = 89,
                HeadDim = 128,
                PageSize = 16,
                Batch = 5,
                RequiredFlags = VariantTraits.GraphCapturable,
            }).Key);
        Assert.Throws<ArgumentException>(() => new ShapeBucket(OpId.Gemm, "has space"));
        Assert.Throws<ArgumentException>(() => new ShapeBucket(OpId.None, "x"));
    }

    [Fact]
    public void NoQualifiedVariantIsProfileUnsupportedByDefault()
    {
        var selector = new VariantSelector(new VariantQualification());
        var e = Assert.Throws<ProfileUnsupportedException>(() => selector.Select(Gemm86, [Candidate("gemm-a")]));
        Assert.Equal(OpId.Gemm, e.Op);
        Assert.Equal(86, e.Sm);
        Assert.Equal(RoostErrorCode.ProfileUnsupported, e.Error.Code);
        Assert.Contains("gemm-a", e.Message);
        Assert.Contains("no conformance pass at SM 86", e.Message);
    }

    [Fact]
    public void NoAdmittedVariantIsProfileUnsupported()
    {
        var e = Assert.Throws<ProfileUnsupportedException>(() => new VariantSelector(new VariantQualification()).Select(Gemm86, []));
        Assert.Contains("no registered variant admits", e.Message);
    }

    [Fact]
    public void APassAtAnotherSmOrForAnotherDigestOrBucketDoesNotQualify()
    {
        var a = Candidate("gemm-a");
        var rebuilt = Candidate("gemm-a", 'd');
        var qualification = new VariantQualification([Pass(a, sm: 89), Pass(rebuilt, bucket: "act=BF16;tokens=1024")]);
        var e = Assert.Throws<ProfileUnsupportedException>(() => new VariantSelector(qualification).Select(Gemm86, [a]));
        Assert.Contains("another digest", e.Message);

        var other = Assert.Throws<ProfileUnsupportedException>(() => new VariantSelector(qualification).Select(Gemm86, [rebuilt]));
        Assert.Contains("only for other shape buckets", other.Message);

        var forBucket = new VariantQualification([Pass(a, bucket: "act=BF16;tokens=256")]);
        Assert.Equal("gemm-a", new VariantSelector(forBucket).Select(Gemm86, [a]).Selected.VariantId);
    }

    [Fact]
    public void UnqualifiedVariantsAreRejectedAndNeverChosen()
    {
        var a = Candidate("gemm-a");
        var b = Candidate("gemm-b");
        var s = new VariantSelector(new VariantQualification([Pass(b)])).Select(Gemm86, [a, b]);
        Assert.Equal(b, s.Selected);
        Assert.Equal(SelectionSource.OnlyCandidate, s.Source);
        Assert.Equal(a, Assert.Single(s.Rejected).Candidate);
    }

    [Fact]
    public void WithoutMeasurementsTheDefaultOrderIsByVariantIdIndependentOfInputOrder()
    {
        var a = Candidate("gemm-a");
        var b = Candidate("gemm-b");
        var z = Candidate("gemm-z");
        var selector = new VariantSelector(new VariantQualification([Pass(a), Pass(b), Pass(z)]), AutotuneCache.InMemory(86, "580.65.06"));
        var first = selector.Select(Gemm86, [z, b, a]);
        var second = selector.Select(Gemm86, [b, a, z]);
        Assert.Equal(a, first.Selected);
        Assert.Equal(SelectionSource.Default, first.Source);
        Assert.Equal([a, b, z], first.Unmeasured);
        Assert.Equal(first.Qualified, second.Qualified);
        Assert.False(second.ChangedFrom(first));
    }

    [Fact]
    public void WhenEveryQualifiedVariantIsMeasuredTheFastestWinsAndTiesGoToTheLowerId()
    {
        var a = Candidate("gemm-a");
        var b = Candidate("gemm-b");
        var c = Candidate("gemm-c");
        var cache = AutotuneCache.InMemory(86, "580.65.06");
        var bucket = ShapeBucket.For(Gemm86);
        cache.Record(bucket, a, new AutotuneMeasurement(12.0, 50));
        cache.Record(bucket, b, new AutotuneMeasurement(9.5, 50));
        var selector = new VariantSelector(new VariantQualification([Pass(a), Pass(b), Pass(c)]), cache);

        var partial = selector.Select(Gemm86, [a, b, c]);
        Assert.Equal(SelectionSource.Default, partial.Source);
        Assert.Equal(a, partial.Selected);
        Assert.Equal(c, Assert.Single(partial.Unmeasured));

        cache.Record(bucket, c, new AutotuneMeasurement(9.5, 50));
        var tuned = selector.Select(Gemm86, [c, b, a]);
        Assert.Equal(SelectionSource.Autotuned, tuned.Source);
        Assert.Equal([b, c, a], tuned.Qualified);
        Assert.True(tuned.ChangedFrom(partial));
    }

    [Fact]
    public void AMeasurementOfAnotherDigestIsInvalidatedNotUsed()
    {
        var a = Candidate("gemm-a");
        var b = Candidate("gemm-b");
        var rebuiltB = Candidate("gemm-b", 'e');
        var cache = AutotuneCache.InMemory(86, "580.65.06");
        var bucket = ShapeBucket.For(Gemm86);
        cache.Record(bucket, a, new AutotuneMeasurement(12.0, 10));
        cache.Record(bucket, b, new AutotuneMeasurement(1.0, 10));
        var selector = new VariantSelector(new VariantQualification([Pass(a), Pass(rebuiltB)]), cache);

        var s = selector.Select(Gemm86, [a, rebuiltB]);
        Assert.Equal(SelectionSource.Default, s.Source);
        Assert.Equal(rebuiltB, Assert.Single(s.Unmeasured));
        Assert.Equal(1, cache.Count);
        Assert.Equal(1, cache.Discarded);
    }

    [Fact]
    public void ACacheForAnotherSmIsIgnored()
    {
        var a = Candidate("gemm-a");
        var b = Candidate("gemm-b");
        var cache = AutotuneCache.InMemory(89, "580.65.06");
        var bucket = ShapeBucket.For(Gemm86);
        cache.Record(bucket, a, new AutotuneMeasurement(12.0, 10));
        cache.Record(bucket, b, new AutotuneMeasurement(1.0, 10));
        var s = new VariantSelector(new VariantQualification([Pass(a), Pass(b)]), cache).Select(Gemm86, [a, b]);
        Assert.Equal(SelectionSource.Default, s.Source);
        Assert.Equal(a, s.Selected);
    }

    [Fact]
    public void CandidatesForAnotherOpOrDuplicateIdsAreRejected()
    {
        var selector = new VariantSelector(new VariantQualification());
        Assert.Throws<ArgumentException>(() => selector.Select(Gemm86, [Candidate("r", op: OpId.Residual)]));
        Assert.Throws<ArgumentException>(() => selector.Select(Gemm86, [Candidate("gemm-a"), Candidate("gemm-a", 'f')]));
    }

    [Fact]
    public void ChangedFromComparesVariantAndDigest()
    {
        var a = Candidate("gemm-a");
        var rebuilt = Candidate("gemm-a", 'd');
        var s1 = new VariantSelector(new VariantQualification([Pass(a)])).Select(Gemm86, [a]);
        var s2 = new VariantSelector(new VariantQualification([Pass(rebuilt)])).Select(Gemm86, [rebuilt]);
        Assert.True(s1.ChangedFrom(null));
        Assert.False(s1.ChangedFrom(s1));
        Assert.True(s2.ChangedFrom(s1));
    }

    [Fact]
    public void VariantDigestsDependOnOpIdAndCode()
    {
        var code = new string('c', 64);
        var d = VariantCandidate.DigestOf(OpId.Gemm, "x", code);
        Assert.Equal(64, d.Length);
        Assert.Equal(d, VariantCandidate.DigestOf(OpId.Gemm, "x", code.ToUpperInvariant()));
        Assert.NotEqual(d, VariantCandidate.DigestOf(OpId.Gemm, "y", code));
        Assert.NotEqual(d, VariantCandidate.DigestOf(OpId.LmHead, "x", code));
        Assert.NotEqual(d, VariantCandidate.DigestOf(OpId.Gemm, "x", new string('d', 64)));
        Assert.Throws<ArgumentException>(() => new VariantCandidate(OpId.Gemm, "x", "ABC"));
    }
}

[Collection("fake kernel library")]
public class VariantSelectorRegistryTests
{
    [Fact]
    public void SelectingFromARegistryPinsTheManifestLibraryDigest()
    {
        FakeKernelLibrary.Reset();
        FakeKernelLibrary.SetEntries(FakeKernelLibrary.Entry(
            OpId.Residual,
            "residual-stub",
            VariantTraits.GraphCapturable | VariantTraits.Deterministic | VariantTraits.BatchInvariant));
        var library = KernelLibrary.Attach("fake.so", new KernelLoadOptions(TestManifest.Build(), 86, true), SmStatus.Experimental, 0, FakeKernelLibrary.Exports);
        var registry = new KernelRegistry();
        registry.Add(library);

        var requirement = new KernelRequirement { Op = OpId.Residual, Sm = 86 };
        var candidate = VariantCandidate.FromKernel(registry.Require(requirement));
        Assert.Equal(VariantCandidate.DigestOf(OpId.Residual, "residual-stub", "00"), candidate.Digest);

        Assert.Throws<ProfileUnsupportedException>(() => new VariantSelector(new VariantQualification()).Select(registry, requirement));
        var qualification = new VariantQualification([new QualifiedVariant(OpId.Residual, "residual-stub", candidate.Digest, 86, null, "evidence/test")]);
        var s = new VariantSelector(qualification).Select(registry, requirement);
        Assert.Equal(candidate, s.Selected);
        Assert.Equal(SelectionSource.OnlyCandidate, s.Source);
    }
}

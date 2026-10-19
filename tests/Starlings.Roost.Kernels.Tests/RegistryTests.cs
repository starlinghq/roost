using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels.Tests;

[Collection("fake kernel library")]
public class RegistryTests
{
    private static ulong Bits(params DType[] types) => types.Aggregate(0UL, (m, t) => m | (1UL << (int)t));

    private static KernelRegistry Registry(params OpEntry[] entries)
    {
        FakeKernelLibrary.Reset();
        FakeKernelLibrary.SetEntries(entries);
        var manifest = TestManifest.Build();
        var lib = KernelLibrary.Attach(
            "fake.so",
            new KernelLoadOptions(manifest, 86, true),
            SmStatus.Experimental,
            0,
            FakeKernelLibrary.Exports);
        var r = new KernelRegistry();
        r.Add(lib);
        return r;
    }

    private static OpEntry Stub(ulong act) =>
        FakeKernelLibrary.Entry(
            OpId.Residual,
            "residual-stub",
            VariantTraits.GraphCapturable | VariantTraits.Deterministic | VariantTraits.BatchInvariant,
            act: act);

    [Fact]
    public void ARequirementThatAVariantAdmitsReturnsIt()
    {
        var r = Registry(Stub(Bits(DType.F32, DType.BF16)));
        var k = r.Require(new KernelRequirement { Op = OpId.Residual, Sm = 86, Activation = DType.BF16 });
        Assert.Equal("residual-stub", k.VariantId);
        Assert.Single(r.RequireAll([new KernelRequirement { Op = OpId.Residual, Sm = 89, Activation = DType.F32 }]));
        Assert.Single(r.Find(new KernelRequirement
        {
            Op = OpId.Residual,
            Sm = 86,
            RequiredFlags = VariantTraits.GraphCapturable,
        }));
    }

    [Fact]
    public void AMissingDtypeIsProfileUnsupportedNamingTheOpAndSm()
    {
        var r = Registry(Stub(Bits(DType.F32, DType.BF16)));
        var e = Assert.Throws<ProfileUnsupportedException>(
            () => r.Require(new KernelRequirement { Op = OpId.Residual, Sm = 86, Activation = DType.F8E4M3 }));
        Assert.Equal(OpId.Residual, e.Op);
        Assert.Equal(86, e.Sm);
        Assert.Equal(RoostErrorCode.ProfileUnsupported, e.Error.Code);
        Assert.Equal(400, e.Error.HttpStatus);
        Assert.Contains("Residual", e.Message);
        Assert.Contains("SM 86", e.Message);
        Assert.Contains("F8E4M3", e.Message);
    }

    [Fact]
    public void AnSmBelowTheVariantsMinimumIsProfileUnsupported()
    {
        var r = Registry(Stub(0));
        var e = Assert.Throws<ProfileUnsupportedException>(() => r.Require(new KernelRequirement { Op = OpId.Residual, Sm = 75 }));
        Assert.Equal(75, e.Sm);
        Assert.Contains("SM 80..up", e.Message);
    }

    [Fact]
    public void AnOpNoLibraryRegistersSaysWhichSmsTheManifestBuildsItFor()
    {
        var r = Registry(Stub(0));
        var e = Assert.Throws<ProfileUnsupportedException>(() => r.Require(new KernelRequirement { Op = OpId.Gemm, Sm = 86 }));
        Assert.Equal(OpId.Gemm, e.Op);
        Assert.Contains("only for SM 89", e.Message);
    }

    [Fact]
    public void RequireAllReportsTheFirstUnmetRequirement()
    {
        var r = Registry(Stub(0));
        var e = Assert.Throws<ProfileUnsupportedException>(() => r.RequireAll(
        [
            new KernelRequirement { Op = OpId.Residual, Sm = 86 },
            new KernelRequirement { Op = OpId.PagedAttention, Sm = 86, HeadDim = 256 },
        ]));
        Assert.Equal(OpId.PagedAttention, e.Op);
    }

    [Fact]
    public unsafe void ShapeConstraintsFollowTheMaskSemanticsOfC1()
    {
        var entry = FakeKernelLibrary.Entry(OpId.Residual, "residual-stub", VariantTraits.GraphCapturable | VariantTraits.Deterministic | VariantTraits.BatchInvariant);
        entry.Constraints.HeadDimMask = (1UL << (64 / 16 - 1)) | (1UL << (256 / 16 - 1)); // 64 and 256
        entry.Constraints.PageSizeMask = 1u << 4; // 16
        entry.Constraints.GroupSizeMask = 1u << 3; // 4
        entry.Constraints.LayoutMask = 1u << (int)TensorLayout.PagedNhd;
        entry.Constraints.MaxBatch = 8;
        entry.Constraints.MaxTokens = 1024;
        var r = Registry(entry);
        KernelRequirement Req(int hd = 64, int page = 16, int group = 4, TensorLayout layout = TensorLayout.PagedNhd, int batch = 8, long tokens = 1024) =>
            new() { Op = OpId.Residual, Sm = 86, HeadDim = hd, PageSize = page, GroupSize = group, Layout = layout, Batch = batch, Tokens = tokens };

        Assert.Single(r.Find(Req()));
        Assert.Single(r.Find(Req(hd: 256)));
        Assert.Empty(r.Find(Req(hd: 80)));
        Assert.Empty(r.Find(Req(hd: 70))); // not a multiple of 16
        Assert.Empty(r.Find(Req(page: 32)));
        Assert.Empty(r.Find(Req(page: 24))); // not a power of two
        Assert.Empty(r.Find(Req(group: 6)));
        Assert.Empty(r.Find(Req(layout: TensorLayout.PagedHnd)));
        Assert.Empty(r.Find(Req(batch: 9)));
        Assert.Empty(r.Find(Req(tokens: 1025)));
    }
}

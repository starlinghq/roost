using System.Security.Cryptography;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels.Tests;

[Collection("fake kernel library")]
public class LoaderTests
{
    private const VariantTraits ResidualFlags = VariantTraits.GraphCapturable | VariantTraits.Deterministic | VariantTraits.BatchInvariant;

    private static KernelLoadOptions Options(KernelManifest? m = null, int sm = 86, bool untested = false) =>
        new(m ?? TestManifest.Build(), sm, untested);

    private static KernelLibrary Attach(KernelLoadOptions o, SmStatus status = SmStatus.Experimental) =>
        KernelLibrary.Attach("fake.so", o, status, 0, FakeKernelLibrary.Exports);

    private static void StubOnly() => FakeKernelLibrary.SetEntries(
        FakeKernelLibrary.Entry(OpId.Residual, "residual-stub", ResidualFlags));

    [Fact]
    public void AMatchingLibraryRegistersItsTable()
    {
        FakeKernelLibrary.Reset();
        StubOnly();
        using var lib = Attach(Options());
        var k = Assert.Single(lib.Kernels);
        Assert.Equal((OpId.Residual, "residual-stub"), (k.Op, k.VariantId));
        Assert.Equal("fake-kernels-sm86", lib.Name);
        using var ctx = lib.CreateContext(0);
        Assert.NotEqual(0, ctx.Handle);
    }

    [Theory]
    [InlineData(KernelAbi.Major + 1, KernelAbi.Minor)]
    [InlineData(KernelAbi.Major + 1, 0)]
    [InlineData(KernelAbi.Major - 1, KernelAbi.Minor + 5)]
    public void AnotherAbiMajorIsAnAssetMismatch(int major, int minor)
    {
        FakeKernelLibrary.Reset();
        StubOnly();
        FakeKernelLibrary.Abi = ((uint)major << 16) | (uint)minor;
        var e = Assert.Throws<KernelLoadException>(() => Attach(Options()));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("ABI", e.Message);
    }

    [Fact]
    public void ALibraryWithAnOlderMinorThanTheCallersIsAnAssetMismatch()
    {
        Assert.True(KernelAbi.Minor >= 1, "the host minor must be at least 1 for an older library minor to exist");
        FakeKernelLibrary.Reset();
        StubOnly();
        FakeKernelLibrary.Abi = FakeKernelLibrary.LibraryAbi = ((uint)KernelAbi.Major << 16) | (uint)(KernelAbi.Minor - 1);
        var e = Assert.Throws<KernelLoadException>(() => Attach(Options()));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("ABI", e.Message);
    }

    [Fact]
    public void ALibraryMinorAboveTheCallersIsAccepted()
    {
        FakeKernelLibrary.Reset();
        StubOnly();
        var abi = ((uint)KernelAbi.Major << 16) | (uint)(KernelAbi.Minor + 2);
        FakeKernelLibrary.Abi = FakeKernelLibrary.LibraryAbi = abi;
        using var lib = Attach(Options());
        Assert.Equal(abi, lib.AbiVersion);
    }

    [Fact]
    public void AbiRuleIsMajorEqualAndMinorAtLeast()
    {
        var major = (uint)KernelAbi.Major;
        var minor = (uint)KernelAbi.Minor;
        Assert.True(KernelAbi.IsCompatible((major << 16) | minor));
        Assert.True(KernelAbi.IsCompatible((major << 16) | (minor + 7)));
        Assert.False(KernelAbi.IsCompatible((major << 16) | (minor - 1)));
        Assert.False(KernelAbi.IsCompatible(((major + 1) << 16) | minor));
        Assert.False(KernelAbi.IsCompatible(((major - 1) << 16) | minor));
        Assert.False(KernelAbi.IsCompatible(0u));
    }

    [Fact]
    public void AManifestForAnOlderMinorThanTheLoaderIsAnAssetMismatch()
    {
        var m = TestManifest.Build(abiMinor: KernelAbi.Minor - 1);
        var e = Assert.Throws<KernelLoadException>(() => KernelLibrary.Load("x", Options(m)));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("manifest is for ABI", e.Message);
    }

    [Fact]
    public void AManifestForANewerMinorThanTheLoaderIsAdmitted()
    {
        // Past the ABI check, the missing file is what stops this load.
        var m = TestManifest.Build(abiMinor: KernelAbi.Minor + 1);
        Assert.Throws<ProfileUnsupportedException>(() => KernelLibrary.Load("/nonexistent/x.so", Options(m)));
    }

    [Fact]
    public void ATableAbiThatDisagreesWithTheExportIsRefused()
    {
        FakeKernelLibrary.Reset();
        StubOnly();
        // Both versions are acceptable on their own; they differ from each other.
        FakeKernelLibrary.LibraryAbi = FakeKernelLibrary.Abi + 1;
        var e = Assert.Throws<KernelLoadException>(() => Attach(Options()));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
    }

    [Fact]
    public void ARegisterAbiMismatchStatusIsAnAssetMismatchAndOtherFailuresAreInternal()
    {
        FakeKernelLibrary.Reset();
        StubOnly();
        FakeKernelLibrary.RegisterStatus = (int)KernelStatus.AbiMismatch;
        Assert.Equal(RoostErrorCode.AssetMismatch, Assert.Throws<KernelLoadException>(() => Attach(Options())).Error.Code);
        FakeKernelLibrary.RegisterStatus = (int)KernelStatus.Internal;
        var e = Assert.Throws<KernelLoadException>(() => Attach(Options()));
        Assert.Equal(RoostErrorCode.Internal, e.Error.Code);
        Assert.Contains("fake error", e.Message);
    }

    [Fact]
    public void AnEntrySizeSmallerThanTheLoadersIsRefusedAndALargerOneWalksWithTheLibrarysStride()
    {
        FakeKernelLibrary.Reset();
        FakeKernelLibrary.EntrySize = 16;
        StubOnly();
        Assert.Equal(RoostErrorCode.AssetMismatch, Assert.Throws<KernelLoadException>(() => Attach(Options())).Error.Code);

        FakeKernelLibrary.Reset();
        unsafe
        {
            FakeKernelLibrary.EntrySize = (uint)sizeof(OpEntry) + 24;
        }

        FakeKernelLibrary.SetEntries(
            FakeKernelLibrary.Entry(OpId.Residual, "residual-stub", ResidualFlags),
            FakeKernelLibrary.Entry(OpId.Gemm, "gemm-x", VariantTraits.Deterministic, smMin: 89));
        using var lib = Attach(Options(TestManifest.Build(gemmFor86: true, digest86: new string('c', 64))));
        Assert.Equal(["gemm-x", "residual-stub"], lib.Kernels.Select(k => k.VariantId).Order());
    }

    [Fact]
    public void MalformedEntriesAreRefused()
    {
        var bad = new Func<OpEntry, OpEntry>[]
        {
            e => { e.Run = 0; return e; },
            e => { e.VariantId = 0; return e; },
            e => { e.Flags |= VariantTraits.NeedsPlan; return e; }, // needs-plan without a plan entry point
            e => { e.Plan = 1; return e; }, // plan entry point without the flag
            e => { e.Constraints.SmMin = 0; return e; },
            e => { e.Constraints.SmMax = 70; return e; },
            e => { e.Op = (OpId)9999; return e; },
            e => { e.Flags = (VariantTraits)0x100; return e; },
            e => { e.Reserved0 = 1; return e; },
        };
        foreach (var mutate in bad)
        {
            FakeKernelLibrary.Reset();
            FakeKernelLibrary.SetEntries(mutate(FakeKernelLibrary.Entry(OpId.Residual, "residual-stub", ResidualFlags)));
            Assert.Equal(RoostErrorCode.AssetMismatch, Assert.Throws<KernelLoadException>(() => Attach(Options())).Error.Code);
        }
    }

    [Fact]
    public void ADuplicateVariantInOneTableIsRefused()
    {
        FakeKernelLibrary.Reset();
        FakeKernelLibrary.SetEntries(
            FakeKernelLibrary.Entry(OpId.Residual, "residual-stub", ResidualFlags),
            FakeKernelLibrary.Entry(OpId.Residual, "residual-stub", ResidualFlags));
        Assert.Equal(RoostErrorCode.AssetMismatch, Assert.Throws<KernelLoadException>(() => Attach(Options())).Error.Code);
    }

    [Fact]
    public void TableAndManifestMustAgree()
    {
        // A variant the manifest does not list for the SM.
        FakeKernelLibrary.Reset();
        FakeKernelLibrary.SetEntries(
            FakeKernelLibrary.Entry(OpId.Residual, "residual-stub", ResidualFlags),
            FakeKernelLibrary.Entry(OpId.Gemm, "gemm-x", VariantTraits.Deterministic, smMin: 89));
        Assert.Contains("does not list", Assert.Throws<KernelLoadException>(() => Attach(Options())).Message);

        // A variant the manifest lists but the library lacks.
        FakeKernelLibrary.Reset();
        FakeKernelLibrary.SetEntries();
        Assert.Contains("does not register", Assert.Throws<KernelLoadException>(() => Attach(Options())).Message);

        // Flags that differ from the manifest's.
        FakeKernelLibrary.Reset();
        FakeKernelLibrary.SetEntries(FakeKernelLibrary.Entry(OpId.Residual, "residual-stub", VariantTraits.Deterministic));
        Assert.Contains("registers flags", Assert.Throws<KernelLoadException>(() => Attach(Options())).Message);
    }

    [Fact]
    public void ASecondLibraryRegisteringTheSameVariantIsRefusedByTheRegistry()
    {
        FakeKernelLibrary.Reset();
        StubOnly();
        using var a = Attach(Options());
        using var b = Attach(Options());
        var registry = new KernelRegistry();
        registry.Add(a);
        var e = Assert.Throws<KernelLoadException>(() => registry.Add(b));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Single(registry.Libraries);
    }

    [Fact]
    public void ALibraryWithAnotherDigestIsRefusedBeforeItIsLoaded()
    {
        var dir = Directory.CreateTempSubdirectory("roost-kn-load-");
        try
        {
            var file = Path.Combine(dir.FullName, "libroost_kernels_sm86.so");
            File.WriteAllBytes(file, [1, 2, 3, 4]);
            var digest = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(file)));

            // Wrong digest: AssetMismatch, and the file is never handed to the OS loader.
            var wrong = TestManifest.Build(digest86: new string('0', 64), bytes86: 4);
            var e = Assert.Throws<KernelLoadException>(() => KernelLibrary.Load(file, Options(wrong)));
            Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
            Assert.Contains("digest mismatch", e.Message);

            // Wrong size.
            var size = TestManifest.Build(digest86: digest, bytes86: 5);
            Assert.Contains("bytes", Assert.Throws<KernelLoadException>(() => KernelLibrary.Load(file, Options(size))).Message);

            // Right digest, but the bytes are not a shared library: the OS refusal is Internal, not a crash.
            var right = TestManifest.Build(digest86: digest, bytes86: 4);
            var os = Assert.Throws<KernelLoadException>(() => KernelLibrary.Load(file, Options(right)));
            Assert.Equal(RoostErrorCode.Internal, os.Error.Code);
        }
        finally
        {
            dir.Delete(true);
        }
    }

    [Fact]
    public void AMissingLibraryFileIsProfileUnsupportedNamingTheSm()
    {
        var e = Assert.Throws<ProfileUnsupportedException>(() => KernelLibrary.Load("/nonexistent/libroost_kernels_sm86.so", Options()));
        Assert.Equal(86, e.Sm);
        Assert.Equal(RoostErrorCode.ProfileUnsupported, e.Error.Code);
        Assert.Contains("SM 86", e.Message);
    }

    [Fact]
    public void SmAdmissionFollowsTheManifestStatus()
    {
        var m = TestManifest.Build();
        var missing = Assert.Throws<ProfileUnsupportedException>(() => KernelLibrary.Load("x", Options(m, sm: 75)));
        Assert.Equal(75, missing.Sm);
        Assert.Equal(OpId.None, missing.Op);
        Assert.Contains("SM 75", missing.Message);
        Assert.DoesNotContain("Residual", missing.Message);

        var unsupported = Assert.Throws<ProfileUnsupportedException>(() => KernelLibrary.Load("x", Options(m, sm: 90, untested: true)));
        Assert.Contains("unsupported", unsupported.Message);

        // Untested is refused unless the caller opts in (the opt-in then fails later, on the missing file).
        var untested = Assert.Throws<ProfileUnsupportedException>(() => KernelLibrary.Load("x", Options(m, sm: 80)));
        Assert.Contains("no hardware evidence", untested.Message);
        var optIn = Assert.Throws<ProfileUnsupportedException>(() => KernelLibrary.Load("/nonexistent/x.so", Options(m, sm: 80, untested: true)));
        Assert.Contains("no kernel library file", optIn.Message);
    }

    [Fact]
    public void AManifestForAnotherAbiMajorIsRefused()
    {
        var toml = "schema_version = 1\ncontract_version = 1\n[abi]\nmajor = " + (KernelAbi.Major + 1) + "\nminor = " + KernelAbi.Minor + "\n[toolkit]\ncuda = \"13.3\"\nmin_driver = \"580\"\n" +
                   "[[sm]]\narch = 86\nstatus = \"experimental\"\n";
        var m = KernelManifest.Parse(System.Text.Encoding.UTF8.GetBytes(toml));
        Assert.Equal(RoostErrorCode.AssetMismatch, Assert.Throws<KernelLoadException>(() => KernelLibrary.Load("x", Options(m))).Error.Code);
    }

    [Fact]
    public void OneSmWithTwoDifferentDigestsIsAnInconsistentManifest()
    {
        var m = TestManifest.Build(gemmFor86: true, digest86: new string('d', 64), gemmDigest86: new string('e', 64));
        var e = Assert.Throws<KernelLoadException>(() => KernelLibrary.Load("x", Options(m)));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("2 different digests", e.Message);
    }
}

using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels.Tests;

public class ManifestAndNamesTests
{
    [Fact]
    public void RepositoryManifestParsesWithTheFieldsTheLoaderReads()
    {
        var m = TestManifest.Repository();
        // The repository manifest records the ABI the committed library was built against. It may lag the header between a
        // contract change and the next kernel rebuild (stream C bumps it then); it can never be ahead of the header.
        Assert.Equal(KernelAbi.Major, m.AbiMajor);
        Assert.InRange(m.AbiMinor, 0, KernelAbi.Minor);
        Assert.Equal("13.3", m.Cuda);
        Assert.Equal(13, m.CudaMajor);
        Assert.Contains(m.Sms, s => s is { Arch: 86, Status: SmStatus.Experimental });
        Assert.Contains(m.Sms, s => s is { Arch: 89, Status: SmStatus.Untested });
        var stub = Assert.Single(m.Variants, v => v.Id == "residual-stub");
        Assert.Equal(OpId.Residual, stub.Op);
        Assert.Equal(VariantTraits.GraphCapturable | VariantTraits.Deterministic | VariantTraits.BatchInvariant, stub.Flags);
        Assert.Equal(80, stub.SmMin);
        Assert.Contains(stub.Binaries, b => b.Arch == 86 && b.Sha256.Length == 64);
        Assert.Equal(64, m.Sha256.Length);
    }

    [Fact]
    public void ManifestDigestMismatchIsAnAssetMismatch()
    {
        var path = Path.Combine(TestManifest.RepoRoot(), "kernels", "manifest.toml");
        var ok = KernelManifest.Load(path).Sha256;
        Assert.Equal(ok, KernelManifest.Load(path, ok.ToUpperInvariant()).Sha256);
        var e = Assert.Throws<KernelLoadException>(() => KernelManifest.Load(path, new string('0', 64)));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Equal("asset_mismatch", e.Error.CodeName);
    }

    [Theory]
    [InlineData("not toml [[[")]
    [InlineData("schema_version = 1")]
    public void MalformedManifestsAreRefused(string text)
    {
        var e = Assert.Throws<KernelLoadException>(() => KernelManifest.Parse(System.Text.Encoding.UTF8.GetBytes(text)));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
    }

    [Fact]
    public void KernelLibraryFileNamesAreResolvedPerOs()
    {
        Assert.Equal("libroost_kernels_sm86.so", KernelLibraryNames.FileName(86, KernelPlatform.Linux));
        Assert.Equal("roost_kernels_sm120.dll", KernelLibraryNames.FileName(120, KernelPlatform.Windows));
        Assert.Throws<ArgumentOutOfRangeException>(() => KernelLibraryNames.FileName(0, KernelPlatform.Linux));
    }

    [Fact]
    public void FindReturnsTheFirstDirectoryThatHoldsTheFile()
    {
        var a = Directory.CreateTempSubdirectory("roost-kn-a-");
        var b = Directory.CreateTempSubdirectory("roost-kn-b-");
        try
        {
            Assert.Null(KernelLibraryNames.Find(86, [a.FullName, b.FullName], KernelPlatform.Linux));
            File.WriteAllText(Path.Combine(b.FullName, "libroost_kernels_sm86.so"), "x");
            Assert.Equal(
                Path.Combine(b.FullName, "libroost_kernels_sm86.so"),
                KernelLibraryNames.Find(86, [a.FullName, b.FullName], KernelPlatform.Linux));
        }
        finally
        {
            a.Delete(true);
            b.Delete(true);
        }
    }

    [Theory]
    [InlineData("cudart", 13, CudaPlatform.Linux, "libcudart.so.13")]
    [InlineData("cudart", 12, CudaPlatform.Linux, "libcudart.so.12")]
    [InlineData("cudart", 13, CudaPlatform.Windows, "cudart64_13.dll")]
    [InlineData("cublasLt", 13, CudaPlatform.Linux, "libcublasLt.so.13")]
    [InlineData("cublasLt", 13, CudaPlatform.Windows, "cublasLt64_13.dll")]
    [InlineData("cuda", 13, CudaPlatform.Linux, "libcuda.so.1")]
    [InlineData("cuda", 13, CudaPlatform.Windows, "nvcuda.dll")]
    [InlineData("nvml", 13, CudaPlatform.Linux, "libnvidia-ml.so.1")]
    [InlineData("nvml", 13, CudaPlatform.Windows, "nvml.dll")]
    public void CudaLibraryNamesTakeTheMajorFromTheCaller(string logical, int major, CudaPlatform os, string first)
    {
        Assert.Equal(first, CudaLibraryNames.Candidates(logical, major, os)[0]);
    }

    [Fact]
    public void ForeignLibraryNamesAreLeftToTheDefaultResolver()
    {
        Assert.Empty(CudaLibraryNames.Candidates("libc", 13, CudaPlatform.Linux));
    }

    [Fact]
    public void ResolverRefusesAMajorTheBindingsWereNotGeneratedFor()
    {
        var other = CudaToolkit.Major + 1;
        Assert.Throws<NotSupportedException>(() => CudaLibraryResolver.Install(other));
    }

    [Fact]
    public void SearchDirectoriesPutTheEnvironmentOverrideFirst()
    {
        var env = new Dictionary<string, string> { ["ROOST_CUDA_LIB_DIR"] = "/x", ["CUDA_HOME"] = "/cuda" };
        var dirs = CudaLibraryResolver.SearchDirectories(CudaPlatform.Linux, k => env.GetValueOrDefault(k));
        Assert.Equal("/x", dirs[0]);
        Assert.Contains(Path.Combine("/cuda", "lib64"), dirs); // the resolver joins with the host separator
        var win = CudaLibraryResolver.SearchDirectories(CudaPlatform.Windows, k => k == "CUDA_PATH" ? @"C:\cuda" : null);
        Assert.Equal(2, win.Count);
    }
}

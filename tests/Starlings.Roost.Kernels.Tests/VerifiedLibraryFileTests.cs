using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels.Tests;

/// <summary>
/// The bytes the OS loader maps are the bytes that were hashed (R-060 review): replacing or rewriting the file after the
/// check changes nothing the loader sees.
/// </summary>
public sealed class VerifiedLibraryFileTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("roost-verified-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private static ManifestBinary Expect(byte[] bytes) => new(86, Convert.ToHexStringLower(SHA256.HashData(bytes)), bytes.Length);

    private string Write(string name, byte[] bytes)
    {
        var path = Path.Combine(_root, name);
        File.WriteAllBytes(path, bytes);
        return path;
    }

    [Fact]
    public void ReplacingOrRewritingTheFileAfterTheCheckDoesNotChangeWhatIsLoaded()
    {
        Assert.SkipUnless(OperatingSystem.IsLinux(), "the /proc/self/fd path is Linux's");
        byte[] good = [.. Enumerable.Range(0, 70000).Select(i => (byte)i)];
        var path = Write("libroost_kernels_sm86.so", good);
        using var verified = VerifiedLibraryFile.Open(path, Expect(good));
        Assert.StartsWith("/proc/self/fd/", verified.LoadPath, StringComparison.Ordinal);

        // Rewrite in place (same inode), then swap another file in by rename.
        File.WriteAllBytes(path, new byte[good.Length]);
        var other = Write("other.so", [.. good.Select(b => (byte)~b)]);
        File.Move(other, path, overwrite: true);

        Assert.Equal(good, File.ReadAllBytes(verified.LoadPath));
    }

    [Fact]
    public void TheSealedCopyCannotBeWritten()
    {
        Assert.SkipUnless(OperatingSystem.IsLinux(), "memfd seals are Linux's");
        byte[] good = [1, 2, 3, 4, 5, 6, 7, 8];
        using var verified = VerifiedLibraryFile.Open(Write("lib.so", good), Expect(good));
        Assert.True(verified.IsSealedCopy, "memfd_create should be available on the test host");
        Assert.ThrowsAny<Exception>(() =>
        {
            using var stream = new FileStream(verified.LoadPath, FileMode.Open, FileAccess.Write);
            stream.Write([9, 9, 9]);
            stream.Flush();
        });
        Assert.Equal(good, File.ReadAllBytes(verified.LoadPath));
    }

    [Fact]
    public void ADigestOrSizeMismatchIsAnAssetMismatch()
    {
        byte[] good = [1, 2, 3, 4];
        var path = Write("lib.so", [1, 2, 3, 5]);
        var digest = Assert.Throws<KernelLoadException>(() => VerifiedLibraryFile.Open(path, Expect(good)));
        Assert.Equal(RoostErrorCode.AssetMismatch, digest.Error.Code);
        var size = Assert.Throws<KernelLoadException>(() => VerifiedLibraryFile.Open(path, Expect([1, 2, 3])));
        Assert.Equal(RoostErrorCode.AssetMismatch, size.Error.Code);
        Assert.Contains("bytes", size.Message);
    }

    [Fact]
    public void TheOsLoaderLoadsTheVerifiedPath()
    {
        Assert.SkipUnless(OperatingSystem.IsLinux(), "the /proc/self/fd path is Linux's");

        // A real shared library that is always present: the runtime's own System.Native.
        var source = Path.Combine(RuntimeEnvironment.GetRuntimeDirectory(), "libSystem.Native.so");
        var copy = Path.Combine(_root, "libSystem.Native.so");
        File.Copy(source, copy);
        var bytes = File.ReadAllBytes(copy);
        using var verified = VerifiedLibraryFile.Open(copy, Expect(bytes));
        var handle = NativeLibrary.Load(verified.LoadPath);
        try
        {
            Assert.True(NativeLibrary.TryGetExport(handle, "SystemNative_GetPid", out _));
        }
        finally
        {
            NativeLibrary.Free(handle);
        }
    }
}

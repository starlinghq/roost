using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Microsoft.Win32.SafeHandles;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels;

/// <summary>
/// A kernel library whose bytes were checked against the manifest, held so that the bytes the OS loader maps are the bytes
/// that were hashed (R-060 review: no window between the digest check and the load in which the file can be swapped).
/// <para>
/// Linux: the file is read once through one handle into a sealed memory file (<c>memfd_create</c>, then
/// <c>F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL</c>) while it is hashed, and the loader opens
/// <c>/proc/self/fd/&lt;memfd&gt;</c>. Nothing can change those bytes after the hash, not even this process. Where
/// <c>memfd_create</c> is unavailable, the file is hashed through an open handle and loaded as <c>/proc/self/fd/&lt;fd&gt;</c>,
/// the same inode, so a rename or replacement of the path after the check has no effect.
/// </para>
/// <para>
/// Other systems: the file stays open with read sharing only while it is hashed and loaded; Windows enforces that share
/// mode, so no writer can open it in between.
/// </para>
/// </summary>
internal sealed partial class VerifiedLibraryFile : IDisposable
{
    private const uint MfdCloexec = 0x1;
    private const uint MfdAllowSealing = 0x2;
    private const int FAddSeals = 1033;
    private const int FGetSeals = 1034;
    private const int SealsAll = 0x1 | 0x2 | 0x4 | 0x8; // F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE

    private readonly IDisposable _keepAlive;

    private VerifiedLibraryFile(string loadPath, bool sealedCopy, IDisposable keepAlive)
    {
        LoadPath = loadPath;
        IsSealedCopy = sealedCopy;
        _keepAlive = keepAlive;
    }

    /// <summary>The path to give the OS loader while this object is alive.</summary>
    public string LoadPath { get; }

    /// <summary>Whether <see cref="LoadPath"/> is a sealed in-memory copy (Linux with <c>memfd_create</c>).</summary>
    public bool IsSealedCopy { get; }

    /// <summary>Opens the file, checks its size and SHA-256 against the manifest through one handle, and holds it for loading.</summary>
    /// <param name="path">The library file.</param>
    /// <param name="expected">The manifest's size (0: not recorded) and digest.</param>
    /// <returns>The verified file; dispose it after the load.</returns>
    /// <exception cref="KernelLoadException">Size or digest differ (<see cref="RoostErrorCode.AssetMismatch"/>).</exception>
    public static VerifiedLibraryFile Open(string path, ManifestBinary expected)
    {
        ArgumentNullException.ThrowIfNull(expected);
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1, FileOptions.SequentialScan);
        try
        {
            CheckSize(path, stream.Length, expected);
            if (OperatingSystem.IsLinux() && Directory.Exists("/proc/self/fd"))
            {
                var copy = TrySealedCopy(path, stream, expected);
                if (copy is not null)
                {
                    stream.Dispose();
                    return copy;
                }

                stream.Position = 0;
                CheckDigest(path, SHA256.HashData(stream), expected);
                return new VerifiedLibraryFile(FdPath(stream.SafeFileHandle), sealedCopy: false, stream);
            }

            CheckDigest(path, SHA256.HashData(stream), expected);
            return new VerifiedLibraryFile(Path.GetFullPath(path), sealedCopy: false, stream);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public void Dispose() => _keepAlive.Dispose();

    private static VerifiedLibraryFile? TrySealedCopy(string path, FileStream source, ManifestBinary expected)
    {
        int fd;
        try
        {
            fd = MemfdCreate("roost-kernels", MfdCloexec | MfdAllowSealing);
        }
        catch (EntryPointNotFoundException)
        {
            return null; // glibc before 2.27; fall back to the open handle
        }

        if (fd < 0)
        {
            return null;
        }

        var memfd = new SafeFileHandle(fd, ownsHandle: true);
        try
        {
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            var buffer = new byte[1 << 20];
            long offset = 0;
            int read;
            while ((read = source.Read(buffer, 0, buffer.Length)) > 0)
            {
                hash.AppendData(buffer, 0, read);
                RandomAccess.Write(memfd, buffer.AsSpan(0, read), offset);
                offset += read;
            }

            CheckSize(path, offset, expected);
            CheckDigest(path, hash.GetHashAndReset(), expected);
            if (Fcntl(fd, FAddSeals, SealsAll) != 0 || (Fcntl(fd, FGetSeals, 0) & SealsAll) != SealsAll)
            {
                throw new KernelLoadException(RoostErrorCode.Internal, $"could not seal the in-memory copy of {path} (errno {Marshal.GetLastPInvokeError()})");
            }

            return new VerifiedLibraryFile(FdPath(memfd), sealedCopy: true, memfd);
        }
        catch
        {
            memfd.Dispose();
            throw;
        }
    }

    private static string FdPath(SafeFileHandle handle) => $"/proc/self/fd/{handle.DangerousGetHandle()}";

    private static void CheckSize(string path, long actual, ManifestBinary expected)
    {
        if (expected.Bytes != 0 && actual != expected.Bytes)
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"kernel library {path} is {actual} bytes; the manifest says {expected.Bytes}");
        }
    }

    private static void CheckDigest(string path, byte[] digest, ManifestBinary expected)
    {
        var actual = Convert.ToHexStringLower(digest);
        if (!string.Equals(actual, expected.Sha256, StringComparison.Ordinal))
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"kernel library digest mismatch for {path}: manifest {expected.Sha256}, file {actual}");
        }
    }

    [LibraryImport("libc", EntryPoint = "memfd_create", StringMarshalling = StringMarshalling.Utf8, SetLastError = true)]
    private static partial int MemfdCreate(string name, uint flags);

    [LibraryImport("libc", EntryPoint = "fcntl", SetLastError = true)]
    private static partial int Fcntl(int fd, int command, int argument);
}

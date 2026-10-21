using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels;

/// <summary>What a load is admitted against.</summary>
/// <param name="Manifest">The manifest (already checked against its pinned digest, if the profile pins one).</param>
/// <param name="Sm">The device's SM as major * 10 + minor.</param>
/// <param name="AllowUntested">
/// Admit an SM whose manifest status is <see cref="SmStatus.Untested"/>. Off by default: missing hardware evidence
/// never means supported (AGENTS.md). <see cref="SmStatus.Unsupported"/> is always refused.
/// </param>
public sealed record KernelLoadOptions(KernelManifest Manifest, int Sm, bool AllowUntested = false);

/// <summary>The five C1 exports, resolved by name.</summary>
internal readonly unsafe struct KernelExports(
    delegate* unmanaged<uint> abiVersion,
    delegate* unmanaged<byte*> lastError,
    delegate* unmanaged<OpTable*, int> register,
    delegate* unmanaged<ContextCreateParams*, nint*, int> contextCreate,
    delegate* unmanaged<nint, int> contextDestroy)
{
    public delegate* unmanaged<uint> AbiVersion { get; } = abiVersion;

    public delegate* unmanaged<byte*> LastError { get; } = lastError;

    public delegate* unmanaged<OpTable*, int> Register { get; } = register;

    public delegate* unmanaged<ContextCreateParams*, nint*, int> ContextCreate { get; } = contextCreate;

    public delegate* unmanaged<nint, int> ContextDestroy { get; } = contextDestroy;
}

/// <summary>
/// A kernel library loaded by its C1 exports (<c>roost_abi_version</c>, <c>roost_last_error</c>, <c>roost_register</c>,
/// <c>roost_context_create</c>, <c>roost_context_destroy</c>) after the file's digest matched the manifest's and the ABI
/// rule held: the major is equal and the library's minor is at least the caller's (contract C1).
/// </summary>
public sealed class KernelLibrary : IDisposable
{
    private const int MaxEntries = 4096;
    private const int MaxVariantIdLength = 128;
    private const uint KnownFlags = 0x1F;

    private readonly nint _handle;
    private readonly KernelExports _exports;
    private bool _disposed;

    private KernelLibrary(
        string path,
        string name,
        int sm,
        SmStatus status,
        uint abiVersion,
        KernelManifest manifest,
        nint handle,
        KernelExports exports)
    {
        Path = path;
        Name = name;
        Sm = sm;
        SmStatus = status;
        AbiVersion = abiVersion;
        Manifest = manifest;
        _handle = handle;
        _exports = exports;
        Kernels = [];
    }

    /// <summary>The file the library was loaded from.</summary>
    public string Path { get; }

    /// <summary>The library's own name from its table (<c>library_name</c>), for diagnostics and evidence.</summary>
    public string Name { get; }

    /// <summary>The SM the library was admitted for.</summary>
    public int Sm { get; }

    /// <summary>The SM's status in the manifest.</summary>
    public SmStatus SmStatus { get; }

    /// <summary>The value <c>roost_abi_version()</c> returned: major in the high 16 bits.</summary>
    public uint AbiVersion { get; }

    /// <summary>The manifest the library was checked against.</summary>
    public KernelManifest Manifest { get; }

    /// <summary>The registration table.</summary>
    public IReadOnlyList<RegisteredKernel> Kernels { get; private set; }

    /// <summary>
    /// Admits an SM and loads its library: checks the manifest and SM status, then the file's size and SHA-256 against the
    /// manifest, and only then loads the file and reads its table.
    /// </summary>
    /// <param name="path">The library file, found by <see cref="KernelLibraryNames.Find"/> or given by the profile.</param>
    /// <param name="options">The manifest and SM to admit against.</param>
    /// <returns>The loaded library.</returns>
    /// <exception cref="ProfileUnsupportedException">The SM is unsupported or untested, or the manifest builds no library for it.</exception>
    /// <exception cref="KernelLoadException">
    /// A digest, ABI or table mismatch (<see cref="RoostErrorCode.AssetMismatch"/>), or the OS could not load the file
    /// (<see cref="RoostErrorCode.Internal"/>).
    /// </exception>
    public static unsafe KernelLibrary Load(string path, KernelLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var status = Admit(options);
        var expected = ExpectedBinary(options);

        if (!File.Exists(path))
        {
            throw new ProfileUnsupportedException(OpId.None, options.Sm, $"no kernel library file at {path}");
        }

        VerifyFile(path, expected);

        nint handle;
        try
        {
            handle = NativeLibrary.Load(path);
        }
        catch (Exception e) when (e is DllNotFoundException or BadImageFormatException)
        {
            throw new KernelLoadException(RoostErrorCode.Internal, $"the OS could not load {path}: {e.Message}", e);
        }

        try
        {
            var exports = new KernelExports(
                (delegate* unmanaged<uint>)Export(handle, "roost_abi_version"),
                (delegate* unmanaged<byte*>)Export(handle, "roost_last_error"),
                (delegate* unmanaged<OpTable*, int>)Export(handle, "roost_register"),
                (delegate* unmanaged<ContextCreateParams*, nint*, int>)Export(handle, "roost_context_create"),
                (delegate* unmanaged<nint, int>)Export(handle, "roost_context_destroy"));
            return Attach(path, options, status, handle, exports);
        }
        catch
        {
            NativeLibrary.Free(handle);
            throw;
        }
    }

    /// <summary>Creates a library context for a device (<c>roost_context_create</c>).</summary>
    /// <param name="device">CUDA device ordinal, current on the calling thread.</param>
    /// <param name="options">Context flags.</param>
    /// <param name="stream">Stream for setup work, or 0 for the legacy default stream.</param>
    /// <returns>The context; dispose it before the library.</returns>
    /// <exception cref="KernelLoadException">The library refused the context.</exception>
    public unsafe KernelContext CreateContext(int device, ContextOptions options = ContextOptions.None, nint stream = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var p = new ContextCreateParams
        {
            StructSize = (uint)sizeof(ContextCreateParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Device = device,
            Flags = options,
            Stream = stream,
        };
        nint ctx = 0;
        var rc = _exports.ContextCreate(&p, &ctx);
        if (rc != (int)KernelStatus.Ok || ctx == 0)
        {
            throw new KernelLoadException(
                rc == (int)KernelStatus.AbiMismatch ? RoostErrorCode.AssetMismatch : RoostErrorCode.Internal,
                $"roost_context_create failed with {(KernelStatus)rc}: {LastError()}");
        }

        return new KernelContext(this, ctx);
    }

    internal unsafe void DestroyContext(nint ctx)
    {
        if (!_disposed)
        {
            _ = _exports.ContextDestroy(ctx);
        }
    }

    /// <summary>Unloads the library. Contexts and function pointers from it are invalid afterwards.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_handle != 0)
        {
            NativeLibrary.Free(_handle);
        }
    }

    internal static unsafe KernelLibrary Attach(string path, KernelLoadOptions options, SmStatus status, nint handle, KernelExports exports)
    {
        var manifest = options.Manifest;
        var abi = exports.AbiVersion();
        if (!KernelAbi.IsCompatible(abi))
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"kernel library ABI {abi >> 16}.{abi & 0xFFFF} is not compatible with the loader's {KernelAbi.Major}.{KernelAbi.Minor}: " +
                "the major must be equal and the library minor at least the loader's");
        }

        var table = new OpTable { StructSize = (uint)sizeof(OpTable), AbiMinor = (uint)KernelAbi.Minor };
        var rc = exports.Register(&table);
        if (rc != (int)KernelStatus.Ok)
        {
            throw new KernelLoadException(
                rc == (int)KernelStatus.AbiMismatch ? RoostErrorCode.AssetMismatch : RoostErrorCode.Internal,
                $"roost_register failed with {(KernelStatus)rc}: {LastError(exports)}");
        }

        var library = new KernelLibrary(path, ReadName(table.LibraryName), options.Sm, status, abi, manifest, handle, exports);
        library.Kernels = ReadTable(library, table, abi);
        CrossCheck(library, manifest, options.Sm);
        return library;
    }

    private static unsafe List<RegisteredKernel> ReadTable(KernelLibrary library, in OpTable table, uint abi)
    {
        if (table.LibraryAbiMajor != KernelAbi.Major || table.LibraryAbiMajor != abi >> 16 || table.LibraryAbiMinor != (abi & 0xFFFF))
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"kernel library table reports ABI {table.LibraryAbiMajor}.{table.LibraryAbiMinor} but roost_abi_version() returned {abi >> 16}.{abi & 0xFFFF}");
        }

        if (table.EntrySize < (uint)sizeof(OpEntry))
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"kernel library entry_size {table.EntrySize} is smaller than the loader's {sizeof(OpEntry)}");
        }

        if (table.EntryCount > MaxEntries || (table.EntryCount > 0 && table.Entries == 0))
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"kernel library table is malformed: {table.EntryCount} entries at {table.Entries:x}");
        }

        var kernels = new List<RegisteredKernel>((int)table.EntryCount);
        var seen = new HashSet<(OpId, string)>();
        for (var i = 0; i < table.EntryCount; i++)
        {
            // The library's stride, not ours: a larger entry_size from a later minor still walks correctly.
            var entry = Unsafe.ReadUnaligned<OpEntry>((void*)(table.Entries + (nint)(i * (long)table.EntrySize)));
            var kernel = ReadEntry(library, entry, i, abi);
            if (!seen.Add((kernel.Op, kernel.VariantId)))
            {
                throw new KernelLoadException(
                    RoostErrorCode.AssetMismatch,
                    $"kernel library registers variant {kernel.VariantId} of {kernel.Op} twice");
            }

            kernels.Add(kernel);
        }

        return kernels;
    }

    private static unsafe RegisteredKernel ReadEntry(KernelLibrary library, in OpEntry entry, int index, uint abi)
    {
        if (!Enum.IsDefined(entry.Op) || entry.Op == OpId.None)
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"kernel library entry {index} has unknown op id {(uint)entry.Op}");
        }

        var id = entry.VariantId == 0 ? null : Marshal.PtrToStringUTF8(entry.VariantId);
        if (string.IsNullOrEmpty(id) || id.Length > MaxVariantIdLength || !id.All(static c => c is > ' ' and < (char)0x7F))
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"kernel library entry {index} ({entry.Op}) has no valid ASCII variant id");
        }

        var what = $"kernel library entry {id} ({entry.Op})";
        if ((uint)entry.Flags is var flags && (flags & ~KnownFlags) != 0)
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"{what} sets unknown flag bits 0x{flags:x}");
        }

        if (entry.Run == 0)
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"{what} has no run entry point");
        }

        if (entry.Flags.HasFlag(VariantTraits.NeedsPlan) != (entry.Plan != 0))
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"{what}: the needs-plan flag and the plan entry point disagree");
        }

        var c = entry.Constraints;
        if (c.SmMin == 0 || (c.SmMax != 0 && c.SmMax < c.SmMin))
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"{what} has an invalid SM range {c.SmMin}..{c.SmMax}");
        }

        if (entry.ParamsAbiMinor > (abi & 0xFFFF) || entry.Reserved0 != 0)
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"{what} has a params minor above the library's or a non-zero reserved field");
        }

        return new RegisteredKernel(entry.Op, id, entry.Flags, c, entry.ParamsStructSize, entry.ParamsAbiMinor, entry.Run, entry.Plan, entry.Workspace, library);
    }

    private static void CrossCheck(KernelLibrary library, KernelManifest manifest, int sm)
    {
        var listed = manifest.Variants
            .Where(v => v.Binaries.Any(b => b.Arch == sm))
            .ToDictionary(v => (v.Op, v.Id));
        foreach (var kernel in library.Kernels)
        {
            if (!listed.TryGetValue((kernel.Op, kernel.VariantId), out var v))
            {
                throw new KernelLoadException(
                    RoostErrorCode.AssetMismatch,
                    $"library {library.Name} registers {kernel.VariantId} ({kernel.Op}), which the manifest does not list for SM {sm}");
            }

            if (v.Flags != kernel.Flags || (v.SmMin != 0 && v.SmMin != (int)kernel.Constraints.SmMin))
            {
                throw new KernelLoadException(
                    RoostErrorCode.AssetMismatch,
                    $"library {library.Name}: {kernel.VariantId} ({kernel.Op}) registers flags {kernel.Flags} and sm_min {kernel.Constraints.SmMin}; " +
                    $"the manifest says {v.Flags} and {v.SmMin}");
            }
        }

        var registered = library.Kernels.Select(k => (k.Op, k.VariantId)).ToHashSet();
        foreach (var key in listed.Keys.Where(k => !registered.Contains(k)))
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"the manifest lists {key.Id} ({key.Op}) for SM {sm}, but library {library.Name} does not register it");
        }
    }

    private static SmStatus Admit(KernelLoadOptions options)
    {
        var manifest = options.Manifest;
        ArgumentNullException.ThrowIfNull(manifest);
        if (manifest.AbiMajor != KernelAbi.Major || manifest.AbiMinor < KernelAbi.Minor)
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"the kernel manifest is for ABI {manifest.AbiMajor}.{manifest.AbiMinor}; the loader needs major {KernelAbi.Major} and minor at least {KernelAbi.Minor}");
        }

        var entry = manifest.Sms.FirstOrDefault(s => s.Arch == options.Sm);
        if (entry is null)
        {
            var built = string.Join(", ", manifest.Sms.Select(s => s.Arch));
            throw new ProfileUnsupportedException(OpId.None, options.Sm, $"the manifest builds no library for SM {options.Sm} (built: {built})");
        }

        return entry.Status switch
        {
            SmStatus.Unsupported => throw new ProfileUnsupportedException(OpId.None, options.Sm, "the manifest marks the SM unsupported"),
            SmStatus.Untested when !options.AllowUntested =>
                throw new ProfileUnsupportedException(OpId.None, options.Sm, "the SM has no hardware evidence (status untested); opt in with AllowUntested"),
            var s => s,
        };
    }

    private static ManifestBinary ExpectedBinary(KernelLoadOptions options)
    {
        var manifest = options.Manifest;
        var binaries = manifest.Variants.SelectMany(v => v.Binaries).Where(b => b.Arch == options.Sm).ToList();
        if (binaries.Count == 0)
        {
            throw new ProfileUnsupportedException(OpId.None, options.Sm, $"no variant has a library digest for SM {options.Sm}");
        }

        // One library per SM holds every variant, so every variant's entry for the SM names the same file.
        var distinct = binaries.Select(b => b.Sha256).Distinct(StringComparer.Ordinal).ToList();
        if (distinct.Count != 1)
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"the manifest lists {distinct.Count} different digests for the one SM {options.Sm} library");
        }

        return binaries.FirstOrDefault(b => b.Bytes != 0) ?? binaries[0];
    }

    private static void VerifyFile(string path, ManifestBinary expected)
    {
        using var stream = File.OpenRead(path);
        if (expected.Bytes != 0 && stream.Length != expected.Bytes)
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"kernel library {path} is {stream.Length} bytes; the manifest says {expected.Bytes}");
        }

        var actual = Convert.ToHexStringLower(SHA256.HashData(stream));
        if (!string.Equals(actual, expected.Sha256, StringComparison.Ordinal))
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"kernel library digest mismatch for {path}: manifest {expected.Sha256}, file {actual}");
        }
    }

    private static nint Export(nint handle, string name)
    {
        if (!NativeLibrary.TryGetExport(handle, name, out var address))
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"the library does not export {name}: not a Roost kernel library (C1)");
        }

        return address;
    }

    private static string ReadName(nint pointer) => pointer == 0 ? "<unnamed>" : Marshal.PtrToStringUTF8(pointer) ?? "<unnamed>";

    private unsafe string LastError() => LastError(_exports);

    /// <summary>The library's last error text on the calling thread (<c>roost_last_error</c>), for the message of a failed run.</summary>
    /// <returns>The text, or an empty string.</returns>
    public unsafe string LastErrorText() => LastError();

    private static unsafe string LastError(KernelExports exports)
    {
        var message = exports.LastError();
        return message == null ? string.Empty : Encoding.UTF8.GetString(MemoryMarshal.CreateReadOnlySpanFromNullTerminated(message));
    }
}

/// <summary>A library context (<c>RoostContext*</c>) for one device, owned by the worker thread that uses it.</summary>
public sealed class KernelContext : IDisposable
{
    private nint _handle;

    internal KernelContext(KernelLibrary library, nint handle)
    {
        Library = library;
        _handle = handle;
    }

    /// <summary>The library that created it.</summary>
    public KernelLibrary Library { get; }

    /// <summary>The native pointer.</summary>
    public nint Handle => _handle;

    /// <summary>Destroys the context (<c>roost_context_destroy</c>).</summary>
    public void Dispose()
    {
        var handle = Interlocked.Exchange(ref _handle, 0);
        if (handle != 0)
        {
            Library.DestroyContext(handle);
        }
    }
}

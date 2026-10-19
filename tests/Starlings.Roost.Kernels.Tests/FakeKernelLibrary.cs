using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Kernels.Tests;

/// <summary>
/// A managed stand-in for a kernel library's C1 exports: the same function-pointer calls the loader makes on a real
/// library, with a registration table built in native memory. State is static because the exports are
/// <c>UnmanagedCallersOnly</c>; the tests that use it live in one class and so run one after another.
/// </summary>
internal static unsafe class FakeKernelLibrary
{
    private static readonly List<nint> Allocations = [];
    private static nint _entries;

    public static uint Abi = KernelAbi.Version;
    public static int RegisterStatus;
    public static uint EntryCount;
    public static uint EntrySize = (uint)sizeof(OpEntry);
    public static uint LibraryAbi = KernelAbi.Version;
    public static string Name = "fake-kernels-sm86";

    public static KernelExports Exports => new(&AbiVersion, &LastError, &Register, &ContextCreate, &ContextDestroy);

    public static void Reset()
    {
        foreach (var p in Allocations)
        {
            NativeMemory.Free((void*)p);
        }

        Allocations.Clear();
        Abi = LibraryAbi = KernelAbi.Version;
        RegisterStatus = 0;
        EntrySize = (uint)sizeof(OpEntry);
        EntryCount = 0;
        _entries = 0;
        Name = "fake-kernels-sm86";
    }

    /// <summary>
    /// Builds the table: entries are laid out at <see cref="EntrySize"/> stride. A stride below <c>sizeof(OpEntry)</c>
    /// copies only the first <see cref="EntrySize"/> bytes of each entry, and the block is always large enough for a
    /// whole <c>OpEntry</c> per slot, so no write leaves the allocation.
    /// </summary>
    public static void SetEntries(params OpEntry[] entries)
    {
        var stride = (int)EntrySize;
        var slot = Math.Max(stride, sizeof(OpEntry));
        var block = (byte*)NativeMemory.AllocZeroed((nuint)(Math.Max(1, entries.Length) * slot));
        Allocations.Add((nint)block);
        for (var i = 0; i < entries.Length; i++)
        {
            var entry = entries[i];
            Buffer.MemoryCopy(&entry, block + (i * stride), slot, Math.Min(stride, sizeof(OpEntry)));
        }

        _entries = (nint)block;
        EntryCount = (uint)entries.Length;
    }

    public static nint Utf8(string s)
    {
        var p = Marshal.StringToCoTaskMemUTF8(s); // test strings live until the process ends
        return p;
    }

    public static OpEntry Entry(OpId op, string variant, VariantTraits flags, uint smMin = 80, uint smMax = 0, ulong act = 0)
    {
        return new OpEntry
        {
            Op = op,
            Flags = flags,
            VariantId = Utf8(variant),
            ParamsStructSize = 64,
            ParamsAbiMinor = 0,
            Constraints = new KernelConstraints { SmMin = smMin, SmMax = smMax, ActDtypeMask = act },
            Run = (nint)(delegate* unmanaged<nint, void*, void*, nint, int>)&Run,
        };
    }

    [UnmanagedCallersOnly]
    private static int Run(nint ctx, void* p, void* ws, nint stream) => 0;

    [UnmanagedCallersOnly]
    private static uint AbiVersion() => Abi;

    [UnmanagedCallersOnly]
    private static byte* LastError() => (byte*)Marshal.StringToCoTaskMemUTF8("fake error");

    [UnmanagedCallersOnly]
    private static int Register(OpTable* table)
    {
        if (RegisterStatus != 0)
        {
            return RegisterStatus;
        }

        table->EntrySize = EntrySize;
        table->EntryCount = EntryCount;
        table->Entries = _entries;
        table->LibraryName = Marshal.StringToCoTaskMemUTF8(Name);
        table->LibraryAbiMajor = LibraryAbi >> 16;
        table->LibraryAbiMinor = LibraryAbi & 0xFFFF;
        return 0;
    }

    [UnmanagedCallersOnly]
    private static int ContextCreate(ContextCreateParams* p, nint* ctx)
    {
        *ctx = 0x1000;
        return 0;
    }

    [UnmanagedCallersOnly]
    private static int ContextDestroy(nint ctx) => 0;
}

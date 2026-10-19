using System.Runtime.InteropServices;
using Starlings.Roost;
using Starlings.Roost.Abi;

namespace Roost.HybridConformance;

/// <summary>A built kernel library, used only through its C1 exports and registration table.</summary>
internal sealed unsafe class Library : IDisposable
{
    private readonly nint _handle;
    private readonly OpEntry[] _entries;
    private nint _ctx;

    private Library(nint handle, OpEntry[] entries, string name, uint major, uint minor)
    {
        _handle = handle;
        _entries = entries;
        Name = name;
        AbiMajor = major;
        AbiMinor = minor;
    }

    public string Name { get; }

    public uint AbiMajor { get; }

    public uint AbiMinor { get; }

    public IReadOnlyList<OpEntry> Entries => _entries;

    public static Library Load(string path, int device)
    {
        var h = NativeLibrary.Load(path);
        var abi = (delegate* unmanaged[Cdecl]<uint>)NativeLibrary.GetExport(h, "roost_abi_version");
        var reg = (delegate* unmanaged[Cdecl]<OpTable*, int>)NativeLibrary.GetExport(h, "roost_register");
        var version = abi();
        var table = new OpTable { StructSize = (uint)sizeof(OpTable), AbiMinor = 1 };
        var rc = reg(&table);
        if (rc != 0)
        {
            throw new InvalidOperationException($"roost_register returned {rc}");
        }

        if (table.EntrySize != (uint)sizeof(OpEntry))
        {
            throw new InvalidOperationException($"entry_size {table.EntrySize} != {sizeof(OpEntry)}");
        }

        var entries = new OpEntry[table.EntryCount];
        for (var i = 0; i < entries.Length; i++)
        {
            entries[i] = ((OpEntry*)table.Entries)[i];
        }

        var lib = new Library(h, entries, Marshal.PtrToStringAnsi(table.LibraryName)!, table.LibraryAbiMajor, table.LibraryAbiMinor);
        if (version != ((table.LibraryAbiMajor << 16) | table.LibraryAbiMinor))
        {
            throw new InvalidOperationException("roost_abi_version disagrees with the table");
        }

        var create = (delegate* unmanaged[Cdecl]<ContextCreateParams*, nint*, int>)NativeLibrary.GetExport(h, "roost_context_create");
        var p = new ContextCreateParams { StructSize = (uint)sizeof(ContextCreateParams), AbiMinor = 1, Device = device };
        nint ctx;
        rc = create(&p, &ctx);
        if (rc != 0)
        {
            throw new InvalidOperationException($"roost_context_create returned {rc}: {lib.LastError()}");
        }

        lib._ctx = ctx;
        return lib;
    }

    public static string VariantId(in OpEntry e) => Marshal.PtrToStringAnsi(e.VariantId)!;

    public OpEntry? Find(string variant)
    {
        foreach (var e in _entries)
        {
            if (VariantId(e) == variant)
            {
                return e;
            }
        }

        return null;
    }

    public string LastError()
    {
        var f = (delegate* unmanaged[Cdecl]<nint>)NativeLibrary.GetExport(_handle, "roost_last_error");
        return Marshal.PtrToStringAnsi(f()) ?? string.Empty;
    }

    public WorkspaceSizes QueryWorkspace<T>(in OpEntry e, in T p)
        where T : unmanaged
    {
        var sizes = new WorkspaceSizes { StructSize = (uint)sizeof(WorkspaceSizes), AbiMinor = 1 };
        if (e.Workspace == 0)
        {
            return sizes;
        }

        fixed (T* pp = &p)
        {
            var rc = ((delegate* unmanaged[Cdecl]<nint, void*, WorkspaceSizes*, int>)e.Workspace)(_ctx, pp, &sizes);
            Throw(rc, "workspace");
        }

        return sizes;
    }

    public int RunRaw<T>(in OpEntry e, in T p, Workspace* ws, nint stream)
        where T : unmanaged
    {
        fixed (T* pp = &p)
        {
            return ((delegate* unmanaged[Cdecl]<nint, void*, Workspace*, nint, int>)e.Run)(_ctx, pp, ws, stream);
        }
    }

    public int PlanRaw<T>(in OpEntry e, in T p, Workspace* ws, nint stream)
        where T : unmanaged
    {
        fixed (T* pp = &p)
        {
            return ((delegate* unmanaged[Cdecl]<nint, void*, Workspace*, nint, int>)e.Plan)(_ctx, pp, ws, stream);
        }
    }

    /// <summary>When set, Run captures the call into a CUDA graph and replays it once instead of launching directly.</summary>
    public bool GraphMode { get; set; }

    public void Run<T>(in OpEntry e, in T p, Workspace* ws, nint stream)
        where T : unmanaged
    {
        if (!GraphMode)
        {
            Throw(RunRaw(e, p, ws, stream), "run");
            return;
        }

        var entry = e;
        var pp = p;
        var w = ws;
        Throw(Cuda.RunGraphed(stream, () => RunRaw(entry, pp, w, stream)), "run (graph capture)");
    }

    public void Plan<T>(in OpEntry e, in T p, Workspace* ws, nint stream)
        where T : unmanaged =>
        Throw(PlanRaw(e, p, ws, stream), "plan");

    private void Throw(int rc, string what)
    {
        if (rc != 0)
        {
            throw new InvalidOperationException($"{what} returned status {rc}: {LastError()}");
        }
    }

    public void Dispose()
    {
        if (_ctx != 0)
        {
            ((delegate* unmanaged[Cdecl]<nint, int>)NativeLibrary.GetExport(_handle, "roost_context_destroy"))(_ctx);
            _ctx = 0;
        }
    }
}

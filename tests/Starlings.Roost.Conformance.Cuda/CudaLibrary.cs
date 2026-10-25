using System.Runtime.InteropServices;
using Starlings.Roost.Abi;

// Origin: kernels/tests/ops-cuda/Library.cs (R-017, stream C), copied here so the conformance runner has no dependency on a
// stream C test project. The class that loads a built library through its C1 exports, the way the worker's loader does.
namespace Starlings.Roost.Conformance.Cuda;

/// <summary>A caller-owned workspace set made from an entry's workspace query (C1).</summary>
internal sealed unsafe class WorkspaceSet : IDisposable
{
    private readonly nint _device;
    private readonly nint _pinned;
    private readonly nint _plan;

    public WorkspaceSet(WorkspaceSizes sizes)
    {
        _device = sizes.DeviceBytes > 0 ? CudaRuntime.Alloc((long)sizes.DeviceBytes) : 0;
        _pinned = sizes.PinnedHostBytes > 0 ? CudaRuntime.AllocPinned((long)sizes.PinnedHostBytes) : 0;
        _plan = sizes.PlanBytes > 0 ? (nint)NativeMemory.AlignedAlloc((nuint)sizes.PlanBytes, 64) : 0;
        if (_plan != 0)
        {
            NativeMemory.Clear((void*)_plan, (nuint)sizes.PlanBytes);
        }

        Sizes = sizes;
        Value = new Workspace
        {
            StructSize = (uint)sizeof(Workspace),
            AbiMinor = 0,
            Device = _device,
            DeviceBytes = sizes.DeviceBytes,
            PinnedHost = _pinned,
            PinnedHostBytes = sizes.PinnedHostBytes,
            Plan = _plan,
            PlanBytes = sizes.PlanBytes,
        };
    }

    public WorkspaceSizes Sizes { get; }

    public Workspace Value;

    public nint Plan => _plan;

    public void Dispose()
    {
        if (_device != 0)
        {
            CudaRuntime.Release(_device);
        }

        if (_pinned != 0)
        {
            CudaRuntime.ReleasePinned(_pinned);
        }

        if (_plan != 0)
        {
            NativeMemory.AlignedFree((void*)_plan);
        }
    }
}

/// <summary>A kernel library loaded through its C1 exports, the way the worker's loader does.</summary>
internal sealed unsafe class Library : IDisposable
{
    private readonly nint _handle;
    private readonly delegate* unmanaged[Cdecl]<nint> _lastError;
    private readonly delegate* unmanaged[Cdecl]<nint, nint> _contextDestroy;
    private readonly Dictionary<string, OpEntry> _entries = [];

    public Library(string path, int device)
    {
        _handle = NativeLibrary.Load(path);
        var abiVersion = (delegate* unmanaged[Cdecl]<uint>)NativeLibrary.GetExport(_handle, "roost_abi_version");
        _lastError = (delegate* unmanaged[Cdecl]<nint>)NativeLibrary.GetExport(_handle, "roost_last_error");
        var register = (delegate* unmanaged[Cdecl]<OpTable*, int>)NativeLibrary.GetExport(_handle, "roost_register");
        var create = (delegate* unmanaged[Cdecl]<ContextCreateParams*, nint*, int>)NativeLibrary.GetExport(_handle, "roost_context_create");
        _contextDestroy = (delegate* unmanaged[Cdecl]<nint, nint>)NativeLibrary.GetExport(_handle, "roost_context_destroy");
        AbiVersion = abiVersion();
        var table = new OpTable { StructSize = (uint)sizeof(OpTable), AbiMinor = 0 };
        CheckStatus(register(&table), "roost_register");
        Name = Marshal.PtrToStringAnsi(table.LibraryName) ?? "?";
        for (uint i = 0; i < table.EntryCount; i++)
        {
            OpEntry e = *(OpEntry*)(table.Entries + (nint)(i * table.EntrySize));
            _entries[Marshal.PtrToStringAnsi(e.VariantId)!] = e;
        }

        var cp = new ContextCreateParams { StructSize = (uint)sizeof(ContextCreateParams), AbiMinor = 0, Device = device };
        nint ctx = 0;
        CheckStatus(create(&cp, &ctx), "roost_context_create");
        Context = ctx;
    }

    public uint AbiVersion { get; }

    public string Name { get; }

    public nint Context { get; }

    public IReadOnlyDictionary<string, OpEntry> Entries => _entries;

    public string LastError => Marshal.PtrToStringAnsi(_lastError()) ?? string.Empty;

    public void CheckStatus(int status, string what)
    {
        if (status != 0)
        {
            throw new KernelException(status, $"{what}: status {status} ({(KernelStatus)status}): {LastError}");
        }
    }

    /// <summary>Calls run() directly (no plan), returning the raw status.</summary>
    public int Run(string variant, void* p, Workspace* ws, nint stream)
        => ((delegate* unmanaged[Cdecl]<nint, void*, Workspace*, nint, int>)_entries[variant].Run)(Context, p, ws, stream);

    public int Plan(string variant, void* p, Workspace* ws, nint stream)
        => ((delegate* unmanaged[Cdecl]<nint, void*, Workspace*, nint, int>)_entries[variant].Plan)(Context, p, ws, stream);

    public int Query(string variant, void* p, WorkspaceSizes* sizes)
        => ((delegate* unmanaged[Cdecl]<nint, void*, WorkspaceSizes*, int>)_entries[variant].Workspace)(Context, p, sizes);

    /// <summary>Runs an op without a workspace and synchronises <paramref name="stream"/>.</summary>
    public void Exec<T>(string variant, ref T p, nint stream) where T : unmanaged
    {
        fixed (T* pp = &p)
        {
            CheckStatus(Run(variant, pp, null, stream), variant);
        }

        CudaRuntime.Sync(stream);
    }

    /// <summary>Raw status of a call without a workspace (for rejection checks); nothing is synchronised.</summary>
    public int Try<T>(string variant, ref T p, nint stream) where T : unmanaged
    {
        fixed (T* pp = &p)
        {
            return Run(variant, pp, null, stream);
        }
    }

    public WorkspaceSet MakeWorkspace<T>(string variant, ref T p) where T : unmanaged
    {
        var sizes = new WorkspaceSizes { StructSize = (uint)sizeof(WorkspaceSizes), AbiMinor = 0 };
        fixed (T* pp = &p)
        {
            CheckStatus(Query(variant, pp, &sizes), variant + " workspace query");
        }

        return new WorkspaceSet(sizes);
    }

    public void PlanWith<T>(string variant, ref T p, WorkspaceSet ws, nint stream) where T : unmanaged
    {
        fixed (T* pp = &p)
        fixed (Workspace* w = &ws.Value)
        {
            CheckStatus(Plan(variant, pp, w, stream), variant + " plan");
        }

        CudaRuntime.Sync(stream);
    }

    public void RunWith<T>(string variant, ref T p, WorkspaceSet ws, nint stream, bool sync = true) where T : unmanaged
    {
        fixed (T* pp = &p)
        fixed (Workspace* w = &ws.Value)
        {
            CheckStatus(Run(variant, pp, w, stream), variant);
        }

        if (sync)
        {
            CudaRuntime.Sync(stream);
        }
    }

    public int TryRunWith<T>(string variant, ref T p, WorkspaceSet ws, nint stream) where T : unmanaged
    {
        fixed (T* pp = &p)
        fixed (Workspace* w = &ws.Value)
        {
            return Run(variant, pp, w, stream);
        }
    }

    public void Dispose()
    {
        _contextDestroy(Context);
        NativeLibrary.Free(_handle);
    }
}

internal sealed class KernelException : Exception
{
    public KernelException(int status, string message)
        : base(message) => Status = status;

    public int Status { get; }
}

using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Kernels;

namespace Starlings.Roost.Worker.Eager;

/// <summary>A kernel's run, plan or workspace query returned a failure status.</summary>
public sealed class KernelRunException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="op">The op.</param>
    /// <param name="variant">The variant id.</param>
    /// <param name="status">The library's status.</param>
    /// <param name="stage">run, plan or workspace.</param>
    /// <param name="detail">The library's last error text.</param>
    public KernelRunException(OpId op, string variant, KernelStatus status, string stage, string detail)
        : base($"{op} variant {variant}: {stage} returned {status}: {detail}")
    {
        Op = op;
        Variant = variant;
        Status = status;
    }

    /// <summary>The op.</summary>
    public OpId Op { get; }

    /// <summary>The variant id.</summary>
    public string Variant { get; }

    /// <summary>The library's status.</summary>
    public KernelStatus Status { get; }
}

/// <summary>How often a variant's <c>plan()</c> runs.</summary>
internal enum PlanScope
{
    /// <summary>The variant has no plan.</summary>
    None,

    /// <summary>Once per shape key (cuBLASLt algorithm choice); never again unless the workspace grows.</summary>
    PerShape,

    /// <summary>Once per step (the batch layout changes every step).</summary>
    PerStep,
}

/// <summary>
/// The state <see cref="CudaOps"/> runs against (R-025). The ops are static members (C3), so the backend finds its
/// library context, stream and per-step host tables here, in a thread-static slot that one worker thread binds for the
/// life of the worker. Nothing here is shared between threads.
/// </summary>
/// <remarks>
/// <para>
/// Workspaces: an op whose registration has a workspace query gets a <see cref="Workspace"/> from the session. Plans
/// (<c>plan()</c>) run on first use of a shape (cuBLASLt algorithm pinning: Gemm and LmHead, keyed by n, k and the shape
/// bucket) or once per step (paged attention, whose plan reads the step's batch tables). This breaks the "calls never
/// allocate" clause of C3 on the first step of each shape: that is the warm-up step graph capture (R-033) removes.
/// </para>
/// </remarks>
public sealed unsafe class CudaOpsSession : IDisposable
{
    [ThreadStatic]
    private static CudaOpsSession? s_current;

    private readonly KernelRegistry _registry;
    private readonly KernelContext _context;
    private readonly DeviceAllocator _allocator;
    private readonly Dictionary<(OpId, DType, DType, DType, int, int, int, string?), RegisteredKernel> _kernels = [];
    private readonly Dictionary<(string Variant, long Key), WorkspaceEntry> _workspaces = [];
    private readonly Dictionary<OpId, long> _launches = [];
    private long _epoch = 1;

    /// <summary>Creates a session over a loaded library's registry.</summary>
    /// <param name="registry">The registered kernels.</param>
    /// <param name="context">A context of one of the registry's libraries, made on the worker thread's device.</param>
    /// <param name="sm">The device's SM (major * 10 + minor).</param>
    /// <param name="allocator">The allocator for workspaces.</param>
    /// <param name="stream">The compute stream.</param>
    public CudaOpsSession(KernelRegistry registry, KernelContext context, int sm, DeviceAllocator allocator, nint stream)
    {
        _registry = registry;
        _context = context;
        Sm = sm;
        _allocator = allocator;
        Stream = stream;
    }

    /// <summary>The session bound to the calling thread.</summary>
    /// <exception cref="InvalidOperationException">No session is bound.</exception>
    public static CudaOpsSession Current => s_current ?? throw new InvalidOperationException("CudaOps used outside a bound CudaOpsSession (call Bind on the worker thread)");

    /// <summary>The device's SM.</summary>
    public int Sm { get; }

    /// <summary>The compute stream every kernel runs on.</summary>
    public nint Stream { get; }

    /// <summary>The device plan region and its pinned host mirror for the current step (see <see cref="SetStepTables"/>).</summary>
    public (nint Device, nint Host, long Bytes) StepTables { get; private set; }

    /// <summary>The number of kernel launches per op since the session was made.</summary>
    public IReadOnlyDictionary<OpId, long> Launches => _launches;

    /// <summary>The variant ids chosen so far, by op.</summary>
    public IEnumerable<(OpId Op, string Variant)> ChosenVariants => _kernels.Values.Select(static k => (k.Op, k.VariantId)).Distinct();

    /// <summary>Binds this session to the calling thread. Dispose the result to unbind.</summary>
    /// <returns>A scope.</returns>
    public IDisposable Bind()
    {
        var previous = s_current;
        s_current = this;
        return new Scope(previous);
    }

    /// <summary>Starts a step: plans that depend on the batch layout run again, and the host mirror of the plan region is set.</summary>
    /// <param name="deviceRegion">The plan region on the device (the address the model's plan inputs point into).</param>
    /// <param name="hostRegion">Its host mirror, with the same contents.</param>
    /// <param name="bytes">Region size.</param>
    public void SetStepTables(nint deviceRegion, nint hostRegion, long bytes)
    {
        StepTables = (deviceRegion, hostRegion, bytes);
        _epoch++;
    }

    /// <summary>The host copy of a plan-input span (its device address lies inside the step's plan region).</summary>
    /// <param name="span">A device span inside the plan region.</param>
    /// <returns>The host address of the same bytes, or 0 when the span is not in the region.</returns>
    public nint HostAddressOf(DeviceSpan span)
    {
        var (dev, host, bytes) = StepTables;
        if (dev == 0 || span.Address < dev || span.Address >= dev + (nint)bytes)
        {
            return 0;
        }

        return host + (span.Address - dev);
    }

    /// <summary>Finds the variant for an op. The first registered match wins unless <paramref name="preferred"/> names one.</summary>
    /// <param name="op">The op.</param>
    /// <param name="activation">Activation dtype.</param>
    /// <param name="weight">Weight dtype, or <see cref="DType.Undefined"/>.</param>
    /// <param name="kv">KV dtype, or <see cref="DType.Undefined"/>.</param>
    /// <param name="headDim">Head size, or 0.</param>
    /// <param name="pageSize">Page size, or 0.</param>
    /// <param name="group">Query heads per KV head, or 0.</param>
    /// <param name="preferred">A variant id to prefer.</param>
    /// <returns>The kernel.</returns>
    /// <exception cref="ProfileUnsupportedException">No loaded variant admits the requirement.</exception>
    internal RegisteredKernel Kernel(OpId op, DType activation, DType weight = DType.Undefined, DType kv = DType.Undefined, int headDim = 0, int pageSize = 0, int group = 0, string? preferred = null)
    {
        var key = (op, activation, weight, kv, headDim, pageSize, group, preferred);
        if (_kernels.TryGetValue(key, out var found))
        {
            return found;
        }

        var requirement = new KernelRequirement
        {
            Op = op,
            Sm = Sm,
            Activation = activation == DType.Undefined ? null : activation,
            Weight = weight == DType.Undefined ? null : weight,
            Kv = kv == DType.Undefined ? null : kv,
            HeadDim = headDim == 0 ? null : headDim,
            PageSize = pageSize == 0 ? null : pageSize,
            GroupSize = group == 0 ? null : group,
        };
        var kernel = preferred is null
            ? _registry.Require(requirement)
            : _registry.Find(requirement).FirstOrDefault(k => k.VariantId == preferred) ?? _registry.Require(requirement);
        _kernels[key] = kernel;
        return kernel;
    }

    /// <summary>Runs a kernel on the compute stream, preparing its workspace and plan first.</summary>
    /// <typeparam name="T">The op's params struct.</typeparam>
    /// <param name="kernel">The variant.</param>
    /// <param name="parameters">The filled params.</param>
    /// <param name="scope">How often the plan runs.</param>
    /// <param name="shapeKey">The plan key within the variant (ignored unless <paramref name="scope"/> is per-shape or per-step).</param>
    internal void Launch<T>(RegisteredKernel kernel, ref T parameters, PlanScope scope = PlanScope.None, long shapeKey = 0)
        where T : unmanaged
    {
        _launches[kernel.Op] = _launches.GetValueOrDefault(kernel.Op) + 1;
        fixed (T* p = &parameters)
        {
            if (kernel.Workspace == 0)
            {
                Check(kernel, "run", (int)kernel.RunKernel(_context, p, null, Stream));
                return;
            }

            var entry = Entry(kernel, p, scope == PlanScope.None ? 0 : shapeKey);
            if ((kernel.Flags & VariantTraits.NeedsPlan) != 0)
            {
                long wanted = scope == PlanScope.PerStep ? _epoch : 1;
                if (entry.PlannedFor != wanted)
                {
                    if (kernel.Plan == 0)
                    {
                        throw new KernelRunException(kernel.Op, kernel.VariantId, KernelStatus.NotPlanned, "plan", "the variant needs a plan but registers none");
                    }

                    fixed (Workspace* w = &entry.Value)
                    {
                        Check(kernel, "plan", ((delegate* unmanaged<nint, void*, Workspace*, nint, int>)kernel.Plan)(_context.Handle, p, w, Stream));
                    }

                    entry.PlannedFor = wanted;
                }
            }

            fixed (Workspace* w = &entry.Value)
            {
                Check(kernel, "run", (int)kernel.RunKernel(_context, p, w, Stream));
            }
        }
    }

    /// <summary>Frees the workspaces.</summary>
    public void Dispose()
    {
        _ = CudaRuntime.cudaStreamSynchronize(Stream);
        foreach (var e in _workspaces.Values)
        {
            e.Dispose();
        }

        _workspaces.Clear();
    }

    private WorkspaceEntry Entry(RegisteredKernel kernel, void* parameters, long key)
    {
        var sizes = new WorkspaceSizes { StructSize = (uint)sizeof(WorkspaceSizes), AbiMinor = (uint)KernelAbi.Minor };
        Check(kernel, "workspace", ((delegate* unmanaged<nint, void*, WorkspaceSizes*, int>)kernel.Workspace)(_context.Handle, parameters, &sizes));
        if (!_workspaces.TryGetValue((kernel.VariantId, key), out var entry))
        {
            entry = new WorkspaceEntry();
            _workspaces[(kernel.VariantId, key)] = entry;
        }

        if (entry.Fits(sizes))
        {
            return entry;
        }

        // In-flight kernels may still read the old buffers.
        CudaCheck.Runtime(CudaRuntime.cudaStreamSynchronize(Stream), "cudaStreamSynchronize");
        entry.Reset(_allocator, sizes);
        return entry;
    }

    private static void Check(RegisteredKernel kernel, string stage, int status)
    {
        if (status != (int)KernelStatus.Ok)
        {
            throw new KernelRunException(kernel.Op, kernel.VariantId, (KernelStatus)status, stage, kernel.Library.LastErrorText());
        }
    }

    private sealed class Scope(CudaOpsSession? previous) : IDisposable
    {
        public void Dispose() => s_current = previous;
    }

    private sealed class WorkspaceEntry : IDisposable
    {
        private DeviceBuffer? _device;
        private PinnedBuffer? _pinned;
        private nint _plan;

        public Workspace Value;
        public long PlannedFor;

        public bool Fits(in WorkspaceSizes s) =>
            Value.StructSize != 0
            && Value.DeviceBytes >= s.DeviceBytes
            && Value.PinnedHostBytes >= s.PinnedHostBytes
            && Value.PlanBytes >= s.PlanBytes;

        public void Reset(DeviceAllocator allocator, in WorkspaceSizes s)
        {
            Dispose();
            _device = s.DeviceBytes > 0 ? allocator.Allocate((long)s.DeviceBytes + Math.Max(s.DeviceAlignment, 256), "kernel workspace") : null;
            _pinned = s.PinnedHostBytes > 0 ? allocator.AllocatePinned((long)s.PinnedHostBytes, "kernel pinned workspace") : null;
            if (s.PlanBytes > 0)
            {
                _plan = (nint)NativeMemory.AlignedAlloc((nuint)s.PlanBytes, 64);
                NativeMemory.Clear((void*)_plan, (nuint)s.PlanBytes);
            }

            nint device = _device?.Address ?? 0;
            if (s.DeviceAlignment > 1 && device != 0)
            {
                device = (device + (nint)s.DeviceAlignment - 1) / (nint)s.DeviceAlignment * (nint)s.DeviceAlignment;
            }

            Value = new Workspace
            {
                StructSize = (uint)sizeof(Workspace),
                AbiMinor = (uint)KernelAbi.Minor,
                Device = device,
                DeviceBytes = s.DeviceBytes,
                PinnedHost = _pinned?.Address ?? 0,
                PinnedHostBytes = s.PinnedHostBytes,
                Plan = _plan,
                PlanBytes = s.PlanBytes,
            };
            PlannedFor = 0;
        }

        public void Dispose()
        {
            _device?.Dispose();
            _device = null;
            _pinned?.Dispose();
            _pinned = null;
            if (_plan != 0)
            {
                NativeMemory.AlignedFree((void*)_plan);
                _plan = 0;
            }

            Value = default;
        }
    }
}

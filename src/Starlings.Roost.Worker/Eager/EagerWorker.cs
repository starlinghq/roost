using System.Diagnostics;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Kernels;
using Starlings.Roost.Ops;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.Worker.Weights;

namespace Starlings.Roost.Worker.Eager;

/// <summary>Capacities and locations for an <see cref="EagerWorker"/>.</summary>
public sealed record EagerWorkerOptions
{
    /// <summary>CUDA ordinal after <c>CUDA_VISIBLE_DEVICES</c>.</summary>
    public int Device { get; init; }

    /// <summary>Largest token count of a step.</summary>
    public int MaxTokens { get; init; } = 128;

    /// <summary>Largest sequence count of a step.</summary>
    public int MaxSequences { get; init; } = 1;

    /// <summary>Largest total of attention block ids in a step.</summary>
    public int MaxKvBlocks { get; init; } = 32;

    /// <summary>Attention pool capacity in blocks.</summary>
    public int AttentionBlocks { get; init; } = 32;

    /// <summary>Recurrent pool capacity in slots.</summary>
    public int RecurrentSlots { get; init; } = 1;

    /// <summary>Directory holding <c>libroost_kernels_sm&lt;N&gt;.so</c>.</summary>
    public required string KernelDirectory { get; init; }

    /// <summary>Whether an SM whose status is <c>untested</c> may load (the run is then evidence for that SM, not a claim).</summary>
    public bool AllowUntested { get; init; }

    /// <summary>The most device bytes the worker may allocate; 0 means the device's capacity.</summary>
    public long BudgetBytes { get; init; }

    /// <summary>Copy each step's selected logits rows to <see cref="EagerStepOutput.Logits"/> (they are copied to the host for the argmax anyway).</summary>
    public bool KeepLogits { get; init; }
}

/// <summary>What one step produced.</summary>
/// <param name="Tokens">The greedy token of each sampling sequence, in plan order.</param>
/// <param name="Logits">The selected rows of float32 logits when <see cref="EagerWorkerOptions.KeepLogits"/> is set, else null.</param>
/// <param name="Timings">Wall-clock phases of the step.</param>
public sealed record EagerStepOutput(int[] Tokens, float[]? Logits, DeviceTimings Timings);

/// <summary>
/// The eager worker (R-025): one device, two streams, a device allocator, the plan copy, the model's generated forward
/// on <see cref="CudaOps"/>, greedy argmax and the C2 result. A step is synchronous: it returns when the tokens are on
/// the host. Async scheduling (R-032) and graphs (R-033) build on this and change no result.
/// </summary>
/// <remarks>
/// A step runs, in order: the plan is checked on the host (<see cref="StepPlanReader"/>); the model's plan inputs are
/// written into a pinned mirror and copied to the device on the copy stream; the compute stream waits for the copy,
/// clears the recurrent state of every sequence that starts at position 0, and runs the forward; the copy stream waits
/// for the forward and brings the selected logits rows back; the argmax runs on the host over that copy (no argmax kernel
/// is registered yet; the fused sampler is R-042). Call it from one thread.
/// </remarks>
public sealed unsafe class EagerWorker : IDisposable
{
    private readonly IEagerModel _model;
    private readonly EagerWorkerOptions _options;
    private readonly KernelLibrary _library;
    private readonly KernelContext _context;
    private readonly DeviceAllocator _allocator;
    private readonly WorkerStreams _streams;
    private readonly CudaOpsSession _session;
    private readonly EagerLayout _layout;
    private readonly DeviceBuffer _arena;
    private readonly DeviceBuffer _planRegion;
    private readonly PinnedBuffer _planHost;
    private readonly DeviceBuffer _stateBuffer;
    private readonly DeviceSpan[] _state;
    private readonly PinnedBuffer _logitsHost;
    private DeviceSpan[] _weights = [];
    private LoadedWeights? _loaded;
    private bool _disposed;

    /// <summary>Opens the device, loads the kernel library for its SM and allocates arena, state and plan regions.</summary>
    /// <param name="model">The model.</param>
    /// <param name="options">Capacities and locations.</param>
    /// <param name="manifest">The kernel manifest (C5), already checked against its pin.</param>
    public EagerWorker(IEagerModel model, EagerWorkerOptions options, KernelManifest manifest)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(manifest);
        _model = model;
        _options = options;
        CudaLibraryResolver.Install(manifest.CudaMajor);
        Sm = CudaCheck.ComputeCapability(options.Device);
        string file = KernelLibraryNames.Find(Sm, [options.KernelDirectory], KernelLibraryNames.Current())
            ?? throw new ProfileUnsupportedException(OpId.None, Sm, $"no kernel library for SM {Sm} in {options.KernelDirectory}");
        _library = KernelLibrary.Load(file, new KernelLoadOptions(manifest, Sm, options.AllowUntested));
        _allocator = new DeviceAllocator(options.Device, options.BudgetBytes);
        _allocator.Select();
        _streams = new WorkerStreams(options.Device);
        _context = _library.CreateContext(options.Device, ContextOptions.Deterministic, _streams.Compute);
        var registry = new KernelRegistry();
        registry.Add(_library);
        _session = new CudaOpsSession(registry, _context, Sm, _allocator, _streams.Compute);
        _layout = model.Layout(options.MaxTokens, options.MaxSequences, options.MaxKvBlocks);
        Limits = new PlanLimits(
            options.MaxTokens,
            options.MaxSequences,
            options.MaxKvBlocks,
            Math.Min(options.MaxSequences, _layout.MaxSelected),
            model.VocabSize,
            model.BlockSize,
            options.AttentionBlocks,
            model.StateSlots.Any(static s => s.Recurrent) ? options.RecurrentSlots : 0,
            model.AttentionGroup,
            model.GroupCount);

        // Fail at start-up, naming the op and the SM, not at the first step (R-023).
        registry.RequireAll(model.KernelRequirements(Sm));

        _arena = _allocator.Allocate(_layout.ArenaBytes, "activation arena");
        _planRegion = _allocator.Allocate(_layout.PlanBytes, "plan region");
        _planHost = _allocator.AllocatePinned(_layout.PlanBytes, "plan mirror");
        _logitsHost = _allocator.AllocatePinned((long)_layout.MaxSelected * model.VocabSize * sizeof(float), "logits download");
        long total = 0;
        var offsets = new long[model.StateSlots.Count];
        for (int i = 0; i < offsets.Length; i++)
        {
            var slot = model.StateSlots[i];
            offsets[i] = total;
            long count = slot.Recurrent ? options.RecurrentSlots : options.AttentionBlocks;
            total += (((count * slot.ElementsPerBlock * ElementBytes(slot.Dtype)) + 255) / 256) * 256;
        }

        _stateBuffer = _allocator.Allocate(total, "state pools", zero: true);
        _state = new DeviceSpan[offsets.Length];
        for (int i = 0; i < offsets.Length; i++)
        {
            var slot = model.StateSlots[i];
            long count = slot.Recurrent ? options.RecurrentSlots : options.AttentionBlocks;
            _state[i] = _stateBuffer.Span(slot.Dtype, count * slot.ElementsPerBlock, offsets[i]);
        }
    }

    /// <summary>The device's SM.</summary>
    public int Sm { get; }

    /// <summary>The output of the most recent step (diagnostics: the executor path returns only the C2 result).</summary>
    public EagerStepOutput? LastOutput { get; private set; }

    /// <summary>What a plan is checked against.</summary>
    public PlanLimits Limits { get; }

    /// <summary>The device allocator, for the allocation report.</summary>
    public DeviceAllocator Allocator => _allocator;

    /// <summary>The kernel session (variants chosen, launches per op).</summary>
    public CudaOpsSession Session => _session;

    /// <summary>The loaded kernel library's name and registration, for evidence.</summary>
    public KernelLibrary Library => _library;

    /// <summary>The verified weights, after <see cref="LoadWeights"/>.</summary>
    public LoadedWeights? Weights => _loaded;

    /// <summary>The device state pools, in the model's state-slot order (for diagnostics and tests).</summary>
    public IReadOnlyList<DeviceSpan> StatePools => _state;

    /// <summary>Streams the locked checkpoint to the device (see <see cref="ModelWeightLoader"/>).</summary>
    /// <param name="path">The safetensors file.</param>
    /// <param name="locked">Its lock entry.</param>
    /// <exception cref="WeightLoadException">The file does not match the lock or the model.</exception>
    public void LoadWeights(string path, LockedFile locked)
    {
        _loaded?.Dispose();
        _loaded = ModelWeightLoader.Load(path, locked, _model.WeightSlots, _allocator);
        _weights = _loaded.Spans;
        _model.CheckBackend<CudaOps>();
    }

    /// <summary>Runs one step on <see cref="CudaOps"/>.</summary>
    /// <param name="plan">A checked plan.</param>
    /// <returns>The tokens (and logits when kept).</returns>
    public EagerStepOutput Step(ParsedPlan plan) => Step<CudaOps>(plan);

    /// <summary>Runs one step with another backend wrapped around the kernels (a tracing backend over <see cref="CudaOps"/>, for example).</summary>
    /// <typeparam name="TOps">A backend whose ops end in <see cref="CudaOps"/>.</typeparam>
    /// <param name="plan">A checked plan.</param>
    /// <returns>The tokens (and logits when kept).</returns>
    public EagerStepOutput Step<TOps>(ParsedPlan plan)
        where TOps : struct, IRoostOps
    {
        ArgumentNullException.ThrowIfNull(plan);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_loaded is null)
        {
            throw new InvalidOperationException("the weights are not loaded");
        }

        _allocator.Select();
        using var bound = _session.Bind();
        var clock = Stopwatch.StartNew();

        // Plan copy: host mirror, then the copy stream.
        var host = new Span<byte>((void*)_planHost.Address, checked((int)_layout.PlanBytes));
        PlanRegionWriter.Fill(host, _layout, plan, _model.BlockSize);
        CudaCheck.Runtime(CudaRuntime.cudaMemcpyAsync((void*)_planRegion.Address, (void*)_planHost.Address, (nuint)_layout.PlanBytes, CudaRuntime.cudaMemcpyHostToDevice, _streams.Copy), "cudaMemcpyAsync(plan)");
        _streams.ComputeWaitsForCopy();
        ulong uploadNs = Ns(clock);

        // A sequence that starts at position 0 starts from an empty recurrent state.
        for (int b = 0; b < plan.Sequences.Length; b++)
        {
            var s = plan.Sequences[b];
            if (s.PositionOffset == 0 && s.RecurrentSlot >= 0)
            {
                for (int i = 0; i < _state.Length; i++)
                {
                    var slot = _model.StateSlots[i];
                    if (slot.Recurrent)
                    {
                        long bytes = slot.ElementsPerBlock * ElementBytes(slot.Dtype);
                        CudaCheck.Runtime(CudaRuntime.cudaMemsetAsync((void*)(_state[i].Address + (nint)(s.RecurrentSlot * bytes)), 0, (nuint)bytes, _streams.Compute), "cudaMemsetAsync(state)");
                    }
                }
            }
        }

        int selected = plan.Selected;
        var input = new EagerForward(
            _arena.Span(DType.U8, _layout.ArenaBytes),
            _planRegion.Span(DType.U8, _layout.PlanBytes),
            _weights,
            _state,
            _options.MaxTokens,
            _options.MaxSequences,
            _options.MaxKvBlocks,
            plan.TokenIds.Length,
            selected,
            plan.Sequences.Length,
            plan.KvBlocks);
        _session.SetStepTables(_planRegion.Address, _planHost.Address, _layout.PlanBytes);
        clock.Restart();
        _model.Forward<TOps>(in input);

        // Logits back on the copy stream, after the forward.
        var logits = _model.Logits(in input);
        long logitBytes = (long)selected * _model.VocabSize * sizeof(float);
        _streams.CopyWaitsForCompute();
        CudaCheck.Runtime(CudaRuntime.cudaMemcpyAsync((void*)_logitsHost.Address, (void*)logits.Address, (nuint)logitBytes, CudaRuntime.cudaMemcpyDeviceToHost, _streams.Copy), "cudaMemcpyAsync(logits)");
        _streams.Synchronize();
        ulong forwardNs = Ns(clock);

        // Greedy argmax on the host over the downloaded rows.
        clock.Restart();
        var tokens = new int[selected];
        fixed (int* pinnedIds = tokens)
        {
            CpuOps.Argmax(new ArgmaxArgs
            {
                Logits = new DeviceSpan { Address = _logitsHost.Address, Length = (ulong)selected * (ulong)_model.VocabSize, Dtype = DType.F32 },
                Ids = new DeviceSpan { Address = (nint)pinnedIds, Length = (ulong)selected, Dtype = DType.I32 },
                Rows = selected,
                Vocab = _model.VocabSize,
                Stride = _model.VocabSize,
            });
        }

        float[]? kept = null;
        if (_options.KeepLogits)
        {
            kept = new float[(long)selected * _model.VocabSize];
            new ReadOnlySpan<float>((void*)_logitsHost.Address, kept.Length).CopyTo(kept);
        }

        ulong sampleNs = Ns(clock);
        var timings = new DeviceTimings { UploadNs = uploadNs, ForwardNs = forwardNs, SampleNs = sampleNs, StepNs = uploadNs + forwardNs + sampleNs };
        LastOutput = new EagerStepOutput(tokens, kept, timings);
        return LastOutput;
    }

    /// <summary>Frees everything on the device.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _ = CudaRuntime.cudaSetDevice(_options.Device);
        _streams.Synchronize();
        _session.Dispose();
        _context.Dispose();
        _library.Dispose();
        _loaded?.Dispose();
        _logitsHost.Dispose();
        _planHost.Dispose();
        _stateBuffer.Dispose();
        _planRegion.Dispose();
        _arena.Dispose();
        _streams.Dispose();
        _allocator.Dispose();
    }

    private static ulong Ns(Stopwatch clock) => (ulong)(clock.ElapsedTicks * 1_000_000_000L / Stopwatch.Frequency);

    private static long ElementBytes(DType dtype) => dtype switch
    {
        DType.F32 or DType.I32 => 4,
        DType.BF16 or DType.F16 => 2,
        DType.I64 => 8,
        _ => 1,
    };
}

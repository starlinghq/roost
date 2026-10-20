using System.Collections.Immutable;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Requests;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Admission;

/// <summary>
/// Admission control (R-031, stream G): decides before any device work whether a request fits, by reserving its worst
/// case in every state group, and refuses with a typed <see cref="RoostError"/> when it does not.
/// </summary>
/// <remarks>
/// <para><b>What is reserved.</b> For each request, per group, <see cref="WorstCaseReservation"/>: blocks for the prompt
/// plus <c>max_tokens</c> (the context length when unset), a recurrent slot per row, and for a request with a LoRA
/// handle the adapter's blocks in the <see cref="StateGroupKind.Adapter"/> group, once per resident adapter however many
/// requests use it. The total over all admitted requests never exceeds a pool's <see cref="BlockPool.Capacity"/>. The
/// workspace is not a pool: <see cref="TryComputePoolBudget"/> takes it off the byte budget the pools are sized from,
/// and <see cref="AdmissionOptions.MinFreeDeviceBytes"/> keeps a floor of free device memory at admission.</para>
/// <para><b>Forked branches count as sequences.</b> A request with <c>n</c>, <c>best_of</c> or a beam width above 1
/// reserves that many rows when it is admitted: its prompt blocks once (branches share them copy-on-write) and, per row,
/// the blocks it can write and its recurrent slot. <see cref="TryAdmitFork"/> claims one of those rows; a fork beyond
/// them reserves one more row, or is refused. A finished row returns its own share at once and the shared prompt blocks
/// when the last row is gone.</para>
/// <para><b>Order of refusals.</b> <see cref="RoostErrorCode.InvalidOption"/>, <see cref="RoostErrorCode.PromptTooLong"/>,
/// <see cref="RoostErrorCode.ProfileUnsupported"/> (an unknown adapter), <see cref="RoostErrorCode.QueueFull"/> (live rows at
/// <see cref="AdmissionOptions.MaxLiveSequences"/>), then <see cref="RoostErrorCode.InsufficientDeviceMemory"/> (the
/// reservation does not fit beside the others, or alone, or the device is below its free floor). Memory refusals are
/// the only ones that clear by themselves when other requests finish; a worst case larger than a pool says so in its message.</para>
/// <para><b>No device work.</b> The controller reads only its own counters, the pools' capacities and the injected
/// <see cref="IDeviceMemorySource"/>; it never allocates from a pool. A refusal changes nothing. When the source cannot
/// be read, the pools alone decide. Engine thread only, like the pools.</para>
/// </remarks>
public sealed class AdmissionController
{
    private readonly ImmutableArray<StateGroupDescriptor> _groups;
    private readonly long[] _capacity;
    private readonly long[] _reserved;
    private readonly AdmissionOptions _options;
    private readonly IDeviceMemorySource? _memory;
    private readonly IAdapterSizeSource? _adapters;
    private readonly int _adapterGroup = -1;
    private readonly Dictionary<ulong, Root> _roots = [];
    private readonly Dictionary<ulong, Root> _rowRoot = [];
    private readonly Dictionary<ulong, AdapterEntry> _residentAdapters = [];
    private int _liveRows;

    /// <summary>Creates a controller over a block manager's pools.</summary>
    /// <param name="blocks">The pools, read for their capacities only.</param>
    /// <param name="options">The limits.</param>
    /// <param name="memory">The device memory source, or null to decide from the pools alone.</param>
    /// <param name="adapters">The adapter sizes, needed only when the manager has an adapter group.</param>
    /// <exception cref="NotSupportedException">A group kind is not supported.</exception>
    public AdmissionController(BlockManager blocks, AdmissionOptions options, IDeviceMemorySource? memory = null, IAdapterSizeSource? adapters = null)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        _options = options;
        _memory = memory;
        _adapters = adapters;
        _groups = blocks.Groups;
        _capacity = new long[_groups.Length];
        _reserved = new long[_groups.Length];
        foreach (var g in _groups)
        {
            _capacity[g.Index] = blocks.GetPool(g.Index).Capacity;
            switch (g.Kind)
            {
                case StateGroupKind.Full or StateGroupKind.Sliding or StateGroupKind.Recurrent or StateGroupKind.MlaLatent:
                    break;
                case StateGroupKind.Adapter:
                    if (_adapterGroup >= 0)
                    {
                        throw new NotSupportedException("Admission supports one adapter group.");
                    }

                    _adapterGroup = g.Index;
                    break;
                default:
                    throw new NotSupportedException($"State group kind {g.Kind} (group {g.Name}) is not supported by admission control.");
            }
        }
    }

    /// <summary>Live rows (sequences, forked branches included) that hold a reservation.</summary>
    public int LiveRows => _liveRows;

    /// <summary>Rows reserved, claimed or not, over all admitted requests.</summary>
    public int ReservedRows
    {
        get
        {
            var rows = 0;
            foreach (var r in _roots.Values)
            {
                rows += r.Rows;
            }

            return rows;
        }
    }

    /// <summary>Blocks (slots) reserved in a group, adapters included.</summary>
    /// <param name="group">Group index.</param>
    /// <returns>The reserved count.</returns>
    public long Reserved(int group) => _reserved[group];

    /// <summary>Blocks (slots) of a group no reservation holds.</summary>
    /// <param name="group">Group index.</param>
    /// <returns>Capacity minus reserved.</returns>
    public long Unreserved(int group) => _capacity[group] - _reserved[group];

    /// <summary>
    /// Computes the byte budget the pools are sized from (the <c>budgetBytes</c> of <see cref="BlockSizing.Compute"/>):
    /// the free device memory, read after the weights are resident, less the workspace and the headroom.
    /// </summary>
    /// <param name="memory">The device memory source.</param>
    /// <param name="options">The limits.</param>
    /// <param name="budgetBytes">The budget when this returns true.</param>
    /// <param name="error">The refusal when this returns false.</param>
    /// <returns>False with <see cref="RoostErrorCode.InsufficientDeviceMemory"/> when the device cannot be read or nothing is left.</returns>
    public static bool TryComputePoolBudget(IDeviceMemorySource memory, AdmissionOptions options, out long budgetBytes, out RoostError? error)
    {
        ArgumentNullException.ThrowIfNull(memory);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        budgetBytes = 0;
        if (!memory.TryRead(out var m))
        {
            error = new RoostError(RoostErrorCode.InsufficientDeviceMemory, "Device memory cannot be read, so the pools cannot be sized.");
            return false;
        }

        var budget = m.FreeBytes - options.WorkspaceBytes - options.HeadroomBytes - options.MinFreeDeviceBytes;
        if (budget <= 0)
        {
            error = new RoostError(
                RoostErrorCode.InsufficientDeviceMemory,
                $"{m.FreeBytes} bytes of device memory are free; the workspace ({options.WorkspaceBytes}), headroom ({options.HeadroomBytes}) and free floor ({options.MinFreeDeviceBytes}) leave nothing for the cache.");
            return false;
        }

        budgetBytes = budget;
        error = null;
        return true;
    }

    /// <summary>Admits a request, reserving its worst case, or refuses it.</summary>
    /// <param name="request">The request. Its id value is the handle.</param>
    /// <param name="error">The refusal when this returns false.</param>
    /// <returns>True when the request is admitted and holds a reservation until <see cref="Release"/>.</returns>
    public bool TryAdmit(Request request, out RoostError? error)
    {
        ArgumentNullException.ThrowIfNull(request);
        var handle = request.Id.Value;
        if (request.PromptTokenIds.IsDefaultOrEmpty)
        {
            return Refuse(RoostErrorCode.InvalidOption, "The prompt is empty.", "prompt", out error);
        }

        var sampling = request.Sampling;
        if (sampling.MaxTokens <= 0)
        {
            return Refuse(RoostErrorCode.InvalidOption, "max_tokens must be positive.", "max_tokens", out error);
        }

        if (sampling.N < 1 || sampling.BeamWidth < 1 || sampling.BestOf < 1)
        {
            return Refuse(RoostErrorCode.InvalidOption, "n, best_of and the beam width must be at least 1.", "n", out error);
        }

        if (_roots.ContainsKey(handle) || _rowRoot.ContainsKey(handle))
        {
            return Refuse(RoostErrorCode.InvalidOption, "A request with this id is already live.", "id", out error);
        }

        var prompt = request.PromptTokenIds.Length;
        if (prompt >= _options.ContextLength)
        {
            return Refuse(RoostErrorCode.PromptTooLong, $"The prompt has {prompt} tokens; the context length is {_options.ContextLength}.", "prompt", out error);
        }

        var perGroup = WorstCaseReservation.Compute(_groups, prompt, sampling.MaxTokens, _options.ContextLength);
        var rows = WorstCaseReservation.Rows(sampling);
        var single = WorstCaseReservation.Compute(_groups, prompt, 0, _options.ContextLength);
        for (var i = 0; i < _groups.Length; i++)
        {
            // The prompt alone, with one row, must fit a pool at all.
            if (single[i].For(1) > _capacity[i])
            {
                return Refuse(RoostErrorCode.PromptTooLong, $"The prompt needs {single[i].For(1)} blocks of group {_groups[i].Name}, which holds {_capacity[i]}.", "prompt", out error);
            }
        }

        long adapterBlocks = 0;
        var lora = request.Lora;
        if (lora.Value != 0)
        {
            if (_adapterGroup < 0 || _adapters is null || !_adapters.TryGetBytes(lora, out var bytes))
            {
                return Refuse(RoostErrorCode.ProfileUnsupported, "The adapter is unknown, or the profile has no adapter group.", "lora", out error);
            }

            var perBlock = _groups[_adapterGroup].BytesPerBlock;
            adapterBlocks = (bytes + perBlock - 1) / perBlock;
        }

        if (_options.MaxLiveSequences > 0 && ReservedRows + rows > _options.MaxLiveSequences)
        {
            return Refuse(RoostErrorCode.QueueFull, $"{ReservedRows} rows are live and the request needs {rows}; the limit is {_options.MaxLiveSequences}.", null, out error);
        }

        var newAdapter = lora.Value != 0 && !_residentAdapters.ContainsKey(lora.Value);
        for (var i = 0; i < _groups.Length; i++)
        {
            var need = perGroup[i].For(rows) + (i == _adapterGroup && newAdapter ? adapterBlocks : 0);
            if (need > _capacity[i])
            {
                return Refuse(
                    RoostErrorCode.InsufficientDeviceMemory,
                    $"The worst case needs {need} blocks of group {_groups[i].Name}, more than the {_capacity[i]} it holds; lower max_tokens or n. Waiting will not help.",
                    "max_tokens",
                    out error);
            }

            if (need > _capacity[i] - _reserved[i])
            {
                return Refuse(
                    RoostErrorCode.InsufficientDeviceMemory,
                    $"The worst case needs {need} blocks of group {_groups[i].Name}; {_capacity[i] - _reserved[i]} are unreserved. Retry when other requests finish.",
                    null,
                    out error);
            }
        }

        if (!FloorHolds(out error))
        {
            return false;
        }

        var root = new Root(handle, perGroup, rows, lora.Value) { Claimed = 1 };
        for (var i = 0; i < _groups.Length; i++)
        {
            _reserved[i] += perGroup[i].For(rows);
        }

        if (lora.Value != 0)
        {
            AcquireAdapter(lora.Value, adapterBlocks);
        }

        _roots.Add(handle, root);
        _rowRoot.Add(handle, root);
        _liveRows++;
        error = null;
        return true;
    }

    /// <summary>
    /// Admits a forked branch row of a live request. It claims a row that <see cref="TryAdmit"/> reserved for
    /// <c>n</c>, <c>best_of</c> or the beam width, or reserves one more row when those are used up.
    /// </summary>
    /// <param name="parent">The handle of a live row of the request the branch belongs to.</param>
    /// <param name="branch">The handle of the branch.</param>
    /// <param name="error">The refusal when this returns false.</param>
    /// <returns>True when the branch holds a row until <see cref="Release"/>.</returns>
    public bool TryAdmitFork(ulong parent, ulong branch, out RoostError? error)
    {
        if (!_rowRoot.TryGetValue(parent, out var root))
        {
            return Refuse(RoostErrorCode.InvalidOption, "The parent request is not live.", "parent", out error);
        }

        if (_rowRoot.ContainsKey(branch) || _roots.ContainsKey(branch))
        {
            return Refuse(RoostErrorCode.InvalidOption, "A request with this id is already live.", "id", out error);
        }

        if (root.Claimed >= root.Rows)
        {
            if (_options.MaxLiveSequences > 0 && ReservedRows + 1 > _options.MaxLiveSequences)
            {
                return Refuse(RoostErrorCode.QueueFull, $"{ReservedRows} rows are live; the limit is {_options.MaxLiveSequences}.", null, out error);
            }

            for (var i = 0; i < _groups.Length; i++)
            {
                if (root.Groups[i].PerRow > _capacity[i] - _reserved[i])
                {
                    return Refuse(RoostErrorCode.InsufficientDeviceMemory, $"A branch row needs {root.Groups[i].PerRow} blocks of group {_groups[i].Name}; {_capacity[i] - _reserved[i]} are unreserved.", null, out error);
                }
            }

            if (!FloorHolds(out error))
            {
                return false;
            }

            for (var i = 0; i < _groups.Length; i++)
            {
                _reserved[i] += root.Groups[i].PerRow;
            }

            root.Rows++;
        }

        root.Claimed++;
        _rowRoot.Add(branch, root);
        _liveRows++;
        error = null;
        return true;
    }

    /// <summary>Returns the reservation of a finished, cancelled or aborted row. Unknown handles are ignored.</summary>
    /// <param name="handle">The handle of a row.</param>
    public void Release(ulong handle)
    {
        if (!_rowRoot.Remove(handle, out var root))
        {
            return;
        }

        _liveRows--;
        root.Claimed--;
        root.Rows--;
        for (var i = 0; i < _groups.Length; i++)
        {
            _reserved[i] -= root.Groups[i].PerRow;
        }

        if (root.Claimed > 0)
        {
            return;
        }

        // The last live row: the shared prompt blocks and the rows nobody forked go back too.
        for (var i = 0; i < _groups.Length; i++)
        {
            _reserved[i] -= root.Groups[i].Shared + (root.Rows * root.Groups[i].PerRow);
        }

        root.Rows = 0;
        _roots.Remove(root.Handle);
        if (root.Adapter != 0)
        {
            ReleaseAdapter(root.Adapter);
        }
    }

    private bool FloorHolds(out RoostError? error)
    {
        if (_memory is not null && _options.MinFreeDeviceBytes > 0 && _memory.TryRead(out var m) && m.FreeBytes < _options.MinFreeDeviceBytes)
        {
            return Refuse(
                RoostErrorCode.InsufficientDeviceMemory,
                $"{m.FreeBytes} bytes of device memory are free; at least {_options.MinFreeDeviceBytes} are needed to admit a request.",
                null,
                out error);
        }

        error = null;
        return true;
    }

    private void AcquireAdapter(ulong adapter, long blocks)
    {
        if (_residentAdapters.TryGetValue(adapter, out var entry))
        {
            entry.Users++;
            return;
        }

        _residentAdapters.Add(adapter, new AdapterEntry(blocks) { Users = 1 });
        _reserved[_adapterGroup] += blocks;
    }

    private void ReleaseAdapter(ulong adapter)
    {
        var entry = _residentAdapters[adapter];
        if (--entry.Users == 0)
        {
            _residentAdapters.Remove(adapter);
            _reserved[_adapterGroup] -= entry.Blocks;
        }
    }

    private static bool Refuse(RoostErrorCode code, string message, string? param, out RoostError? error)
    {
        error = new RoostError(code, message, param);
        return false;
    }

    private sealed class Root(ulong handle, ImmutableArray<GroupReservation> groups, int rows, ulong adapter)
    {
        public ulong Handle { get; } = handle;

        public ImmutableArray<GroupReservation> Groups { get; } = groups;

        public int Rows { get; set; } = rows;

        public int Claimed { get; set; }

        public ulong Adapter { get; } = adapter;
    }

    private sealed class AdapterEntry(long blocks)
    {
        public long Blocks { get; } = blocks;

        public int Users { get; set; }
    }
}

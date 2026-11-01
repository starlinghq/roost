using System.Collections.Immutable;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Ops;
using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.EngineGates.Tests.Support;

/// <summary>The two CI models the gates run (R-021).</summary>
public enum TinyProfile
{
    /// <summary><c>roost-tiny</c>: Qwen3-shaped, float32, one full-attention group.</summary>
    Dense,

    /// <summary><c>roost-tiny-hybrid</c>: Qwen3.5-shaped, bfloat16, paged attention plus a recurrent group with snapshots.</summary>
    Hybrid,
}

/// <summary>One row of a forward: a prefill chunk or a decode token of one sequence.</summary>
/// <param name="TokenStart">First token in the flat token table.</param>
/// <param name="TokenCount">Tokens in the row.</param>
/// <param name="PositionOffset">Position of the first token.</param>
/// <param name="Blocks">The attention block table (at least the pages the row reaches).</param>
/// <param name="RecurrentSlot">Recurrent pool slot, or -1.</param>
/// <param name="Sample">Whether the row's last token selects logits.</param>
internal readonly record struct ForwardRow(int TokenStart, int TokenCount, int PositionOffset, int[] Blocks, int RecurrentSlot, bool Sample);

/// <summary>
/// A tiny model on <see cref="CpuOps"/> as a worker holds it: generated weights, the paged attention pool, the recurrent pool
/// and its snapshot pool in native memory (host addresses stand in for device addresses), one activation arena and one
/// plan-input region. <see cref="Forward"/> runs the generated forward over any mix of rows, and <see cref="Apply"/> runs
/// the C2 cache ops (block copy, recurrent fork, snapshot write and restore) on the pools.
/// </summary>
/// <remarks>
/// A row that starts at position 0 with a recurrent slot starts from a zero state, as the eager worker does
/// (<c>EagerWorker.Step</c>), because the generated delta rule and convolution always read their initial state.
/// </remarks>
internal sealed unsafe class TinyCpuModel : IDisposable
{
    private static readonly Lazy<byte[][]> s_denseWeights = new(() => Encode(TinyConfig.Dense, [.. RoostTiny.WeightSlots.Select(s => (s.Name, s.Shape, s.Dtype, s.Elements))]));
    private static readonly Lazy<byte[][]> s_hybridWeights = new(() => Encode(TinyConfig.Hybrid, [.. RoostTinyHybrid.WeightSlots.Select(s => (s.Name, s.Shape, s.Dtype, s.Elements))]));

    private readonly List<nint> _allocations = [];
    private readonly TinyProfile _profile;
    private readonly DeviceSpan[] _weights;
    private readonly DeviceSpan[] _state;
    private readonly DeviceSpan[] _snapshots;
    private readonly (int Group, long Bytes)[] _slots;
    private readonly DeviceSpan _arena;
    private readonly DeviceSpan _planRegion;
    private readonly RoostTiny.ForwardLayout _dense;
    private readonly RoostTinyHybrid.ForwardLayout _hybrid;

    /// <summary>Allocates the model for a worker of the given capacities.</summary>
    /// <param name="profile">The model.</param>
    /// <param name="attentionBlocks">Blocks in the attention pool.</param>
    /// <param name="recurrentSlots">Slots in the recurrent pool (ignored by the dense model).</param>
    /// <param name="snapshotSlots">Slots in the snapshot pool (ignored by the dense model).</param>
    /// <param name="maxTokens">Most tokens in one forward.</param>
    /// <param name="maxSequences">Most rows in one forward.</param>
    public TinyCpuModel(TinyProfile profile, int attentionBlocks, int recurrentSlots, int snapshotSlots, int maxTokens, int maxSequences)
    {
        _profile = profile;
        byte[][] weights = profile == TinyProfile.Dense ? s_denseWeights.Value : s_hybridWeights.Value;
        DType[] weightTypes = profile == TinyProfile.Dense
            ? [.. RoostTiny.WeightSlots.Select(s => s.Dtype)]
            : [.. RoostTinyHybrid.WeightSlots.Select(s => s.Dtype)];
        _weights = new DeviceSpan[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            _weights[i] = Alloc(weights[i].Length, weightTypes[i], weights[i].Length / Size(weightTypes[i]));
            weights[i].CopyTo(new Span<byte>((void*)_weights[i].Address, weights[i].Length));
        }

        (int Group, DType Dtype, long Elements)[] slots = profile == TinyProfile.Dense
            ? [.. RoostTiny.StateSlots.Select(s => (s.Group, s.Dtype, s.ElementsPerBlock))]
            : [.. RoostTinyHybrid.StateSlots.Select(s => (s.Group, s.Dtype, s.ElementsPerBlock))];
        StateGroups = profile == TinyProfile.Dense ? RoostTiny.CreateStateGroups() : RoostTinyHybrid.CreateStateGroups();
        _slots = new (int, long)[slots.Length];
        _state = new DeviceSpan[slots.Length];
        _snapshots = new DeviceSpan[slots.Length];
        for (int i = 0; i < slots.Length; i++)
        {
            var (group, dtype, elements) = slots[i];
            bool recurrent = StateGroups[group].Kind == StateGroupKind.Recurrent;
            long count = recurrent ? recurrentSlots : attentionBlocks;
            _slots[i] = (group, elements * Size(dtype));
            _state[i] = Alloc(count * elements * Size(dtype), dtype, count * elements);
            if (recurrent && snapshotSlots > 0)
            {
                _snapshots[i] = Alloc(snapshotSlots * elements * Size(dtype), dtype, snapshotSlots * elements);
            }
        }

        // A step may name a shared prefix block once per row, so the block-index input holds a full table per row.
        int maxKvBlocks = attentionBlocks * maxSequences;
        long arenaBytes, planBytes;
        if (profile == TinyProfile.Dense)
        {
            _dense = new RoostTiny.ForwardLayout(maxTokens, maxSequences, maxKvBlocks);
            (arenaBytes, planBytes) = (_dense.ArenaBytes, _dense.PlanBytes);
        }
        else
        {
            _hybrid = new RoostTinyHybrid.ForwardLayout(maxTokens, maxSequences, maxKvBlocks);
            (arenaBytes, planBytes) = (_hybrid.ArenaBytes, _hybrid.PlanBytes);
        }

        _arena = Alloc(arenaBytes, DType.U8, arenaBytes);
        _planRegion = Alloc(planBytes, DType.U8, planBytes);
    }

    /// <summary>The model's state groups (C4), as its generated <c>CreateStateGroups()</c> returns them.</summary>
    public ImmutableArray<StateGroupDescriptor> StateGroups { get; }

    /// <summary>Vocabulary size (both models: 256).</summary>
    public static int Vocab => TinyConfig.Dense.Vocab;

    /// <summary>The model's state groups without allocating a model.</summary>
    /// <param name="profile">The model.</param>
    /// <returns>The descriptors.</returns>
    public static ImmutableArray<StateGroupDescriptor> GroupsOf(TinyProfile profile) =>
        profile == TinyProfile.Dense ? RoostTiny.CreateStateGroups() : RoostTinyHybrid.CreateStateGroups();

    /// <summary>A digest that names the generated weights (seed and slot order), for handshakes and cache salts.</summary>
    /// <param name="profile">The model.</param>
    /// <returns>32 bytes.</returns>
    public static ImmutableArray<byte> WeightsDigest(TinyProfile profile)
    {
        using var sha = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
        foreach (var w in profile == TinyProfile.Dense ? s_denseWeights.Value : s_hybridWeights.Value)
        {
            sha.AppendData(w);
        }

        return [.. sha.GetHashAndReset()];
    }

    /// <summary>Runs one C2 cache op on the pools.</summary>
    /// <param name="op">The op.</param>
    public void Apply(in PlanCacheOp op)
    {
        for (int i = 0; i < _slots.Length; i++)
        {
            var (group, bytes) = _slots[i];
            if (group != (int)op.Group)
            {
                continue;
            }

            byte* live = (byte*)_state[i].Address;
            byte* snap = (byte*)_snapshots[i].Address;
            for (long k = 0; k < op.Count; k++)
            {
                long src = (long)op.Src + k, dst = (long)op.Dst + k;
                switch (op.Kind)
                {
                    case CacheOpKind.Copy:
                    case CacheOpKind.Fork:
                        Buffer.MemoryCopy(live + (src * bytes), live + (dst * bytes), bytes, bytes);
                        break;
                    case CacheOpKind.SnapshotWrite:
                        Buffer.MemoryCopy(live + (src * bytes), snap + (dst * bytes), bytes, bytes);
                        break;
                    case CacheOpKind.SnapshotRestore:
                        Buffer.MemoryCopy(snap + (src * bytes), live + (dst * bytes), bytes, bytes);
                        break;
                    default:
                        throw new InvalidOperationException($"The CPU gate model does not run {op.Kind} ops.");
                }
            }
        }
    }

    /// <summary>One generated forward; returns the float32 logits of each sampling row, in row order.</summary>
    /// <param name="tokenIds">The flat token table.</param>
    /// <param name="rows">The rows.</param>
    /// <returns>One logits row (vocabulary 256) per sampling row.</returns>
    public float[][] Forward(ReadOnlySpan<int> tokenIds, IReadOnlyList<ForwardRow> rows)
    {
        foreach (var row in rows)
        {
            if (row.PositionOffset == 0 && row.RecurrentSlot >= 0)
            {
                ZeroRecurrent(row.RecurrentSlot);
            }
        }

        int bs = StateGroups.First(g => g.Kind != StateGroupKind.Recurrent).BlockSize;
        int total = tokenIds.Length;
        int selected = rows.Count(r => r.Sample);
        int pages = rows.Sum(r => (r.PositionOffset + r.TokenCount + bs - 1) / bs);
        DeviceSpan logits;
        if (_profile == TinyProfile.Dense)
        {
            var plan = RoostTiny.PlanInputs.Bind(_planRegion, _dense, total, selected, rows.Count, pages);
            Fill(tokenIds, rows, bs, plan.Ids, plan.Positions, plan.SlotMapping, plan.QIndptr, plan.KvIndptr, plan.KvIndices, plan.KvLastPageLength, plan.Rows, default, default);
            var ctx = new RoostTiny.StepContext(_dense, _arena, _weights, _state, plan, total, selected, rows.Count, pages);
            RoostTiny.Forward<CpuOps>(in ctx);
            logits = RoostTiny.Output_logits(in ctx);
        }
        else
        {
            var plan = RoostTinyHybrid.PlanInputs.Bind(_planRegion, _hybrid, total, selected, rows.Count, pages);
            Fill(tokenIds, rows, bs, plan.Ids, plan.Positions, plan.SlotMapping, plan.QIndptr, plan.KvIndptr, plan.KvIndices, plan.KvLastPageLength, plan.Rows, plan.StateSlots, plan.CuSeqlens);
            var ctx = new RoostTinyHybrid.StepContext(_hybrid, _arena, _weights, _state, plan, total, selected, rows.Count, pages);
            RoostTinyHybrid.Forward<CpuOps>(in ctx);
            logits = RoostTinyHybrid.Output_logits(in ctx);
        }

        var result = new float[selected][];
        for (int r = 0; r < selected; r++)
        {
            result[r] = new ReadOnlySpan<float>((float*)logits.Address + ((long)r * Vocab), Vocab).ToArray();
        }

        return result;
    }

    /// <summary>Greedy choice over one logits row with <see cref="CpuOps.Argmax"/>, the op the eager worker samples with.</summary>
    /// <param name="logits">One row.</param>
    /// <returns>The token.</returns>
    public static int Argmax(float[] logits)
    {
        int id = 0;
        fixed (float* l = logits)
        {
            CpuOps.Argmax(new ArgmaxArgs
            {
                Logits = new DeviceSpan { Address = (nint)l, Length = (ulong)logits.Length, Dtype = DType.F32 },
                Ids = new DeviceSpan { Address = (nint)(&id), Length = 1, Dtype = DType.I32 },
                Rows = 1,
                Vocab = logits.Length,
                Stride = logits.Length,
            });
        }

        return id;
    }

    public void Dispose()
    {
        foreach (var p in _allocations)
        {
            NativeMemory.AlignedFree((void*)p);
        }

        _allocations.Clear();
    }

    private static void Fill(
        ReadOnlySpan<int> tokenIds,
        IReadOnlyList<ForwardRow> rows,
        int bs,
        DeviceSpan idsSpan,
        DeviceSpan positionsSpan,
        DeviceSpan slotSpan,
        DeviceSpan qIndptrSpan,
        DeviceSpan kvIndptrSpan,
        DeviceSpan kvIndicesSpan,
        DeviceSpan lastSpan,
        DeviceSpan rowsSpan,
        DeviceSpan stateSlotsSpan,
        DeviceSpan cuSeqlensSpan)
    {
        int* ids = (int*)idsSpan.Address;
        int* positions = (int*)positionsSpan.Address;
        long* slots = (long*)slotSpan.Address;
        int* qIndptr = (int*)qIndptrSpan.Address;
        int* kvIndptr = (int*)kvIndptrSpan.Address;
        int* kvIndices = (int*)kvIndicesSpan.Address;
        int* last = (int*)lastSpan.Address;
        int* selectedRows = (int*)rowsSpan.Address;
        int* stateSlots = (int*)stateSlotsSpan.Address;
        int* cuSeqlens = (int*)cuSeqlensSpan.Address;
        int page = 0, selected = 0;
        for (int b = 0; b < rows.Count; b++)
        {
            var row = rows[b];
            qIndptr[b] = row.TokenStart;
            kvIndptr[b] = page;
            if (cuSeqlens != null)
            {
                cuSeqlens[b] = row.TokenStart;
                stateSlots[b] = row.RecurrentSlot;
            }

            for (int j = 0; j < row.TokenCount; j++)
            {
                int t = row.TokenStart + j;
                int p = row.PositionOffset + j;
                ids[t] = tokenIds[t];
                positions[t] = p;
                slots[t] = ((long)row.Blocks[p / bs] * bs) + (p % bs);
            }

            int after = row.PositionOffset + row.TokenCount;
            int count = (after + bs - 1) / bs;
            for (int i = 0; i < count; i++)
            {
                kvIndices[page++] = row.Blocks[i];
            }

            last[b] = ((after - 1) % bs) + 1;
            if (row.Sample)
            {
                selectedRows[selected++] = row.TokenStart + row.TokenCount - 1;
            }
        }

        qIndptr[rows.Count] = tokenIds.Length;
        kvIndptr[rows.Count] = page;
        if (cuSeqlens != null)
        {
            cuSeqlens[rows.Count] = tokenIds.Length;
        }
    }

    private static byte[][] Encode(TinyConfig config, (string Name, string Shape, DType Dtype, long Elements)[] slots)
    {
        var result = new byte[slots.Length][];
        for (int i = 0; i < slots.Length; i++)
        {
            var (name, shape, dtype, elements) = slots[i];
            double[] values = TinyWeights.Generate(config.Seed, name, shape, dtype);
            var bytes = new byte[elements * Size(dtype)];
            fixed (byte* p = bytes)
            {
                TinyWeights.Write(values, dtype, (nint)p);
            }

            result[i] = bytes;
        }

        return result;
    }

    private static int Size(DType t) => t switch
    {
        DType.F64 or DType.I64 or DType.U64 => 8,
        DType.F32 or DType.I32 or DType.U32 => 4,
        DType.F16 or DType.BF16 or DType.I16 => 2,
        _ => 1,
    };

    private void ZeroRecurrent(int slot)
    {
        for (int i = 0; i < _slots.Length; i++)
        {
            var (group, bytes) = _slots[i];
            if (StateGroups[group].Kind == StateGroupKind.Recurrent)
            {
                new Span<byte>((byte*)_state[i].Address + (slot * bytes), checked((int)bytes)).Clear();
            }
        }
    }

    private DeviceSpan Alloc(long bytes, DType dtype, long elements)
    {
        bytes = Math.Max(bytes, 64);
        void* p = NativeMemory.AlignedAlloc((nuint)bytes, 64);
        new Span<byte>(p, checked((int)bytes)).Clear();
        _allocations.Add((nint)p);
        return new DeviceSpan { Address = (nint)p, Length = (ulong)elements, Dtype = dtype };
    }
}

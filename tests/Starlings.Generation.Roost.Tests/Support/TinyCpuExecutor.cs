using System.Buffers.Binary;
using System.Collections.Immutable;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Roost;
using Starlings.Roost.Abi;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Ops;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.State;
using Starlings.Roost.Transport;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Weights;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost.Tests.Support;

/// <summary>A roost-tiny-hybrid checkpoint written at test time (nothing is committed): one tensor per weight slot, generated from a seed.</summary>
internal static class TinyCheckpoint
{
    /// <summary>Writes the checkpoint for <paramref name="seed"/> to <paramref name="path"/> and returns its lock entry.</summary>
    public static LockedFile Write(string path, ulong seed)
    {
        var tensors = RoostTinyHybrid.WeightSlots
            .Select(w =>
            {
                var values = TinyWeights.Generate(seed, w.Name, w.Shape, w.Dtype);
                var data = new byte[values.Length * (w.Dtype == DType.F32 ? 4 : 2)];
                unsafe
                {
                    fixed (byte* p = data)
                    {
                        TinyWeights.Write(values, w.Dtype, (nint)p);
                    }
                }

                return (w.Name, Dtype: w.Dtype == DType.F32 ? "F32" : "BF16", w.Elements, Data: data);
            })
            .ToList();

        using var json = new MemoryStream();
        using (var writer = new Utf8JsonWriter(json))
        {
            writer.WriteStartObject();
            long cursor = 0;
            foreach (var t in tensors)
            {
                writer.WriteStartObject(t.Name);
                writer.WriteString("dtype", t.Dtype);
                writer.WriteStartArray("shape");
                writer.WriteNumberValue(t.Elements);
                writer.WriteEndArray();
                writer.WriteStartArray("data_offsets");
                writer.WriteNumberValue(cursor);
                writer.WriteNumberValue(cursor + t.Data.Length);
                writer.WriteEndArray();
                writer.WriteEndObject();
                cursor += t.Data.Length;
            }

            writer.WriteEndObject();
        }

        var header = json.ToArray();
        var file = new byte[8 + header.Length + tensors.Sum(static t => t.Data.Length)];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)header.Length);
        header.CopyTo(file, 8);
        var at = 8 + header.Length;
        foreach (var t in tensors)
        {
            t.Data.CopyTo(file, at);
            at += t.Data.Length;
        }

        File.WriteAllBytes(path, file);
        return new LockedFile(Path.GetFileName(path), file.Length, Convert.ToHexStringLower(SHA256.HashData(file)));
    }

    /// <summary>The lock entry as a <see cref="WeightSource"/>.</summary>
    public static WeightSource Source(string path, LockedFile locked) =>
        new(Path.GetFullPath(path), locked.Size, ImmutableArray.Create(Convert.FromHexString(locked.Sha256)));
}

/// <summary>
/// A CPU worker that runs a real model: roost-tiny-hybrid (R-021), its weights streamed and verified from a locked safetensors
/// file by <see cref="HostWeightResidency"/> (R-024 and R-100), one generated forward per plan on <see cref="CpuOps"/> over the
/// plan's block ids and recurrent slot, and greedy sampling (first maximum). Plans are read with the eager worker's
/// <see cref="StepPlanReader"/> and results written with its <see cref="StepResultWriter"/>. It implements
/// <see cref="IWeightLifecycle"/>: a sleep frees the KV and recurrent state pools (and at level 2 the weights), a wake allocates
/// them again zeroed and restores the weights with the digest check, and a replacement swaps in a verified file.
/// </summary>
/// <remarks>
/// The test tokenizer has 264 ids and the model 256 embeddings, so input ids are taken modulo 256; generated ids are 0 to 255,
/// and the stop id (263) is never sampled, so requests end on their length limit. A recurrent slot is zeroed when a sequence
/// starts (<c>position_offset == 0</c>). Host addresses stand in for device addresses.
/// </remarks>
internal sealed unsafe class TinyCpuExecutor : IStepExecutor, IWeightLifecycle
{
    private static readonly TinyConfig Config = TinyConfig.Hybrid;

    private static readonly EagerWeightSlot[] Slots =
        [.. RoostTinyHybrid.WeightSlots.Select(static w => new EagerWeightSlot(w.Name, w.Dtype, w.Elements, [w.Name], null))];

    private readonly string _path;
    private readonly LockedFile _locked;
    private readonly PlanLimits _limits;
    private readonly int _recurrentGroup;
    private readonly RoostTinyHybrid.ForwardLayout _layout;
    private readonly nint _region;
    private readonly nint _arena;
    private HostWeightResidency? _weights;
    private DeviceSpan[]? _state;
    private long _stateBytes;

    public TinyCpuExecutor(string path, LockedFile locked, PlanLimits limits)
    {
        _path = path;
        _locked = locked;
        _limits = limits;
        _recurrentGroup = RoostTinyHybrid.CreateStateGroups().Single(g => g.Kind == StateGroupKind.Recurrent).Index;
        _layout = new RoostTinyHybrid.ForwardLayout(limits.MaxTokens, limits.MaxSequences, maxKvBlocks: limits.AttentionBlocks);
        _region = Alloc(_layout.PlanBytes);
        _arena = Alloc(_layout.ArenaBytes);
    }

    /// <summary>Bytes of the KV and recurrent state pools while allocated.</summary>
    public long StateBytes => _stateBytes;

    /// <summary>Whether the state pools are allocated.</summary>
    public bool HasState => _state is not null;

    /// <summary>The weights, after the handshake.</summary>
    public HostWeightResidency? Weights => _weights;

    public WorkerIdentity Identity => new(KernelAbi.Version, ScriptedExecutor.ManifestDigest, ImmutableArray.Create(Convert.FromHexString(_locked.Sha256)));

    public ImmutableArray<byte> ResidentWeightsDigest => _weights?.ResidentDigest ?? default;

    public HandshakeReject OpenDevice(int deviceOrdinal, Span<byte> deviceUuid)
    {
        deviceUuid.Fill(0x7C);
        return deviceOrdinal == 0 ? HandshakeReject.None : HandshakeReject.Device;
    }

    public void RegisterSegment(nint address, long bytes)
    {
    }

    public HandshakeReject LoadWeights()
    {
        try
        {
            _weights = new HostWeightResidency(Slots, _path, _locked);
        }
        catch (WeightLoadException e) when (e.Error.Code == RoostErrorCode.AssetMismatch)
        {
            return HandshakeReject.WeightsDigest;
        }

        AllocateState();
        return HandshakeReject.None;
    }

    public StepOutcome Execute(ReadOnlySpan<byte> plan, Span<byte> result)
    {
        ParsedPlan parsed;
        try
        {
            parsed = StepPlanReader.Parse(plan, _limits);
        }
        catch (PlanRejectedException e)
        {
            return StepOutcome.Recoverable((int)e.Reason);
        }

        var state = _state ?? throw new InvalidOperationException("The state pools are released.");
        var weights = _weights?.Weights ?? throw new InvalidOperationException("The weights are released.");
        var logits = Forward(parsed, state, [.. weights.Spans]);
        var emitted = new List<int>();
        var row = 0;
        foreach (var s in parsed.Sequences)
        {
            if (!s.Sample)
            {
                continue;
            }

            // The forward selects every row's last token; row r of the logits is sequence r.
            var values = new ReadOnlySpan<float>((float*)logits + ((long)row * Config.Vocab), Config.Vocab);
            var best = 0;
            for (var v = 1; v < values.Length; v++)
            {
                if (values[v] > values[best])
                {
                    best = v;
                }
            }

            emitted.Add(best);
            row++;
        }

        return StepOutcome.Ok(StepResultWriter.Write(result, parsed, [.. emitted], default));
    }

    public ControlOutcome Sleep(SleepLevel level)
    {
        var bytes = FreeState();
        if (level == SleepLevel.Weights)
        {
            bytes += _weights!.Release().Bytes;
        }

        return ControlOutcome.Ok(bytes);
    }

    public ControlOutcome Wake(ReadOnlySpan<byte> expectedDigest)
    {
        var restored = _weights!.Restore(expectedDigest);
        if (restored.Status != ControlStatus.Ok)
        {
            return restored; // the state pools stay released with the weights
        }

        return ControlOutcome.Ok(restored.Bytes + AllocateState());
    }

    public ControlOutcome ReplaceWeights(WeightSource source) => _weights!.Replace(source);

    public void Dispose()
    {
        _ = FreeState();
        _weights?.Dispose();
        NativeMemory.AlignedFree((void*)_region);
        NativeMemory.AlignedFree((void*)_arena);
    }

    private static int Size(DType t) => t == DType.F32 ? 4 : 2;

    private static nint Alloc(long bytes)
    {
        var size = Math.Max(bytes, 64);
        var p = NativeMemory.AlignedAlloc((nuint)size, 64);
        NativeMemory.Clear(p, (nuint)size);
        return (nint)p;
    }

    private long AllocateState()
    {
        if (_state is not null)
        {
            return 0;
        }

        var slots = RoostTinyHybrid.StateSlots;
        var state = new DeviceSpan[slots.Length];
        long total = 0;
        for (var i = 0; i < state.Length; i++)
        {
            var slot = slots[i];
            long count = (slot.Group == _recurrentGroup ? _limits.RecurrentSlots : _limits.AttentionBlocks) * slot.ElementsPerBlock;
            long bytes = count * Size(slot.Dtype);
            state[i] = new DeviceSpan { Address = Alloc(bytes), Length = (ulong)count, Dtype = slot.Dtype };
            total += bytes;
        }

        _state = state;
        _stateBytes = total;
        return total;
    }

    private long FreeState()
    {
        var state = _state;
        if (state is null)
        {
            return 0;
        }

        foreach (var span in state)
        {
            NativeMemory.AlignedFree((void*)span.Address);
        }

        _state = null;
        var bytes = _stateBytes;
        _stateBytes = 0;
        return bytes;
    }

    private nint Forward(ParsedPlan plan, DeviceSpan[] state, DeviceSpan[] weights)
    {
        var bs = Config.BlockSize;
        var n = plan.Sequences.Length;
        var total = plan.TokenIds.Length;
        var pages = plan.Sequences.Sum(s => (s.KvLength + bs - 1) / bs);
        var region = new DeviceSpan { Address = _region, Length = (ulong)_layout.PlanBytes, Dtype = DType.U8 };
        var arena = new DeviceSpan { Address = _arena, Length = (ulong)_layout.ArenaBytes, Dtype = DType.U8 };
        var p = RoostTinyHybrid.PlanInputs.Bind(region, _layout, total, n, n, pages);
        int* ids = (int*)p.Ids.Address, positions = (int*)p.Positions.Address, qIndptr = (int*)p.QIndptr.Address;
        int* kvIndptr = (int*)p.KvIndptr.Address, kvIndices = (int*)p.KvIndices.Address, last = (int*)p.KvLastPageLength.Address;
        int* rows = (int*)p.Rows.Address, stateSlots = (int*)p.StateSlots.Address, cuSeqlens = (int*)p.CuSeqlens.Address;
        var slots = (long*)p.SlotMapping.Address;
        var page = 0;
        for (var b = 0; b < n; b++)
        {
            var s = plan.Sequences[b];
            if (s.PositionOffset == 0)
            {
                ClearRecurrentSlot(state, s.RecurrentSlot);
            }

            qIndptr[b] = s.TokenStart;
            kvIndptr[b] = page;
            cuSeqlens[b] = s.TokenStart;
            stateSlots[b] = s.RecurrentSlot;
            for (var j = 0; j < s.TokenCount; j++)
            {
                var pos = s.PositionOffset + j;
                ids[s.TokenStart + j] = plan.TokenIds[s.TokenStart + j] % Config.Vocab;
                positions[s.TokenStart + j] = pos;
                slots[s.TokenStart + j] = ((long)s.Blocks[pos / bs] * bs) + (pos % bs);
            }

            foreach (var block in s.Blocks)
            {
                kvIndices[page++] = block;
            }

            last[b] = ((s.KvLength - 1) % bs) + 1;
            rows[b] = s.TokenStart + s.TokenCount - 1;
        }

        qIndptr[n] = total;
        kvIndptr[n] = page;
        cuSeqlens[n] = total;
        var ctx = new RoostTinyHybrid.StepContext(_layout, arena, weights, state, p, total, n, n, pages);
        RoostTinyHybrid.Forward<CpuOps>(in ctx);
        return RoostTinyHybrid.Output_logits(in ctx).Address;
    }

    private void ClearRecurrentSlot(DeviceSpan[] state, int slot)
    {
        var slots = RoostTinyHybrid.StateSlots;
        for (var i = 0; i < state.Length; i++)
        {
            if (slots[i].Group == _recurrentGroup)
            {
                var bytes = slots[i].ElementsPerBlock * Size(slots[i].Dtype);
                new Span<byte>((byte*)state[i].Address + (slot * bytes), (int)bytes).Clear();
            }
        }
    }
}

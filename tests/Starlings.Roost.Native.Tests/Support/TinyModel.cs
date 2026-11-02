using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Native.Tests.Support;

/// <summary>The tiny CI models these tests decode with (both full-attention; the hybrid's recurrent state is not needed for masks).</summary>
public enum TinyKind
{
    /// <summary>roost-tiny, dense.</summary>
    Dense,

    /// <summary>roost-tiny-moe.</summary>
    Moe,
}

/// <summary>
/// One sequence of a tiny model on <see cref="CpuOps"/>, as <c>tests/Starlings.Roost.Models.Tiny.Tests</c> runs it: seeded weights in
/// native memory, a paged KV cache, one generated forward per step. Returns the logits of the last token, widened from the
/// model's 256 columns to the tiny tokenizer's 264 ids: the special ids 256 to 262 at -inf (never chosen) and the stop id 263 at
/// 0, so that a mask that allows only the stop id can choose it.
/// </summary>
internal sealed unsafe class TinyModel : IDisposable
{
    public const int ModelVocab = 256;

    private readonly TinyKind _kind;
    private readonly TinyConfig _config;
    private readonly List<nint> _blocks = [];
    private readonly DeviceSpan[] _weights;
    private readonly DeviceSpan[] _state;
    private readonly object _layout;
    private readonly DeviceSpan _arena;
    private readonly DeviceSpan _planRegion;
    private readonly List<int> _kvBlocks = [];
    private readonly int _capacity;
    private int _length;

    public TinyModel(TinyKind kind, int blocks = 48, int maxTokens = 128)
    {
        _kind = kind;
        _config = kind == TinyKind.Dense ? TinyConfig.Dense : TinyConfig.Moe;
        _capacity = blocks;
        var weightSlots = kind == TinyKind.Dense
            ? RoostTiny.WeightSlots.Select(w => (w.Name, w.Shape, w.Dtype, w.Elements)).ToArray()
            : RoostTinyMoe.WeightSlots.Select(w => (w.Name, w.Shape, w.Dtype, w.Elements)).ToArray();
        var stateSlots = kind == TinyKind.Dense
            ? RoostTiny.StateSlots.Select(st => (st.Dtype, st.ElementsPerBlock)).ToArray()
            : RoostTinyMoe.StateSlots.Select(st => (st.Dtype, st.ElementsPerBlock)).ToArray();
        _weights = new DeviceSpan[weightSlots.Length];
        for (var i = 0; i < _weights.Length; i++)
        {
            var (name, shape, dtype, elements) = weightSlots[i];
            _weights[i] = Alloc(elements, dtype);
            TinyWeights.Write(TinyWeights.Generate(_config.Seed, name, shape, dtype), dtype, _weights[i].Address);
        }

        _state = new DeviceSpan[stateSlots.Length];
        for (var i = 0; i < _state.Length; i++)
        {
            _state[i] = Alloc(blocks * stateSlots[i].ElementsPerBlock, stateSlots[i].Dtype);
        }

        if (kind == TinyKind.Dense)
        {
            var layout = new RoostTiny.ForwardLayout(maxTokens, 1, maxKvBlocks: blocks);
            _layout = layout;
            _arena = Alloc(layout.ArenaBytes, DType.U8);
            _planRegion = Alloc(layout.PlanBytes, DType.U8);
        }
        else
        {
            var layout = new RoostTinyMoe.ForwardLayout(maxTokens, 1, maxKvBlocks: blocks);
            _layout = layout;
            _arena = Alloc(layout.ArenaBytes, DType.U8);
            _planRegion = Alloc(layout.PlanBytes, DType.U8);
        }
    }

    /// <summary>Runs the tokens through the model after what it has seen; returns the widened logits of the last one.</summary>
    public float[] Step(ReadOnlySpan<int> tokens)
    {
        int bs = _config.BlockSize;
        int after = _length + tokens.Length;
        int pages = (after + bs - 1) / bs;
        if (pages > _capacity)
        {
            throw new InvalidOperationException("The test's KV cache is full.");
        }

        while (_kvBlocks.Count < pages)
        {
            _kvBlocks.Add(_kvBlocks.Count);
        }

        var region = BindPlan(tokens.Length, pages);
        int* ids = (int*)region.Ids;
        int* positions = (int*)region.Positions;
        long* slots = (long*)region.SlotMapping;
        for (var j = 0; j < tokens.Length; j++)
        {
            int p = _length + j;
            ids[j] = tokens[j];
            positions[j] = p;
            slots[j] = ((long)_kvBlocks[p / bs] * bs) + (p % bs);
        }

        ((int*)region.QIndptr)[0] = 0;
        ((int*)region.QIndptr)[1] = tokens.Length;
        ((int*)region.KvIndptr)[0] = 0;
        ((int*)region.KvIndptr)[1] = pages;
        for (var i = 0; i < pages; i++)
        {
            ((int*)region.KvIndices)[i] = _kvBlocks[i];
        }

        ((int*)region.KvLastPageLength)[0] = ((after - 1) % bs) + 1;
        ((int*)region.Rows)[0] = tokens.Length - 1;
        _length = after;

        var logits = Forward(tokens.Length, pages);
        var widened = new float[Grammar.TokenizerVocab];
        new ReadOnlySpan<float>((float*)logits, ModelVocab).CopyTo(widened);
        for (var v = ModelVocab; v < widened.Length; v++)
        {
            widened[v] = float.NegativeInfinity;
        }

        widened[Grammar.StopId] = 0f;
        return widened;
    }

    public void Dispose()
    {
        foreach (var p in _blocks)
        {
            NativeMemory.AlignedFree((void*)p);
        }

        _blocks.Clear();
    }

    private (nint Ids, nint Positions, nint SlotMapping, nint QIndptr, nint KvIndptr, nint KvIndices, nint KvLastPageLength, nint Rows) BindPlan(int tokens, int pages)
    {
        if (_kind == TinyKind.Dense)
        {
            var p = RoostTiny.PlanInputs.Bind(_planRegion, (RoostTiny.ForwardLayout)_layout, tokens, 1, 1, pages);
            return (p.Ids.Address, p.Positions.Address, p.SlotMapping.Address, p.QIndptr.Address, p.KvIndptr.Address, p.KvIndices.Address, p.KvLastPageLength.Address, p.Rows.Address);
        }

        var m = RoostTinyMoe.PlanInputs.Bind(_planRegion, (RoostTinyMoe.ForwardLayout)_layout, tokens, 1, 1, pages);
        return (m.Ids.Address, m.Positions.Address, m.SlotMapping.Address, m.QIndptr.Address, m.KvIndptr.Address, m.KvIndices.Address, m.KvLastPageLength.Address, m.Rows.Address);
    }

    private nint Forward(int tokens, int pages)
    {
        if (_kind == TinyKind.Dense)
        {
            var layout = (RoostTiny.ForwardLayout)_layout;
            var plan = RoostTiny.PlanInputs.Bind(_planRegion, layout, tokens, 1, 1, pages);
            var ctx = new RoostTiny.StepContext(layout, _arena, _weights, _state, plan, tokens, 1, 1, pages);
            RoostTiny.Forward<CpuOps>(in ctx);
            return RoostTiny.Output_logits(in ctx).Address;
        }

        var moeLayout = (RoostTinyMoe.ForwardLayout)_layout;
        var moePlan = RoostTinyMoe.PlanInputs.Bind(_planRegion, moeLayout, tokens, 1, 1, pages);
        var moeCtx = new RoostTinyMoe.StepContext(moeLayout, _arena, _weights, _state, moePlan, tokens, 1, 1, pages);
        RoostTinyMoe.Forward<CpuOps>(in moeCtx);
        return RoostTinyMoe.Output_logits(in moeCtx).Address;
    }

    private DeviceSpan Alloc(long elements, DType dtype)
    {
        long size = dtype switch
        {
            DType.F64 or DType.I64 or DType.U64 => 8,
            DType.F32 or DType.I32 or DType.U32 => 4,
            DType.F16 or DType.BF16 or DType.I16 => 2,
            _ => 1,
        };
        long bytes = Math.Max(elements * size, 64);
        void* p = NativeMemory.AlignedAlloc((nuint)bytes, 64);
        NativeMemory.Clear(p, (nuint)bytes);
        _blocks.Add((nint)p);
        return new DeviceSpan { Address = (nint)p, Length = (ulong)elements, Dtype = dtype };
    }
}

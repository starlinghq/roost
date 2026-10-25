using Starlings.Roost.Abi;
using System.Runtime.InteropServices;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Models.Qwen35.Tests;

/// <summary>
/// Runs the generated Qwen3.5-0.8B forward on the CPU over the locked checkpoint's weights: one sequence, native memory,
/// weights held as the checkpoint's bfloat16 (a backend upcasts per op). Test-time only; used by the smoke test (R-035) and
/// the per-layer parity run (R-022). Not thread-safe.
/// </summary>
internal sealed unsafe class Qwen35CpuRunner : IDisposable
{
    public const string FileName = "model.safetensors-00001-of-00001.safetensors";
    public const int BlockSize = 16;
    public const int Vocab = 248320;

    private readonly List<nint> _blocks = [];
    private readonly DeviceSpan _arena;
    private readonly DeviceSpan _planRegion;
    private DeviceSpan[] _state = [];

    /// <summary>Loads every weight slot from <paramref name="checkpoint"/> and sizes the arena for <paramref name="maxTokens"/> tokens per step
    /// and <paramref name="maxContext"/> tokens of context.</summary>
    public Qwen35CpuRunner(SafetensorsFile checkpoint, int maxTokens, int maxContext)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        MaxKvBlocks = (maxContext + BlockSize - 1) / BlockSize;
        Layout = new Qwen35Model.ForwardLayout(maxTokens, maxSequences: 1, maxKvBlocks: MaxKvBlocks);
        Weights = LoadWeights(checkpoint);
        _arena = Alloc(Layout.ArenaBytes, DType.U8);
        _planRegion = Alloc(Layout.PlanBytes, DType.U8);
        ResetState();
    }

    public Qwen35Model.ForwardLayout Layout { get; }

    public int MaxKvBlocks { get; }

    public DeviceSpan[] Weights { get; }

    /// <summary>The state spans of the one sequence, in <see cref="Qwen35Model.StateSlots"/> order.</summary>
    public ReadOnlySpan<DeviceSpan> State => _state;

    public void Dispose()
    {
        foreach (nint p in _blocks)
        {
            NativeMemory.AlignedFree((void*)p);
        }

        _blocks.Clear();
    }

    /// <summary>Starts a new sequence: fresh, zeroed K/V pages, recurrent state and convolution state (the old ones are kept until disposal).</summary>
    public void ResetState()
    {
        var state = new DeviceSpan[Qwen35Model.StateSlots.Length];
        for (int i = 0; i < state.Length; i++)
        {
            var slot = Qwen35Model.StateSlots[i];
            long blocks = slot.Role is "k" or "v" ? MaxKvBlocks : 1;
            state[i] = Alloc(blocks * slot.ElementsPerBlock, slot.Dtype);
        }

        _state = state;
    }

    /// <summary>Runs <paramref name="ids"/> at positions <paramref name="start"/>.. of the sequence on <typeparamref name="TOps"/> and returns
    /// the last token's logits (float32, as LmHead writes them).</summary>
    public float[] Step<TOps>(ReadOnlySpan<int> ids, int start)
        where TOps : struct, IRoostOps
    {
        int n = ids.Length;
        int context = start + n;
        int pages = (context + BlockSize - 1) / BlockSize;
        if (n > Layout.MaxTokens || pages > MaxKvBlocks)
        {
            throw new ArgumentException($"step of {n} tokens at {start} exceeds the layout ({Layout.MaxTokens} tokens, {MaxKvBlocks} pages)");
        }

        var plan = Qwen35Model.PlanInputs.Bind(_planRegion, Layout, n, 1, 1, pages);
        for (int t = 0; t < n; t++)
        {
            SetInt(plan.Ids, t, ids[t]);
            SetInt(plan.Positions, t, start + t);
            SetInt(plan.SlotMapping, t, start + t);
        }

        SetInt(plan.Rows, 0, n - 1);
        SetInt(plan.QIndptr, 0, 0);
        SetInt(plan.QIndptr, 1, n);
        SetInt(plan.KvIndptr, 0, 0);
        SetInt(plan.KvIndptr, 1, pages);
        for (int p = 0; p < pages; p++)
        {
            SetInt(plan.KvIndices, p, p);
        }

        SetInt(plan.KvLastPageLength, 0, context - ((pages - 1) * BlockSize));
        SetInt(plan.StateSlots, 0, 0);
        SetInt(plan.CuSeqlens, 0, 0);
        SetInt(plan.CuSeqlens, 1, n);

        var ctx = new Qwen35Model.StepContext(Layout, _arena, Weights, _state, plan, n, 1, 1, pages);
        Qwen35Model.Forward<TOps>(in ctx);
        var logits = Qwen35Model.Output_logits(in ctx);
        if (logits.Dtype != DType.F32)
        {
            throw new InvalidOperationException($"logits are {logits.Dtype}; LmHead writes float32 (C3)");
        }

        return new ReadOnlySpan<float>((void*)logits.Address, Vocab).ToArray();
    }

    private DeviceSpan Alloc(long elements, DType dtype)
    {
        long bytes = Math.Max(elements * (dtype is DType.BF16 ? 2 : dtype is DType.F32 or DType.I32 ? 4 : dtype is DType.I64 ? 8 : 1), 64);
        void* p = NativeMemory.AlignedAlloc((nuint)bytes, 64);
        NativeMemory.Clear(p, (nuint)bytes);
        _blocks.Add((nint)p);
        return new DeviceSpan { Address = (nint)p, Length = (ulong)elements, Dtype = dtype };
    }

    private DeviceSpan[] LoadWeights(SafetensorsFile file)
    {
        var weights = new DeviceSpan[Qwen35Model.WeightSlots.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            var slot = Qwen35Model.WeightSlots[i];
            var src = Qwen35WeightMap.For(slot);
            var span = Alloc(slot.Elements, slot.Dtype);
            weights[i] = span;
            int elementBytes = slot.Dtype == DType.F32 ? 4 : 2;
            var dst = new Span<byte>((void*)span.Address, checked((int)(slot.Elements * elementBytes)));
            switch (src.Transform)
            {
                case WeightTransform.None or WeightTransform.SqueezeConv:
                    file.Bytes(src.Tensors[0]).CopyTo(dst);
                    break;
                case WeightTransform.ConcatRows:
                    int at = 0;
                    foreach (string t in src.Tensors)
                    {
                        file.Bytes(t).CopyTo(dst[at..]);
                        at += file.Bytes(t).Length;
                    }

                    if (at != dst.Length)
                    {
                        throw new InvalidOperationException($"slot {slot.Name}: {at} bytes for {dst.Length}");
                    }

                    break;
                case WeightTransform.OnePlus:
                    Qwen35WeightMap.OnePlus(MemoryMarshal.Cast<byte, ushort>(file.Bytes(src.Tensors[0])), MemoryMarshal.Cast<byte, ushort>(dst));
                    break;
                case WeightTransform.QueryRows or WeightTransform.GateRows:
                    Qwen35WeightMap.SplitQueryRows(
                        MemoryMarshal.Cast<byte, ushort>(file.Bytes(src.Tensors[0])),
                        MemoryMarshal.Cast<byte, ushort>(dst),
                        heads: 8,
                        headDim: 256,
                        hidden: 1024,
                        gate: src.Transform == WeightTransform.GateRows);
                    break;
                default:
                    throw new InvalidOperationException($"slot {slot.Name}: unknown transform {src.Transform}");
            }
        }

        return weights;
    }

    private static void SetInt(DeviceSpan span, int index, long value)
    {
        if (span.Dtype == DType.I64)
        {
            ((long*)span.Address)[index] = value;
        }
        else
        {
            ((int*)span.Address)[index] = (int)value;
        }
    }
}

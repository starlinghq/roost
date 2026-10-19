using Starlings.Roost.Abi;
using Starlings.Roost.Kernels;
using Starlings.Roost.Models.Qwen35;
using Starlings.Roost.Ops;
using Starlings.Roost.Worker.Eager;
using System.Runtime.InteropServices;

namespace Starlings.Roost.Worker.Tests.Support;

/// <summary>
/// The adapter between the eager worker and the generated Qwen3.5-0.8B forward: slot tables, weight assembly from the
/// weight map, layout offsets and the call into <c>Qwen35Model.Forward</c>. The forward runs the generated code directly on
/// the backend; only when a <see cref="Recorder"/> is set (the per-layer comparison) does it go through <see cref="Probe{TInner}"/>.
/// </summary>
public sealed class Qwen35EagerModel : IEagerModel
{
    private static readonly string[] PlanInputNames = ["Ids", "Positions", "SlotMapping", "QIndptr", "KvIndptr", "KvIndices", "KvLastPageLength", "Rows", "StateSlots", "CuSeqlens"];

    private readonly List<EagerWeightSlot> _weights;
    private readonly List<EagerStateSlot> _state;

    /// <summary>Builds the tables from the generated definition and the weight map.</summary>
    public Qwen35EagerModel()
    {
        _weights = [.. Qwen35Model.WeightSlots.Select(Slot)];
        _state = [.. Qwen35Model.StateSlots.Select(static s => new EagerStateSlot(s.Name, s.Group, s.Dtype, s.ElementsPerBlock, s.Group == 1))];
    }

    /// <inheritdoc/>
    public string Name => Qwen35Model.ModelName;

    /// <inheritdoc/>
    public int VocabSize => 248320;

    /// <inheritdoc/>
    public int BlockSize => 16;

    /// <inheritdoc/>
    public int AttentionGroup => 0;

    /// <inheritdoc/>
    public int GroupCount => 2;

    /// <inheritdoc/>
    public IReadOnlyList<EagerWeightSlot> WeightSlots => _weights;

    /// <inheritdoc/>
    public IReadOnlyList<EagerStateSlot> StateSlots => _state;

    /// <inheritdoc/>
    public EagerLayout Layout(int maxTokens, int maxSequences, int maxKvBlocks)
    {
        var l = new Qwen35Model.ForwardLayout(maxTokens, maxSequences, maxKvBlocks);
        var offsets = new Dictionary<string, long>
        {
            ["Ids"] = l.IdsOffset,
            ["Positions"] = l.PositionsOffset,
            ["SlotMapping"] = l.SlotMappingOffset,
            ["QIndptr"] = l.QIndptrOffset,
            ["KvIndptr"] = l.KvIndptrOffset,
            ["KvIndices"] = l.KvIndicesOffset,
            ["KvLastPageLength"] = l.KvLastPageLengthOffset,
            ["Rows"] = l.RowsOffset,
            ["StateSlots"] = l.StateSlotsOffset,
            ["CuSeqlens"] = l.CuSeqlensOffset,
        };
        var dtypes = Qwen35Model.PlanInputs.Layout.ToDictionary(static e => e.Name, static e => e.Dtype);
        foreach (string name in PlanInputNames)
        {
            if (!dtypes.ContainsKey(name))
            {
                throw new InvalidOperationException($"the generated plan inputs have no {name}");
            }
        }

        // The generated logits value is F32 (R-035), the dtype LmHead writes (C1), so every token row has room for a logits row.
        return new EagerLayout(l.PlanBytes, l.ArenaBytes, offsets, dtypes, l.CapacityTokens);
    }

    /// <inheritdoc/>
    public IReadOnlyList<KernelRequirement> KernelRequirements(int sm) =>
    [
        new() { Op = OpId.Embedding, Sm = sm, Activation = DType.BF16 },
        new() { Op = OpId.RmsNorm, Sm = sm, Activation = DType.BF16 },
        new() { Op = OpId.Gemm, Sm = sm, Activation = DType.BF16, Weight = DType.BF16 },
        new() { Op = OpId.CausalConv1d, Sm = sm, Activation = DType.BF16 },
        new() { Op = OpId.GatedDeltaRuleGates, Sm = sm, Activation = DType.BF16 },
        new() { Op = OpId.GatedDeltaRule, Sm = sm, Activation = DType.BF16 },
        new() { Op = OpId.GatedRmsNorm, Sm = sm, Activation = DType.BF16 },
        new() { Op = OpId.QkNorm, Sm = sm, Activation = DType.BF16, HeadDim = 256 },
        new() { Op = OpId.Rope, Sm = sm, Activation = DType.BF16, HeadDim = 256 },
        new() { Op = OpId.CacheWrite, Sm = sm, Kv = DType.BF16, HeadDim = 256, PageSize = 16 },
        new() { Op = OpId.PagedAttention, Sm = sm, Activation = DType.BF16, Kv = DType.BF16, HeadDim = 256, PageSize = 16, GroupSize = 4 },
        new() { Op = OpId.SigmoidMul, Sm = sm, Activation = DType.BF16 },
        new() { Op = OpId.Residual, Sm = sm, Activation = DType.BF16 },
        new() { Op = OpId.SiluAndMul, Sm = sm, Activation = DType.BF16 },
        new() { Op = OpId.LmHead, Sm = sm, Activation = DType.BF16 },
    ];

    /// <inheritdoc/>
    public void CheckBackend<TOps>()
        where TOps : struct, IRoostOps => Qwen35Model.CheckBackend<TOps>();

    /// <inheritdoc/>
    public void Forward<TOps>(in EagerForward input)
        where TOps : struct, IRoostOps
    {
        var layout = new Qwen35Model.ForwardLayout(input.MaxTokens, input.MaxSequences, input.MaxKvBlocks);
        var plan = Qwen35Model.PlanInputs.Bind(input.PlanRegion, layout, input.Tokens, input.Selected, input.Sequences, input.KvBlocks);
        var ctx = new Qwen35Model.StepContext(layout, input.Arena, input.Weights, input.State, plan, input.Tokens, input.Selected, input.Sequences, input.KvBlocks);
        if (ProbeState.Recorder is null)
        {
            Qwen35Model.Forward<TOps>(in ctx);
        }
        else
        {
            Qwen35Model.Forward<Probe<TOps>>(in ctx);
        }
    }

    /// <inheritdoc/>
    public DeviceSpan Logits(in EagerForward input)
    {
        var layout = new Qwen35Model.ForwardLayout(input.MaxTokens, input.MaxSequences, input.MaxKvBlocks);
        var plan = Qwen35Model.PlanInputs.Bind(input.PlanRegion, layout, input.Tokens, input.Selected, input.Sequences, input.KvBlocks);
        var ctx = new Qwen35Model.StepContext(layout, input.Arena, input.Weights, input.State, plan, input.Tokens, input.Selected, input.Sequences, input.KvBlocks);
        return Qwen35Model.Output_logits(in ctx);
    }

    private static EagerWeightSlot Slot(Qwen35Model.WeightSlot slot)
    {
        var src = Qwen35WeightMap.For(slot);
        WeightAssembler? assembler = src.Transform switch
        {
            WeightTransform.None or WeightTransform.SqueezeConv => null,
            WeightTransform.ConcatRows => static (sources, dst) =>
            {
                int at = 0;
                foreach (var s in sources)
                {
                    s.Span.CopyTo(dst[at..]);
                    at += s.Length;
                }
            },
            WeightTransform.OnePlus => static (sources, dst) =>
                Qwen35WeightMap.OnePlus(MemoryMarshal.Cast<byte, ushort>(sources[0].Span), MemoryMarshal.Cast<byte, ushort>(dst)),
            WeightTransform.QueryRows or WeightTransform.GateRows => src.Transform == WeightTransform.GateRows ? Split(true) : Split(false),
            _ => throw new InvalidOperationException($"unmapped transform {src.Transform}"),
        };
        return new EagerWeightSlot(slot.Name, slot.Dtype, slot.Elements, src.Tensors, assembler);
    }

    private static WeightAssembler Split(bool gate) => (sources, dst) =>
        Qwen35WeightMap.SplitQueryRows(MemoryMarshal.Cast<byte, ushort>(sources[0].Span), MemoryMarshal.Cast<byte, ushort>(dst), heads: 8, headDim: 256, hidden: 1024, gate: gate);
}

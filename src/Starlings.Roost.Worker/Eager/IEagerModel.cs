using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Worker.Eager;

/// <summary>Bytes of one checkpoint tensor in host memory, as the weight loader hands them to an assembler.</summary>
/// <param name="Address">First byte; valid for the duration of the assembler call.</param>
/// <param name="Length">Byte length.</param>
public readonly record struct WeightBytes(nint Address, int Length)
{
    /// <summary>The bytes as a span.</summary>
    public unsafe ReadOnlySpan<byte> Span => new((void*)Address, Length);
}

/// <summary>Builds a model weight slot's bytes from its source tensors (norm offsets, row splits and concatenations).</summary>
/// <param name="sources">The slot's source tensors, in <see cref="EagerWeightSlot.Sources"/> order.</param>
/// <param name="destination">The slot's bytes, <see cref="EagerWeightSlot.Bytes"/> long.</param>
public delegate void WeightAssembler(IReadOnlyList<WeightBytes> sources, Span<byte> destination);

/// <summary>One slot of a model's weight table, with where its bytes come from.</summary>
/// <param name="Name">The slot name.</param>
/// <param name="Dtype">The slot's element type.</param>
/// <param name="Elements">The element count.</param>
/// <param name="Sources">Checkpoint tensor names that fill it.</param>
/// <param name="Assembler">How; null means the slot is the single source tensor byte for byte.</param>
public sealed record EagerWeightSlot(string Name, DType Dtype, long Elements, IReadOnlyList<string> Sources, WeightAssembler? Assembler)
{
    /// <summary>Bytes of the slot.</summary>
    public long Bytes => Elements * Dtype switch { DType.F32 or DType.I32 => 4, DType.BF16 or DType.F16 => 2, DType.I64 => 8, _ => 1 };
}

/// <summary>One per-layer state tensor of a model: a K or V pool, a recurrent state or a convolution state.</summary>
/// <param name="Name">The slot name.</param>
/// <param name="Group">The state group index (C4).</param>
/// <param name="Dtype">The element type.</param>
/// <param name="ElementsPerBlock">Elements of one block (paged groups) or one slot (recurrent groups).</param>
/// <param name="Recurrent">Whether the group is recurrent (slot-indexed and zeroed when a sequence starts).</param>
public readonly record struct EagerStateSlot(string Name, int Group, DType Dtype, long ElementsPerBlock, bool Recurrent);

/// <summary>Where a model's plan inputs sit in the plan region, by the names of the generated <c>PlanInputs.Layout</c>.</summary>
/// <param name="PlanBytes">Size of the plan region.</param>
/// <param name="ArenaBytes">Size of the activation arena.</param>
/// <param name="Offsets">Byte offset of each plan input (Ids, Positions, SlotMapping, QIndptr, KvIndptr, KvIndices, KvLastPageLength, Rows, StateSlots, CuSeqlens).</param>
/// <param name="Dtypes">Element type of each plan input.</param>
/// <param name="MaxSelected">The most logits rows the arena can hold (float32 rows, one per token row of the generated buffer).</param>
public sealed record EagerLayout(long PlanBytes, long ArenaBytes, IReadOnlyDictionary<string, long> Offsets, IReadOnlyDictionary<string, DType> Dtypes, int MaxSelected);

/// <summary>The memory and counts of one forward call, handed to the model.</summary>
public readonly ref struct EagerForward
{
    /// <summary>Creates the input.</summary>
    /// <param name="arena">The activation arena.</param>
    /// <param name="planRegion">The plan region (device).</param>
    /// <param name="weights">One span per weight slot.</param>
    /// <param name="state">One span per state slot.</param>
    /// <param name="maxTokens">Capacity the layout was made for.</param>
    /// <param name="maxSequences">Capacity the layout was made for.</param>
    /// <param name="maxKvBlocks">Capacity the layout was made for.</param>
    /// <param name="tokens">Tokens in this step.</param>
    /// <param name="selected">Logits rows in this step.</param>
    /// <param name="sequences">Sequences in this step.</param>
    /// <param name="kvBlocks">Block ids in this step.</param>
    public EagerForward(
        DeviceSpan arena,
        DeviceSpan planRegion,
        ReadOnlySpan<DeviceSpan> weights,
        ReadOnlySpan<DeviceSpan> state,
        int maxTokens,
        int maxSequences,
        int maxKvBlocks,
        int tokens,
        int selected,
        int sequences,
        int kvBlocks)
    {
        Arena = arena;
        PlanRegion = planRegion;
        Weights = weights;
        State = state;
        MaxTokens = maxTokens;
        MaxSequences = maxSequences;
        MaxKvBlocks = maxKvBlocks;
        Tokens = tokens;
        Selected = selected;
        Sequences = sequences;
        KvBlocks = kvBlocks;
    }

    /// <summary>The activation arena.</summary>
    public DeviceSpan Arena { get; }

    /// <summary>The plan region.</summary>
    public DeviceSpan PlanRegion { get; }

    /// <summary>One span per weight slot, in slot order.</summary>
    public ReadOnlySpan<DeviceSpan> Weights { get; }

    /// <summary>One span per state slot, in slot order.</summary>
    public ReadOnlySpan<DeviceSpan> State { get; }

    /// <summary>Layout capacity.</summary>
    public int MaxTokens { get; }

    /// <summary>Layout capacity.</summary>
    public int MaxSequences { get; }

    /// <summary>Layout capacity.</summary>
    public int MaxKvBlocks { get; }

    /// <summary>Tokens in this step.</summary>
    public int Tokens { get; }

    /// <summary>Logits rows in this step.</summary>
    public int Selected { get; }

    /// <summary>Sequences in this step.</summary>
    public int Sequences { get; }

    /// <summary>Block ids in this step.</summary>
    public int KvBlocks { get; }
}

/// <summary>
/// A model as the eager worker sees it (R-025). Generated forwards are per-model static types, so a thin adapter per
/// model implements this over the generated members (weight slots, state slots, layout, <c>Forward&lt;TOps&gt;</c>);
/// the worker stays model-agnostic.
/// </summary>
public interface IEagerModel
{
    /// <summary>The model name.</summary>
    string Name { get; }

    /// <summary>Vocabulary size (token ids are in [0, VocabSize)).</summary>
    int VocabSize { get; }

    /// <summary>Tokens per attention block.</summary>
    int BlockSize { get; }

    /// <summary>The attention state group's index in the plan's group tables.</summary>
    int AttentionGroup { get; }

    /// <summary>State groups in the plan.</summary>
    int GroupCount { get; }

    /// <summary>The weight slots, in the order of the forward's weight span.</summary>
    IReadOnlyList<EagerWeightSlot> WeightSlots { get; }

    /// <summary>The state slots, in the order of the forward's state span.</summary>
    IReadOnlyList<EagerStateSlot> StateSlots { get; }

    /// <summary>Sizes and offsets for a worker's capacities.</summary>
    /// <param name="maxTokens">Largest token count of a step.</param>
    /// <param name="maxSequences">Largest sequence count of a step.</param>
    /// <param name="maxKvBlocks">Largest total of block ids in a step.</param>
    /// <returns>The layout.</returns>
    EagerLayout Layout(int maxTokens, int maxSequences, int maxKvBlocks);

    /// <summary>What the forward needs from the kernel libraries at an SM, one requirement per op and specialisation it uses.</summary>
    /// <param name="sm">The device's SM.</param>
    /// <returns>The requirements; the worker refuses to start when one is unmet.</returns>
    IReadOnlyList<Kernels.KernelRequirement> KernelRequirements(int sm);

    /// <summary>Whether a backend implements every op the forward uses (throws <see cref="NotSupportedException"/> naming the missing ones).</summary>
    /// <typeparam name="TOps">The backend.</typeparam>
    void CheckBackend<TOps>()
        where TOps : struct, IRoostOps;

    /// <summary>Runs the forward on a backend.</summary>
    /// <typeparam name="TOps">The backend.</typeparam>
    /// <param name="input">Memory and counts.</param>
    void Forward<TOps>(in EagerForward input)
        where TOps : struct, IRoostOps;

    /// <summary>The logits after <see cref="Forward{TOps}"/>: <c>Selected</c> rows of <see cref="VocabSize"/> float32 values.</summary>
    /// <param name="input">The same input as the forward.</param>
    /// <returns>The span (its dtype names what the model declared; the data is float32, see <see cref="CudaOps"/>).</returns>
    DeviceSpan Logits(in EagerForward input);
}

namespace Starlings.Roost.Models.Qwen35;

/// <summary>What a loader does to checkpoint tensors to fill a weight slot of <see cref="Qwen35Model"/>.</summary>
public enum WeightTransform
{
    /// <summary>Copy the tensor as it is stored.</summary>
    None = 0,

    /// <summary>
    /// The checkpoint stores an RMS norm weight zero-centred (transformers computes <c>x_norm * (1 + w)</c>); the slot holds
    /// <c>1 + w</c>, computed in float32 and rounded to the slot's element type, so the op multiplies by the slot as it does for any norm.
    /// </summary>
    OnePlus = 1,

    /// <summary>
    /// The query rows of the full-attention <c>q_proj</c>. The checkpoint's [heads * 2 * head_dim, hidden] holds, for head h, rows
    /// <c>h * 2 * head_dim ..+ head_dim</c> (the query) followed by <c>.. + head_dim</c> (the output gate); the slot takes the query rows of every head in order.
    /// </summary>
    QueryRows = 2,

    /// <summary>The output-gate rows of the same <c>q_proj</c>, the other half of every head's 2 * head_dim rows.</summary>
    GateRows = 3,

    /// <summary>The depthwise <c>conv1d.weight</c> [channels, 1, width] is the slot [channels, width]: the same bytes, the unit dimension dropped.</summary>
    SqueezeConv = 4,

    /// <summary>The slot's rows are the first tensor's rows followed by the second's (<c>gate_proj</c> then <c>up_proj</c>), the layout SiluAndMul reads.</summary>
    ConcatRows = 5,
}

/// <summary>Where a weight slot comes from: its checkpoint tensors and what to do to them.</summary>
/// <param name="Slot">The slot name in <see cref="Qwen35Model.WeightSlots"/>.</param>
/// <param name="Tensors">Checkpoint tensor names, in the order they are concatenated.</param>
/// <param name="Transform">The transform.</param>
public readonly record struct SlotSource(string Slot, IReadOnlyList<string> Tensors, WeightTransform Transform);

/// <summary>
/// The weight map of the locked Qwen3.5-0.8B checkpoint (<c>Qwen/Qwen3.5-0.8B</c> at 2fc06364715b967f1860aea9cf38778875588b17):
/// slot to tensors and transform, the tensors the M0 forward does not read, and the transforms themselves. The checkpoint has 488
/// tensors: 153 under <c>model.visual.</c> (vision, not loaded), 15 under <c>mtp.</c> (the MTP head, mapped here and unused in M0),
/// and 320 text tensors, all of which fill a slot. There is no <c>lm_head.weight</c>: the head is tied to the embedding.
/// </summary>
public static class Qwen35WeightMap
{
    private static readonly string[] MtpTensors =
    [
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ];

    /// <summary>The MTP head's tensors (one speculative layer plus its projections). Mapped for the speculation item (R-084); no slot reads them in M0.</summary>
    public static IReadOnlyList<string> Mtp => MtpTensors;

    /// <summary>The prefix of the vision tower's tensors, which this text-only definition does not load.</summary>
    public const string VisionPrefix = "model.visual.";

    /// <summary>The checkpoint tensors and transform for a slot of <see cref="Qwen35Model.WeightSlots"/>.</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>Its source.</returns>
    public static SlotSource For(Qwen35Model.WeightSlot slot)
    {
        string source = slot.Source ?? throw new ArgumentException($"Slot '{slot.Name}' has no checkpoint source.", nameof(slot));
        string[] tensors = source.Split('+');
        string role = slot.Layer >= 0 ? slot.Name.Substring(slot.Name.IndexOf('.', StringComparison.Ordinal) + 1) : slot.Name;
        WeightTransform transform = role switch
        {
            "input_norm" or "post_norm" or "q_norm" or "k_norm" or "final_norm" => WeightTransform.OnePlus,
            "q_proj" => WeightTransform.QueryRows,
            "q_gate_proj" => WeightTransform.GateRows,
            "conv" => WeightTransform.SqueezeConv,
            "gate_up" => WeightTransform.ConcatRows,
            _ => WeightTransform.None,
        };
        return new SlotSource(slot.Name, tensors, transform);
    }

    /// <summary>Every slot's source, in slot order.</summary>
    /// <returns>The sources.</returns>
    public static SlotSource[] All() => [.. Qwen35Model.WeightSlots.Select(For)];

    /// <summary><c>dst[i] = bfloat16(1 + float(src[i]))</c>, rounded to nearest even.</summary>
    /// <param name="src">Checkpoint bfloat16 bits.</param>
    /// <param name="dst">Slot bfloat16 bits, the same length.</param>
    public static void OnePlus(ReadOnlySpan<ushort> src, Span<ushort> dst)
    {
        if (dst.Length != src.Length)
        {
            throw new ArgumentException("dst must be as long as src.", nameof(dst));
        }

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = ToBFloat16Bits(1f + BFloat16ToSingle(src[i]));
        }
    }

    /// <summary>Takes the query rows or the gate rows of the full-attention q_proj (see <see cref="WeightTransform.QueryRows"/>).</summary>
    /// <param name="src">The checkpoint matrix [heads * 2 * headDim, hidden], row-major.</param>
    /// <param name="dst">The slot matrix [heads * headDim, hidden].</param>
    /// <param name="heads">Query heads.</param>
    /// <param name="headDim">Head size.</param>
    /// <param name="hidden">Columns.</param>
    /// <param name="gate">True for the gate rows, false for the query rows.</param>
    public static void SplitQueryRows<T>(ReadOnlySpan<T> src, Span<T> dst, int heads, int headDim, int hidden, bool gate)
    {
        long rowLength = hidden;
        if (src.Length != 2L * heads * headDim * rowLength || dst.Length != (long)heads * headDim * rowLength)
        {
            throw new ArgumentException("src must be [heads*2*headDim, hidden] and dst [heads*headDim, hidden].");
        }

        for (int h = 0; h < heads; h++)
        {
            long from = ((2L * h) + (gate ? 1 : 0)) * headDim * rowLength;
            src.Slice((int)from, (int)(headDim * rowLength)).CopyTo(dst.Slice((int)(h * headDim * rowLength), (int)(headDim * rowLength)));
        }
    }

    /// <summary>Widens bfloat16 bits to a float.</summary>
    /// <param name="bits">The bits.</param>
    /// <returns>The value.</returns>
    public static float BFloat16ToSingle(ushort bits) => BitConverter.Int32BitsToSingle(bits << 16);

    /// <summary>Rounds a float to bfloat16, to nearest even (NaN stays NaN).</summary>
    /// <param name="value">The value.</param>
    /// <returns>The bits.</returns>
    public static ushort ToBFloat16Bits(float value)
    {
        int bits = BitConverter.SingleToInt32Bits(value);
        if (float.IsNaN(value))
        {
            return 0x7FC0;
        }

        int rounded = bits + 0x7FFF + ((bits >> 16) & 1);
        return (ushort)(rounded >>> 16);
    }
}

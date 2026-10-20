using System.Collections.Immutable;

namespace Starlings.Roost.Models.Tiny;

/// <summary>The token mixer of one layer.</summary>
public enum TinyLayerType
{
    /// <summary>Grouped-query attention over a paged KV cache.</summary>
    FullAttention,

    /// <summary>Gated DeltaNet: a convolution and a delta-rule recurrence over a per-sequence state.</summary>
    LinearAttention,
}

/// <summary>
/// A tiny CI model's configuration: the seed its weights are generated from and the widths its definition declares.
/// The definitions (<see cref="RoostTiny"/>, <see cref="RoostTinyHybrid"/>) are the source of truth for shapes; these
/// numbers repeat them for tests and tools, and a test compares the two so they cannot drift.
/// </summary>
/// <param name="Name">Model name, as in the definition's <c>RoostModel</c> attribute.</param>
/// <param name="Seed">Seed of the weight generator (<see cref="TinyWeights"/>).</param>
/// <param name="Layers">Number of decoder layers.</param>
/// <param name="FullAttentionInterval">Every <c>FullAttentionInterval</c>-th layer (the last of each group) is full attention and the others are linear; 1 means all layers are full attention.</param>
/// <param name="Hidden">Hidden size.</param>
/// <param name="Intermediate">MLP width (per gate and per up projection).</param>
/// <param name="Vocab">Vocabulary size.</param>
/// <param name="Heads">Query heads of a full-attention layer.</param>
/// <param name="KvHeads">KV heads of a full-attention layer.</param>
/// <param name="HeadDim">Attention head size.</param>
/// <param name="BlockSize">Tokens per KV block.</param>
/// <param name="LinearKeyHeads">Key heads of a linear-attention layer (0 when there are none).</param>
/// <param name="LinearValueHeads">Value heads of a linear-attention layer (0 when there are none).</param>
/// <param name="LinearKeyDim">Key head size of a linear-attention layer.</param>
/// <param name="LinearValueDim">Value head size of a linear-attention layer.</param>
/// <param name="ConvWidth">Kernel width of the linear-attention convolution.</param>
/// <param name="TiedHead">Whether the language-model head is the embedding table.</param>
public sealed record TinyConfig(
    string Name,
    ulong Seed,
    int Layers,
    int FullAttentionInterval,
    int Hidden,
    int Intermediate,
    int Vocab,
    int Heads,
    int KvHeads,
    int HeadDim,
    int BlockSize,
    int LinearKeyHeads,
    int LinearValueHeads,
    int LinearKeyDim,
    int LinearValueDim,
    int ConvWidth,
    bool TiedHead)
{
    /// <summary>roost-tiny: Qwen3-shaped, four full-attention layers.</summary>
    public static TinyConfig Dense { get; } = new(
        "roost-tiny", 0x524F4F5354_0001UL, Layers: 4, FullAttentionInterval: 1, Hidden: 64, Intermediate: 128, Vocab: 256,
        Heads: 4, KvHeads: 2, HeadDim: 16, BlockSize: 16,
        LinearKeyHeads: 0, LinearValueHeads: 0, LinearKeyDim: 0, LinearValueDim: 0, ConvWidth: 0, TiedHead: false);

    /// <summary>roost-tiny-hybrid: Qwen3.5-shaped, full attention every fourth layer and Gated DeltaNet elsewhere.</summary>
    public static TinyConfig Hybrid { get; } = new(
        "roost-tiny-hybrid", 0x524F4F5354_0002UL, Layers: 8, FullAttentionInterval: 4, Hidden: 64, Intermediate: 128, Vocab: 256,
        Heads: 4, KvHeads: 2, HeadDim: 16, BlockSize: 16,
        LinearKeyHeads: 4, LinearValueHeads: 4, LinearKeyDim: 16, LinearValueDim: 16, ConvWidth: 4, TiedHead: true);

    /// <summary>Every tiny configuration.</summary>
    public static ImmutableArray<TinyConfig> All { get; } = [Dense, Hybrid];

    /// <summary>Channels of the fused linear-attention projection and of its convolution: q, k and v side by side.</summary>
    public int LinearConvDim => (2 * LinearKeyHeads * LinearKeyDim) + (LinearValueHeads * LinearValueDim);

    /// <summary>The mixer of layer <paramref name="layer"/>: the last layer of each <see cref="FullAttentionInterval"/> group is full attention.</summary>
    /// <param name="layer">Layer index.</param>
    /// <returns>The layer's token mixer.</returns>
    public TinyLayerType LayerType(int layer) =>
        (uint)layer >= (uint)Layers ? throw new ArgumentOutOfRangeException(nameof(layer))
        : layer % FullAttentionInterval == FullAttentionInterval - 1 ? TinyLayerType.FullAttention : TinyLayerType.LinearAttention;
}

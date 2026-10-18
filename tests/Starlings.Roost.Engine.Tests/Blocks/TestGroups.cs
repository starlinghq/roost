using System.Collections.Immutable;
using Starlings.Roost.Abi;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Tests.Blocks;

/// <summary>State group descriptors for the block manager tests. All shapes are synthetic unless a comment names a source.</summary>
internal static class TestGroups
{
    public static StateGroupDescriptor Full(int index = 0, int blockSize = 4, long bytesPerBlock = 1024) => new()
    {
        Index = index,
        Name = "full",
        Kind = StateGroupKind.Full,
        Layers = [new LayerKvShape(0, 2, 8)],
        BlockSize = blockSize,
        DType = DType.BF16,
        BytesPerBlock = bytesPerBlock,
    };

    public static StateGroupDescriptor Sliding(int index = 1, int blockSize = 2, int window = 5, long bytesPerBlock = 512) => new()
    {
        Index = index,
        Name = "sliding",
        Kind = StateGroupKind.Sliding,
        Layers = [new LayerKvShape(1, 2, 8)],
        BlockSize = blockSize,
        DType = DType.BF16,
        BytesPerBlock = bytesPerBlock,
        Window = window,
    };

    public static StateGroupDescriptor Recurrent(int index = 2, long bytesPerSlot = 4096, int snapshotSlots = 4) => new()
    {
        Index = index,
        Name = "recurrent",
        Kind = StateGroupKind.Recurrent,
        Layers = [],
        BlockSize = 1,
        DType = DType.F32,
        BytesPerBlock = bytesPerSlot,
        RecurrentStateShape = [16, 128, 128],
        SnapshotSlots = snapshotSlots,
    };

    /// <summary>
    /// A Qwen3.5-0.8B-shaped hybrid: 6 full-attention layers in one paged group and 18 Gated DeltaNet layers in one
    /// recurrent group (build plan decision 15). The recurrent side is from the S3 report (16 key and value heads, head
    /// dim 128, fp32 state; conv state is not counted). The attention KV heads and head size are NOT sourced for this
    /// model, so the 2 x 256 below are a stand-in, and only the ratios in the tests depend on them.
    /// </summary>
    public static ImmutableArray<StateGroupDescriptor> Qwen35Shaped(int attentionBlockSize = 16, int snapshotSlots = 8)
    {
        const int layers = 6;
        const int kvHeads = 2;
        const int headSize = 256;
        const int bf16 = 2;
        var attentionBlock = (long)layers * 2 * kvHeads * headSize * bf16 * attentionBlockSize;
        var recurrentSlot = 18L * 16 * 128 * 128 * sizeof(float);
        return
        [
            Full(0, attentionBlockSize, attentionBlock) with { Name = "attention", Layers = [.. Enumerable.Range(0, layers).Select(l => new LayerKvShape((l * 4) + 3, kvHeads, headSize))] },
            Recurrent(1, recurrentSlot, snapshotSlots) with { Name = "gated-deltanet" },
        ];
    }
}

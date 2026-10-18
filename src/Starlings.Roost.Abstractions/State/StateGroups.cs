using System.Collections.Immutable;
using Starlings.Roost.Abi;
using Starlings.Roost.Mesh;
using Starlings.Roost.Plans;

namespace Starlings.Roost.State;

/// <summary>Kinds of state group (contract C4).</summary>
public enum StateGroupKind
{
    /// <summary>Paged full-attention KV.</summary>
    Full = 0,

    /// <summary>Paged sliding-window KV; blocks leaving the window are released.</summary>
    Sliding,

    /// <summary>Fixed-size recurrent state per sequence (Gated DeltaNet, conv state), one slot per sequence.</summary>
    Recurrent,

    /// <summary>Cross-attention KV over encoder outputs.</summary>
    Cross,

    /// <summary>KV of a draft model (speculation).</summary>
    Draft,

    /// <summary>MLA latent KV (M4).</summary>
    MlaLatent,

    /// <summary>LoRA adapter weights paged like blocks (M3).</summary>
    Adapter,

    /// <summary>Expert weights paged between tiers (M2, bet X6).</summary>
    Expert,

    /// <summary>Encoder outputs kept for the decoder (M5).</summary>
    EncoderOutput,
}

/// <summary>KV shape of one layer in a group.</summary>
/// <param name="Layer">Model layer index.</param>
/// <param name="KvHeads">KV heads on this rank.</param>
/// <param name="HeadSize">Head size.</param>
public readonly record struct LayerKvShape(int Layer, int KvHeads, int HeadSize);

/// <summary>How a group is sharded over the mesh (C9).</summary>
/// <param name="TensorParallel">KV heads are split over this many tensor-parallel ranks; 1 means not split.</param>
/// <param name="PipelineStage">The pipeline stage that owns the group's layers; 0 on one stage.</param>
public readonly record struct GroupSharding(int TensorParallel, int PipelineStage)
{
    /// <summary>A single rank.</summary>
    public static GroupSharding SingleRank { get; } = new(1, 0);
}

/// <summary>
/// A state group (contract C4): a set of layers whose per-sequence state shares one pool, block size and element type.
/// Emitted by the model compiler (stream E), allocated by the engine (stream G), read by the worker (F) and the mesh (K).
/// </summary>
/// <remarks>
/// Recurrent snapshots are their own pool (<see cref="SnapshotSlots"/>), so a hybrid model's attention block size is
/// never inflated to the recurrent state size. <see cref="Index"/> is the group index used by C2 cache ops and
/// <c>RoostPlanGroupBlocks</c>.
/// </remarks>
public sealed record StateGroupDescriptor
{
    /// <summary>Group index in the profile, as used by the step plan.</summary>
    public required int Index { get; init; }

    /// <summary>Stable name, for evidence and diagnostics.</summary>
    public required string Name { get; init; }

    /// <summary>The kind.</summary>
    public required StateGroupKind Kind { get; init; }

    /// <summary>Per-layer KV shapes; empty for <see cref="StateGroupKind.Recurrent"/>.</summary>
    public required ImmutableArray<LayerKvShape> Layers { get; init; }

    /// <summary>Tokens per block for paged kinds; 1 for recurrent (one slot is one block).</summary>
    public required int BlockSize { get; init; }

    /// <summary>Element type of the stored state.</summary>
    public required DType DType { get; init; }

    /// <summary>Bytes per block (per slot for recurrent), across all layers of the group on this rank.</summary>
    public required long BytesPerBlock { get; init; }

    /// <summary>The tier the pool lives in.</summary>
    public BlockTier Tier { get; init; } = BlockTier.Device;

    /// <summary>Window in tokens for <see cref="StateGroupKind.Sliding"/>; 0 otherwise.</summary>
    public int Window { get; init; }

    /// <summary>Per-sequence state shape for <see cref="StateGroupKind.Recurrent"/> (for example [heads, k, v]); empty otherwise.</summary>
    public ImmutableArray<long> RecurrentStateShape { get; init; } = [];

    /// <summary>Slots of the separate snapshot pool for recurrent state (prefix caching, speculation rollback); 0 for none.</summary>
    public int SnapshotSlots { get; init; }

    /// <summary>Sharding over the mesh.</summary>
    public GroupSharding Sharding { get; init; } = GroupSharding.SingleRank;
}

/// <summary>A pool of blocks (or recurrent slots) for one state group and tier. Allocation never blocks or allocates managed memory.</summary>
public interface IBlockPool
{
    /// <summary>The group.</summary>
    StateGroupDescriptor Group { get; }

    /// <summary>The tier.</summary>
    BlockTier Tier { get; }

    /// <summary>Total blocks.</summary>
    int Capacity { get; }

    /// <summary>Free blocks.</summary>
    int FreeCount { get; }

    /// <summary>Allocates <c>blocks.Length</c> blocks, all or none, with a reference count of one.</summary>
    /// <param name="blocks">Receives the block ids.</param>
    /// <returns>False, writing nothing, when too few are free.</returns>
    bool TryAllocate(Span<int> blocks);

    /// <summary>Drops one reference to each block; a block whose count reaches zero is free.</summary>
    /// <param name="blocks">Block ids.</param>
    void Release(ReadOnlySpan<int> blocks);

    /// <summary>
    /// Forks a sequence's blocks: paged groups share the blocks copy-on-write (reference counts rise and nothing is
    /// copied until a shared block is written); recurrent groups copy the state into newly allocated slots.
    /// </summary>
    /// <param name="source">The source sequence's blocks.</param>
    /// <param name="forked">Receives the fork's blocks; same length as <paramref name="source"/>.</param>
    /// <param name="copies">Receives the cache ops the next step plan must carry (C2); may be empty for paged groups.</param>
    /// <returns>The number of cache ops written, or -1 when too few blocks are free (nothing changed).</returns>
    int Fork(ReadOnlySpan<int> source, Span<int> forked, Span<PlanCacheOp> copies);
}

/// <summary>Blocks exported from one pool for import into another process, host or tier (C4, used by K in M4).</summary>
/// <param name="TransferId">Correlates export and import.</param>
/// <param name="Group">Group index.</param>
/// <param name="BlockCount">Blocks.</param>
/// <param name="BytesPerBlock">Bytes per block; the importer's group must match.</param>
/// <param name="SourceRank">Mesh rank that exported them.</param>
/// <param name="SourceTier">Tier they were read from.</param>
public sealed record BlockExport(ulong TransferId, int Group, int BlockCount, long BytesPerBlock, int SourceRank, BlockTier SourceTier);

/// <summary>Moves blocks between pools in different processes, hosts or tiers.</summary>
public interface IBlockTransfer
{
    /// <summary>Exports blocks; the blocks stay allocated until the import completes or the export is cancelled.</summary>
    /// <param name="group">Group index.</param>
    /// <param name="blocks">Block ids.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The export descriptor.</returns>
    ValueTask<BlockExport> ExportBlocksAsync(int group, ReadOnlyMemory<int> blocks, CancellationToken cancellationToken);

    /// <summary>Imports exported blocks into newly allocated blocks.</summary>
    /// <param name="blockExport">The export.</param>
    /// <param name="destination">Receives the new block ids.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>A task that completes when the data is in place.</returns>
    ValueTask ImportBlocksAsync(BlockExport blockExport, Memory<int> destination, CancellationToken cancellationToken);
}

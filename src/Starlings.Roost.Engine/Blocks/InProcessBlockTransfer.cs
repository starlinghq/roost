using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Blocks;

/// <summary>
/// In-process export and import of blocks between two managers (C4 <see cref="IBlockTransfer"/>), for the case where
/// one worker owns both pools: a tier move, a copy into another pool, or a test. The data moves as cache ops that the
/// next plans carry; nothing here touches device memory.
/// </summary>
/// <remarks>
/// <para><b>Threading.</b> Engine thread only, like the pools. The methods return completed <see cref="ValueTask"/>s.</para>
/// <para><b>Lifetime.</b> A transfer is in one of three states: <i>exported</i> (the source blocks carry one extra
/// reference, the pin), <i>imported</i> (destination ids allocated, copy ops queued, the pin still held) and
/// <i>done</i> (the plan that carried the ops has completed, the pin is released). Invariants:</para>
/// <list type="number">
/// <item><description>Every pin in <c>_exports</c> or <c>_inFlight</c> or queued is one reference on each of its
/// blocks, and is released exactly once: by <see cref="CancelExport"/>, by <see cref="Complete"/> or by
/// <see cref="Dispose"/>. Removing an entry from its collection is what makes the release happen once.</description></item>
/// <item><description>A pinned block cannot be freed or reallocated, so the source of a queued copy op is intact until
/// the worker has executed the op and the caller has called <see cref="Complete"/> with the ticket of the plan that
/// carried it. Paged blocks are never written in place while pinned (the pin makes them shared, so a writer gets a
/// copy). Recurrent live slots are written in place, so a recurrent export is valid only while the sequence is not
/// scheduled; export a snapshot-pool copy when it must keep running.</description></item>
/// <item><description>A failed or cancelled <see cref="ImportBlocksAsync"/> changes nothing: the destination ids are
/// not allocated and the export stays exported, so the import can be retried or the export cancelled.</description></item>
/// <item><description>An export is imported at most once: a successful import removes it from <c>_exports</c>.</description></item>
/// </list>
/// <para>"Import complete" means the ids are allocated and the ops are queued: the data is in place once the worker has
/// run the plan that carries them (before that plan's forward, in the order <see cref="Drain"/> returned them).</para>
/// </remarks>
public sealed class InProcessBlockTransfer : IBlockTransfer, IDisposable
{
    private readonly BlockManager _source;
    private readonly BlockManager _destination;
    private readonly int _sourceRank;
    private readonly Dictionary<ulong, Export> _exports = [];
    private readonly Queue<Batch> _queued = new();
    private readonly List<Batch> _inFlight = [];
    private ulong _nextTransferId = 1;
    private long _nextTicket = 1;

    /// <summary>Creates a transfer from one manager to another (they may be the same).</summary>
    /// <param name="source">Exports read here.</param>
    /// <param name="destination">Imports allocate here.</param>
    /// <param name="sourceRank">Mesh rank recorded in the export.</param>
    public InProcessBlockTransfer(BlockManager source, BlockManager destination, int sourceRank = 0)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        _source = source;
        _destination = destination;
        _sourceRank = sourceRank;
    }

    /// <summary>Exports whose import has not completed and that were not cancelled.</summary>
    public int PendingExports => _exports.Count;

    /// <summary>Imports whose copy ops have not been drained yet.</summary>
    public int QueuedImports => _queued.Count;

    /// <summary>Drained imports whose plan has not been completed.</summary>
    public int InFlightImports => _inFlight.Count;

    /// <inheritdoc/>
    public ValueTask<BlockExport> ExportBlocksAsync(int group, ReadOnlyMemory<int> blocks, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var pool = _source.GetPool(group);
        pool.Retain(blocks.Span); // validates all, then pins all
        var id = _nextTransferId++;
        _exports.Add(id, new Export(pool, blocks.ToArray()));
        return ValueTask.FromResult(new BlockExport(id, group, blocks.Length, pool.Group.BytesPerBlock, _sourceRank, pool.Tier));
    }

    /// <inheritdoc/>
    public ValueTask ImportBlocksAsync(BlockExport blockExport, Memory<int> destination, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(blockExport);
        cancellationToken.ThrowIfCancellationRequested();
        if (!_exports.TryGetValue(blockExport.TransferId, out var export))
        {
            throw new InvalidOperationException($"Transfer {blockExport.TransferId} is unknown, cancelled or already imported.");
        }

        var target = _destination.GetPool(blockExport.Group);
        if (export.Pool.Group.Index != blockExport.Group
            || export.Blocks.Length != blockExport.BlockCount
            || export.Pool.Group.BytesPerBlock != blockExport.BytesPerBlock
            || target.Group.BytesPerBlock != blockExport.BytesPerBlock
            || target.Group.BlockSize != export.Pool.Group.BlockSize
            || target.Group.Kind != export.Pool.Group.Kind)
        {
            throw new ArgumentException("The export does not match the destination group.", nameof(blockExport));
        }

        if (destination.Length != blockExport.BlockCount)
        {
            throw new ArgumentException("The destination must hold one id per exported block.", nameof(destination));
        }

        // Null entries (released out of a sliding window) carry no data: they stay null in the destination and
        // only the real blocks are allocated.
        var real = 0;
        for (var i = 0; i < export.Blocks.Length; i++)
        {
            if (export.Blocks[i] != BlockPool.NullBlock)
            {
                real++;
            }
        }

        var allocated = new int[real];
        if (!target.TryAllocate(allocated))
        {
            throw new InvalidOperationException($"Group {blockExport.Group} has too few free blocks for the import.");
        }

        var next = 0;
        for (var i = 0; i < export.Blocks.Length; i++)
        {
            destination.Span[i] = export.Blocks[i] == BlockPool.NullBlock ? BlockPool.NullBlock : allocated[next++];
        }

        // From here nothing can fail: the entry moves from _exports to _queued, which keeps the pin held once.
        var kind = export.Pool.Tier == target.Tier ? CacheOpKind.Copy : CacheOpKind.TierMove;
        var ops = new PlanCacheOp[export.Blocks.Length];
        var n = 0;
        for (var i = 0; i < export.Blocks.Length; i++)
        {
            if (export.Blocks[i] == BlockPool.NullBlock)
            {
                continue;
            }

            var op = new PlanCacheOp
            {
                Kind = kind,
                Group = (uint)blockExport.Group,
                Src = (uint)export.Blocks[i],
                Dst = (uint)destination.Span[i],
                Count = 1,
                SrcTier = export.Pool.Tier,
                DstTier = target.Tier,
            };
            BlockPool.Append(ops, ref n, op);
        }

        _exports.Remove(blockExport.TransferId);
        _queued.Enqueue(new Batch(export, ops.AsMemory(0, n).ToArray()));
        return ValueTask.CompletedTask;
    }

    /// <summary>Cancels an export that has not been imported, releasing its pin.</summary>
    /// <param name="transferId">The export's id.</param>
    /// <returns>False when the id is unknown or already imported.</returns>
    public bool CancelExport(ulong transferId)
    {
        if (!_exports.Remove(transferId, out var export))
        {
            return false;
        }

        export.Pool.Release(export.Blocks);
        return true;
    }

    /// <summary>
    /// Moves the copy ops of whole queued imports into <paramref name="ops"/>, in import order, stopping before an
    /// import whose ops do not fit.
    /// </summary>
    /// <param name="ops">Receives the ops for the next plan.</param>
    /// <param name="ticket">Identifies this drain for <see cref="Complete"/>; 0 when nothing was drained.</param>
    /// <returns>The number of ops written.</returns>
    public int Drain(Span<PlanCacheOp> ops, out long ticket)
    {
        var written = 0;
        ticket = 0;
        while (_queued.TryPeek(out var batch) && batch.Ops.Length <= ops.Length - written)
        {
            _queued.Dequeue();
            batch.Ops.CopyTo(ops[written..]);
            written += batch.Ops.Length;
            if (ticket == 0)
            {
                ticket = _nextTicket++;
            }

            _inFlight.Add(batch with { Ticket = ticket });
        }

        return written;
    }

    /// <summary>Releases the pins of every import drained up to and including <paramref name="ticket"/>: the worker has executed their ops.</summary>
    /// <param name="ticket">A ticket from <see cref="Drain"/>.</param>
    public void Complete(long ticket)
    {
        for (var i = 0; i < _inFlight.Count;)
        {
            if (_inFlight[i].Ticket <= ticket)
            {
                var batch = _inFlight[i];
                _inFlight.RemoveAt(i); // removed before released: released exactly once
                batch.Export.Pool.Release(batch.Export.Blocks);
            }
            else
            {
                i++;
            }
        }
    }

    /// <summary>Releases every pin that is still held.</summary>
    public void Dispose()
    {
        foreach (var id in _exports.Keys.ToArray())
        {
            CancelExport(id);
        }

        while (_queued.TryDequeue(out var batch))
        {
            batch.Export.Pool.Release(batch.Export.Blocks);
        }

        Complete(long.MaxValue);
    }

    private sealed record Export(BlockPool Pool, int[] Blocks);

    private sealed record Batch(Export Export, PlanCacheOp[] Ops)
    {
        public long Ticket { get; init; }
    }
}

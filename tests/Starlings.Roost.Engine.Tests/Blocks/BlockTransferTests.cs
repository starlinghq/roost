using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Tests.Blocks;

public sealed class BlockTransferTests
{
    private static BlockManager Manager(int blocks) => new([TestGroups.Full()], [new GroupCapacity(0, blocks, 0)]);

    [Fact]
    public async Task AnExportPinsBlocksUntilTheImportedOpsHaveRun()
    {
        var src = Manager(8);
        var dst = Manager(8);
        using var transfer = new InProcessBlockTransfer(src, dst, sourceRank: 3);
        var blocks = new int[3];
        src.GetPool(0).TryAllocate(blocks);

        var export = await transfer.ExportBlocksAsync(0, blocks, CancellationToken.None);
        Assert.Equal(3, export.BlockCount);
        Assert.Equal(3, export.SourceRank);
        Assert.Equal(TestGroups.Full().BytesPerBlock, export.BytesPerBlock);
        Assert.All(blocks, b => Assert.Equal(2, src.GetPool(0).RefCount(b)));

        src.GetPool(0).Release(blocks); // the owner finishes while the export is pending
        Assert.Equal(5, src.GetPool(0).FreeCount); // still allocated: the pin holds them

        var imported = new int[3];
        await transfer.ImportBlocksAsync(export, imported, CancellationToken.None);
        Assert.Equal(5, dst.GetPool(0).FreeCount);
        Assert.Equal(5, src.GetPool(0).FreeCount); // the pin is held until the ops have run
        Assert.Equal(0, transfer.PendingExports);

        var ops = new PlanCacheOp[4];
        var n = transfer.Drain(ops, out var ticket);
        Assert.Equal(1, n); // three adjacent blocks coalesce into one op
        Assert.Equal(CacheOpKind.Copy, ops[0].Kind);
        Assert.Equal((uint)blocks[0], ops[0].Src);
        Assert.Equal((uint)imported[0], ops[0].Dst);
        Assert.Equal(3u, ops[0].Count);
        Assert.NotEqual(0, ticket);
        Assert.Equal(5, src.GetPool(0).FreeCount);

        transfer.Complete(ticket);
        Assert.Equal(8, src.GetPool(0).FreeCount);
        Assert.Equal(0, transfer.InFlightImports);
        src.GetPool(0).CheckInvariants();
    }

    [Fact]
    public async Task ACancelledExportReleasesItsPin()
    {
        var src = Manager(4);
        using var transfer = new InProcessBlockTransfer(src, src);
        var blocks = new int[2];
        src.GetPool(0).TryAllocate(blocks);
        var export = await transfer.ExportBlocksAsync(0, blocks, CancellationToken.None);
        Assert.True(transfer.CancelExport(export.TransferId));
        Assert.False(transfer.CancelExport(export.TransferId));
        Assert.All(blocks, b => Assert.Equal(1, src.GetPool(0).RefCount(b)));
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await transfer.ImportBlocksAsync(export, new int[2], CancellationToken.None));
    }

    [Fact]
    public async Task AFailedImportChangesNothingAndCanBeRetried()
    {
        var src = Manager(4);
        var dst = Manager(1);
        using var transfer = new InProcessBlockTransfer(src, dst);
        var blocks = new int[2];
        src.GetPool(0).TryAllocate(blocks);
        var export = await transfer.ExportBlocksAsync(0, blocks, CancellationToken.None);

        await Assert.ThrowsAsync<InvalidOperationException>(async () => await transfer.ImportBlocksAsync(export, new int[2], CancellationToken.None));
        Assert.Equal(1, dst.GetPool(0).FreeCount);
        Assert.Equal(1, transfer.PendingExports);
        await Assert.ThrowsAsync<ArgumentException>(async () => await transfer.ImportBlocksAsync(export, new int[1], CancellationToken.None));
        using var cancelled = new CancellationTokenSource();
        await cancelled.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await transfer.ImportBlocksAsync(export, new int[2], cancelled.Token));
        Assert.Equal(1, transfer.PendingExports);
        Assert.Equal(1, dst.GetPool(0).FreeCount);
        Assert.All(blocks, b => Assert.Equal(2, src.GetPool(0).RefCount(b)));
    }

    [Fact]
    public async Task ImportingTwiceOrIntoAMismatchedGroupIsRefused()
    {
        var src = Manager(4);
        var other = new BlockManager([TestGroups.Full(blockSize: 8, bytesPerBlock: 2048)], [new GroupCapacity(0, 4, 0)]);
        using var transfer = new InProcessBlockTransfer(src, src);
        using var mismatched = new InProcessBlockTransfer(src, other);
        var blocks = new int[1];
        src.GetPool(0).TryAllocate(blocks);

        var export = await transfer.ExportBlocksAsync(0, blocks, CancellationToken.None);
        await transfer.ImportBlocksAsync(export, new int[1], CancellationToken.None);
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await transfer.ImportBlocksAsync(export, new int[1], CancellationToken.None));

        var export2 = await mismatched.ExportBlocksAsync(0, blocks, CancellationToken.None);
        await Assert.ThrowsAsync<ArgumentException>(async () => await mismatched.ImportBlocksAsync(export2, new int[1], CancellationToken.None));
        Assert.Equal(4, other.GetPool(0).FreeCount);
    }

    [Fact]
    public async Task ADifferentTierBecomesATierMoveAndDisposeReleasesEveryPin()
    {
        var host = TestGroups.Full() with { Tier = BlockTier.Host };
        var src = Manager(4);
        var dst = new BlockManager([host], [new GroupCapacity(0, 4, 0)]);
        var transfer = new InProcessBlockTransfer(src, dst);
        var a = new int[1];
        var b = new int[1];
        src.GetPool(0).TryAllocate(a);
        src.GetPool(0).TryAllocate(b);
        var e1 = await transfer.ExportBlocksAsync(0, a, CancellationToken.None);
        await transfer.ExportBlocksAsync(0, b, CancellationToken.None);
        await transfer.ImportBlocksAsync(e1, new int[1], CancellationToken.None);
        var ops = new PlanCacheOp[1];
        transfer.Drain(ops, out _);
        Assert.Equal(CacheOpKind.TierMove, ops[0].Kind);
        Assert.Equal(BlockTier.Device, ops[0].SrcTier);
        Assert.Equal(BlockTier.Host, ops[0].DstTier);

        transfer.Dispose();
        Assert.Equal(1, src.GetPool(0).RefCount(a[0]));
        Assert.Equal(1, src.GetPool(0).RefCount(b[0]));
        src.GetPool(0).CheckInvariants();
    }

    [Fact]
    public async Task DrainStopsBeforeAnImportThatDoesNotFit()
    {
        var src = Manager(8);
        using var transfer = new InProcessBlockTransfer(src, src);
        var x = new int[1];
        var y = new int[3];
        src.GetPool(0).TryAllocate(x);
        src.GetPool(0).TryAllocate(y);
        // Make y's ops non-contiguous so it needs more than one op.
        var yGap = new[] { y[0], y[2] };
        await transfer.ImportBlocksAsync(await transfer.ExportBlocksAsync(0, x, CancellationToken.None), new int[1], CancellationToken.None);
        await transfer.ImportBlocksAsync(await transfer.ExportBlocksAsync(0, yGap, CancellationToken.None), new int[2], CancellationToken.None);

        var ops = new PlanCacheOp[2];
        Assert.Equal(1, transfer.Drain(ops, out var t1));
        Assert.Equal(1, transfer.QueuedImports);
        Assert.Equal(2, transfer.Drain(ops, out var t2));
        Assert.True(t2 > t1);
        Assert.Equal(0, transfer.Drain(ops, out var t3));
        Assert.Equal(0, t3);
    }
}

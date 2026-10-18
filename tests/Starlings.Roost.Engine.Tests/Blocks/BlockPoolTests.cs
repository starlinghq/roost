using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Engine.Tests.Blocks;

public sealed class BlockPoolTests
{
    private static BlockPool Paged(int capacity = 8) => new(TestGroups.Full(), capacity);

    private static BlockPool Recurrent(int capacity = 4) => new(TestGroups.Recurrent(0), capacity);

    [Fact]
    public void AllocationIsDeterministicAndAllOrNone()
    {
        var pool = Paged(4);
        Span<int> a = stackalloc int[3];
        Assert.True(pool.TryAllocate(a));
        Assert.Equal([0, 1, 2], a.ToArray());
        Assert.Equal(1, pool.FreeCount);

        Span<int> tooMany = stackalloc int[2];
        tooMany.Fill(-7);
        Assert.False(pool.TryAllocate(tooMany));
        Assert.Equal([-7, -7], tooMany.ToArray());
        Assert.Equal(1, pool.FreeCount);

        pool.Release(a[1..2]);
        Span<int> again = stackalloc int[1];
        Assert.True(pool.TryAllocate(again));
        Assert.Equal(1, again[0]); // LIFO: the id freed last is handed out first
        pool.CheckInvariants();
    }

    [Fact]
    public void DoubleReleaseThrowsAndLeavesThePoolAsItWas()
    {
        var pool = Paged(4);
        var a = new int[3];
        pool.TryAllocate(a);
        var bad = new[] { a[0], a[1], a[0] }; // the third release is a double free
        Assert.Throws<InvalidOperationException>(() => pool.Release(bad));
        Assert.Equal(1, pool.FreeCount);
        Assert.All(a, b => Assert.Equal(1, pool.RefCount(b)));
        pool.CheckInvariants();

        Assert.Throws<ArgumentOutOfRangeException>(() => pool.Release([99]));
        Assert.Equal(1, pool.FreeCount);
        pool.CheckInvariants();
    }

    [Fact]
    public void NullBlocksAreIgnored()
    {
        var pool = Paged(2);
        pool.Release([BlockPool.NullBlock, BlockPool.NullBlock]);
        Span<int> forked = stackalloc int[2];
        Assert.Equal(0, pool.Fork([BlockPool.NullBlock, BlockPool.NullBlock], forked, []));
        Assert.Equal([-1, -1], forked.ToArray());
        Assert.Equal(2, pool.FreeCount);
    }

    [Fact]
    public void PagedForkSharesBlocksAndCopiesNothing()
    {
        var pool = Paged();
        var parent = new int[3];
        pool.TryAllocate(parent);
        var child = new int[3];
        Assert.Equal(0, pool.Fork(parent, child, []));
        Assert.Equal(parent, child);
        Assert.All(parent, b => Assert.Equal(2, pool.RefCount(b)));
        Assert.Equal(5, pool.FreeCount);

        pool.Release(parent);
        Assert.Equal(5, pool.FreeCount); // the child still holds them
        pool.Release(child);
        Assert.Equal(8, pool.FreeCount);
        pool.CheckInvariants();
    }

    [Fact]
    public void CopyOnWriteGivesTheWriterAPrivateTail()
    {
        var pool = Paged();
        var parent = new int[2];
        pool.TryAllocate(parent);
        var child = new int[2];
        pool.Fork(parent, child, []);

        Assert.Equal(ExclusiveResult.Copied, pool.TryMakeExclusive(child[1], out var writable, out var op));
        Assert.NotEqual(child[1], writable);
        Assert.Equal(CacheOpKind.Copy, op.Kind);
        Assert.Equal((uint)child[1], op.Src);
        Assert.Equal((uint)writable, op.Dst);
        Assert.Equal(1u, op.Count);
        Assert.Equal(pool.Tier, op.SrcTier);
        Assert.Equal(1, pool.RefCount(writable));
        Assert.Equal(1, pool.RefCount(parent[1])); // the parent is now the only holder of the old tail

        Assert.Equal(ExclusiveResult.AlreadyExclusive, pool.TryMakeExclusive(parent[1], out var same, out _));
        Assert.Equal(parent[1], same);
        pool.CheckInvariants();
    }

    [Fact]
    public void CopyOnWriteReportsExhaustionWithoutChangingAnything()
    {
        var pool = Paged(2);
        var parent = new int[2];
        pool.TryAllocate(parent);
        var child = new int[2];
        pool.Fork(parent, child, []);
        Assert.Equal(ExclusiveResult.OutOfBlocks, pool.TryMakeExclusive(child[0], out var writable, out _));
        Assert.Equal(child[0], writable);
        Assert.Equal(2, pool.RefCount(child[0]));
        pool.CheckInvariants();
    }

    [Fact]
    public void RecurrentForkCopiesIntoNewSlots()
    {
        var pool = Recurrent();
        Span<int> parent = stackalloc int[1];
        pool.TryAllocate(parent);
        Span<int> child = stackalloc int[1];
        Span<PlanCacheOp> ops = stackalloc PlanCacheOp[1];
        Assert.Equal(1, pool.Fork(parent, child, ops));
        Assert.NotEqual(parent[0], child[0]);
        Assert.Equal(CacheOpKind.Fork, ops[0].Kind);
        Assert.Equal((uint)parent[0], ops[0].Src);
        Assert.Equal((uint)child[0], ops[0].Dst);
        Assert.Equal(1, pool.RefCount(parent[0]));
        Assert.Equal(1, pool.RefCount(child[0]));
        pool.CheckInvariants();
    }

    [Fact]
    public void RecurrentForkWithTooFewSlotsChangesNothing()
    {
        var pool = Recurrent(3);
        var parent = new int[2];
        pool.TryAllocate(parent);
        var child = new int[2];
        var ops = new PlanCacheOp[2];
        Assert.Equal(-1, pool.Fork(parent, child, ops));
        Assert.Equal(1, pool.FreeCount);
        Assert.Throws<ArgumentException>(() => pool.Fork(parent, child, new PlanCacheOp[1]));
        pool.CheckInvariants();
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(3, 0)]
    [InlineData(4, 0)]
    [InlineData(5, 1)] // window 3, block 2 (the case the rule must get right when the window is not a multiple)
    [InlineData(6, 1)]
    [InlineData(7, 2)]
    public void SlidingWindowRule(int tokens, int expected)
    {
        Assert.Equal(expected, BlockPool.LeadingReleasableBlocks(tokens, window: 3, blockSize: 2));
    }

    [Fact]
    public void SlidingWindowReleaseNullsLeadingEntriesAndNeverReleasesAReadableBlock()
    {
        var group = TestGroups.Sliding(0, blockSize: 2, window: 3);
        var pool = new BlockPool(group, 8);
        Span<int> table = stackalloc int[4];
        pool.TryAllocate(table);

        Assert.Equal(0, pool.ReleaseOutOfWindow(table, tokens: 3));
        Assert.Equal(1, pool.ReleaseOutOfWindow(table, tokens: 6));
        Assert.Equal(BlockPool.NullBlock, table[0]);
        Assert.Equal(5, pool.FreeCount);
        Assert.Equal(1, pool.ReleaseOutOfWindow(table, tokens: 7)); // block 0 is already null; block 1 goes
        Assert.Equal(BlockPool.NullBlock, table[1]);
        Assert.Equal(0, pool.ReleaseOutOfWindow(table, tokens: 7));

        // Every key a query at position `tokens` can read (positions >= tokens - window + 1) is in a block still held.
        for (var tokens = 1; tokens <= 8; tokens++)
        {
            var releasable = BlockPool.LeadingReleasableBlocks(tokens, 3, 2);
            var firstReadable = Math.Max(0, tokens - 3 + 1);
            Assert.True(releasable * 2 <= firstReadable);
        }

        pool.CheckInvariants();
    }

    [Fact]
    public void ReleasingOutOfWindowOnAForkedTableOnlyDropsOneReference()
    {
        var pool = new BlockPool(TestGroups.Sliding(0, blockSize: 2, window: 3), 8);
        var table = new int[3];
        pool.TryAllocate(table);
        var child = new int[3];
        pool.Fork(table, child, []);
        var first = table[0];
        pool.ReleaseOutOfWindow(table, tokens: 6);
        Assert.Equal(1, pool.RefCount(first)); // the child still reads it
        pool.CheckInvariants();
    }

    [Fact]
    public void ReleaseOutOfWindowRejectsNonSlidingGroups()
    {
        Assert.Throws<InvalidOperationException>(() => Paged().ReleaseOutOfWindow(new int[2], 100));
    }

    [Fact]
    public void ThePoolRejectsAGroupWithoutSizes()
    {
        Assert.Throws<ArgumentException>(() => new BlockPool(TestGroups.Full(bytesPerBlock: 0), 1));
        Assert.Throws<ArgumentException>(() => new BlockPool(TestGroups.Full(), 1, BlockPoolRole.Snapshot));
    }

    [Fact]
    public void MakingANullEntryExclusiveIsRejectedClearly()
    {
        var pool = new BlockPool(TestGroups.Full(), 4);
        Assert.Throws<ArgumentOutOfRangeException>(() => pool.TryMakeExclusive(BlockPool.NullBlock, out _, out _));
    }
}

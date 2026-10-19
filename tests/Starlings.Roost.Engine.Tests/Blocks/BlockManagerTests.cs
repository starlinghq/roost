using System.Collections.Immutable;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Tests.Blocks;

public sealed class BlockManagerTests
{
    [Fact]
    public void TheAttentionBlockSizeIsNotInflatedByTheRecurrentState()
    {
        var hybrid = TestGroups.Qwen35Shaped(attentionBlockSize: 16);
        var attentionOnly = hybrid.RemoveAt(1);
        var withRecurrent = BlockManager.Create(hybrid, budgetBytes: 4L << 30, maxSequences: 8);
        var without = BlockManager.Create(attentionOnly, budgetBytes: 4L << 30, maxSequences: 8);

        var group = withRecurrent.GetPool(0).Group;
        Assert.Equal(16, group.BlockSize);
        Assert.Equal(attentionOnly[0].BytesPerBlock, group.BytesPerBlock);
        Assert.True(hybrid[1].BytesPerBlock > 50 * group.BytesPerBlock); // a recurrent slot is far larger than an attention block (about 96x with the stand-in KV shape)
        Assert.Equal(8 + 8, withRecurrent.GetPool(1).Capacity + withRecurrent.GetSnapshotPool(1).Capacity);
        Assert.True(withRecurrent.GetPool(0).Capacity < without.GetPool(0).Capacity); // it costs budget, not block size
    }

    [Fact]
    public void SizingSpendsTheRestOfTheBudgetByBytesPerToken()
    {
        ImmutableArray<StateGroupDescriptor> groups =
        [
            TestGroups.Full(0, blockSize: 4, bytesPerBlock: 1024),     // 256 bytes per token
            TestGroups.Full(1, blockSize: 16, bytesPerBlock: 16384),   // 1024 bytes per token
            TestGroups.Recurrent(2, bytesPerSlot: 1000, snapshotSlots: 2),
        ];
        var capacities = BlockSizing.Compute(groups, budgetBytes: 4 * 1000 + 2 * 1000 + 128_000, maxSequences: 4);

        // rest = 128000 bytes; per token 256 + 1024 = 1280 -> T = 100 tokens
        Assert.Equal(25, capacities[0].Blocks);  // 100 / 4
        Assert.Equal(6, capacities[1].Blocks);   // 100 / 16
        Assert.Equal(4, capacities[2].Blocks);
        Assert.Equal(2, capacities[2].SnapshotSlots);
        var spent = (25 * 1024L) + (6 * 16384L) + (6 * 1000L);
        Assert.True(spent <= 4 * 1000 + 2 * 1000 + 128_000);
    }

    [Fact]
    public void ASlidingGroupIsCappedAndTheOthersGetItsShare()
    {
        ImmutableArray<StateGroupDescriptor> groups =
        [
            TestGroups.Full(0, blockSize: 4, bytesPerBlock: 1024),
            TestGroups.Sliding(1, blockSize: 4, window: 8, bytesPerBlock: 1024),
        ];
        // 2 sequences x (ceil(8/4) + 1) = 6 blocks cap for the sliding group.
        var capacities = BlockSizing.Compute(groups, budgetBytes: 100 * 1024, maxSequences: 2);
        Assert.Equal(6, capacities[1].Blocks);
        Assert.Equal(94, capacities[0].Blocks);
    }

    [Fact]
    public void SizingRefusesABudgetThatCannotHoldTheRecurrentSlots()
    {
        Assert.Throws<ArgumentException>(() => BlockSizing.Compute([TestGroups.Recurrent(0, 1000, 0)], 999, 1));
        Assert.Throws<ArgumentException>(() => BlockSizing.Compute([TestGroups.Full(3)], 1000, 1)); // index must match position
    }

    [Fact]
    public void SizingNeverOverspends()
    {
        var rng = new Random(11);
        for (var i = 0; i < 500; i++)
        {
            var groups = ImmutableArray.CreateBuilder<StateGroupDescriptor>();
            var n = rng.Next(1, 4);
            for (var g = 0; g < n; g++)
            {
                var bs = rng.Next(1, 33);
                groups.Add(rng.Next(3) switch
                {
                    0 => TestGroups.Recurrent(g, rng.Next(1, 5000), rng.Next(0, 4)),
                    1 => TestGroups.Sliding(g, bs, rng.Next(1, 64), rng.Next(1, 100_000)),
                    _ => TestGroups.Full(g, bs, rng.Next(1, 100_000)),
                });
            }

            var budget = rng.NextInt64(50_000, 20_000_000);
            var maxSeq = rng.Next(0, 6);
            ImmutableArray<GroupCapacity> caps;
            try
            {
                caps = BlockSizing.Compute(groups.ToImmutable(), budget, maxSeq);
            }
            catch (ArgumentException)
            {
                continue; // recurrent slots alone exceed the budget
            }

            long spent = 0;
            for (var g = 0; g < n; g++)
            {
                spent += (caps[g].Blocks + (long)caps[g].SnapshotSlots) * groups[g].BytesPerBlock;
            }

            Assert.True(spent <= budget, $"iteration {i}: spent {spent} of {budget}");
        }
    }

    [Fact]
    public void SnapshotsLiveInTheirOwnPoolAndEmitOps()
    {
        var manager = new BlockManager([TestGroups.Full(0), TestGroups.Recurrent(1, snapshotSlots: 2)], [new GroupCapacity(0, 4, 0), new GroupCapacity(1, 3, 2)]);
        var live = manager.GetPool(1);
        var snaps = manager.GetSnapshotPool(1);
        Span<int> slot = stackalloc int[1];
        live.TryAllocate(slot);

        Assert.True(manager.TrySnapshot(1, slot[0], out var s1, out var write));
        Assert.Equal(CacheOpKind.SnapshotWrite, write.Kind);
        Assert.Equal((uint)slot[0], write.Src);
        Assert.Equal((uint)s1, write.Dst);
        Assert.Equal(2, live.FreeCount);   // the snapshot did not take a live slot
        Assert.Equal(1, snaps.FreeCount);
        Assert.Equal(4, manager.GetPool(0).FreeCount); // nor an attention block

        Assert.True(manager.TrySnapshot(1, slot[0], out _, out _));
        Assert.False(manager.TrySnapshot(1, slot[0], out var none, out _)); // pool of 2 is full
        Assert.Equal(BlockPool.NullBlock, none);

        var restore = manager.Restore(1, s1, slot[0]);
        Assert.Equal(CacheOpKind.SnapshotRestore, restore.Kind);
        Assert.Equal((uint)s1, restore.Src);
        Assert.Equal((uint)slot[0], restore.Dst);
        Assert.Equal(1, snaps.RefCount(s1)); // restoring leaves it held

        Assert.Throws<InvalidOperationException>(() => manager.GetSnapshotPool(0));
        Assert.Throws<InvalidOperationException>(() => manager.TrySnapshot(1, 2, out _, out _)); // slot 2 is not allocated
    }

    [Fact]
    public void ASharedSnapshotIsFreedByItsLastHolder()
    {
        var manager = new BlockManager([TestGroups.Recurrent(0, snapshotSlots: 1)], [new GroupCapacity(0, 2, 1)]);
        var snaps = manager.GetSnapshotPool(0);
        Span<int> slot = stackalloc int[1];
        manager.GetPool(0).TryAllocate(slot);
        manager.TrySnapshot(0, slot[0], out var s, out _);
        snaps.Retain([s]);
        snaps.Release([s]);
        Assert.Equal(0, snaps.FreeCount);
        snaps.Release([s]);
        Assert.Equal(1, snaps.FreeCount);
    }
}

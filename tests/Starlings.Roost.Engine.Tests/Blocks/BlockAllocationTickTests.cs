using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Engine.Tests.Blocks;

/// <summary>
/// The pools promise not to allocate managed memory (C4 <c>IBlockPool</c>). Each tick runs the steady-state hot path:
/// allocate, fork, copy-on-write, sliding release, snapshot and restore, and release. After a warm-up the thread's
/// allocated-byte counter must not move.
/// </summary>
public sealed class BlockAllocationTickTests
{
    private const int Warmup = 5_000;
    private const int Ticks = 50_000;

    [Fact]
    public void TheBlockHotPathAllocatesNothing()
    {
        var manager = new BlockManager(
            [TestGroups.Full(0, 4), TestGroups.Sliding(1, 2, 5), TestGroups.Recurrent(2, snapshotSlots: 2)],
            [new GroupCapacity(0, 16, 0), new GroupCapacity(1, 16, 0), new GroupCapacity(2, 4, 2)]);
        var full = manager.GetPool(0);
        var sliding = manager.GetPool(1);
        var recurrent = manager.GetPool(2);
        var snapshots = manager.GetSnapshotPool(2);
        var a = new int[4];
        var b = new int[4];
        var s = new int[4];
        var r = new int[1];
        var r2 = new int[1];
        var ops = new PlanCacheOp[4];

        void Tick()
        {
            full.TryAllocate(a);
            full.Fork(a, b, ops);
            full.TryMakeExclusive(b[3], out var writable, out _);
            b[3] = writable;
            full.Release(a);
            full.Release(b);

            sliding.TryAllocate(s);
            sliding.ReleaseOutOfWindow(s, 11);
            sliding.Release(s);

            recurrent.TryAllocate(r);
            recurrent.Fork(r, r2, ops);
            if (manager.TrySnapshot(2, r[0], out var snap, out _))
            {
                manager.Restore(2, snap, r2[0]);
                snapshots.Release([snap]);
            }

            recurrent.Release(r);
            recurrent.Release(r2);
        }

        for (var i = 0; i < Warmup; i++)
        {
            Tick();
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < Ticks; i++)
        {
            Tick();
        }

        var bytes = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, bytes);
        Assert.Equal(16, full.FreeCount);
        Assert.Equal(16, sliding.FreeCount);
        Assert.Equal(4, recurrent.FreeCount);
        Assert.Equal(2, snapshots.FreeCount);
    }
}

using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Caching;
using static Starlings.Roost.Engine.Tests.Caching.CacheRun;

namespace Starlings.Roost.Engine.Tests.Caching;

public sealed class PrefixCacheEvictionTests
{
    // Adds a finished two-block prompt whose blocks only the cache holds.
    private static void AddPrompt(BlockManager blocks, PrefixCache cache, int from)
    {
        var tokens = Seq(from, 8).ToArray();
        var ids = new int[2];
        Assert.True(blocks.GetPool(0).TryAllocate(ids));
        List<int>[] tables = [[.. ids]];
        Assert.True(cache.Sync(tokens, 8, 0, tables, []));
        blocks.GetPool(0).Release(ids); // the sequence is gone
    }

    private static int Hit(PrefixCache cache, BlockManager blocks, int from, bool hold = false)
    {
        List<int>[] tables = [[]];
        var found = new List<PrefixCache.Node>();
        var hit = cache.TryAttach(Seq(from, 9).ToArray(), 0, tables, found, out var computed, out _);
        if (hit && !hold)
        {
            blocks.GetPool(0).Release(tables[0].ToArray());
        }

        return hit ? computed : 0;
    }

    [Fact]
    public void EvictsLeastRecentlyUsedLeavesFirstAndNeverAnInnerNodeBeforeItsChild()
    {
        var blocks = Attention(20);
        var cache = new PrefixCache(blocks, new PrefixCacheOptions(), _ => false);
        AddPrompt(blocks, cache, 0);
        AddPrompt(blocks, cache, 100);
        AddPrompt(blocks, cache, 200);
        Assert.Equal(8, Hit(cache, blocks, 0)); // touch A: B is now least recent
        Assert.Equal(14, blocks.GetPool(0).FreeCount);

        Assert.True(cache.EvictOne()); // B's leaf: the oldest node with no child
        Assert.Equal(5, cache.NodeCount);
        Assert.True(cache.EvictOne()); // B's head, only now a leaf
        Assert.Equal(4, cache.NodeCount);
        Assert.Equal(0, Hit(cache, blocks, 100));
        Assert.Equal(8, Hit(cache, blocks, 200));
        Assert.Equal(8, Hit(cache, blocks, 0));
        Assert.Equal(16, blocks.GetPool(0).FreeCount);
        cache.CheckInvariants();
    }

    [Fact]
    public void OnlyZeroReferenceUnpinnedLeavesAreEvicted()
    {
        var blocks = Attention(20);
        var pinned = new HashSet<long>();
        var cache = new PrefixCache(blocks, new PrefixCacheOptions(), pinned.Contains);
        AddPrompt(blocks, cache, 0);
        Assert.Equal(8, Hit(cache, blocks, 0, hold: true)); // a sequence now holds both blocks
        Assert.False(cache.EvictOne());
        Assert.Equal(0, cache.EvictableNodes());

        // Give the sequence's references back, then pin one block as an uncompleted plan would.
        blocks.GetPool(0).Release([0, 1]);
        pinned.Add(BlockKey.Live(0, 1));
        Assert.False(cache.EvictOne()); // the leaf is pinned and the head has a child
        pinned.Clear();
        Assert.True(cache.EvictOne());
        Assert.True(cache.EvictOne());
        Assert.False(cache.EvictOne());
        Assert.Equal(20, blocks.GetPool(0).FreeCount);
    }
}

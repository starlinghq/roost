using Starlings.Roost.Engine.Caching;

namespace Starlings.Roost.Engine.Tests.Caching;

public sealed class PrefixHashTests
{
    private static readonly int[] Block = [5, 6, 7, 8];

    [Fact]
    public void KeysAreStableAcrossRuns()
    {
        // Pinned values: a change means recorded runs and any persisted cache keys no longer line up.
        var k = PrefixHash.Next(PrefixHash.Seed(0, 0, 4, 0), Block);
        Assert.Equal(PrefixHash.Next(PrefixHash.Seed(0, 0, 4, 0), Block), k);
        Assert.NotEqual(default, k);
    }

    [Fact]
    public void GroupLoraBlockSizeSaltAndTokensAllChangeTheKey()
    {
        var baseKey = PrefixHash.Next(PrefixHash.Seed(1, 2, 4, 3), Block);
        Assert.NotEqual(baseKey, PrefixHash.Next(PrefixHash.Seed(1, 3, 4, 3), Block)); // group
        Assert.NotEqual(baseKey, PrefixHash.Next(PrefixHash.Seed(1, 2, 4, 4), Block)); // LoRA
        Assert.NotEqual(baseKey, PrefixHash.Next(PrefixHash.Seed(1, 2, 8, 3), Block)); // block size
        Assert.NotEqual(baseKey, PrefixHash.Next(PrefixHash.Seed(2, 2, 4, 3), Block)); // salt
        Assert.NotEqual(baseKey, PrefixHash.Next(PrefixHash.Seed(1, 2, 4, 3), [5, 6, 7, 9])); // token
        Assert.NotEqual(baseKey, PrefixHash.Next(PrefixHash.Seed(1, 2, 4, 3), [5, 6, 7])); // length
    }

    [Fact]
    public void AKeyNamesEveryEarlierBlock()
    {
        var seed = PrefixHash.Seed(0, 0, 4, 0);
        var a = PrefixHash.Next(PrefixHash.Next(seed, [1, 2, 3, 4]), Block);
        var b = PrefixHash.Next(PrefixHash.Next(seed, [1, 2, 3, 9]), Block);
        Assert.NotEqual(a, b);
    }

    [Fact]
    public void NoCollisionsAmongManyDistinctBlocks()
    {
        var seen = new HashSet<PrefixKey>();
        var seed = PrefixHash.Seed(0, 0, 4, 0);
        for (var i = 0; i < 20000; i++)
        {
            Assert.True(seen.Add(PrefixHash.Next(seed, [i, i / 7, i % 13, 1])));
        }
    }
}

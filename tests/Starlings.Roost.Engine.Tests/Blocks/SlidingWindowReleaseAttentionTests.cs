using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Engine.Tests.Blocks;

/// <summary>
/// Sliding-window release (R-028) against sliding-window attention on <see cref="CpuOps"/> (R-113): the blocks the pool releases
/// are never read by a query whose <c>WindowLeft</c> is the group's <c>Window</c>, and released entries (-1) in a block table are
/// accepted by the attention op.
/// </summary>
public sealed class SlidingWindowReleaseAttentionTests
{
    [Fact]
    public void TheRuleReleasesEveryBlockAndOnlyTheBlocksNoQueryOfTheNextChunkReads()
    {
        for (var window = 1; window <= 12; window++)
        {
            for (var bs = 1; bs <= 5; bs++)
            {
                for (var computed = 0; computed <= 48; computed++)
                {
                    // The planner releases at the sequence's computed count, before the chunk's queries (positions computed and on)
                    // run. The lowest key any of them reads with WindowLeft = window is computed - window.
                    var released = BlockPool.LeadingReleasableBlocks(computed, window, bs);
                    var lowestRead = Math.Max(0, computed - window);
                    Assert.True(released * bs <= lowestRead, $"window {window} bs {bs} computed {computed}: block {released - 1} is still read");
                    if (computed > window)
                    {
                        // The next block holds key computed - window, so nothing more could go.
                        Assert.True((released + 1) * bs > lowestRead, $"window {window} bs {bs} computed {computed}: block {released} could go");
                    }
                    else
                    {
                        Assert.Equal(0, released);
                    }
                }
            }
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 4)]
    [InlineData(3, 2)] // window not a multiple of the block size
    [InlineData(4, 2)]
    [InlineData(5, 4)]
    [InlineData(8, 3)]
    [InlineData(40, 4)] // longer than every context below
    public void TwoSequencesDecodingThroughAPoolAtTheSizingCapReadTheSameKeysAsWithTheWholeHistory(int window, int bs)
    {
        // Two sequences decode in turn over one pool sized at the sliding cap (BlockSizing: ceil(window / bs) + 1 per sequence).
        // Each step releases out of the window, then allocates; the pool hands freed blocks out again last-in first-out, so a
        // block released too early would be overwritten by the other sequence's next key before it is read. Each step's attention
        // over the released tables must equal, bit for bit, attention over a copy that keeps the whole history.
        const int qHeads = 2;
        const int hd = 4;
        const int steps = 30;
        var group = TestGroups.Sliding(0, blockSize: bs, window: window);
        var capacity = 2 * (((window + bs - 1) / bs) + 1);
        var pool = new BlockPool(group, capacity);
        var rng = new Random((window * 100) + bs);
        var poolK = new double[capacity * bs * hd];
        var poolV = new double[poolK.Length];
        var histK = new[] { new double[steps * hd], new double[steps * hd] };
        var histV = new[] { new double[steps * hd], new double[steps * hd] };
        var tables = new[] { new List<int>(), new List<int>() };
        for (var t = 0; t < steps; t++)
        {
            for (var s = 0; s < 2; s++)
            {
                var table = tables[s];
                pool.ReleaseOutOfWindow(CollectionsMarshal.AsSpan(table), t);
                if (t % bs == 0)
                {
                    var fresh = new int[1];
                    Assert.True(pool.TryAllocate(fresh), $"step {t}: the pool at its cap has no block");
                    table.Add(fresh[0]);
                }

                for (var d = 0; d < hd; d++)
                {
                    double k = (rng.NextDouble() * 2) - 1, v = (rng.NextDouble() * 2) - 1;
                    var row = ((table[t / bs] * bs) + (t % bs)) * hd;
                    poolK[row + d] = histK[s][(t * hd) + d] = k;
                    poolV[row + d] = histV[s][(t * hd) + d] = v;
                }
            }

            var q = Enumerable.Range(0, 2 * qHeads * hd).Select(_ => (rng.NextDouble() * 2) - 1).ToArray();
            var released = Decode(q, poolK, poolV, [.. tables[0], .. tables[1]], [0, tables[0].Count, tables[0].Count + tables[1].Count], t + 1, bs, window);
            int[] identity = [.. Enumerable.Range(0, 2 * steps)];
            int[] histIndptr = [0, t + 1, 2 * (t + 1)];
            var dense = Decode(q, Interleave(histK, t + 1, hd), Interleave(histV, t + 1, hd), identity[..(2 * (t + 1))], histIndptr, t + 1, 1, window);
            Assert.Equal(dense, released);
            pool.CheckInvariants();
        }

        // At steady state each sequence holds at most the cap's share, and the leading entries are released.
        Assert.All(tables, t => Assert.True(t.Count(b => b != BlockPool.NullBlock) <= capacity / 2));
        if (steps - 1 > window + bs)
        {
            Assert.All(tables, t => Assert.Equal(BlockPool.NullBlock, t[0]));
        }
    }

    // Both sequences' K or V rows of the first `len` positions, sequence 0 then 1, as a block-size-1 pool.
    private static double[] Interleave(double[][] hist, int len, int hd) => [.. hist[0].AsSpan(0, len * hd), .. hist[1].AsSpan(0, len * hd)];

    private static double[] Decode(double[] q, double[] k, double[] v, int[] indices, int[] kvIndptr, int kvLen, int bs, int window)
    {
        var o = new double[q.Length];
        int[] last = [kvLen - ((kvIndptr[1] - 1) * bs), kvLen - ((kvIndptr[2] - kvIndptr[1] - 1) * bs)];
        int[] qIndptr = [0, 1, 2];
        var handles = new[] { Pin(q), Pin(k), Pin(v), Pin(o), Pin(indices), Pin(kvIndptr), Pin(last), Pin(qIndptr) };
        try
        {
            CpuOps.PagedAttention(new PagedAttentionArgs
            {
                Q = Span(handles[0], q.Length, DType.F64),
                KCache = Span(handles[1], k.Length, DType.F64),
                VCache = Span(handles[2], v.Length, DType.F64),
                Output = Span(handles[3], o.Length, DType.F64),
                KvIndices = Span(handles[4], indices.Length, DType.I32),
                KvIndptr = Span(handles[5], 3, DType.I32),
                KvLastPageLength = Span(handles[6], 2, DType.I32),
                QIndptr = Span(handles[7], 3, DType.I32),
                Mode = AttentionMode.Decode,
                Batch = 2,
                Tokens = 2,
                QHeads = 2,
                KvHeads = 1,
                HeadDim = 4,
                BlockSize = bs,
                KvLayout = TensorLayout.PagedNhd,
                WindowLeft = window,
                Causal = true,
                KvDType = DType.F64,
                DType = DType.F64,
            });
        }
        finally
        {
            foreach (var h in handles)
            {
                h.Free();
            }
        }

        return o;
    }

    private static GCHandle Pin(Array a) => GCHandle.Alloc(a, GCHandleType.Pinned);

    private static DeviceSpan Span(GCHandle h, int count, DType t) => new() { Address = h.AddrOfPinnedObject(), Length = (ulong)count, Dtype = t };
}

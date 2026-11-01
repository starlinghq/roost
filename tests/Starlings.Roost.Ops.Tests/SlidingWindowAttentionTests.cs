using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>
/// Sliding-window paged attention on <see cref="CpuOps"/> (R-113): key j is visible to the query at position p when
/// <c>p - WindowLeft &lt;= j</c> (and <c>j &lt;= p</c> when causal), so a window of w spans w + 1 keys. Hand-computed cases use equal
/// scores (q = 0), where the output is the mean of the visible values, so the visible set can be read off the result.
/// </summary>
public sealed class SlidingWindowAttentionTests
{
    // One sequence, one head of width 1, block size 4. Values v[j] = j + 1, keys 0, query 0: every visible key weighs the same.
    private static double[] MeanAttention(int kvLen, int queries, int window, bool causal, int blockSize = 4)
    {
        var pages = (kvLen + blockSize - 1) / blockSize;
        using var q = Mem.Zeros(DType.F64, queries);
        using var kc = Mem.Zeros(DType.F64, pages * blockSize);
        using var vc = Mem.F64(Enumerable.Range(0, pages * blockSize).Select(j => j + 1.0).ToArray());
        using var o = Mem.Zeros(DType.F64, queries);
        using var qi = Mem.I32(0, queries);
        using var ki = Mem.I32(0, pages);
        using var idx = Mem.I32(Enumerable.Range(0, pages).ToArray());
        using var last = Mem.I32(kvLen - ((pages - 1) * blockSize));
        CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = q.Span, Output = o.Span, KCache = kc.Span, VCache = vc.Span, QIndptr = qi.Span, KvIndptr = ki.Span, KvIndices = idx.Span, KvLastPageLength = last.Span,
            Mode = queries == 1 ? AttentionMode.Decode : AttentionMode.Prefill, Batch = 1, Tokens = queries, QHeads = 1, KvHeads = 1, HeadDim = 1, BlockSize = blockSize,
            KvLayout = TensorLayout.PagedNhd, Scale = 1f, WindowLeft = window, Causal = causal, KvDType = DType.F64, DType = DType.F64,
        });
        return o.ReadF64();
    }

    private static double Mean(int from, int to) => Enumerable.Range(from, to - from + 1).Average(j => j + 1.0);

    [Theory]
    [InlineData(1, 4, 5)] // keys 4..5, across no page boundary
    [InlineData(2, 3, 5)] // keys 3..5, across the page boundary at 4
    [InlineData(4, 1, 5)]
    [InlineData(5, 0, 5)] // the window reaches key 0 exactly
    [InlineData(6, 0, 5)] // one past the context
    [InlineData(1000, 0, 5)]
    [InlineData(-1, 0, 5)]
    public void ADecodeQuerySeesExactlyTheKeysOfItsWindow(int window, int first, int last)
    {
        // Six keys (positions 0..5), the decode query at position 5.
        Assert.Equal(Mean(first, last), MeanAttention(6, 1, window, causal: true)[0], 1e-13);
        // A decode query is the last key, so the causal flag does not change what it sees.
        Assert.Equal(Mean(first, last), MeanAttention(6, 1, window, causal: false)[0], 1e-13);
    }

    [Fact]
    public void APrefillChunkSlidesTheWindowPerQuery()
    {
        // Six keys, the last four are queries (positions 2..5), window 1: each sees the key before it and its own.
        Near.Eq([Mean(1, 2), Mean(2, 3), Mean(3, 4), Mean(4, 5)], MeanAttention(6, 4, 1, causal: true), 1e-13);
        // Window 2.
        Near.Eq([Mean(0, 2), Mean(1, 3), Mean(2, 4), Mean(3, 5)], MeanAttention(6, 4, 2, causal: true), 1e-13);
        // A fresh prefill of three tokens: the first query has nothing before it, the window is clipped at key 0.
        Near.Eq([Mean(0, 0), Mean(0, 1), Mean(1, 2)], MeanAttention(3, 3, 1, causal: true), 1e-13);
        // A window at least as long as the chunk's context is no window.
        Near.Eq(MeanAttention(6, 4, -1, causal: true), MeanAttention(6, 4, 5, causal: true), 0);
        Near.Eq(MeanAttention(6, 4, -1, causal: true), MeanAttention(6, 4, 99, causal: true), 0);
    }

    [Fact]
    public void ANonCausalWindowKeepsTheLeftBoundAndSeesEveryLaterKey()
    {
        // Four keys, queries at positions 2 and 3, window 1, not causal: from position - 1 to the last key.
        Near.Eq([Mean(1, 3), Mean(2, 3)], MeanAttention(4, 2, 1, causal: false), 1e-13);
    }

    [Fact]
    public void AKeyOutsideTheWindowIsIgnoredWhateverItsScore()
    {
        // Scores 100, 0 and ln 3 with scale 1: the key at 0 would take all the weight, but window 1 hides it, leaving weights
        // 1/4 and 3/4 on the values 4 and 8.
        using var q = Mem.F64(1);
        using var kc = Mem.F64(100, 0, Math.Log(3), 0);
        using var vc = Mem.F64(1000, 4, 8, 0);
        using var o = Mem.Zeros(DType.F64, 1);
        using var qi = Mem.I32(0, 1);
        using var ki = Mem.I32(0, 2);
        using var idx = Mem.I32(0, 1);
        using var last = Mem.I32(1);
        CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = q.Span, Output = o.Span, KCache = kc.Span, VCache = vc.Span, QIndptr = qi.Span, KvIndptr = ki.Span, KvIndices = idx.Span, KvLastPageLength = last.Span,
            Mode = AttentionMode.Decode, Batch = 1, Tokens = 1, QHeads = 1, KvHeads = 1, HeadDim = 1, BlockSize = 2, KvLayout = TensorLayout.PagedHnd, Scale = 1f,
            WindowLeft = 1, Causal = true, KvDType = DType.F64, DType = DType.F64,
        });
        Assert.Equal((0.25 * 4) + (0.75 * 8), o.ReadF64()[0], 1e-13);
    }

    [Fact]
    public void GroupedQueryHeadsShareTheirKvHeadsWindow()
    {
        // Four query heads over two KV heads, width 1, q = 0. KV head 0 holds j + 1 and KV head 1 holds 10 (j + 1); three keys,
        // window 1: positions 1 and 2 are visible, so heads 0 and 1 give 2.5 and heads 2 and 3 give 25.
        using var q = Mem.Zeros(DType.F64, 4);
        using var kc = Mem.Zeros(DType.F64, 4 * 2);
        using var vc = Mem.F64(1, 10, 2, 20, 3, 30, 0, 0); // NHD, block size 4: [slot][kv head]
        using var o = Mem.Zeros(DType.F64, 4);
        using var qi = Mem.I32(0, 1);
        using var ki = Mem.I32(0, 1);
        using var idx = Mem.I32(0);
        using var last = Mem.I32(3);
        CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = q.Span, Output = o.Span, KCache = kc.Span, VCache = vc.Span, QIndptr = qi.Span, KvIndptr = ki.Span, KvIndices = idx.Span, KvLastPageLength = last.Span,
            Mode = AttentionMode.Decode, Batch = 1, Tokens = 1, QHeads = 4, KvHeads = 2, HeadDim = 1, BlockSize = 4, KvLayout = TensorLayout.PagedNhd,
            WindowLeft = 1, Causal = true, KvDType = DType.F64, DType = DType.F64,
        });
        Near.Eq([2.5, 2.5, 25, 25], o.ReadF64(), 1e-13);
    }

    [Fact]
    public void AReleasedPageOutsideEveryWindowIsNeverReadAndOneInsideIsRejected()
    {
        // Ten keys in pages of 2 (KvIndices entries 0..4), a chunk of 3 queries (positions 7..9), window 3: the lowest key read is
        // 4, so pages 0 and 1 (keys 0..3) may be released (-1) and page 2 (keys 4..5) may not.
        using var p = Paged.Build(new Random(7), [new Sequence(10, 3)], qHeads: 2, kvHeads: 1, hd: 3, bs: 2, TensorLayout.PagedHnd);
        double[] full = p.Run(3, causal: true);
        Assert.Equal(full, p.Run(3, causal: true, released: [0, 1]));

        var ex = Assert.Throws<ArgumentException>(() => p.Run(3, causal: true, released: [2]));
        Assert.Contains("PagedAttention", ex.Message, StringComparison.Ordinal);
        Assert.Contains("released", ex.Message, StringComparison.Ordinal);
        Assert.All(p.LastOutput, x => Assert.Equal(Paged.Untouched, x));
        // Without a window every page is read, so even page 0 released is rejected.
        Assert.Throws<ArgumentException>(() => p.Run(-1, causal: true, released: [0]));
    }

    [Fact]
    public void ARejectedSequenceLaterInTheBatchLeavesTheEarlierSequencesOutputUntouched()
    {
        // Window 2. Sequence 0 (5 keys, 1 query, entries 0..2) reads from key 2 (page 1), so entry 0 may go; sequence 1 (9 keys, 2
        // queries, entries 3..7) reads from key 5 (its page 2, entry 5), which is released here.
        using var p = Paged.Build(new Random(11), [new Sequence(5, 1), new Sequence(9, 2)], qHeads: 1, kvHeads: 1, hd: 2, bs: 2, TensorLayout.PagedNhd);
        Assert.Throws<ArgumentException>(() => p.Run(2, causal: true, released: [0, 5]));
        Assert.All(p.LastOutput, x => Assert.Equal(Paged.Untouched, x));
        // Entries 3 and 4 (keys 0..3 of sequence 1) are outside both of its queries' windows.
        Assert.Equal(p.Run(2, causal: true), p.Run(2, causal: true, released: [0, 3, 4]));
    }

    [Fact]
    public void RandomizedWindowsMatchADenseDefinition()
    {
        var rng = new Random(20261101);
        (int Q, int Kv)[] heads = [(1, 1), (2, 1), (4, 2), (8, 2), (3, 3)];
        int[] blockSizes = [1, 2, 3, 4, 16];
        for (int trial = 0; trial < 300; trial++)
        {
            (int qh, int kvh) = heads[rng.Next(heads.Length)];
            int hd = rng.Next(1, 9);
            int bs = blockSizes[rng.Next(blockSizes.Length)];
            bool causal = rng.Next(4) != 0;
            var layout = rng.Next(2) == 0 ? TensorLayout.PagedNhd : TensorLayout.PagedHnd;
            var seqs = Enumerable.Range(0, rng.Next(1, 4)).Select(_ =>
            {
                int kv = rng.Next(0, 40);
                int n = kv == 0 ? 0 : rng.Next(3) == 0 ? 1 : rng.Next(1, kv + 1);
                return new Sequence(kv, n);
            }).ToArray();
            int maxKv = seqs.Max(s => s.KvLen);
            int window = rng.Next(5) switch
            {
                0 => -1,
                1 => 1,
                2 => maxKv + rng.Next(0, 4), // at least the context
                _ => rng.Next(1, Math.Max(2, maxKv)),
            };

            using var p = Paged.Build(rng, seqs, qh, kvh, hd, bs, layout);
            // Release a random number of the leading pages that no query of the sequence can read.
            var released = new List<int>();
            for (int b = 0; b < seqs.Length; b++)
            {
                int firstKey = window < 0 || seqs[b].QLen == 0 ? 0 : Math.Max(0, seqs[b].KvLen - seqs[b].QLen - window);
                int releasable = seqs[b].QLen == 0 ? 0 : firstKey / bs;
                released.AddRange(Enumerable.Range(p.FirstEntry(b), rng.Next(0, releasable + 1)));
            }

            float scale = rng.Next(2) == 0 ? 0f : (float)(0.5 + rng.NextDouble());
            float cap = rng.Next(3) == 0 ? (float)(2.0 + rng.NextDouble()) : 0f;
            double[] got = p.Run(window, causal, [.. released], scale, cap);
            double[] want = p.Dense(window, causal, scale == 0f ? 1.0 / Math.Sqrt(hd) : scale, cap);
            Assert.True(want.Length == got.Length, $"trial {trial}");
            for (int i = 0; i < want.Length; i++)
            {
                Assert.True(Math.Abs(want[i] - got[i]) <= 1e-12, $"trial {trial} window {window} causal {causal} bs {bs} heads {qh}/{kvh} [{i}]: {want[i]:R} vs {got[i]:R}");
            }
        }
    }

    [Fact]
    public void AWindowAtLeastAsLongAsTheContextIsBitIdenticalToNoWindow()
    {
        using var p = Paged.Build(new Random(5), [new Sequence(13, 4), new Sequence(30, 1), new Sequence(7, 7)], 4, 2, 8, 4, TensorLayout.PagedNhd);
        double[] none = p.Run(-1, causal: true);
        Assert.Equal(none, p.Run(29, causal: true));
        Assert.Equal(none, p.Run(int.MaxValue, causal: true));
        Assert.NotEqual(none, p.Run(5, causal: true));
    }

    private sealed record Sequence(int KvLen, int QLen);

    /// <summary>Sequences with dense K and V (rows are positions, columns kvHeads * hd) laid out in a pool of shuffled blocks.</summary>
    private sealed class Paged : IDisposable
    {
        public const double Untouched = -7.0;

        private readonly Bag _bag = new();
        private readonly Sequence[] _seqs;
        private readonly double[][] _q;
        private readonly double[][,] _k;
        private readonly double[][,] _v;
        private readonly int[] _indices;
        private readonly int[] _kvIndptr;
        private readonly int _qh;
        private readonly int _kvh;
        private readonly int _hd;
        private readonly int _bs;
        private readonly TensorLayout _layout;
        private readonly Mem _kPool;
        private readonly Mem _vPool;
        private readonly Mem _qMem;
        private readonly Mem _qIndptr;
        private readonly Mem _kvIndptrMem;
        private readonly Mem _last;

        private Paged(Random rng, Sequence[] seqs, int qHeads, int kvHeads, int hd, int bs, TensorLayout layout)
        {
            (_seqs, _qh, _kvh, _hd, _bs, _layout) = (seqs, qHeads, kvHeads, hd, bs, layout);
            double R() => (rng.NextDouble() * 2) - 1;
            _q = Enumerable.Range(0, Tokens).Select(_ => Enumerable.Range(0, qHeads * hd).Select(_ => R()).ToArray()).ToArray();
            _k = seqs.Select(s => new double[s.KvLen, kvHeads * hd]).ToArray();
            _v = seqs.Select(s => new double[s.KvLen, kvHeads * hd]).ToArray();
            foreach (var arr in _k.Concat(_v))
            {
                for (int j = 0; j < arr.GetLength(0); j++)
                {
                    for (int c = 0; c < arr.GetLength(1); c++)
                    {
                        arr[j, c] = R();
                    }
                }
            }

            int[] pagesPer = seqs.Select(s => (s.KvLen + bs - 1) / bs).ToArray();
            int total = pagesPer.Sum();
            int poolBlocks = total + 2;
            _indices = Enumerable.Range(0, poolBlocks).OrderBy(_ => rng.Next()).Take(total).ToArray();
            _kvIndptr = new int[seqs.Length + 1];
            var qIndptr = new int[seqs.Length + 1];
            var last = new int[seqs.Length];
            for (int b = 0; b < seqs.Length; b++)
            {
                _kvIndptr[b + 1] = _kvIndptr[b] + pagesPer[b];
                qIndptr[b + 1] = qIndptr[b] + seqs[b].QLen;
                last[b] = pagesPer[b] == 0 ? 1 : seqs[b].KvLen - ((pagesPer[b] - 1) * bs);
            }

            // Pool rows that hold no key are NaN, so a read of a wrong block shows in the output.
            var kPool = Enumerable.Repeat(double.NaN, poolBlocks * bs * kvHeads * hd).ToArray();
            var vPool = (double[])kPool.Clone();
            for (int b = 0; b < seqs.Length; b++)
            {
                for (int j = 0; j < seqs[b].KvLen; j++)
                {
                    int blk = _indices[_kvIndptr[b] + (j / bs)];
                    int off = j % bs;
                    for (int h = 0; h < kvHeads; h++)
                    {
                        for (int d = 0; d < hd; d++)
                        {
                            int i = layout == TensorLayout.PagedNhd
                                ? (((((blk * bs) + off) * kvHeads) + h) * hd) + d
                                : (((((blk * kvHeads) + h) * bs) + off) * hd) + d;
                            kPool[i] = _k[b][j, (h * hd) + d];
                            vPool[i] = _v[b][j, (h * hd) + d];
                        }
                    }
                }
            }

            _kPool = _bag.Add(Mem.F64(kPool));
            _vPool = _bag.Add(Mem.F64(vPool));
            _qMem = _bag.Add(Mem.F64(_q.SelectMany(x => x).ToArray()));
            _qIndptr = _bag.Add(Mem.I32(qIndptr));
            _kvIndptrMem = _bag.Add(Mem.I32(_kvIndptr));
            _last = _bag.Add(Mem.I32(last));
        }

        /// <summary>The whole output buffer of the last <see cref="Run"/>, also when it threw.</summary>
        public double[] LastOutput { get; private set; } = [];

        private int Tokens => _seqs.Sum(s => s.QLen);

        public static Paged Build(Random rng, Sequence[] seqs, int qHeads, int kvHeads, int hd, int bs, TensorLayout layout)
            => new(rng, seqs, qHeads, kvHeads, hd, bs, layout);

        public int FirstEntry(int b) => _kvIndptr[b];

        /// <summary>Runs CpuOps with the KvIndices entries at <paramref name="released"/> set to -1 (released out of the window).</summary>
        public double[] Run(int window, bool causal, int[]? released = null, float scale = 0f, float cap = 0f)
        {
            int[] idx = (int[])_indices.Clone();
            foreach (int e in released ?? [])
            {
                idx[e] = -1;
            }

            using var idxMem = Mem.I32(idx.Length == 0 ? [0] : idx);
            using var o = Mem.F64(Enumerable.Repeat(Untouched, Math.Max(1, Tokens * _qh * _hd)).ToArray());
            try
            {
                CpuOps.PagedAttention(new PagedAttentionArgs
                {
                    Q = _qMem.Span, Output = o.Span, KCache = _kPool.Span, VCache = _vPool.Span, QIndptr = _qIndptr.Span, KvIndptr = _kvIndptrMem.Span,
                    KvIndices = idxMem.Span, KvLastPageLength = _last.Span, Mode = AttentionMode.Prefill, Batch = _seqs.Length, Tokens = Tokens,
                    QHeads = _qh, KvHeads = _kvh, HeadDim = _hd, BlockSize = _bs, KvLayout = _layout, Scale = scale, SoftCap = cap, WindowLeft = window,
                    Causal = causal, KvDType = DType.F64, DType = DType.F64,
                });
            }
            finally
            {
                LastOutput = o.ReadF64();
            }

            return LastOutput.Take(Tokens * _qh * _hd).ToArray();
        }

        /// <summary>The definition, written out per query: the visible set, then a two-pass softmax.</summary>
        public double[] Dense(int window, bool causal, double scale, double cap)
        {
            var o = new List<double>();
            int qi = 0;
            for (int b = 0; b < _seqs.Length; b++)
            {
                int kvLen = _seqs[b].KvLen;
                for (int i = 0; i < _seqs[b].QLen; i++, qi++)
                {
                    int pos = kvLen - _seqs[b].QLen + i;
                    int[] visible = Enumerable.Range(0, kvLen).Where(j => (!causal || j <= pos) && (window < 0 || pos - j <= window)).ToArray();
                    for (int h = 0; h < _qh; h++)
                    {
                        int g = h / (_qh / _kvh);
                        double[] s = visible.Select(j =>
                        {
                            double dot = 0;
                            for (int d = 0; d < _hd; d++)
                            {
                                dot += _q[qi][(h * _hd) + d] * _k[b][j, (g * _hd) + d];
                            }

                            dot *= scale;
                            return cap != 0 ? cap * Math.Tanh(dot / cap) : dot;
                        }).ToArray();
                        double mx = s.Length == 0 ? 0 : s.Max();
                        double z = s.Sum(x => Math.Exp(x - mx));
                        for (int d = 0; d < _hd; d++)
                        {
                            double acc = 0;
                            for (int t = 0; t < visible.Length; t++)
                            {
                                acc += Math.Exp(s[t] - mx) / z * _v[b][visible[t], (g * _hd) + d];
                            }

                            o.Add(acc);
                        }
                    }
                }
            }

            return o.ToArray();
        }

        public void Dispose() => _bag.Dispose();
    }
}

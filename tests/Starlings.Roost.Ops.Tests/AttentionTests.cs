using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

public sealed class AttentionTests
{
    // Independent reference: dense per-sequence K and V, three-pass softmax, no online rescaling.
    private static double[] Dense(double[][] q, double[][,] k, double[][,] v, int[] qLens, int qHeads, int kvHeads, int hd, bool causal, int window, double scale, double cap)
    {
        var o = new List<double>();
        int qi = 0;
        for (int b = 0; b < qLens.Length; b++)
        {
            int kvLen = k[b].GetLength(0);
            for (int i = 0; i < qLens[b]; i++, qi++)
            {
                int qpos = kvLen - qLens[b] + i;
                for (int h = 0; h < qHeads; h++)
                {
                    int kvh = h / (qHeads / kvHeads);
                    var scores = new double[kvLen];
                    var ok = new bool[kvLen];
                    for (int j = 0; j < kvLen; j++)
                    {
                        ok[j] = (!causal || j <= qpos) && (window < 0 || j >= qpos - window);
                        double s = 0;
                        for (int d = 0; d < hd; d++)
                        {
                            s += q[qi][(h * hd) + d] * k[b][j, (kvh * hd) + d];
                        }

                        s *= scale;
                        scores[j] = cap != 0 ? cap * Math.Tanh(s / cap) : s;
                    }

                    double mx = Enumerable.Range(0, kvLen).Where(j => ok[j]).Select(j => scores[j]).DefaultIfEmpty(0).Max();
                    double z = Enumerable.Range(0, kvLen).Where(j => ok[j]).Sum(j => Math.Exp(scores[j] - mx));
                    for (int d = 0; d < hd; d++)
                    {
                        double acc = 0;
                        for (int j = 0; j < kvLen; j++)
                        {
                            if (ok[j])
                            {
                                acc += Math.Exp(scores[j] - mx) / z * v[b][j, (kvh * hd) + d];
                            }
                        }

                        o.Add(acc);
                    }
                }
            }
        }

        return o.ToArray();
    }

    [Theory]
    [InlineData(TensorLayout.PagedNhd, false, -1, 0.0, 0.0)]
    [InlineData(TensorLayout.PagedHnd, false, -1, 0.0, 0.0)]
    [InlineData(TensorLayout.PagedNhd, true, -1, 0.0, 0.0)]
    [InlineData(TensorLayout.PagedHnd, true, -1, 0.0, 0.0)]
    [InlineData(TensorLayout.PagedNhd, true, 3, 0.0, 0.0)]
    [InlineData(TensorLayout.PagedHnd, true, 3, 0.0, 0.0)]
    [InlineData(TensorLayout.PagedNhd, true, -1, 0.0, 5.0)]
    [InlineData(TensorLayout.PagedHnd, true, 2, 0.37, 3.0)]
    [InlineData(TensorLayout.PagedNhd, false, 4, 0.0, 0.0)]
    public void PrefillAndDecodeMatchADenseReferenceThroughShuffledBlocks(TensorLayout layout, bool causal, int window, double scaleArg, double cap)
    {
        const int qHeads = 4;
        const int kvHeads = 2;
        const int hd = 8;
        const int bs = 4;
        // Two sequences: 7 keys with 3 queries (prefill chunk) and 10 keys with 1 query (decode).
        int[] kvLens = [7, 10];
        int[] qLens = [3, 1];
        var rng = new Random(1234);
        double R() => (rng.NextDouble() * 2) - 1;
        var q = qLens.SelectMany((n, b) => Enumerable.Range(0, n)).Select(_ => Enumerable.Range(0, qHeads * hd).Select(_ => R()).ToArray()).ToArray();
        var k = kvLens.Select(n => new double[n, kvHeads * hd]).ToArray();
        var v = kvLens.Select(n => new double[n, kvHeads * hd]).ToArray();
        foreach (var arr in k.Concat(v))
        {
            for (int j = 0; j < arr.GetLength(0); j++)
            {
                for (int c = 0; c < arr.GetLength(1); c++)
                {
                    arr[j, c] = R();
                }
            }
        }

        // Pool of 8 blocks; the sequences use shuffled block ids.
        int[][] table = [[5, 2], [7, 0, 3]];
        var kPool = new double[8 * bs * kvHeads * hd];
        var vPool = new double[kPool.Length];
        for (int b = 0; b < 2; b++)
        {
            for (int j = 0; j < kvLens[b]; j++)
            {
                int blk = table[b][j / bs];
                int off = j % bs;
                for (int h = 0; h < kvHeads; h++)
                {
                    for (int d = 0; d < hd; d++)
                    {
                        int idx = layout == TensorLayout.PagedNhd
                            ? ((((blk * bs) + off) * kvHeads) + h) * hd + d
                            : ((((blk * kvHeads) + h) * bs) + off) * hd + d;
                        kPool[idx] = k[b][j, (h * hd) + d];
                        vPool[idx] = v[b][j, (h * hd) + d];
                    }
                }
            }
        }

        using var qm = Mem.F64(q.SelectMany(x => x).ToArray());
        using var km = Mem.F64(kPool);
        using var vm = Mem.F64(vPool);
        using var o = Mem.Zeros(DType.F64, 4 * qHeads * hd);
        using var qIndptr = Mem.I32(0, 3, 4);
        using var kvIndptr = Mem.I32(0, 2, 5);
        using var kvIndices = Mem.I32(5, 2, 7, 0, 3);
        using var last = Mem.I32(3, 2);
        CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = qm.Span, Output = o.Span, KCache = km.Span, VCache = vm.Span, QIndptr = qIndptr.Span, KvIndptr = kvIndptr.Span, KvIndices = kvIndices.Span,
            KvLastPageLength = last.Span, Mode = AttentionMode.Prefill, Batch = 2, Tokens = 4, QHeads = qHeads, KvHeads = kvHeads, HeadDim = hd, BlockSize = bs,
            KvLayout = layout, Scale = (float)scaleArg, SoftCap = (float)cap, WindowLeft = window, Causal = causal, KvDType = DType.F64, DType = DType.F64,
        });
        double scale = scaleArg == 0 ? 1.0 / Math.Sqrt(hd) : (float)scaleArg;
        Near.Eq(Dense(q, k, v, qLens, qHeads, kvHeads, hd, causal, window, scale, (float)cap), o.ReadF64(), 1e-12);
    }

    [Fact]
    public void HandComputedTwoKeyAttention()
    {
        // One head of width 1, scale 1: scores 0 and ln 3, so weights 1/4 and 3/4.
        using var q = Mem.F64(1);
        using var kc = Mem.F64(0, Math.Log(3));
        using var vc = Mem.F64(4, 8);
        using var o = Mem.Zeros(DType.F64, 1);
        using var qi = Mem.I32(0, 1);
        using var ki = Mem.I32(0, 1);
        using var idx = Mem.I32(0);
        using var last = Mem.I32(2);
        CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = q.Span, Output = o.Span, KCache = kc.Span, VCache = vc.Span, QIndptr = qi.Span, KvIndptr = ki.Span, KvIndices = idx.Span, KvLastPageLength = last.Span,
            Mode = AttentionMode.Decode, Batch = 1, Tokens = 1, QHeads = 1, KvHeads = 1, HeadDim = 1, BlockSize = 2, KvLayout = TensorLayout.PagedNhd, Scale = 1f,
            WindowLeft = -1, Causal = true, KvDType = DType.F64, DType = DType.F64,
        });
        Assert.Equal((0.25 * 4) + (0.75 * 8), o.ReadF64()[0], 1e-13);
    }

    [Fact]
    public void ABf16CacheIsReadWithoutIntermediateRoundingOfTheResultUntilTheEnd()
    {
        using var q = Mem.Bf16(1, 0);
        using var kc = Mem.Bf16(1, 0, 1, 0);
        using var vc = Mem.Bf16(1, 2, 3, 4);
        using var o = Mem.Zeros(DType.BF16, 2);
        using var qi = Mem.I32(0, 1);
        using var ki = Mem.I32(0, 1);
        using var idx = Mem.I32(0);
        using var last = Mem.I32(2);
        CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = q.Span, Output = o.Span, KCache = kc.Span, VCache = vc.Span, QIndptr = qi.Span, KvIndptr = ki.Span, KvIndices = idx.Span, KvLastPageLength = last.Span,
            Mode = AttentionMode.Decode, Batch = 1, Tokens = 1, QHeads = 1, KvHeads = 1, HeadDim = 2, BlockSize = 2, KvLayout = TensorLayout.PagedNhd,
            WindowLeft = -1, Causal = true, KvDType = DType.BF16, DType = DType.BF16,
        });
        // Equal scores: the mean of the two value rows.
        Assert.Equal([2f, 3f], o.ReadBf16());
    }

    [Fact]
    public void AWindowOfOneAndNoPagesGiveZeros()
    {
        using var q = Mem.F64(1);
        using var kc = Mem.F64(1, 1);
        using var vc = Mem.F64(5, 5);
        using var o = Mem.F64(99);
        using var qi = Mem.I32(0, 1);
        using var ki = Mem.I32(0, 1);
        using var idx = Mem.I32(0);
        using var last = Mem.I32(2);
        // Window 1 keeps the query's own key and the one before it (both hold 5 here).
        CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = q.Span, Output = o.Span, KCache = kc.Span, VCache = vc.Span, QIndptr = qi.Span, KvIndptr = ki.Span, KvIndices = idx.Span, KvLastPageLength = last.Span,
            Mode = AttentionMode.Decode, Batch = 1, Tokens = 1, QHeads = 1, KvHeads = 1, HeadDim = 1, BlockSize = 2, KvLayout = TensorLayout.PagedNhd,
            WindowLeft = 1, Causal = true, KvDType = DType.F64, DType = DType.F64,
        });
        Assert.Equal(5.0, o.ReadF64()[0], 1e-13);
        // No pages at all: zero keys, zeros out.
        using var empty = Mem.I32(0, 0);
        CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = q.Span, Output = o.Span, KCache = kc.Span, VCache = vc.Span, QIndptr = qi.Span, KvIndptr = empty.Span, KvIndices = idx.Span, KvLastPageLength = last.Span,
            Mode = AttentionMode.Decode, Batch = 1, Tokens = 1, QHeads = 1, KvHeads = 1, HeadDim = 1, BlockSize = 2, KvLayout = TensorLayout.PagedNhd,
            WindowLeft = -1, Causal = false, KvDType = DType.F64, DType = DType.F64,
        });
        Assert.Equal(0.0, o.ReadF64()[0]);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(0)]
    [InlineData(-1)]
    public void AnOutOfRangeKvLastPageLengthThrows(int lastLen)
    {
        using var q = Mem.F64(1);
        using var kc = Mem.F64(1, 1);
        using var vc = Mem.F64(5, 5);
        using var o = Mem.F64(99);
        using var qi = Mem.I32(0, 1);
        using var ki = Mem.I32(0, 1);
        using var idx = Mem.I32(0);
        using var last = Mem.I32(lastLen);
        Assert.Throws<ArgumentException>(() => CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = q.Span, Output = o.Span, KCache = kc.Span, VCache = vc.Span, QIndptr = qi.Span, KvIndptr = ki.Span, KvIndices = idx.Span, KvLastPageLength = last.Span,
            Mode = AttentionMode.Decode, Batch = 1, Tokens = 1, QHeads = 1, KvHeads = 1, HeadDim = 1, BlockSize = 2, KvLayout = TensorLayout.PagedNhd,
            WindowLeft = -1, Causal = true, KvDType = DType.F64, DType = DType.F64,
        }));
        Assert.Equal(99.0, o.ReadF64()[0]);
    }

    [Fact]
    public void AWindowOfZeroThrowsNamingTheOpAndLeavesTheOutputUntouched()
    {
        using var q = Mem.F64(1);
        using var kc = Mem.F64(1, 1);
        using var vc = Mem.F64(5, 5);
        using var o = Mem.F64(99);
        using var qi = Mem.I32(0, 1);
        using var ki = Mem.I32(0, 1);
        using var idx = Mem.I32(0);
        using var last = Mem.I32(2);
        var ex = Assert.Throws<ArgumentException>(() => CpuOps.PagedAttention(new PagedAttentionArgs
        {
            Q = q.Span, Output = o.Span, KCache = kc.Span, VCache = vc.Span, QIndptr = qi.Span, KvIndptr = ki.Span, KvIndices = idx.Span, KvLastPageLength = last.Span,
            Mode = AttentionMode.Decode, Batch = 1, Tokens = 1, QHeads = 1, KvHeads = 1, HeadDim = 1, BlockSize = 2, KvLayout = TensorLayout.PagedNhd,
            WindowLeft = 0, Causal = true, KvDType = DType.F64, DType = DType.F64,
        }));
        Assert.Contains("PagedAttention", ex.Message, StringComparison.Ordinal);
        Assert.Contains("WindowLeft", ex.Message, StringComparison.Ordinal);
        Assert.Equal(99.0, o.ReadF64()[0]);
    }
}

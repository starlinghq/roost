using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

public sealed class BasicOpsTests
{
    private const double Tight = 1e-13;

    [Fact]
    public void RmsNormMatchesHandComputedRows()
    {
        using var x = Mem.F64(3, 4, 6, 8);
        using var w = Mem.F64(1, 2);
        using var y = Mem.Zeros(DType.F64, 4);
        CpuOps.RmsNorm(new RmsNormArgs { Input = x.Span, Weight = w.Span, Output = y.Span, Rows = 2, Dim = 2, Eps = 0f, DType = DType.F64 });
        double inv = 1.0 / Math.Sqrt(12.5);
        Near.Eq([3 * inv, 8 * inv, 3 * inv, 8 * inv], y.ReadF64(), Tight);
    }

    [Fact]
    public void RmsNormEpsIsInsideTheRootAndBf16OutputIsRoundedOnce()
    {
        using var x = Mem.Bf16(1, 1, 1, 1);
        using var w = Mem.Bf16(1, 1, 1, 1);
        using var y = Mem.Zeros(DType.BF16, 4);
        CpuOps.RmsNorm(new RmsNormArgs { Input = x.Span, Weight = w.Span, Output = y.Span, Rows = 1, Dim = 4, Eps = 3f, DType = DType.BF16 });
        // 1 / sqrt(1 + 3) = 0.5 exactly.
        Assert.Equal([0.5f, 0.5f, 0.5f, 0.5f], y.ReadBf16());
        // Input aliasing Output is allowed.
        CpuOps.RmsNorm(new RmsNormArgs { Input = x.Span, Weight = w.Span, Output = x.Span, Rows = 1, Dim = 4, Eps = 3f, DType = DType.BF16 });
        Assert.Equal(y.ReadBf16(), x.ReadBf16());
    }

    [Fact]
    public void QkNormNormalisesEachHeadWithItsOwnWeightInPlace()
    {
        // One token, 2 query heads and 1 key head of width 2.
        using var q = Mem.F64(3, 4, 6, 8);
        using var k = Mem.F64(0, 5);
        using var qw = Mem.F64(1, 1);
        using var kw = Mem.F64(2, 3);
        CpuOps.QkNorm(new QkNormArgs { Q = q.Span, K = k.Span, QWeight = qw.Span, KWeight = kw.Span, Tokens = 1, QHeads = 2, KvHeads = 1, HeadDim = 2, Eps = 0f, DType = DType.F64 });
        double r = 1.0 / Math.Sqrt(12.5);
        double r2 = 1.0 / Math.Sqrt(50);
        Near.Eq([3 * r, 4 * r, 6 * r2 * 1.0 * 1.0, 8 * r2 * 1.0], q.ReadF64(), Tight);
        Near.Eq([0.0, 5 * (1.0 / Math.Sqrt(12.5)) * 3], k.ReadF64(), Tight);
    }

    private static double[] RunRope(double[] q, int pos, int headDim, int rotaryDim, bool interleaved, double theta, RopeKind kind = RopeKind.Default, float factor = 0, float low = 0, float high = 0, int orig = 0)
    {
        using var qm = Mem.F64(q);
        using var km = Mem.F64(q);
        using var p = Mem.I32(pos);
        CpuOps.Rope(new RopeArgs
        {
            Q = qm.Span, K = km.Span, Positions = p.Span, Tokens = 1, QHeads = 1, KvHeads = 1, HeadDim = headDim, RotaryDim = rotaryDim,
            Kind = kind, Interleaved = interleaved, Theta = (float)theta, Llama3Factor = factor, Llama3LowFreqFactor = low, Llama3HighFreqFactor = high,
            Llama3OriginalMaxPosition = orig, DType = DType.F64,
        });
        Assert.Equal(qm.ReadF64(), km.ReadF64());
        return qm.ReadF64();
    }

    [Fact]
    public void RopeNeoxRotatesHalvesByHandComputedAngles()
    {
        // HeadDim 4, theta 100: frequencies 1 and 0.1; pairs (0,2) and (1,3); position 2.
        double[] y = RunRope([1, 2, 3, 4], 2, 4, 0, false, 100);
        double a0 = 2.0;
        double a1 = 0.2;
        Near.Eq(
            [Math.Cos(a0) - (3 * Math.Sin(a0)), (2 * Math.Cos(a1)) - (4 * Math.Sin(a1)), (3 * Math.Cos(a0)) + Math.Sin(a0), (4 * Math.Cos(a1)) + (2 * Math.Sin(a1))],
            y,
            Tight);
    }

    [Fact]
    public void RopeInterleavedPairsNeighbours()
    {
        double[] y = RunRope([1, 2, 3, 4], 2, 4, 0, true, 100);
        double a0 = 2.0;
        double a1 = 0.2;
        Near.Eq(
            [Math.Cos(a0) - (2 * Math.Sin(a0)), (2 * Math.Cos(a0)) + Math.Sin(a0), (3 * Math.Cos(a1)) - (4 * Math.Sin(a1)), (4 * Math.Cos(a1)) + (3 * Math.Sin(a1))],
            y,
            Tight);
    }

    [Fact]
    public void RopePartialRotaryLeavesTheTailAndPositionZeroIsIdentity()
    {
        double[] y = RunRope([1, 2, 3, 4], 1, 4, 2, false, 100);
        Assert.Equal(3.0, y[2]);
        Assert.Equal(4.0, y[3]);
        Assert.Equal(Math.Cos(1.0) - (2 * Math.Sin(1.0)), y[0], Tight);
        Assert.Equal([1.0, 2.0, 3.0, 4.0], RunRope([1, 2, 3, 4], 0, 4, 0, false, 100));
    }

    [Fact]
    public void RopeLlama3ScalesOnlyLowFrequencies()
    {
        // HeadDim 8, theta 1e6: f = 1, 1e-1.5, 1e-3, 1e-4.5. Original 8192, low 1, high 4, factor 8.
        // Wavelengths 6.3, 198, 6283, 198692. Keep below 2048; divide by 8 above 8192; blend in between.
        double[] x = [1, 1, 1, 1, 0, 0, 0, 0];
        double[] y = RunRope(x, 1000, 8, 0, false, 1_000_000, RopeKind.Llama3, 8f, 1f, 4f, 8192);
        double theta = 1_000_000;
        double[] f = Enumerable.Range(0, 4).Select(i => Math.Pow((double)(float)theta, -2.0 * i / 8)).ToArray();
        double smooth = ((8192.0 / (2 * Math.PI / f[2])) - 1) / (4 - 1);
        double[] scaled = [f[0], f[1], (((1 - smooth) * f[2]) / 8) + (smooth * f[2]), f[3] / 8];
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(Math.Cos(1000 * scaled[i]), y[i], 1e-11);
            Assert.Equal(Math.Sin(1000 * scaled[i]), y[i + 4], 1e-11);
        }
    }

    [Fact]
    public void SiluAndMulHandValues()
    {
        using var x = Mem.F64(1, 2, 0, 7);
        using var y = Mem.Zeros(DType.F64, 2);
        CpuOps.SiluAndMul(new SiluAndMulArgs { Input = x.Span, Output = y.Span, Rows = 2, Dim = 1, DType = DType.F64 });
        Near.Eq([0.7310585786300049 * 2, 0.0], y.ReadF64(), Tight);
    }

    [Fact]
    public void GeluAndMulExactAndTanh()
    {
        using var x = Mem.F64(1, 2, -3, 1);
        using var exact = Mem.Zeros(DType.F64, 2);
        using var tanh = Mem.Zeros(DType.F64, 2);
        CpuOps.GeluAndMul(new GeluAndMulArgs { Input = x.Span, Output = exact.Span, Rows = 2, Dim = 1, Approximation = GeluApproximation.Exact, DType = DType.F64 });
        CpuOps.GeluAndMul(new GeluAndMulArgs { Input = x.Span, Output = tanh.Span, Rows = 2, Dim = 1, Approximation = GeluApproximation.Tanh, DType = DType.F64 });
        // gelu(1) = 0.8413447460685429, gelu(-3) = -0.004050858825143345; tanh form: 0.8411919906082768, -0.003637...
        Assert.Equal(0.8413447460685429 * 2, exact.ReadF64()[0], 1e-12);
        Assert.Equal(-3 * 0.5 * 0.0026997960632601866 * 1.0, exact.ReadF64()[1], 1e-12);
        Assert.Equal(0.8411919906082768 * 2, tanh.ReadF64()[0], 1e-12);
        double g = -3;
        Assert.Equal(0.5 * g * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (g + (0.044715 * g * g * g)))), tanh.ReadF64()[1], 1e-14);
    }

    [Fact]
    public void GemmMatchesHandProductWithAlphaBetaBias()
    {
        // A [2,3], B [3,2] row-major.
        using var a = Mem.F64(1, 2, 3, 4, 5, 6);
        using var b = Mem.F64(7, 8, 9, 10, 11, 12);
        using var c = Mem.F64(1, 1, 1, 1);
        using var bias = Mem.F64(100, 200);
        using var d = Mem.Zeros(DType.F64, 4);
        CpuOps.Gemm(new GemmArgs { A = a.Span, B = b.Span, C = c.Span, D = d.Span, Bias = bias.Span, M = 2, N = 2, K = 3, Alpha = 2f, Beta = 3f, DType = DType.F64, WeightDType = DType.F64 });
        // A@B = [[58, 64], [139, 154]]
        Near.Eq([(2 * 58) + 3 + 100, (2 * 64) + 3 + 200, (2 * 139) + 3 + 100, (2 * 154) + 3 + 200], d.ReadF64(), Tight);
    }

    [Fact]
    public void GemmTransposedBf16WeightsAndNoOptionalInputs()
    {
        using var a = Mem.F32(1, 2, 3);
        using var bt = Mem.Bf16(1, 0, 1, 0, 2, 0); // [N=2, K=3]: rows (1,0,1) and (0,2,0)
        using var d = Mem.Zeros(DType.F32, 2);
        CpuOps.Gemm(new GemmArgs { A = a.Span, B = bt.Span, D = d.Span, M = 1, N = 2, K = 3, TransposeB = true, Alpha = 1f, Beta = 0f, DType = DType.F32, WeightDType = DType.BF16 });
        Assert.Equal([4f, 4f], d.ReadF32());
    }

    [Fact]
    public void EmbeddingCopiesRowsBitExactlyAndRejectsBadIds()
    {
        using var table = Mem.Bf16(1, 2, 3, 4, 5, 6);
        using var ids = Mem.I32(2, 0, 2);
        using var o = Mem.Zeros(DType.BF16, 6);
        CpuOps.Embedding(new EmbeddingArgs { Ids = ids.Span, Table = table.Span, Output = o.Span, Tokens = 3, Dim = 2, Vocab = 3, DType = DType.BF16 });
        Assert.Equal([5f, 6f, 1f, 2f, 5f, 6f], o.ReadBf16());
        using var bad = Mem.I32(3);
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuOps.Embedding(new EmbeddingArgs { Ids = bad.Span, Table = table.Span, Output = o.Span, Tokens = 1, Dim = 2, Vocab = 3, DType = DType.BF16 }));
    }

    [Fact]
    public void LmHeadSelectsRowsSoftCapsAndLeavesPaddingAlone()
    {
        using var hidden = Mem.F64(1, 0, 0, 2);
        using var rows = Mem.I32(1);
        using var w = Mem.F64(1, 1, 0, 3, 5, 5);
        using var logits = Mem.F32(-7, -7, -7, -7);
        CpuOps.LmHead(new LmHeadArgs { Hidden = hidden.Span, Rows = rows.Span, Weight = w.Span, Logits = logits.Span, Selected = 1, Dim = 2, Vocab = 3, VocabPadded = 4, DType = DType.F64 });
        Assert.Equal([2f, 6f, 10f, -7f], logits.ReadF32());
        CpuOps.LmHead(new LmHeadArgs { Hidden = hidden.Span, Rows = rows.Span, Weight = w.Span, Logits = logits.Span, Selected = 1, Dim = 2, Vocab = 3, VocabPadded = 4, SoftCap = 4f, DType = DType.F64 });
        Assert.Equal((float)(4 * Math.Tanh(0.5)), logits.ReadF32()[0]);
        Assert.Equal((float)(4 * Math.Tanh(2.5)), logits.ReadF32()[2]);
        Assert.Equal(-7f, logits.ReadF32()[3]);
    }

    [Fact]
    public void ResidualAddsAndMayAliasTheFirstAddend()
    {
        using var a = Mem.F64(1, 2, 3);
        using var b = Mem.F64(10, 20, 30);
        CpuOps.Residual(new ResidualArgs { A = a.Span, B = b.Span, Output = a.Span, Elements = 3, DType = DType.F64 });
        Assert.Equal([11.0, 22.0, 33.0], a.ReadF64());
    }

    [Theory]
    [InlineData(TensorLayout.PagedNhd)]
    [InlineData(TensorLayout.PagedHnd)]
    public void CacheWriteScattersTokensAndSkipsNegativeSlots(TensorLayout layout)
    {
        // 3 tokens, 2 kv heads, head dim 2, block size 2, 2 blocks.
        using var k = Mem.Bf16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        using var v = Mem.Bf16(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12);
        using var kc = Mem.Zeros(DType.BF16, 16);
        using var vc = Mem.Zeros(DType.BF16, 16);
        using var slots = Mem.I64(3, -1, 0);
        CpuOps.CacheWrite(new CacheWriteArgs { K = k.Span, V = v.Span, KCache = kc.Span, VCache = vc.Span, SlotMapping = slots.Span, Tokens = 3, KvHeads = 2, HeadDim = 2, BlockSize = 2, KvLayout = layout, KScale = 1f, VScale = 1f, KvDType = DType.BF16 });
        float[] got = kc.ReadBf16();
        var expected = new float[16];
        // Token 0 -> slot 3 (block 1, offset 1); token 2 -> slot 0 (block 0, offset 0); token 1 skipped.
        for (int h = 0; h < 2; h++)
        {
            for (int d = 0; d < 2; d++)
            {
                int Index(int block, int off) => layout == TensorLayout.PagedNhd
                    ? ((((block * 2) + off) * 2) + h) * 2 + d
                    : ((((block * 2) + h) * 2) + off) * 2 + d;
                expected[Index(1, 1)] = (h * 2) + d + 1;
                expected[Index(0, 0)] = 8 + (h * 2) + d + 1;
            }
        }

        Assert.Equal(expected, got);
        Assert.Equal(expected.Select(x => -x).ToArray(), vc.ReadBf16().Select(x => x == 0 ? 0f : x).ToArray());
    }

    [Fact]
    public void CacheWriteQuantisesByTheScaleWhenTheTypesDiffer()
    {
        using var k = Mem.F32(3, 5);
        using var v = Mem.F32(7, 9);
        using var kc = Mem.Zeros(DType.F16, 2);
        using var vc = Mem.Zeros(DType.F16, 2);
        using var slots = Mem.I64(0);
        CpuOps.CacheWrite(new CacheWriteArgs { K = k.Span, V = v.Span, KCache = kc.Span, VCache = vc.Span, SlotMapping = slots.Span, Tokens = 1, KvHeads = 1, HeadDim = 2, BlockSize = 1, KvLayout = TensorLayout.PagedNhd, KScale = 2f, VScale = 0.5f, KvDType = DType.F16 });
        Assert.Equal([1.5, 2.5], kc.ReadU16().Select(u => (double)BitConverter.UInt16BitsToHalf(u)).ToArray());
        Assert.Equal([14.0, 18.0], vc.ReadU16().Select(u => (double)BitConverter.UInt16BitsToHalf(u)).ToArray());
    }

    [Fact]
    public void CacheCopyCopiesBlocksInPairOrderAndMayOverlapASharedPool()
    {
        using var pool = Mem.F32(1, 2, 3, 4, 5, 6);
        using var pairs = Mem.I32(0, 1, 1, 2);
        CpuOps.CacheCopy(new CacheCopyArgs { Source = pool.Span, Destination = pool.Span, Pairs = pairs.Span, BlockBytes = 8, Count = 2 });
        // Block 1 <- block 0, then block 2 <- (new) block 1.
        Assert.Equal([1f, 2f, 1f, 2f, 1f, 2f], pool.ReadF32());
        using var dst = Mem.F32(0, 0);
        using var p2 = Mem.I32(2, 0);
        CpuOps.CacheCopy(new CacheCopyArgs { Source = pool.Span, Destination = dst.Span, Pairs = p2.Span, BlockBytes = 8, Count = 1 });
        Assert.Equal([1f, 2f], dst.ReadF32());
        using var oob = Mem.I32(3, 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuOps.CacheCopy(new CacheCopyArgs { Source = pool.Span, Destination = dst.Span, Pairs = oob.Span, BlockBytes = 8, Count = 1 }));
    }

    [Fact]
    public void ArgmaxTiesGoToTheLowestIndexAndHonourStrideAndNaN()
    {
        // Stride 4, vocab 3: the fourth column of each row is padding that must be ignored.
        using var logits = Mem.F32(1, 5, 5, 99, float.NaN, -2, -1, 99, float.NaN, float.NaN, float.NaN, 0);
        using var ids = Mem.Zeros(DType.I32, 3);
        CpuOps.Argmax(new ArgmaxArgs { Logits = logits.Span, Ids = ids.Span, Rows = 3, Vocab = 3, Stride = 4 });
        Assert.Equal([1, 2, 0], ids.ReadI32());
        using var flat = Mem.F32(2, 2, 1);
        using var id1 = Mem.Zeros(DType.I32, 1);
        CpuOps.Argmax(new ArgmaxArgs { Logits = flat.Span, Ids = id1.Span, Rows = 1, Vocab = 3 });
        Assert.Equal([0], id1.ReadI32());
    }

    [Fact]
    public void UndersizedSpansThrowBeforeAnyWrite()
    {
        using var x = Mem.F64(1, 2);
        using var w = Mem.F64(1, 1);
        using var y = Mem.F64(-1, -1);
        Assert.Throws<ArgumentException>(() => CpuOps.RmsNorm(new RmsNormArgs { Input = x.Span, Weight = w.Span, Output = y.Span, Rows = 2, Dim = 2, Eps = 0f, DType = DType.F64 }));
        Assert.Equal([-1.0, -1.0], y.ReadF64());
    }
}

using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>Quantize, Dequantize, GemmFp8, GemmW4A16 and the FP8 KV cache path of CpuOps (R-077, R-043; CPU half only).</summary>
public sealed class QuantOpsTests
{
    private static float[] Values(int n, int seed, double scale = 1.0)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)(((rng.NextDouble() * 2) - 1) * scale)).ToArray();
    }

    private static byte[] Packed4(params int[] v)
    {
        var b = new byte[(v.Length + 1) / 2];
        for (int i = 0; i < v.Length; i++)
        {
            b[i / 2] |= (byte)((v[i] & 0xF) << (4 * (i % 2)));
        }

        return b;
    }

    [Theory]
    [InlineData(DType.F8E4M3, 0)]
    [InlineData(DType.F8E4M3, 4)]
    [InlineData(DType.F8E5M2, 0)]
    [InlineData(DType.F8E5M2, 8)]
    public void QuantizeFp8MatchesTheScaleAndConversionRule(DType outType, int groupSize)
    {
        const int rows = 3, dim = 16;
        float[] x = Values(rows * dim, 7, 30);
        x[5] = 0f;
        using var input = Mem.F32(x);
        using var output = Mem.Zeros(outType, rows * dim);
        int groups = groupSize == 0 ? 1 : dim / groupSize;
        using var scales = Mem.Zeros(DType.F32, rows * groups);
        CpuOps.Quantize(new QuantizeArgs { Input = input.Span, Output = output.Span, Scales = scales.Span, Rows = rows, Dim = dim, GroupSize = groupSize, DType = DType.F32, OutputDType = outType });

        byte[] q = output.ReadBytes();
        float[] s = scales.ReadF32();
        int g = dim / groups;
        for (int r = 0; r < rows; r++)
        {
            for (int gi = 0; gi < groups; gi++)
            {
                double amax = Enumerable.Range(0, g).Max(i => Math.Abs((double)x[(r * dim) + (gi * g) + i]));
                float want = (float)(amax / Fp8.MaxOf(outType));
                Assert.Equal(want, s[(r * groups) + gi]);
                for (int i = 0; i < g; i++)
                {
                    int at = (r * dim) + (gi * g) + i;
                    Assert.Equal(Fp8.Encode(x[at] / (double)want, outType, saturate: true), q[at]);
                }
            }
        }

        // The group's largest magnitude lands on (or, after the scale's rounding, saturates to) the format's largest value.
        Assert.Contains(q, b => (b & 0x7F) == (outType == DType.F8E4M3 ? 0x7E : 0x7B));
    }

    [Fact]
    public void QuantizeInt4IsSymmetricAndPackedLowNibbleFirst()
    {
        using var input = Mem.F32(0.7f, -1.4f, 3.5f, 0f);
        using var output = Mem.Zeros(DType.I4, 4);
        using var scales = Mem.Zeros(DType.F32, 1);
        CpuOps.Quantize(new QuantizeArgs { Input = input.Span, Output = output.Span, Scales = scales.Span, Rows = 1, Dim = 4, DType = DType.F32, OutputDType = DType.I4 });

        // s = 3.5 / 7 = 0.5; q = round(1.4) = 1, round(-2.8) = -3 (0xD), 7, 0.
        Assert.Equal(0.5f, scales.ReadF32()[0]);
        Assert.Equal(new byte[] { 0xD1, 0x07 }, output.ReadBytes()[..2]);
    }

    [Fact]
    public void QuantizeInt8RoundsHalfToEvenAndClampsToTheTypeRange()
    {
        // Scales stored as F64 so the arithmetic is the hand arithmetic: s = 127 / 127 = 1.
        using var input = Mem.F32(127f, 2.5f, -3.5f, 0.5f, -127f, 1.5f);
        using var output = Mem.Zeros(DType.I8, 6);
        using var scales = Mem.Zeros(DType.F64, 1);
        CpuOps.Quantize(new QuantizeArgs { Input = input.Span, Output = output.Span, Scales = scales.Span, Rows = 1, Dim = 6, DType = DType.F32, OutputDType = DType.I8 });
        Assert.Equal(1.0, scales.ReadF64()[0]);
        Assert.Equal(new sbyte[] { 127, 2, -4, 0, -127, 2 }, output.ReadBytes().Select(b => unchecked((sbyte)b)).ToArray());
    }

    [Fact]
    public void QuantizeHandlesZeroGroupsAndNaN()
    {
        using var input = Mem.F32(0f, 0f, 0f, 0f, 1f, float.NaN, -2f, float.PositiveInfinity);
        using var output = Mem.Zeros(DType.F8E4M3, 8);
        using var scales = Mem.Zeros(DType.F32, 2);
        CpuOps.Quantize(new QuantizeArgs { Input = input.Span, Output = output.Span, Scales = scales.Span, Rows = 2, Dim = 4, DType = DType.F32, OutputDType = DType.F8E4M3 });
        float[] s = scales.ReadF32();
        byte[] q = output.ReadBytes();
        Assert.Equal(0f, s[0]);
        Assert.Equal(new byte[] { 0, 0, 0, 0 }, q[..4]);
        Assert.Equal((float)(2.0 / 448), s[1]); // amax ignores NaN and infinity
        Assert.Equal(0x7F, q[5]);
        Assert.Equal(0xFE, q[6]);
        Assert.Equal(0x7E, q[7]); // infinity saturates
    }

    [Fact]
    public void QuantizeRejectsUnsupportedTypesAndGroups()
    {
        using var input = Mem.F32(1, 2, 3, 4);
        using var output = Mem.Zeros(DType.U8, 4);
        using var scales = Mem.Zeros(DType.F32, 4);
        var a = new QuantizeArgs { Input = input.Span, Output = output.Span, Scales = scales.Span, Rows = 1, Dim = 4, DType = DType.F32, OutputDType = DType.U4 };
        Assert.Throws<ArgumentException>(() => CpuOps.Quantize(a));
        a.OutputDType = DType.I8;
        a.GroupSize = 3;
        Assert.Throws<ArgumentException>(() => CpuOps.Quantize(a));
        a.GroupSize = 0;
        using var intScales = Mem.Zeros(DType.I32, 1);
        a.Scales = intScales.Span;
        Assert.Throws<ArgumentException>(() => CpuOps.Quantize(a));
    }

    [Theory]
    [InlineData(DType.F8E4M3, DType.BF16)]
    [InlineData(DType.F8E5M2, DType.F16)]
    [InlineData(DType.I8, DType.F32)]
    [InlineData(DType.I4, DType.F32)]
    public void QuantizeThenDequantizeStaysWithinHalfAStep(DType q, DType outType)
    {
        const int rows = 4, dim = 32, group = 8;
        float[] x = Values(rows * dim, 11, 5);
        using var input = Mem.F32(x);
        using var packed = Mem.Zeros(q, rows * dim);
        using var scales = Mem.Zeros(DType.F32, rows * dim / group);
        using var output = Mem.Zeros(outType, rows * dim);
        CpuOps.Quantize(new QuantizeArgs { Input = input.Span, Output = packed.Span, Scales = scales.Span, Rows = rows, Dim = dim, GroupSize = group, DType = DType.F32, OutputDType = q });
        CpuOps.Dequantize(new DequantizeArgs { Input = packed.Span, Scales = scales.Span, Output = output.Span, Rows = rows, Dim = dim, GroupSize = group, InputDType = q, DType = outType });
        float[] s = scales.ReadF32();
        for (int i = 0; i < x.Length; i++)
        {
            double got = CpuElements.Load(output.Address, outType, i);
            double step = s[i / group];

            // Integers: half a step. FP8: half an ulp relative (2^-4 for E4M3, 2^-3 for E5M2) or half the subnormal quantum.
            double bound = q switch
            {
                DType.F8E4M3 => Math.Max(Math.Abs(x[i]) * Math.Pow(2, -4), step * Math.Pow(2, -10)),
                DType.F8E5M2 => Math.Max(Math.Abs(x[i]) * Math.Pow(2, -3), step * Math.Pow(2, -17)),
                _ => step / 2,
            };
            double outRounding = Math.Abs(got) * (outType == DType.BF16 ? Math.Pow(2, -8) : outType == DType.F16 ? Math.Pow(2, -11) : Math.Pow(2, -24));
            Assert.True(Math.Abs(got - x[i]) <= bound + outRounding + 1e-12, $"{q} [{i}] {x[i]} -> {got}");
        }
    }

    [Fact]
    public void DequantizeUsesZeroPointsAndTheirDefaults()
    {
        // U4 values 0, 3, 15, 8 over two groups of two; scales 0.5 and 2.
        using var q = Mem.Raw(DType.U4, 4, Packed4(0, 3, 15, 8));
        using var scales = Mem.F32(0.5f, 2f);
        using var zeros = Mem.Raw(DType.U4, 2, Packed4(1, 9));
        using var output = Mem.Zeros(DType.F32, 4);
        var a = new DequantizeArgs { Input = q.Span, Scales = scales.Span, Zeros = zeros.Span, Output = output.Span, Rows = 1, Dim = 4, GroupSize = 2, InputDType = DType.U4, DType = DType.F32 };
        CpuOps.Dequantize(a);
        Assert.Equal(new[] { -0.5f, 1f, 12f, -2f }, output.ReadF32());

        a.Zeros = default; // U4 without zero points: 8 (uint4b8)
        CpuOps.Dequantize(a);
        Assert.Equal(new[] { -4f, -2.5f, 14f, 0f }, output.ReadF32());

        using var i4 = Mem.Raw(DType.I4, 4, Packed4(-8, -1, 7, 1));
        CpuOps.Dequantize(a with { Input = i4.Span, InputDType = DType.I4 });
        Assert.Equal(new[] { -4f, -0.5f, 14f, 2f }, output.ReadF32());

        using var u8 = Mem.Raw(DType.U8, 4, [0, 128, 255, 130]);
        CpuOps.Dequantize(a with { Input = u8.Span, InputDType = DType.U8 });
        Assert.Equal(new[] { -64f, 0f, 254f, 4f }, output.ReadF32());

        // Zero points wider than U4 (a GPTQ v1 checkpoint's 16) and float zero points.
        using var z8 = Mem.Raw(DType.U8, 2, [16, 0]);
        CpuOps.Dequantize(a with { Zeros = z8.Span });
        Assert.Equal(new[] { -8f, -6.5f, 30f, 16f }, output.ReadF32());
        using var zf = Mem.F32(0.5f, -1f);
        CpuOps.Dequantize(a with { Zeros = zf.Span });
        Assert.Equal(new[] { -0.25f, 1.25f, 32f, 18f }, output.ReadF32());
    }

    [Fact]
    public void DequantizeFp8PerChannelAndRejectsZeroPoints()
    {
        byte[] codes = [Fp8.Encode(1.5, DType.F8E4M3, true), Fp8.Encode(-448, DType.F8E4M3, true), Fp8.Encode(0.0625, DType.F8E4M3, true), 0x7F];
        using var q = Mem.Raw(DType.F8E4M3, 4, codes);
        using var scales = Mem.F32(2f, 0.25f);
        using var output = Mem.Zeros(DType.F32, 4);
        var a = new DequantizeArgs { Input = q.Span, Scales = scales.Span, Output = output.Span, Rows = 2, Dim = 2, InputDType = DType.F8E4M3, DType = DType.F32 };
        CpuOps.Dequantize(a);
        float[] o = output.ReadF32();
        Assert.Equal(new[] { 3f, -896f, 0.015625f }, o[..3]);
        Assert.True(float.IsNaN(o[3]));

        using var zeros = Mem.F32(0f, 0f);
        Assert.Throws<ArgumentException>(() => CpuOps.Dequantize(a with { Zeros = zeros.Span }));
        Assert.Throws<ArgumentException>(() => CpuOps.Dequantize(a with { InputDType = DType.F4E2M1 }));
    }

    [Fact]
    public void GemmFp8MatchesAHandComputedCase()
    {
        static byte E(double v) => Fp8.Encode(v, DType.F8E4M3, true);
        using var am = Mem.Raw(DType.F8E4M3, 2, [E(1), E(2)]);
        using var bm = Mem.Raw(DType.F8E4M3, 4, [E(0.5), E(0.25), E(3), E(-1)]);
        using var aScale = Mem.F32(2f);
        using var bScale = Mem.F32(1f, 0.5f);
        using var d = Mem.Zeros(DType.F32, 2);
        var a = new GemmFp8Args { A = am.Span, B = bm.Span, AScale = aScale.Span, BScale = bScale.Span, D = d.Span, M = 1, N = 2, K = 2, DType = DType.F32 };
        CpuOps.GemmFp8(a);
        Assert.Equal([2f, 1f], d.ReadF32());

        using var badScale = Mem.F32(1f, 1f, 1f);
        Assert.Throws<ArgumentException>(() => CpuOps.GemmFp8(a with { BScale = badScale.Span }));
        using var notFp8 = Mem.F32(1f, 2f);
        Assert.Throws<ArgumentException>(() => CpuOps.GemmFp8(a with { A = notFp8.Span }));
    }

    [Theory]
    [InlineData(DType.F8E4M3, DType.F8E4M3, true, true, DType.BF16)]
    [InlineData(DType.F8E4M3, DType.F8E5M2, false, true, DType.F16)]
    [InlineData(DType.F8E5M2, DType.F8E4M3, true, false, DType.F32)]
    public void GemmFp8EqualsDequantizeThenGemm(DType at, DType bt, bool perToken, bool perChannel, DType outType)
    {
        const int m = 5, n = 6, k = 24;
        using var af = Mem.F32(Values(m * k, 3, 4));
        using var bf = Mem.F32(Values(n * k, 4, 0.5));
        using var aq = Mem.Zeros(at, m * k);
        using var bq = Mem.Zeros(bt, n * k);
        using var aScale = Mem.Zeros(DType.F32, perToken ? m : 1);
        using var bScale = Mem.Zeros(DType.F32, perChannel ? n : 1);
        CpuOps.Quantize(new QuantizeArgs { Input = af.Span, Output = aq.Span, Scales = aScale.Span, Rows = perToken ? m : 1, Dim = perToken ? k : m * k, DType = DType.F32, OutputDType = at });
        CpuOps.Quantize(new QuantizeArgs { Input = bf.Span, Output = bq.Span, Scales = bScale.Span, Rows = perChannel ? n : 1, Dim = perChannel ? k : n * k, DType = DType.F32, OutputDType = bt });
        using var d = Mem.Zeros(outType, m * n);
        CpuOps.GemmFp8(new GemmFp8Args { A = aq.Span, B = bq.Span, AScale = aScale.Span, BScale = bScale.Span, D = d.Span, M = m, N = n, K = k, DType = outType });

        // Reference: dequantise both to double, Gemm in double, round once into the output type.
        using var ad = Mem.Zeros(DType.F64, m * k);
        using var bd = Mem.Zeros(DType.F64, n * k);
        CpuOps.Dequantize(new DequantizeArgs { Input = aq.Span, Scales = aScale.Span, Output = ad.Span, Rows = perToken ? m : 1, Dim = perToken ? k : m * k, InputDType = at, DType = DType.F64 });
        CpuOps.Dequantize(new DequantizeArgs { Input = bq.Span, Scales = bScale.Span, Output = bd.Span, Rows = perChannel ? n : 1, Dim = perChannel ? k : n * k, InputDType = bt, DType = DType.F64 });
        using var dd = Mem.Zeros(DType.F64, m * n);
        CpuOps.Gemm(new GemmArgs { A = ad.Span, B = bd.Span, D = dd.Span, M = m, N = n, K = k, TransposeB = true, Alpha = 1f, DType = DType.F64, WeightDType = DType.F64 });
        double[] want = dd.ReadF64();
        for (int i = 0; i < m * n; i++)
        {
            // Only the order of the scale products differs (in double), so the results agree to within one step of the output type.
            double got = CpuElements.Load(d.Address, outType, i);
            double ulp = outType == DType.BF16 ? Math.Pow(2, -7) : outType == DType.F16 ? Math.Pow(2, -10) : Math.Pow(2, -23);
            Assert.True(Math.Abs(got - want[i]) <= Math.Abs(want[i]) * ulp, $"[{i}] {got} vs {want[i]}");
        }
    }

    [Theory]
    [InlineData(DType.BF16, DType.U4, 8, true)]
    [InlineData(DType.BF16, DType.U4, 0, false)]
    [InlineData(DType.F16, DType.U4, 16, true)]
    [InlineData(DType.F16, DType.I4, 8, false)]
    public void GemmW4A16IsBitForBitDequantizeThenGemm(DType act, DType wt, int group, bool withZeros)
    {
        const int m = 3, n = 8, k = 32;
        int groups = group == 0 ? 1 : k / group;
        var rng = new Random(5);
        int[] qv = Enumerable.Range(0, n * k).Select(_ => wt == DType.U4 ? rng.Next(0, 16) : rng.Next(-8, 8)).ToArray();
        using var b = Mem.Raw(wt, n * k, Packed4(qv));
        using var scales = Mem.Zeros(DType.F16, n * groups);
        for (int i = 0; i < n * groups; i++)
        {
            CpuElements.Store(scales.Address, DType.F16, i, 0.01 + (rng.NextDouble() * 0.05));
        }

        using var zeros = withZeros ? Mem.Raw(DType.U4, n * groups, Packed4(Enumerable.Range(0, n * groups).Select(_ => rng.Next(0, 16)).ToArray())) : null;
        using var a32 = Mem.F32(Values(m * k, 6, 2));
        using var a = Mem.Zeros(act, m * k);
        for (int i = 0; i < m * k; i++)
        {
            CpuElements.Store(a.Address, act, i, CpuElements.Load(a32.Address, DType.F32, i));
        }

        using var d = Mem.Zeros(act, m * n);
        CpuOps.GemmW4A16(new GemmW4A16Args { A = a.Span, B = b.Span, Scales = scales.Span, Zeros = zeros?.Span ?? default, D = d.Span, M = m, N = n, K = k, GroupSize = group, DType = act });

        using var w = Mem.Zeros(act, n * k);
        CpuOps.Dequantize(new DequantizeArgs { Input = b.Span, Scales = scales.Span, Zeros = zeros?.Span ?? default, Output = w.Span, Rows = n, Dim = k, GroupSize = group, InputDType = wt, DType = act });
        using var want = Mem.Zeros(act, m * n);
        CpuOps.Gemm(new GemmArgs { A = a.Span, B = w.Span, D = want.Span, M = m, N = n, K = k, TransposeB = true, Alpha = 1f, DType = act, WeightDType = act });
        Assert.Equal(want.ReadBytes(), d.ReadBytes());
    }

    [Fact]
    public void GemmW4A16RejectsANon4BitWeightAndABadGroup()
    {
        using var a = Mem.Zeros(DType.F16, 8);
        using var b = Mem.Zeros(DType.U8, 8);
        using var s = Mem.Zeros(DType.F16, 1);
        using var d = Mem.Zeros(DType.F16, 1);
        var args = new GemmW4A16Args { A = a.Span, B = b.Span, Scales = s.Span, D = d.Span, M = 1, N = 1, K = 8, DType = DType.F16 };
        Assert.Throws<ArgumentException>(() => CpuOps.GemmW4A16(args));
        using var b4 = Mem.Zeros(DType.U4, 8);
        Assert.Throws<ArgumentException>(() => CpuOps.GemmW4A16(args with { B = b4.Span, GroupSize = 3 }));
    }

    [Theory]
    [InlineData(DType.F8E4M3, TensorLayout.PagedNhd)]
    [InlineData(DType.F8E5M2, TensorLayout.PagedHnd)]
    public void Fp8CacheWriteScalesAndSaturatesAndAttentionReadsTheCodes(DType kv, TensorLayout layout)
    {
        const int tokens = 6, kvHeads = 1, qHeads = 2, hd = 4, bs = 4, blocks = 2;
        float[] k = Values(tokens * kvHeads * hd, 8, 3);
        float[] v = Values(tokens * kvHeads * hd, 9, 3);
        k[1] = 1e6f; // saturates
        using var kIn = Mem.F32(k);
        using var vIn = Mem.F32(v);
        using var slots = Mem.I64(4, 5, 6, 7, 0, 1); // block 1 then block 0
        using var kPool = Mem.Zeros(kv, blocks * bs * kvHeads * hd);
        using var vPool = Mem.Zeros(kv, blocks * bs * kvHeads * hd);
        CpuOps.CacheWrite(new CacheWriteArgs
        {
            K = kIn.Span, V = vIn.Span, KCache = kPool.Span, VCache = vPool.Span, SlotMapping = slots.Span, Tokens = tokens, KvHeads = kvHeads, HeadDim = hd,
            BlockSize = bs, KvLayout = layout, KScale = 2f, VScale = 0.5f, KvDType = kv,
        });
        byte[] kb = kPool.ReadBytes(), vb = vPool.ReadBytes();
        long[] slotOf = [4, 5, 6, 7, 0, 1];
        for (int t = 0; t < tokens; t++)
        {
            for (int d = 0; d < hd; d++)
            {
                long slot = slotOf[t];
                long at = layout == TensorLayout.PagedNhd ? (slot * hd) + d : ((((slot / bs) * kvHeads * bs) + (slot % bs)) * hd) + d;
                Assert.Equal(Fp8.Encode(k[(t * hd) + d] / 2.0, kv, true), kb[at]);
                Assert.Equal(Fp8.Encode(v[(t * hd) + d] / 0.5, kv, true), vb[at]);
            }
        }

        Assert.Equal(kv == DType.F8E4M3 ? 0x7E : 0x7B, kb[(4 * hd) + 1]); // token 0 (slot 4), element 1: 5e5 saturates

        // Attention over the FP8 pools equals attention over float32 pools holding the decoded codes (unit scale), bit for bit.
        using var kDec = Mem.F32(kb.Select(c => (float)Fp8.Decode(c, kv)).ToArray());
        using var vDec = Mem.F32(vb.Select(c => (float)Fp8.Decode(c, kv)).ToArray());
        using var q = Mem.F32(Values(2 * qHeads * hd, 10));
        using var o8 = Mem.Zeros(DType.F32, 2 * qHeads * hd);
        using var o32 = Mem.Zeros(DType.F32, 2 * qHeads * hd);
        using var qIndptr = Mem.I32(0, 2);
        using var kvIndptr = Mem.I32(0, 2);
        using var kvIndices = Mem.I32(1, 0);
        using var last = Mem.I32(2);
        var pa = new PagedAttentionArgs
        {
            Q = q.Span, Output = o8.Span, KCache = kPool.Span, VCache = vPool.Span, QIndptr = qIndptr.Span, KvIndptr = kvIndptr.Span, KvIndices = kvIndices.Span,
            KvLastPageLength = last.Span, Mode = AttentionMode.Prefill, Batch = 1, Tokens = 2, QHeads = qHeads, KvHeads = kvHeads, HeadDim = hd, BlockSize = bs,
            KvLayout = layout, WindowLeft = -1, Causal = true, KvDType = kv, DType = DType.F32,
        };
        CpuOps.PagedAttention(pa);
        CpuOps.PagedAttention(pa with { KCache = kDec.Span, VCache = vDec.Span, KvDType = DType.F32, Output = o32.Span });
        Assert.Equal(o32.ReadBytes(), o8.ReadBytes());
        Assert.All(o8.ReadF32(), x => Assert.True(float.IsFinite(x)));
    }
}

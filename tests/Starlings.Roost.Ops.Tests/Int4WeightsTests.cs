using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>AWQ and GPTQ scale and zero-point rules and checkpoint packing, against hand-computed groups and words.</summary>
public sealed class Int4WeightsTests
{
    private static (byte[] Q, double[] S, byte[] Z) Awq(double[] w, int n, int k, int g, DType st = DType.F64)
    {
        int groups = k / (g == 0 ? k : g);
        var q = new byte[n * k];
        var s = new double[n * groups];
        var z = new byte[n * groups];
        Int4Weights.QuantizeAwq(w, n, k, g, st, q, s, z);
        return (q, s, z);
    }

    private static (byte[] Q, double[] S, byte[] Z) Gptq(double[] w, int n, int k, int g, bool sym, DType st = DType.F64)
    {
        int groups = k / (g == 0 ? k : g);
        var q = new byte[n * k];
        var s = new double[n * groups];
        var z = new byte[n * groups];
        Int4Weights.QuantizeGptq(w, n, k, g, sym, st, q, s, z);
        return (q, s, z);
    }

    [Fact]
    public void AwqGroupMatchesHandArithmetic()
    {
        // max 6, min -1.5: s = 7.5 / 15 = 0.5, z = -round(-3) = 3; w / s = -3, 0, 0.5, 1.5, 12 -> round half even -3, 0, 0, 2, 12.
        var (q, s, z) = Awq([-1.5, 0, 0.25, 0.75, 6], 1, 5, 0);
        Assert.Equal(0.5, Assert.Single(s));
        Assert.Equal(new byte[] { 3 }, z);
        Assert.Equal(new byte[] { 0, 3, 3, 5, 15 }, q);
    }

    [Fact]
    public void AwqClampsTheRangeAndRoundsTheScaleToItsStorageType()
    {
        var (q, s, z) = Awq([2, 2, 2, 2], 1, 4, 0);
        Assert.Equal(1e-5 / 15, s[0]);
        Assert.Equal(0, z[0]); // -round(2 / s) is far below 0
        Assert.All(q, x => Assert.Equal(15, x));

        // 7 / 15 rounded to half first: q and z are computed with the stored scale.
        var (_, sh, _) = Awq([0, 7], 1, 2, 0, DType.F16);
        Assert.Equal((double)(Half)(7.0 / 15), sh[0]);
    }

    [Fact]
    public void GptqSymmetricAndAsymmetricGroupsMatchHandArithmetic()
    {
        // Symmetric: lo -3.75, hi 2 -> hi 3.75, lo -3.75: s = 0.5, z = 8; w / s = -2, 0.5, 4, -7.5 -> -2, 0, 4, -8.
        var (q, s, z) = Gptq([-1, 0.25, 2, -3.75], 1, 4, 0, sym: true);
        Assert.Equal(0.5, Assert.Single(s));
        Assert.Equal(new byte[] { 8 }, z);
        Assert.Equal(new byte[] { 6, 8, 12, 0 }, q);

        // Asymmetric: lo -1, hi 6.5: s = 0.5, z = round(2) = 2.
        (q, s, z) = Gptq([-1, 0.5, 2, 6.5], 1, 4, 0, sym: false);
        Assert.Equal(0.5, Assert.Single(s));
        Assert.Equal(new byte[] { 2 }, z);
        Assert.Equal(new byte[] { 0, 3, 6, 15 }, q);
    }

    [Fact]
    public void GptqKeepsItsQuirks()
    {
        // Symmetric with no negative weight: lo stays 0, s = 7.5 / 15, z = 8, so the top half clips at 15.
        var (q, s, z) = Gptq([0, 1, 2, 7.5], 1, 4, 0, sym: true);
        Assert.Equal(0.5, Assert.Single(s));
        Assert.Equal(new byte[] { 8 }, z);
        Assert.Equal(new byte[] { 8, 10, 12, 15 }, q);

        // An all-zero group uses lo -1, hi 1: s = 2 / 15, asymmetric z = round(7.5) = 8 (half to even).
        (q, s, z) = Gptq([0, 0], 1, 2, 0, sym: false);
        Assert.Equal(2.0 / 15, s[0]);
        Assert.Equal(new byte[] { 8 }, z);
        Assert.Equal(new byte[] { 8, 8 }, q);
    }

    [Fact]
    public void GroupsAreConsecutiveAlongK()
    {
        double[] w = [-1.5, 6, 0, 1, 0, 0.5, 0, 7.5]; // two rows of K = 4, group 2
        var (q, s, z) = Awq(w, 2, 4, 2);
        Assert.Equal(new[] { 0.5, 1.0 / 15, 0.5 / 15, 0.5 }, s);
        Assert.Equal(new byte[] { 3, 0, 0, 0 }, z);
        Assert.Equal(new byte[] { 0, 15, 0, 15, 0, 15, 0, 15 }, q);
    }

    [Fact]
    public void PackU4IsLowNibbleFirstAndRoundTrips()
    {
        var packed = new byte[3];
        Int4Weights.PackU4([1, 2, 3, 4, 15], packed);
        Assert.Equal(new byte[] { 0x21, 0x43, 0x0F }, packed);
        var back = new byte[5];
        Int4Weights.UnpackU4(packed, back);
        Assert.Equal(new byte[] { 1, 2, 3, 4, 15 }, back);
        Assert.Throws<ArgumentOutOfRangeException>(() => Int4Weights.PackU4([16], new byte[1]));
    }

    [Fact]
    public void AwqWordsUseTheInterleavedColumnOrder()
    {
        // K = 1, N = 8: q[n, 0] = n + 1. Nibble i holds column {0, 2, 4, 6, 1, 3, 5, 7}[i]: 1, 3, 5, 7, 2, 4, 6, 8.
        byte[] q = [1, 2, 3, 4, 5, 6, 7, 8];
        byte[] z = [8, 9, 10, 11, 12, 13, 14, 15];
        var qw = new int[1];
        var qz = new int[1];
        Int4Weights.PackAwq(q, z, 8, 1, 0, qw, qz);
        Assert.Equal(unchecked((int)0x86427531u), qw[0]);
        Assert.Equal(unchecked((int)0xFDB9ECA8u), qz[0]);
    }

    [Fact]
    public void GptqWordsRunAlongKAndV1StoresZeroMinusOne()
    {
        // N = 8, K = 8: row 0 of q is 1..8 along K, so word (0, 0) is 0x87654321; zeros of 8 are stored 7 in v1, 8 in v2.
        var q = new byte[64];
        for (int k = 0; k < 8; k++)
        {
            q[k] = (byte)(k + 1);
        }

        byte[] z = [8, 8, 8, 8, 8, 8, 8, 8];
        var qw = new int[8];
        var qz = new int[1];
        Int4Weights.PackGptq(q, z, 8, 8, 0, v1: true, qw, qz);
        Assert.Equal(unchecked((int)0x87654321u), qw[0]);
        Assert.All(qw[1..], x => Assert.Equal(0, x));
        Assert.Equal(0x77777777, qz[0]);
        Int4Weights.PackGptq(q, z, 8, 8, 0, v1: false, qw, qz);
        Assert.Equal(unchecked((int)0x88888888u), qz[0]);
        Assert.Throws<ArgumentException>(() => Int4Weights.PackGptq(q, new byte[8], 8, 8, 0, v1: true, qw, qz));

        var zv1 = new byte[8];
        Int4Weights.UnpackGptq(qw, [unchecked((int)0xF0000000u)], [], 8, 8, 0, v1: true, new byte[64], zv1);
        Assert.Equal(new byte[] { 1, 1, 1, 1, 1, 1, 1, 16 }, zv1);
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(false, true)]
    public void CheckpointPackingRoundTripsBitForBit(bool awq, bool v1)
    {
        const int n = 16, k = 64, g = 16;
        var rng = new Random(42);
        byte[] q = Enumerable.Range(0, n * k).Select(_ => (byte)rng.Next(0, 16)).ToArray();
        byte[] z = Enumerable.Range(0, n * (k / g)).Select(_ => (byte)rng.Next(v1 ? 1 : 0, 16)).ToArray();
        var q2 = new byte[q.Length];
        var z2 = new byte[z.Length];
        if (awq)
        {
            var qw = new int[k * n / 8];
            var qz = new int[k / g * n / 8];
            Int4Weights.PackAwq(q, z, n, k, g, qw, qz);
            Int4Weights.UnpackAwq(qw, qz, n, k, g, q2, z2);
        }
        else
        {
            var qw = new int[k / 8 * n];
            var qz = new int[k / g * n / 8];
            int[] gIdx = Enumerable.Range(0, k).Select(i => i / g).ToArray();
            Int4Weights.PackGptq(q, z, n, k, g, v1, qw, qz);
            Int4Weights.UnpackGptq(qw, qz, gIdx, n, k, g, v1, q2, z2);
        }

        Assert.Equal(q, q2);
        Assert.Equal(z, z2);
    }

    [Fact]
    public void ActOrderGptqIsRefused()
    {
        int[] gIdx = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
        Assert.Throws<NotSupportedException>(() =>
            Int4Weights.UnpackGptq(new int[16], new int[2], gIdx, 8, 16, 8, v1: false, new byte[128], new byte[16]));
    }

    [Fact]
    public void TransposeScalesMovesCheckpointOrderToPlainOrder()
    {
        ushort[] src = [1, 2, 3, 4, 5, 6]; // [groups 2, N 3]
        var dst = new ushort[6];
        Int4Weights.TransposeScales<ushort>(src, 2, 3, dst);
        Assert.Equal(new ushort[] { 1, 4, 2, 5, 3, 6 }, dst);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void QuantizedWeightsDequantizeThroughCpuOpsWithinOneStep(bool awq)
    {
        const int n = 8, k = 32, g = 8;
        var rng = new Random(3);
        double[] w = Enumerable.Range(0, n * k).Select(_ => (rng.NextDouble() * 2) - 1).ToArray();
        var (q, s, z) = awq ? Awq(w, n, k, g, DType.F16) : Gptq(w, n, k, g, sym: false, DType.F16);

        var qp = new byte[n * k / 2];
        var zp = new byte[n * (k / g) / 2];
        Int4Weights.PackU4(q, qp);
        Int4Weights.PackU4(z, zp);
        using var qm = Mem.Raw(DType.U4, n * k, qp);
        using var zm = Mem.Raw(DType.U4, n * (k / g), zp);
        using var sm = Mem.Zeros(DType.F16, s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            CpuElements.Store(sm.Address, DType.F16, i, s[i]);
        }

        using var out32 = Mem.Zeros(DType.F32, n * k);
        CpuOps.Dequantize(new DequantizeArgs { Input = qm.Span, Scales = sm.Span, Zeros = zm.Span, Output = out32.Span, Rows = n, Dim = k, GroupSize = g, InputDType = DType.U4, DType = DType.F32 });
        float[] got = out32.ReadF32();
        for (int i = 0; i < w.Length; i++)
        {
            Assert.Equal((float)((q[i] - z[i / g]) * s[i / g]), got[i]);
            // Half a step, plus up to half a step more where the rounded zero point pushes a group end past 0 or 15 and clips.
            Assert.True(Math.Abs(got[i] - w[i]) <= s[i / g] + 1e-6, $"[{i}] {w[i]} -> {got[i]}");
        }
    }
}

using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>
/// FP8 conversion against tables built here from the OFP8 bit-field definitions, independently of <see cref="Fp8"/>: the
/// decode table of all 256 codes, and an encoder that searches that table for the nearest value (ties to the even code).
/// Encoding is checked exhaustively over every binary16 and every bfloat16 value, at every midpoint and its double
/// neighbours, and over a fixed sample of binary32 values, in both formats, saturating and not.
/// </summary>
public sealed class Fp8Tests
{
    public static TheoryData<DType> Formats => [DType.F8E4M3, DType.F8E5M2];

    // From the bit fields: E4M3 bias 7, NaN only at S.1111.111; E5M2 bias 15, Inf at S.11111.00, NaN at S.11111.xx.
    private static double RefDecode(int code, DType t)
    {
        bool neg = (code & 0x80) != 0;
        double v;
        if (t == DType.F8E4M3)
        {
            int e = (code >> 3) & 0xF, m = code & 0x7;
            if (e == 15 && m == 7)
            {
                return double.NaN;
            }

            v = e == 0 ? m / 8.0 * Math.Pow(2, -6) : (1 + (m / 8.0)) * Math.Pow(2, e - 7);
        }
        else
        {
            int e = (code >> 2) & 0x1F, m = code & 0x3;
            if (e == 31)
            {
                return m == 0 ? (neg ? double.NegativeInfinity : double.PositiveInfinity) : double.NaN;
            }

            v = e == 0 ? m / 4.0 * Math.Pow(2, -14) : (1 + (m / 4.0)) * Math.Pow(2, e - 15);
        }

        return neg ? -v : v;
    }

    private static int MaxCode(DType t) => t == DType.F8E4M3 ? 0x7E : 0x7B;

    // The value the next code would have if the exponent range went on (E4M3: 480, E5M2: 65536).
    private static double Beyond(DType t) => t == DType.F8E4M3 ? 480.0 : 65536.0;

    private static double Magnitude(int code, DType t) => code <= MaxCode(t) ? RefDecode(code, t) : Beyond(t);

    // Nearest magnitude among codes 0..max and the virtual one beyond, ties to the even code; the virtual code means overflow.
    private static byte RefEncode(double x, DType t, bool saturate)
    {
        if (double.IsNaN(x))
        {
            return 0x7F;
        }

        int sign = double.IsNegative(x) ? 0x80 : 0;
        double a = Math.Abs(x);
        int best = MaxCode(t) + 1;
        // At or beyond the virtual value the overflow is certain (and far beyond it the distances below would round equal).
        if (a < Beyond(t))
        {
            double bestDist = double.PositiveInfinity;
            for (int c = 0; c <= MaxCode(t) + 1; c++)
            {
                double d = Math.Abs(a - Magnitude(c, t));
                if (d < bestDist || (d == bestDist && (c & 1) == 0))
                {
                    best = c;
                    bestDist = d;
                }
            }
        }

        if (best <= MaxCode(t))
        {
            return (byte)(sign | best);
        }

        return saturate ? (byte)(sign | MaxCode(t)) : t == DType.F8E4M3 ? (byte)(sign | 0x7F) : (byte)(sign | 0x7C);
    }

    private static void Same(double x, DType t)
    {
        foreach (bool sat in new[] { true, false })
        {
            byte want = RefEncode(x, t, sat);
            byte got = Fp8.Encode(x, t, sat);
            if (want != got)
            {
                Assert.Fail($"{t} {(sat ? "sat" : "nosat")} {x:R} (0x{BitConverter.DoubleToInt64Bits(x):X16}): expected 0x{want:X2}, got 0x{got:X2}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void DecodeMatchesTheBitFieldTableForAll256Codes(DType t)
    {
        for (int c = 0; c < 256; c++)
        {
            double want = RefDecode(c, t);
            double got = Fp8.Decode((byte)c, t);
            Assert.True(want.Equals(got), $"{t} 0x{c:X2}: expected {want:R}, got {got:R}");
            Assert.Equal(BitConverter.DoubleToInt64Bits(want), BitConverter.DoubleToInt64Bits(got));
        }
    }

    [Fact]
    public void AnchorValuesOfBothFormats()
    {
        Assert.Equal(448.0, Fp8.Decode(0x7E, DType.F8E4M3));
        Assert.Equal(-448.0, Fp8.Decode(0xFE, DType.F8E4M3));
        Assert.Equal(Math.Pow(2, -9), Fp8.Decode(0x01, DType.F8E4M3));
        Assert.Equal(Math.Pow(2, -6), Fp8.Decode(0x08, DType.F8E4M3));
        Assert.True(double.IsNaN(Fp8.Decode(0x7F, DType.F8E4M3)));
        Assert.True(double.IsNaN(Fp8.Decode(0xFF, DType.F8E4M3)));
        Assert.Equal(240.0, Fp8.Decode(0x77, DType.F8E4M3)); // S.1110.111: the top binade is finite except its last code
        Assert.Equal(57344.0, Fp8.Decode(0x7B, DType.F8E5M2));
        Assert.Equal(Math.Pow(2, -16), Fp8.Decode(0x01, DType.F8E5M2));
        Assert.Equal(double.PositiveInfinity, Fp8.Decode(0x7C, DType.F8E5M2));
        Assert.Equal(double.NegativeInfinity, Fp8.Decode(0xFC, DType.F8E5M2));
        Assert.True(double.IsNaN(Fp8.Decode(0x7D, DType.F8E5M2)));
        Assert.True(double.IsNegative(Fp8.Decode(0x80, DType.F8E4M3)));
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void EveryFiniteCodeRoundTrips(DType t)
    {
        for (int c = 0; c < 256; c++)
        {
            double v = RefDecode(c, t);
            if (double.IsFinite(v))
            {
                Assert.Equal(c, Fp8.Encode(v, t, saturate: true));
                Assert.Equal(c, Fp8.Encode(v, t, saturate: false));
            }
        }
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void EveryBinary16ValueEncodesToTheNearestCode(DType t)
    {
        for (int bits = 0; bits < 0x10000; bits++)
        {
            Same((double)BitConverter.UInt16BitsToHalf((ushort)bits), t);
        }
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void EveryBfloat16ValueEncodesToTheNearestCode(DType t)
    {
        for (int bits = 0; bits < 0x10000; bits++)
        {
            Same(BitConverter.Int32BitsToSingle(bits << 16), t);
        }
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void EveryMidpointTiesToEvenAndItsNeighboursRoundAway(DType t)
    {
        for (int c = 0; c <= MaxCode(t); c++)
        {
            double lo = Magnitude(c, t), hi = Magnitude(c + 1, t);
            double mid = (lo + hi) / 2;
            foreach (double s in new[] { 1.0, -1.0 })
            {
                Same(s * mid, t);
                Same(s * Math.BitIncrement(mid), t);
                Same(s * Math.BitDecrement(mid), t);
            }

            int even = (c & 1) == 0 ? c : c + 1;
            Assert.Equal(Math.Min(even, MaxCode(t)), Fp8.Encode(mid, t, saturate: true));
            Assert.Equal(c, Fp8.Encode(Math.BitDecrement(mid), t, saturate: true));
        }
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void SampledBinary32ValuesEncodeToTheNearestCode(DType t)
    {
        var rng = new Random(20261101);
        for (int i = 0; i < 200_000; i++)
        {
            Same(BitConverter.Int32BitsToSingle(rng.Next(int.MinValue, int.MaxValue)), t);
        }
    }

    [Fact]
    public void OverflowSaturatesOrFollowsOfp8()
    {
        // E4M3: 464 is the tie between 448 (even code 0x7E) and the absent 480, so it stays finite; anything above overflows.
        Assert.Equal(0x7E, Fp8.Encode(464.0, DType.F8E4M3, saturate: false));
        Assert.Equal(0x7F, Fp8.Encode(Math.BitIncrement(464.0), DType.F8E4M3, saturate: false));
        Assert.Equal(0x7E, Fp8.Encode(Math.BitIncrement(464.0), DType.F8E4M3, saturate: true));
        Assert.Equal(0xFE, Fp8.Encode(-1e30, DType.F8E4M3, saturate: true));
        Assert.Equal(0x7E, Fp8.Encode(double.PositiveInfinity, DType.F8E4M3, saturate: true));
        Assert.Equal(0xFF, Fp8.Encode(double.NegativeInfinity, DType.F8E4M3, saturate: false));

        // E5M2: 61440 is the tie between 57344 (odd code 0x7B) and 65536, so it rounds up and overflows.
        Assert.Equal(0x7B, Fp8.Encode(Math.BitDecrement(61440.0), DType.F8E5M2, saturate: false));
        Assert.Equal(0x7C, Fp8.Encode(61440.0, DType.F8E5M2, saturate: false));
        Assert.Equal(0x7B, Fp8.Encode(61440.0, DType.F8E5M2, saturate: true));
        Assert.Equal(0xFC, Fp8.Encode(double.NegativeInfinity, DType.F8E5M2, saturate: false));
        Assert.Equal(0xFB, Fp8.Encode(double.NegativeInfinity, DType.F8E5M2, saturate: true));

        Assert.Equal(0x7F, Fp8.Encode(double.NaN, DType.F8E4M3, saturate: true));
        Assert.Equal(0x7F, Fp8.Encode(double.NaN, DType.F8E5M2, saturate: true));
        Assert.Equal(0x80, Fp8.Encode(-0.0, DType.F8E4M3, saturate: true));
        Assert.Equal(0x80, Fp8.Encode(-1e-30, DType.F8E5M2, saturate: true));
    }

    [Fact]
    public void SubnormalsRoundToNearestEven()
    {
        double q = Math.Pow(2, -9); // E4M3 quantum below 2^-6
        Assert.Equal(0x00, Fp8.Encode(0.5 * q, DType.F8E4M3, saturate: true)); // tie with 0: even
        Assert.Equal(0x01, Fp8.Encode(Math.BitIncrement(0.5 * q), DType.F8E4M3, saturate: true));
        Assert.Equal(0x02, Fp8.Encode(1.5 * q, DType.F8E4M3, saturate: true)); // tie 1 / 2: even
        Assert.Equal(0x08, Fp8.Encode(7.5 * q, DType.F8E4M3, saturate: true)); // carries into the smallest normal
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void StoreSaturatesAndLoadDecodes(DType t)
    {
        using var b = Mem.Zeros(t, 4);
        CpuElements.Store(b.Address, t, 0, 1.0);
        CpuElements.Store(b.Address, t, 1, -1e9);
        CpuElements.Store(b.Address, t, 2, double.NaN);
        CpuElements.Store(b.Address, t, 3, 0.3);
        byte[] raw = b.ReadBytes();
        Assert.Equal(Fp8.Encode(1.0, t, true), raw[0]);
        Assert.Equal((byte)(0x80 | MaxCode(t)), raw[1]);
        Assert.Equal(0x7F, raw[2]);
        Assert.Equal(-Fp8.MaxOf(t), CpuElements.Load(b.Address, t, 1));
        Assert.Equal(RefDecode(raw[3], t), CpuElements.Load(b.Address, t, 3));
    }
}

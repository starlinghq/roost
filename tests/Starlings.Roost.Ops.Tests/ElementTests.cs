using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

public sealed class ElementTests
{
    [Theory]
    [InlineData(0x3F808000u, 0x3F80)] // 1 + 2^-8: tie, rounds to even (down)
    [InlineData(0x3F818000u, 0x3F82)] // tie, rounds to even (up)
    [InlineData(0x3F808001u, 0x3F81)] // just above a tie
    [InlineData(0x3F807FFFu, 0x3F80)] // just below a tie
    [InlineData(0x3FFFC000u, 0x4000)] // tie whose carry reaches the exponent
    [InlineData(0xBF818000u, 0xBF82)] // sign is kept
    [InlineData(0x00000000u, 0x0000)]
    [InlineData(0x80000000u, 0x8000)]
    public void BFloat16RoundsFloatsToNearestEven(uint floatBits, int expected) =>
        Assert.Equal((ushort)expected, CpuElements.ToBFloat16(BitConverter.UInt32BitsToSingle(floatBits)));

    [Fact]
    public void BFloat16DoesNotDoubleRoundThroughFloat()
    {
        // 1 + 2^-8 + 2^-40 is above the bfloat16 tie, but rounds to the tie when narrowed to float first.
        double v = 1.0 + Math.Pow(2, -8) + Math.Pow(2, -40);
        Assert.Equal(0x3F80, (int)CpuElements.ToBFloat16((float)v));
        Assert.Equal(0x3F81, (int)CpuElements.ToBFloat16(v));
        Assert.Equal(0xBF81, (int)CpuElements.ToBFloat16(-v));
    }

    [Fact]
    public void BFloat16HandlesSpecialValuesAndOverflow()
    {
        Assert.Equal(0x7F80, (int)CpuElements.ToBFloat16(double.PositiveInfinity));
        Assert.Equal(0xFF80, (int)CpuElements.ToBFloat16(double.NegativeInfinity));
        Assert.Equal(0x7F80, (int)CpuElements.ToBFloat16(1e300));
        Assert.Equal(0xFF80, (int)CpuElements.ToBFloat16(-1e300));
        Assert.True(float.IsNaN(BitConverter.Int32BitsToSingle(CpuElements.ToBFloat16(double.NaN) << 16)));
        Assert.Equal(0x0000, (int)CpuElements.ToBFloat16(0.0));
        Assert.Equal(0x3F80, (int)CpuElements.ToBFloat16(1.0));
    }

    [Fact]
    public void StoreAndLoadRoundTripEveryFloatType()
    {
        foreach (var t in new[] { DType.F64, DType.F32, DType.F16, DType.BF16 })
        {
            using var b = Mem.Zeros(t, 2);
            CpuElements.Store(b.Address, t, 1, 0.5);
            Assert.Equal(0.5, CpuElements.Load(b.Address, t, 1));
            Assert.Equal(0.0, CpuElements.Load(b.Address, t, 0));
        }
    }

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(0.5, 0.5204998778130465)]
    [InlineData(1.0, 0.8427007929497149)]
    [InlineData(2.0, 0.9953222650189527)]
    [InlineData(3.0, 0.9999779095030014)]
    [InlineData(-1.0, -0.8427007929497149)]
    public void ErfMatchesKnownValues(double x, double expected) => Assert.Equal(expected, CpuElements.Erf(x), 1e-14);

    [Theory]
    [InlineData(1.0, 0.15729920705028513)]
    [InlineData(3.0, 2.209049699858544e-05)]
    [InlineData(5.0, 1.5374597944280347e-12)]
    [InlineData(10.0, 2.088487583762545e-45)]
    public void ErfcKeepsRelativeAccuracyInTheTail(double x, double expected) =>
        Assert.Equal(1.0, CpuElements.Erfc(x) / expected, 10);

    [Fact]
    public void PhiloxMatchesRandom123KnownAnswers()
    {
        Assert.Equal((0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u), CpuPhilox.Block(0, 0, 0, 0, 0, 0));
        Assert.Equal(
            (0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu),
            CpuPhilox.Block(uint.MaxValue, uint.MaxValue, uint.MaxValue, uint.MaxValue, uint.MaxValue, uint.MaxValue));
        Assert.Equal(
            (0xd16cfe09u, 0x94fdcceb, 0x5001e420u, 0x24126ea1u),
            CpuPhilox.Block(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0));
    }

    [Fact]
    public void PhiloxUniformIsTheTop24BitsOfTheFirstWord()
    {
        var (r0, _, _, _) = CpuPhilox.Block(7, 0, 0, 0, 3, 0);
        Assert.Equal((r0 >> 8) / 16777216.0, CpuPhilox.Uniform(3, 7));
        Assert.InRange(CpuPhilox.Uniform(ulong.MaxValue, ulong.MaxValue), 0.0, 1.0 - (1.0 / 16777216.0));
    }
}

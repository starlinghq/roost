namespace Starlings.Roost.Ops.Tests;

/// <summary>The float32 exp, log and lane sum behind <see cref="Abi.SampleMath.Fp32"/>: accuracy against double and the documented edge cases.</summary>
public sealed class Fp32MathTests
{
    private static double Ulps(float actual, double exact)
    {
        float rounded = (float)exact;
        float ulp = MathF.BitIncrement(MathF.Abs(rounded)) - MathF.Abs(rounded);
        return Math.Abs(actual - exact) / ulp;
    }

    [Fact]
    public void ExpIsWithinTwoUlpsOverTheSamplersRange()
    {
        var rng = new Random(1);
        for (int i = 0; i < 200_000; i++)
        {
            float x = (float)((rng.NextDouble() * 120) - 100); // softmax arguments are <= 0; check both sides
            Assert.True(Ulps(Fp32Math.Exp(x), Math.Exp(x)) <= 2, $"exp({x:R}) = {Fp32Math.Exp(x):R}, exact {Math.Exp(x):R}");
        }
    }

    [Fact]
    public void ExpEdgeCases()
    {
        Assert.Equal(1f, Fp32Math.Exp(0f));
        Assert.Equal(0f, Fp32Math.Exp(float.NegativeInfinity));
        Assert.Equal(0f, Fp32Math.Exp(-200f));
        Assert.Equal(float.PositiveInfinity, Fp32Math.Exp(100f));
        Assert.True(float.IsNaN(Fp32Math.Exp(float.NaN)));
        Assert.True(Fp32Math.Exp(-90f) > 0f); // subnormal results are produced, not flushed
    }

    [Fact]
    public void LogIsWithinTwoUlpsAndExactAtOne()
    {
        Assert.Equal(0f, Fp32Math.Log(1f));
        var rng = new Random(2);
        for (int i = 0; i < 200_000; i++)
        {
            float x = (float)Math.Exp((rng.NextDouble() * 60) - 20);
            double exact = Math.Log(x);
            if (Math.Abs(exact) < 1e-3)
            {
                continue; // relative ulps blow up near 1; absolute error there is checked below
            }

            Assert.True(Ulps(Fp32Math.Log(x), exact) <= 2, $"log({x:R}) = {Fp32Math.Log(x):R}, exact {exact:R}");
        }

        Assert.True(Math.Abs(Fp32Math.Log(1.0001f) - Math.Log(1.0001f)) < 1e-9);
        Assert.Equal(float.NegativeInfinity, Fp32Math.Log(0f));
        Assert.True(float.IsNaN(Fp32Math.Log(-1f)));
        Assert.Equal((float)Math.Log(1e-40), Fp32Math.Log(1e-40f), 4);
    }

    [Fact]
    public void LaneSumAddsByIndexModulo32ThenHalves()
    {
        // 1e8 in lane 0 and 1 in lane 32's slot (also lane 0): sequential float addition loses the 1; lane 1 then adds
        // -1e8 at the w = 1 step, so the result is 0 rather than 1.
        Span<float> lanes = stackalloc float[32];
        lanes.Clear();
        Fp32Math.AddToLane(lanes, 0, 1e8f);
        Fp32Math.AddToLane(lanes, 1, -1e8f);
        Fp32Math.AddToLane(lanes, 32, 1f);
        Assert.Equal(0f, Fp32Math.ReduceLanes(lanes));

        // Lane 16 joins lane 0 first: (1e8 + -1e8) + 1 = 1.
        lanes.Clear();
        Fp32Math.AddToLane(lanes, 0, 1e8f);
        Fp32Math.AddToLane(lanes, 16, -1e8f);
        Fp32Math.AddToLane(lanes, 1, 1f);
        Assert.Equal(1f, Fp32Math.ReduceLanes(lanes));
    }
}

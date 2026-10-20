using Starlings.Roost.Abi;

namespace Starlings.Roost.Models.Tiny.Tests;

public class TinyWeightsTests
{
    [Fact]
    public void ASlotIsAFunctionOfSeedAndNameOnly()
    {
        double[] a = TinyWeights.Generate(1, "l0.q_proj", "64,64", DType.F32);
        double[] b = TinyWeights.Generate(1, "l0.q_proj", "64,64", DType.F32);
        Assert.Equal(a, b);
        Assert.NotEqual(a, TinyWeights.Generate(2, "l0.q_proj", "64,64", DType.F32));
        Assert.NotEqual(a, TinyWeights.Generate(1, "l1.q_proj", "64,64", DType.F32));
    }

    [Fact]
    public void GeneratedValuesAreExactlyRepresentableInTheSlotType()
    {
        foreach (var dtype in new[] { DType.F32, DType.BF16 })
        {
            double[] v = TinyWeights.Generate(7, "l0.gate_up", "256,64", dtype);
            Assert.All(v, x => Assert.Equal(x, TinyWeights.Round(x, dtype)));
        }

        // BF16 keeps 8 bits of mantissa: it must differ from the F32 values and be within 2^-8 relative.
        double[] f32 = TinyWeights.Generate(7, "l0.gate_up", "256,64", DType.F32);
        double[] bf16 = TinyWeights.Generate(7, "l0.gate_up", "256,64", DType.BF16);
        Assert.NotEqual(f32, bf16);
        for (int i = 0; i < f32.Length; i++)
        {
            Assert.InRange(Math.Abs(f32[i] - bf16[i]), 0, Math.Abs(f32[i]) / 128);
        }
    }

    [Fact]
    public void ValueRangesFollowTheNamingRules()
    {
        double[] norm = TinyWeights.Generate(3, "l2.post_norm", "64", DType.F32);
        Assert.All(norm, x => Assert.InRange(x, 0.899, 1.101));
        double[] embed = TinyWeights.Generate(3, "embed", "256,64", DType.F32);
        Assert.All(embed, x => Assert.InRange(x, -1.0, 1.0));
        Assert.Contains(embed, x => x > 0.9);
        double[] proj = TinyWeights.Generate(3, "l0.down", "64,128", DType.F32);
        double a = 1.5 / Math.Sqrt(128);
        Assert.All(proj, x => Assert.InRange(x, -a, a));
        Assert.InRange(proj.Average(), -0.02, 0.02);
    }

    // The bytes are the same on every host: the generator uses SplitMix64 and IEEE double arithmetic only. If one of these
    // digests changes, the generator changed, and so did every tiny-model result pinned against it.
    [Theory]
    [InlineData(DType.F32, 10427801740649689927UL)]
    [InlineData(DType.BF16, 6253068743922294065UL)]
    public void DigestsOfAReferenceSlotArePinned(DType dtype, ulong expected)
    {
        double[] v = TinyWeights.Generate(TinyConfig.Dense.Seed, "l0.q_proj", "64,64", dtype);
        Assert.Equal(expected, TinyWeights.Digest(v, dtype));
    }

    [Fact]
    public unsafe void WriteStoresWhatGenerateReturns()
    {
        using var native = new Starlings.Roost.ModelCompiler.Tests.Native();
        double[] v = TinyWeights.Generate(5, "l0.k_proj", "32,64", DType.BF16);
        var span = native.Alloc(v.Length, DType.BF16);
        TinyWeights.Write(v, DType.BF16, span.Address);
        var raw = new ReadOnlySpan<ushort>((void*)span.Address, v.Length);
        for (int i = 0; i < v.Length; i++)
        {
            Assert.Equal(v[i], BitConverter.UInt32BitsToSingle((uint)raw[i] << 16));
        }
    }
}

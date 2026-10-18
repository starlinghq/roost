using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

public sealed class SamplingTests
{
    private static int[] Run(float[] logits, int vocab, float temp, int topK, float topP, float minP, ulong seed, ulong offset, uint[]? mask = null, float[]? logprobOut = null, int stride = 0)
    {
        int rowLen = stride == 0 ? vocab : stride;
        int rows = logits.Length / rowLen;
        using var l = Mem.F32(logits);
        using var t = Mem.F32(Enumerable.Repeat(temp, rows).ToArray());
        using var k = Mem.I32(Enumerable.Repeat(topK, rows).ToArray());
        using var p = Mem.F32(Enumerable.Repeat(topP, rows).ToArray());
        using var mp = Mem.F32(Enumerable.Repeat(minP, rows).ToArray());
        using var s = Mem.U64(Enumerable.Repeat(seed, rows).ToArray());
        using var o = Mem.U64(Enumerable.Repeat(offset, rows).ToArray());
        using var ids = Mem.Zeros(DType.I32, rows);
        using var lp = Mem.Zeros(DType.F32, rows);
        using var mk = mask is null ? null : Mem.U32(mask);
        CpuOps.Sample(new SampleArgs
        {
            Logits = l.Span, Temperature = t.Span, TopK = k.Span, TopP = p.Span, MinP = mp.Span, Seeds = s.Span, Offsets = o.Span,
            Mask = mk?.Span ?? default, Ids = ids.Span, Logprobs = logprobOut is null ? default : lp.Span, Rows = rows, Vocab = vocab, Stride = stride,
        });
        if (logprobOut is not null)
        {
            lp.ReadF32().CopyTo(logprobOut, 0);
        }

        return ids.ReadI32();
    }

    [Fact]
    public void StrideZeroMeansVocabAndAStrideBelowVocabThrows()
    {
        // Two rows, greedy: row 0 -> 0, row 1 -> 1 (matches Argmax with Stride 0).
        Assert.Equal([0, 1], Run([9, 0, 0, 9], 2, 0f, 0, 1f, 0f, 1, 0, stride: 0));
        Assert.Throws<ArgumentException>(() => Run([9, 0, 0, 9], 2, 0f, 0, 1f, 0f, 1, 0, stride: 1));
    }

    [Fact]
    public void ZeroTemperatureIsGreedyWithLowestIndexTies()
    {
        Assert.Equal([1], Run([0, 3, 3, 1], 4, 0f, 0, 1f, 0f, 1, 0));
        Assert.Equal([2, 0], Run([0, 1, 9, 5, 0, 5], 3, 0f, 0, 1f, 0f, 1, 0));
    }

    [Fact]
    public void MaskRemovesTokensBeforeEverythingElse()
    {
        // Token 1 has the top logit but its mask bit is 0; bit v of word 0.
        Assert.Equal([2], Run([0, 9, 4, 1], 4, 0f, 0, 1f, 0f, 1, 0, mask: [0b1101u]));
        Assert.Equal([2], Run([0, 9, 4, 1], 4, 1f, 0, 1f, 0f, 1, 0, mask: [0b0100u]));
        Assert.Equal([-1], Run([0, 9, 4, 1], 4, 1f, 0, 1f, 0f, 1, 0, mask: [0u]));
    }

    [Fact]
    public void MaskWordsAreIndexedPerRowAcrossMoreThan32Tokens()
    {
        var logits = new float[2 * 40];
        logits[33] = 5; // row 0
        logits[40 + 3] = 5; // row 1
        // Row 0 allows only token 33 (word 1 bit 1); row 1 allows only token 3 (word 0 bit 3).
        Assert.Equal([33, 3], Run(logits, 40, 0f, 0, 1f, 0f, 1, 0, mask: [0u, 0b10u, 0b1000u, 0u]));
    }

    [Fact]
    public void TwoTokenDrawFollowsThePhiloxUniformAndTheDescendingOrder()
    {
        // Probabilities 0.25 (token 0) and 0.75 (token 1); order is token 1 then token 0.
        float[] logits = [0f, (float)Math.Log(3)];
        double p1 = Math.Exp(logits[1]) / (1 + Math.Exp(logits[1]));
        for (ulong offset = 0; offset < 40; offset++)
        {
            double u = CpuPhilox.Uniform(99, offset);
            int expected = u < p1 ? 1 : 0;
            Assert.Equal([expected], Run(logits, 2, 1f, 0, 1f, 0f, 99, offset));
        }
    }

    [Fact]
    public void SameSeedAndOffsetGiveTheSameDrawWhateverTheBatch()
    {
        var rng = new Random(5);
        float[] row = Enumerable.Range(0, 64).Select(_ => (float)((rng.NextDouble() * 6) - 3)).ToArray();
        int alone = Run(row, 64, 0.8f, 0, 1f, 0f, 1234, 77)[0];
        float[] batch = [.. new float[64], .. row, .. new float[64]];
        // Same row in the middle of a 3-row batch (seeds are per row; all rows share the seed here).
        Assert.Equal(alone, Run(batch, 64, 0.8f, 0, 1f, 0f, 1234, 77)[1]);
    }

    [Fact]
    public void EmpiricalFrequenciesMatchTheTemperatureSoftmax()
    {
        float[] logits = [1f, 2f, 3f];
        double[] p = [Math.Exp(1), Math.Exp(2), Math.Exp(3)];
        double z = p.Sum();
        var counts = new int[3];
        const int n = 6000;
        for (ulong i = 0; i < n; i++)
        {
            counts[Run(logits, 3, 1f, 0, 1f, 0f, 2024, i)[0]]++;
        }

        for (int v = 0; v < 3; v++)
        {
            Assert.InRange(counts[v] / (double)n, (p[v] / z) - 0.03, (p[v] / z) + 0.03);
        }
    }

    [Fact]
    public void TopKKeepsOnlyTheBestTokens()
    {
        float[] logits = [0f, 1f, 5f, 4f, 2f];
        for (ulong i = 0; i < 200; i++)
        {
            Assert.Equal(2, Run(logits, 5, 1f, 1, 1f, 0f, 3, i)[0]);
            Assert.InRange(Run(logits, 5, 1f, 2, 1f, 0f, 3, i)[0], 2, 3);
        }
    }

    [Fact]
    public void TopPKeepsTheShortestPrefixReachingTheMass()
    {
        // p = 0.5, 0.3, 0.2 at temperature 1; top_p 0.7 keeps the first two (0.5 < 0.7 <= 0.8).
        float[] logits = [(float)Math.Log(0.2), (float)Math.Log(0.5), (float)Math.Log(0.3)];
        var seen = new HashSet<int>();
        for (ulong i = 0; i < 300; i++)
        {
            seen.Add(Run(logits, 3, 1f, 0, 0.7f, 0f, 8, i)[0]);
        }

        Assert.Equal([1, 2], seen.OrderBy(x => x));
        // top_p 0.5 keeps exactly the first token (cumulative 0.5 reaches 0.5).
        Assert.Equal(1, Run(logits, 3, 1f, 0, 0.5f, 0f, 8, 0)[0]);
    }

    [Fact]
    public void TopKTopPAndMinPComposeInOrderWithRenormalisedTopP()
    {
        // Probabilities by token: 0.06, 0.40, 0.10, 0.25, 0.04, 0.15 (descending order is tokens 1, 3, 5, 2, 0, 4).
        double[] want = [0.06, 0.40, 0.10, 0.25, 0.04, 0.15];
        float[] logits = want.Select(p => (float)Math.Log(p)).ToArray();
        double z = logits.Sum(l => Math.Exp(l));
        double[] p = logits.Select(l => Math.Exp(l) / z).ToArray();
        // top_k 4 keeps tokens 1, 3, 5, 2 (mass 0.90). top_p 0.85 on the renormalised mass: 0.444, 0.722, 0.889 reaches it, so token 2 goes
        // (on the raw mass the cut would sit at 0.90 and keep token 2). min_p 0.5 then drops token 5 (0.15 < 0.5 * 0.40). Tokens 1 and 3 remain.
        double mass = p[1] + p[3];
        var seen = new HashSet<int>();
        for (ulong offset = 0; offset < 200; offset++)
        {
            double u = CpuPhilox.Uniform(31, offset);
            int expected = (u * mass) < p[1] ? 1 : 3;
            int got = Run(logits, 6, 1f, 4, 0.85f, 0.5f, 31, offset)[0];
            Assert.Equal(expected, got);
            seen.Add(got);
        }

        Assert.Equal([1, 3], seen.OrderBy(x => x));
    }

    [Fact]
    public void MinPDropsTokensBelowAFractionOfTheLargest()
    {
        float[] logits = [(float)Math.Log(0.6), (float)Math.Log(0.3), (float)Math.Log(0.1)];
        var seen = new HashSet<int>();
        for (ulong i = 0; i < 300; i++)
        {
            seen.Add(Run(logits, 3, 1f, 0, 1f, 0.4f, 8, i)[0]); // floor 0.24: tokens 0 and 1 stay
        }

        Assert.Equal([0, 1], seen.OrderBy(x => x));
    }

    [Fact]
    public void LogprobIsTheRawLogSoftmaxOfTheChosenToken()
    {
        var lp = new float[1];
        int id = Run([1f, 2f, 3f], 3, 0f, 0, 1f, 0f, 1, 0, logprobOut: lp)[0];
        Assert.Equal(2, id);
        Assert.Equal((float)(3 - Math.Log(Math.Exp(1) + Math.Exp(2) + Math.Exp(3))), lp[0], 5);
    }

    [Fact]
    public void StrideSkipsPaddingColumns()
    {
        // Stride 4, vocab 3: column 3 holds a huge value that must never be chosen.
        Assert.Equal([1, 0], Run([0, 2, 1, 1e9f, 7, 0, 0, 1e9f], 3, 0f, 0, 1f, 0f, 1, 0, stride: 4));
    }
}

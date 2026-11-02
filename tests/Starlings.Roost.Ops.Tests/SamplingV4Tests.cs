using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>
/// The contract version 4 sampling features (R-042, CPU half): penalties over token counts and a prompt mask, stop flags,
/// top-k logprobs, the float32 math mode and per-row Philox draws under batch reordering.
/// </summary>
public sealed class SamplingV4Tests
{
    /// <summary>One sampling row's parameters.</summary>
    private sealed record Row(float[] Logits)
    {
        public float Temperature { get; init; }

        public int TopK { get; init; }

        public float TopP { get; init; } = 1f;

        public float MinP { get; init; }

        public ulong Seed { get; init; } = 1;

        public ulong Offset { get; init; }

        public float Repetition { get; init; } = 1f;

        public float Presence { get; init; }

        public float Frequency { get; init; }

        public int[]? Counts { get; init; }

        public uint[]? Prompt { get; init; }

        public int[] Stops { get; init; } = [];
    }

    private sealed record Output(int[] Ids, float[] Logprobs, int[] TopIds, float[] TopValues, int[] StopFlags);

    private static Output Run(Row[] rows, int top = 0, SampleMath math = SampleMath.Reference, bool penalties = false)
    {
        int vocab = rows[0].Logits.Length;
        int words = (vocab + 31) / 32;
        int n = rows.Length;
        var stopIds = rows.SelectMany(r => r.Stops).ToArray();
        var ranges = new int[2 * n];
        int at = 0;
        for (int r = 0; r < n; r++)
        {
            ranges[2 * r] = at;
            ranges[(2 * r) + 1] = rows[r].Stops.Length;
            at += rows[r].Stops.Length;
        }

        using var l = Mem.F32([.. rows.SelectMany(r => r.Logits)]);
        using var t = Mem.F32([.. rows.Select(r => r.Temperature)]);
        using var k = Mem.I32([.. rows.Select(r => r.TopK)]);
        using var p = Mem.F32([.. rows.Select(r => r.TopP)]);
        using var mp = Mem.F32([.. rows.Select(r => r.MinP)]);
        using var s = Mem.U64([.. rows.Select(r => r.Seed)]);
        using var o = Mem.U64([.. rows.Select(r => r.Offset)]);
        using var rep = Mem.F32([.. rows.Select(r => r.Repetition)]);
        using var pres = Mem.F32([.. rows.Select(r => r.Presence)]);
        using var freq = Mem.F32([.. rows.Select(r => r.Frequency)]);
        using var counts = Mem.I32([.. rows.SelectMany(r => r.Counts ?? new int[vocab])]);
        using var prompt = Mem.U32([.. rows.SelectMany(r => r.Prompt ?? new uint[words])]);
        using var stops = Mem.I32(stopIds.Length == 0 ? new int[1] : stopIds);
        using var stopRanges = Mem.I32(ranges);
        using var ids = Mem.Zeros(DType.I32, n);
        using var lp = Mem.Zeros(DType.F32, n);
        using var flags = Mem.Zeros(DType.I32, n);
        using var topIds = Mem.Zeros(DType.I32, Math.Max(1, n * top));
        using var topValues = Mem.Zeros(DType.F32, Math.Max(1, n * top));
        CpuOps.Sample(new SampleArgs
        {
            Logits = l.Span, Temperature = t.Span, TopK = k.Span, TopP = p.Span, MinP = mp.Span, Seeds = s.Span, Offsets = o.Span,
            Ids = ids.Span, Logprobs = lp.Span, Rows = n, Vocab = vocab, Stride = vocab,
            RepetitionPenalty = rep.Span, PresencePenalty = pres.Span, FrequencyPenalty = freq.Span,
            TokenCounts = penalties ? counts.Span : default, PromptMask = penalties ? prompt.Span : default,
            StopIds = stops.Span, StopRanges = stopRanges.Span, StopFlags = flags.Span,
            TopLogprobIds = top > 0 ? topIds.Span : default, TopLogprobValues = top > 0 ? topValues.Span : default, TopLogprobs = top,
            Math = math,
        });
        return new Output(ids.ReadI32(), lp.ReadF32(), topIds.ReadI32(), topValues.ReadF32(), flags.ReadI32());
    }

    [Theory]
    [InlineData(SampleMath.Reference)]
    [InlineData(SampleMath.Fp32)]
    public void RepetitionPenaltyDividesPositiveAndMultipliesNegativeSeenLogits(SampleMath math)
    {
        // Greedy over [4, 3, -1, -1.5]: token 0 wins. Penalising token 0 (count 1) by 2 makes it 2, so token 1 (3) wins.
        var row = new Row([4f, 3f, -1f, -1.5f]) { Repetition = 2f, Counts = [1, 0, 0, 0] };
        Assert.Equal(1, Run([row], math: math, penalties: true).Ids[0]);

        // A seen negative logit is multiplied: token 2 (in the prompt) at -1 becomes -2, below token 3's -1.5; divided it would
        // be -0.5 and win. Tokens 0 and 1 are pushed down by the frequency penalty.
        var negative = new Row([4f, 3f, -1f, -1.5f]) { Repetition = 2f, Frequency = 2f, Counts = [10, 10, 0, 0], Prompt = [0b100u] };
        Assert.Equal(3, Run([negative], math: math, penalties: true).Ids[0]);
    }

    [Theory]
    [InlineData(SampleMath.Reference)]
    [InlineData(SampleMath.Fp32)]
    public void ThePromptMaskFeedsTheRepetitionPenaltyOnly(SampleMath math)
    {
        // Token 0 is in the prompt only (count 0): the repetition penalty applies (4 / 2 = 2 < 3), presence and frequency do not.
        var rep = new Row([4f, 3f]) { Repetition = 2f, Counts = [0, 0], Prompt = [0b01u] };
        Assert.Equal(1, Run([rep], math: math, penalties: true).Ids[0]);
        var presence = new Row([4f, 3f]) { Presence = 2f, Frequency = 2f, Counts = [0, 0], Prompt = [0b01u] };
        Assert.Equal(0, Run([presence], math: math, penalties: true).Ids[0]);
    }

    [Theory]
    [InlineData(SampleMath.Reference)]
    [InlineData(SampleMath.Fp32)]
    public void FrequencyScalesWithTheCountAndPresenceDoesNot(SampleMath math)
    {
        // Token 0 leads by 1.5 and occurred 3 times. Frequency 0.4: 4 - 1.2 = 2.8 > 2.5, token 0 stays; frequency 0.6: 2.2 < 2.5.
        float[] logits = [4f, 2.5f];
        Assert.Equal(0, Run([new Row(logits) { Frequency = 0.4f, Counts = [3, 0] }], math: math, penalties: true).Ids[0]);
        Assert.Equal(1, Run([new Row(logits) { Frequency = 0.6f, Counts = [3, 0] }], math: math, penalties: true).Ids[0]);

        // Presence 1.4 removes 1.4 once whatever the count: 2.6 > 2.5.
        Assert.Equal(0, Run([new Row(logits) { Presence = 1.4f, Counts = [30, 0] }], math: math, penalties: true).Ids[0]);
        Assert.Equal(1, Run([new Row(logits) { Presence = 1.6f, Counts = [1, 0] }], math: math, penalties: true).Ids[0]);
    }

    [Fact]
    public void PenaltiesMoveTheSampledDistribution()
    {
        // Two equal logits; a presence penalty of ln 3 on token 0 gives probabilities 1/4 and 3/4.
        float[] logits = [0f, 0f];
        int ones = 0;
        const int n = 4000;
        for (ulong i = 0; i < n; i++)
        {
            ones += Run([new Row(logits) { Temperature = 1f, Seed = 7, Offset = i, Presence = (float)Math.Log(3), Counts = [1, 0] }], penalties: true).Ids[0];
        }

        Assert.InRange(ones / (double)n, 0.72, 0.78);
    }

    [Fact]
    public void LogprobsAreOfTheRawLogitsBeforePenalties()
    {
        float[] logits = [4f, 3f, 0f];
        var o = Run([new Row(logits) { Repetition = 4f, Counts = [1, 0, 0] }], top: 2, penalties: true);
        double z = Math.Log(Math.Exp(4) + Math.Exp(3) + 1);
        Assert.Equal(1, o.Ids[0]);
        Assert.Equal(3 - z, o.Logprobs[0], 5);
        Assert.Equal([0, 1], o.TopIds);
        Assert.Equal(4 - z, o.TopValues[0], 5);
        Assert.Equal(3 - z, o.TopValues[1], 5);
    }

    [Theory]
    [InlineData(SampleMath.Reference)]
    [InlineData(SampleMath.Fp32)]
    public void TopLogprobsAreTheHighestRawLogitsWithLowerIndexTiesAndPadding(SampleMath math)
    {
        float[] logits = [1f, 5f, 3f, 5f, float.NaN];
        var o = Run([new Row(logits)], top: 3, math: math);
        Assert.Equal([1, 3, 2], o.TopIds);
        Assert.Equal(o.TopValues[0], o.TopValues[1]);
        Assert.True(o.TopValues[1] > o.TopValues[2]);

        // Five tokens, one NaN: asking for five pads the last with -1 and -inf.
        var all = Run([new Row(logits)], top: 5, math: math);
        Assert.Equal([1, 3, 2, 0, -1], all.TopIds);
        Assert.Equal(float.NegativeInfinity, all.TopValues[4]);
    }

    [Fact]
    public void StopFlagsMarkRowsWhoseTokenIsOneOfTheirOwnStopIds()
    {
        Row[] rows =
        [
            new Row([0f, 9f, 0f]) { Stops = [1] },
            new Row([0f, 9f, 0f]) { Stops = [2] },
            new Row([9f, 0f, 0f]) { Stops = [] },
            new Row([0f, 0f, 9f]) { Stops = [0, 2] },
        ];
        var o = Run(rows);
        Assert.Equal([1, 1, 0, 2], o.Ids);
        Assert.Equal([1, 0, 0, 1], o.StopFlags);
    }

    [Theory]
    [InlineData(SampleMath.Reference)]
    [InlineData(SampleMath.Fp32)]
    public void EachRowDrawsFromItsOwnSeedAndOffsetInAnyBatchOrder(SampleMath math)
    {
        var rng = new Random(11);
        var rows = Enumerable.Range(0, 6).Select(i => new Row([.. Enumerable.Range(0, 50).Select(_ => (float)((rng.NextDouble() * 6) - 3))])
        {
            Temperature = 0.9f,
            TopK = i % 2 == 0 ? 20 : 0,
            TopP = 0.95f,
            MinP = 0.01f,
            Seed = (ulong)(1000 + i),
            Offset = (ulong)(3 * i),
            Frequency = 0.3f,
            Counts = [.. Enumerable.Range(0, 50).Select(v => (v + i) % 7 == 0 ? 2 : 0)],
        }).ToArray();
        var forward = Run(rows, top: 3, math: math, penalties: true);
        var reversed = Run([.. rows.Reverse()], top: 3, math: math, penalties: true);
        for (int i = 0; i < rows.Length; i++)
        {
            int j = rows.Length - 1 - i;
            Assert.Equal(forward.Ids[i], reversed.Ids[j]);
            Assert.Equal(forward.Logprobs[i], reversed.Logprobs[j]);
            Assert.Equal(forward.TopIds.AsSpan(3 * i, 3).ToArray(), reversed.TopIds.AsSpan(3 * j, 3).ToArray());
            Assert.Equal(forward.TopValues.AsSpan(3 * i, 3).ToArray(), reversed.TopValues.AsSpan(3 * j, 3).ToArray());
            var alone = Run([rows[i]], top: 3, math: math, penalties: true);
            Assert.Equal(forward.Ids[i], alone.Ids[0]);
        }
    }

    [Fact]
    public void Fp32DrawsAgreeWithTheReferenceAwayFromProbabilityBoundaries()
    {
        // The two modes differ only in rounding, so a draw can differ only when u * mass falls within rounding of a cumulative
        // boundary. With 2000 draws over a smooth distribution that never happens at these magnitudes.
        var rng = new Random(3);
        float[] logits = [.. Enumerable.Range(0, 100).Select(_ => (float)((rng.NextDouble() * 8) - 4))];
        int same = 0;
        for (ulong i = 0; i < 2000; i++)
        {
            var row = new Row(logits) { Temperature = 0.7f, TopP = 0.9f, Seed = 42, Offset = i };
            same += Run([row]).Ids[0] == Run([row], math: SampleMath.Fp32).Ids[0] ? 1 : 0;
        }

        Assert.True(same >= 1998, $"{same} of 2000 draws agree");
    }

    [Fact]
    public void Fp32LogprobsAreWithinFloatRoundingOfTheReference()
    {
        var rng = new Random(8);
        float[] logits = [.. Enumerable.Range(0, 1000).Select(_ => (float)((rng.NextDouble() * 20) - 10))];
        var reference = Run([new Row(logits)], top: 5);
        var fp32 = Run([new Row(logits)], top: 5, math: SampleMath.Fp32);
        Assert.Equal(reference.TopIds, fp32.TopIds);
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(reference.TopValues[i], fp32.TopValues[i], 5);
        }
    }

    [Fact]
    public void Fp32IsBitwiseRepeatable()
    {
        var rng = new Random(21);
        var rows = Enumerable.Range(0, 3).Select(i => new Row([.. Enumerable.Range(0, 300).Select(_ => (float)((rng.NextDouble() * 10) - 5))])
        {
            Temperature = 1.3f, Seed = 9, Offset = (ulong)i,
        }).ToArray();
        var a = Run(rows, top: 4, math: SampleMath.Fp32);
        var b = Run(rows, top: 4, math: SampleMath.Fp32);
        Assert.Equal(a.Ids, b.Ids);
        Assert.Equal(a.Logprobs.Select(BitConverter.SingleToInt32Bits), b.Logprobs.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(a.TopValues.Select(BitConverter.SingleToInt32Bits), b.TopValues.Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void Fp32LogprobFollowsTheDocumentedOperationSequence()
    {
        // (x - max) - log(sum) with the 32-lane sum of Fp32Math.Exp, recomputed here from the documented definition.
        var rng = new Random(4);
        float[] logits = [.. Enumerable.Range(0, 77).Select(_ => (float)((rng.NextDouble() * 12) - 6))];
        float max = logits.Max();
        Span<float> lanes = stackalloc float[32];
        lanes.Clear();
        for (int v = 0; v < logits.Length; v++)
        {
            lanes[v % 32] += Fp32Math.Exp(logits[v] - max);
        }

        for (int w = 16; w >= 1; w /= 2)
        {
            for (int j = 0; j < w; j++)
            {
                lanes[j] += lanes[j + w];
            }
        }

        float lse = Fp32Math.Log(lanes[0]);
        var o = Run([new Row(logits)], math: SampleMath.Fp32);
        Assert.Equal(BitConverter.SingleToInt32Bits((logits[o.Ids[0]] - max) - lse), BitConverter.SingleToInt32Bits(o.Logprobs[0]));
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        float[] logits = [0f, 1f];
        Assert.Throws<ArgumentException>(() => Run([new Row(logits)], top: 3));
        Assert.Throws<ArgumentException>(() => Run([new Row(logits)], math: (SampleMath)7));
        using var l = Mem.F32(logits);
        using var f = Mem.F32(0f);
        using var i = Mem.I32(0);
        using var u = Mem.U64(0);
        using var ids = Mem.Zeros(DType.I32, 1);
        using var flags = Mem.Zeros(DType.I32, 1);
        using var ranges = Mem.I32(0, 2);
        using var stops = Mem.I32(1);
        Assert.Throws<ArgumentException>(() => CpuOps.Sample(new SampleArgs
        {
            Logits = l.Span, Temperature = f.Span, TopK = i.Span, TopP = f.Span, MinP = f.Span, Seeds = u.Span, Offsets = u.Span, Ids = ids.Span,
            Rows = 1, Vocab = 2, StopFlags = flags.Span, StopRanges = ranges.Span, StopIds = stops.Span,
        }));
    }
}

using Starlings.Roost.Ops;

namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>
/// roost-tiny-hybrid (Qwen3.5-shaped) run on CpuOps: the generated forward with the fused q/k/v views and the v3 gating ops, over a
/// paged KV cache and a recurrent state pool, against an independent double-precision transcription
/// (<see cref="TinyHybridReference"/>), then against itself: chunked prefill against one call, and several sequences in one batch
/// against each alone.
/// </summary>
public sealed class RoostTinyHybridNumericTests : IDisposable
{
    private static readonly TinyConfig C = TinyConfig.Hybrid;
    private readonly HybridRunner _runner = new();

    public void Dispose() => _runner.Dispose();

    private static int[] Prompt(int length, int salt) => Enumerable.Range(0, length).Select(i => ((i * 37) + (salt * 101) + 11) % C.Vocab).ToArray();

    private static void AssertMatchesReference(double[] want, float[] got, string what) => Tolerances.AssertMatchesReference(want, got, what);

    [Fact]
    public void PrefillLogitsMatchTheIndependentReference()
    {
        int[] prompt = Prompt(20, 1);
        var reference = new TinyHybridReference(C, _runner.Weights, roundActivations: true);
        var seq = new HybridRunner.Sequence();
        float[] got = _runner.Step((seq, prompt))[0];
        AssertMatchesReference(reference.Logits(prompt)[^1], got, "prefill of 20 tokens (two blocks)");
        Assert.Equal(2, seq.Blocks.Count);
    }

    [Fact]
    public void DecodeOneTokenAtATimeMatchesTheReferenceAtEveryPosition()
    {
        int[] tokens = Prompt(21, 2);
        var reference = new TinyHybridReference(C, _runner.Weights, roundActivations: true);
        double[][] want = reference.Logits(tokens);
        var seq = new HybridRunner.Sequence();
        for (int p = 0; p < tokens.Length; p++)
        {
            AssertMatchesReference(want[p], _runner.Step((seq, new[] { tokens[p] }))[0], $"position {p}");
        }
    }

    [Fact]
    public void ChunkedPrefillAndMixedPrefillDecodeBatchesMatchTheReference()
    {
        int[] a = Prompt(23, 3);
        int[] b = Prompt(9, 4);
        var reference = new TinyHybridReference(C, _runner.Weights, roundActivations: true);
        double[][] wantA = reference.Logits(a);
        double[][] wantB = reference.Logits(b);
        var sa = new HybridRunner.Sequence();
        var sb = new HybridRunner.Sequence();

        float[][] r = _runner.Step((sa, a[..11]), (sb, b));
        AssertMatchesReference(wantA[10], r[0], "A chunk 1");
        AssertMatchesReference(wantB[^1], r[1], "B prompt");

        int[] bMore = Prompt(3, 5);
        double[][] wantBMore = reference.Logits([.. b, .. bMore]);
        r = _runner.Step((sa, a[11..19]), (sb, bMore[..1]));
        AssertMatchesReference(wantA[18], r[0], "A chunk 2");
        AssertMatchesReference(wantBMore[b.Length], r[1], "B decode 1");
        r = _runner.Step((sb, bMore[1..2]), (sa, a[19..]));
        AssertMatchesReference(wantBMore[b.Length + 1], r[0], "B decode 2");
        AssertMatchesReference(wantA[^1], r[1], "A chunk 3");

        Assert.Empty(sa.Blocks.Intersect(sb.Blocks));
    }

    [Fact]
    public void GreedyGenerationFollowsTheReferenceAndIsPinned()
    {
        int[] prompt = Prompt(18, 6);
        var reference = new TinyHybridReference(C, _runner.Weights, roundActivations: true);
        int[] want = reference.Greedy(prompt, 8);
        int[] got = _runner.Generate(prompt, 8);
        Assert.Equal(want, got);

        // No step was a near-tie, so the equality above does not hinge on bf16 rounding.
        var seq = prompt.ToList();
        foreach (int next in want)
        {
            double[] logits = reference.Logits(seq.ToArray())[^1];
            double top = logits.Max();
            double second = logits.Where(x => x < top).Max();
            Assert.True(top - second > Tolerances.GreedyMargin, $"top two logits are {top} and {second}");
            seq.Add(next);
        }

        Assert.True(PinnedGeneration.SequenceEqual(got), string.Join(",", got));
    }

    [Fact]
    public void Bfloat16ActivationsStayNearTheUnroundedReference()
    {
        // The same forward in plain double, nothing rounded: what is left is the bfloat16 noise of the activations, which is
        // large at these magnitudes (logits of a few units) but must not drift with position.
        int[] tokens = Prompt(21, 2);
        var plain = new TinyHybridReference(C, _runner.Weights);
        double[][] want = plain.Logits(tokens);
        var seq = new HybridRunner.Sequence();
        double worst = 0;
        for (int p = 0; p < tokens.Length; p++)
        {
            worst = Math.Max(worst, Tolerances.Worst(want[p], _runner.Step((seq, new[] { tokens[p] }))[0]));
        }

        TestContext.Current.SendDiagnosticMessage($"MEASURED unrounded hybrid decode 21 positions: worst scaled {worst:E3}");
        Assert.True(worst < Tolerances.HybridUnrounded, $"worst scaled difference {worst}");
    }

    [Fact]
    public void TheRecurrentStateCarriesHistory()
    {
        // The same last token after two different histories gives different logits: the delta-rule and convolution state are read.
        var a = new HybridRunner.Sequence();
        var b = new HybridRunner.Sequence();
        int[] h1 = Prompt(12, 8);
        int[] h2 = Prompt(12, 9);
        _runner.Step((a, h1), (b, h2));
        float[][] r = _runner.Step((a, SameToken), (b, SameToken));
        double worst = 0;
        for (int i = 0; i < r[0].Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(r[0][i] - r[1][i]));
        }

        Assert.True(worst > 0.1, $"the logits after different histories differ by only {worst}");
    }

    [Fact]
    public void ChunkedPrefillEqualsOneCallWithinTheSelfConsistencyTolerance()
    {
        int[] prompt = Prompt(37, 10);
        int[] next = Prompt(4, 11);
        int[][] chunkings = Chunkings;
        float[][] baseline = Chunked(_runner, prompt, next, chunkings[0]);
        for (int c = 1; c < chunkings.Length; c++)
        {
            using var other = new HybridRunner();
            float[][] got = Chunked(other, prompt, next, chunkings[c]);
            for (int step = 0; step < baseline.Length; step++)
            {
                Tolerances.AssertSelfConsistent(baseline[step], got[step], $"chunking {string.Join('+', chunkings[c])}, step {step}");
            }
        }
    }

    [Fact]
    public void SequencesInOneBatchEqualEachAloneWithinTheSelfConsistencyTolerance()
    {
        int[] a = Prompt(25, 12);
        int[] b = Prompt(7, 13);
        int[] c = Prompt(18, 14);
        int[] aNext = Prompt(3, 15);
        int[] bNext = Prompt(3, 16);
        int[] cNext = Prompt(3, 17);

        // Together: one prefill batch of three, then three decode steps with all of them in each plan.
        var sa = new HybridRunner.Sequence();
        var sb = new HybridRunner.Sequence();
        var sc = new HybridRunner.Sequence();
        var together = new List<float[][]> { _runner.Step((sa, a), (sb, b), (sc, c)) };
        for (int i = 0; i < 3; i++)
        {
            together.Add(_runner.Step((sa, aNext[i..(i + 1)]), (sb, bNext[i..(i + 1)]), (sc, cNext[i..(i + 1)])));
        }

        // Alone: each on a fresh runner (its own pools), the same steps.
        int[][] prompts = [a, b, c];
        int[][] nexts = [aNext, bNext, cNext];
        for (int s = 0; s < 3; s++)
        {
            using var alone = new HybridRunner();
            var seq = new HybridRunner.Sequence();
            Tolerances.AssertSelfConsistent(alone.Step((seq, prompts[s]))[0], together[0][s], $"sequence {s} prefill");
            for (int i = 0; i < 3; i++)
            {
                Tolerances.AssertSelfConsistent(alone.Step((seq, nexts[s][i..(i + 1)]))[0], together[i + 1][s], $"sequence {s} decode {i}");
            }
        }
    }

    // Logits after each of: the whole prompt in the given chunks, then each of the continuation tokens one at a time.
    private static float[][] Chunked(HybridRunner runner, int[] prompt, int[] next, int[] chunks)
    {
        Assert.Equal(prompt.Length, chunks.Sum());
        var seq = new HybridRunner.Sequence();
        var result = new List<float[]>();
        int at = 0;
        float[] last = [];
        foreach (int n in chunks)
        {
            last = runner.Step((seq, prompt[at..(at + n)]))[0];
            at += n;
        }

        result.Add(last);
        foreach (int token in next)
        {
            result.Add(runner.Step((seq, new[] { token }))[0]);
        }

        return result.ToArray();
    }

    private static readonly int[] SameToken = [5];

    private static readonly int[][] Chunkings = [[37], [20, 17], [1, 16, 1, 19], [5, 5, 5, 5, 5, 5, 5, 2], Enumerable.Repeat(1, 37).ToArray()];

    // Greedy ids for Prompt(18, 6) from the seed in TinyConfig.Hybrid; any change to the weights, the definition or CpuOps moves them.
    private static readonly int[] PinnedGeneration = [222, 222, 222, 222, 222, 73, 46, 250];
}

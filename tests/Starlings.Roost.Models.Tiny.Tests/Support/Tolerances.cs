namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>
/// The tolerances of the tiny-model tests, with the measured differences next to them in the diagnostics of every run.
/// A tolerance is a worst difference scaled as |got - want| / (1 + |want|) over a whole logits row.
/// </summary>
public static class Tolerances
{
    /// <summary>roost-tiny (float32) generated forward against the double transcription.</summary>
    public const double DenseReference = 1e-5;

    /// <summary>roost-tiny-hybrid (bfloat16 activations) generated forward against the double transcription.</summary>
    public const double HybridReference = 0.25;

    /// <summary>The median scaled difference over a logits row, same comparison: tie flips move a row by about 0.01, a wrong op by much more.</summary>
    public const double HybridReferenceMedian = 0.03;

    /// <summary>roost-tiny-hybrid against the plain double transcription with nothing rounded: the bfloat16 noise of its activations.</summary>
    public const double HybridUnrounded = 3.0;

    /// <summary>roost-tiny, one call against chunked or batched runs of the same forward.</summary>
    public const double DenseSelf = 1e-6;

    /// <summary>roost-tiny-hybrid, one call against chunked or batched runs of the same forward.</summary>
    public const double HybridSelf = 0.25;

    /// <summary>The median of the same comparison.</summary>
    public const double HybridSelfMedian = 0.03;

    /// <summary>Smallest gap between the top two reference logits for a greedy step not to count as a near-tie.</summary>
    public const double GreedyMargin = 1e-3;

    public static double Worst(double[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        double worst = 0;
        for (int i = 0; i < want.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(got[i] - want[i]) / (1 + Math.Abs(want[i])));
        }

        return worst;
    }

    public static double Median(double[] want, float[] got)
    {
        double[] d = want.Select((w, i) => Math.Abs(got[i] - w) / (1 + Math.Abs(w))).Order().ToArray();
        return d[d.Length / 2];
    }

    public static void AssertMatchesReference(double[] want, float[] got, string what)
    {
        double worst = Worst(want, got);
        TestContext.Current.SendDiagnosticMessage($"MEASURED reference hybrid {what}: worst scaled {worst:E3}, median {Median(want, got):E3}");
        Assert.True(worst < HybridReference, $"{what}: worst scaled difference {worst}");
        Assert.True(Median(want, got) < HybridReferenceMedian, $"{what}: median scaled difference {Median(want, got)}");
    }

    public static void AssertSelfConsistent(float[] want, float[] got, string what)
    {
        AssertSelfConsistent(want, got, what, "hybrid", HybridSelf);
        double[] w = want.Select(x => (double)x).ToArray();
        Assert.True(Median(w, got) < HybridSelfMedian, $"{what}: median scaled difference {Median(w, got)}");
    }

    public static void AssertSelfConsistent(float[] want, float[] got, string what, string model, double tolerance)
    {
        double worst = Worst(want.Select(x => (double)x).ToArray(), got);
        TestContext.Current.SendDiagnosticMessage($"MEASURED self {model} {what}: worst scaled {worst:E3}");
        Assert.True(worst < tolerance, $"{what}: worst scaled difference {worst}");
    }
}

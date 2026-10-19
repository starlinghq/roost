using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.Ops;
using Starlings.Roost.State;

namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>roost-tiny (Qwen3-shaped): the generated structure, then the generated forward on CpuOps against an independent transcription.</summary>
public sealed class RoostTinyTests : IDisposable
{
    private static readonly TinyConfig C = TinyConfig.Dense;
    private readonly TinyRunner _runner = new();

    public void Dispose() => _runner.Dispose();

    private static int[] Prompt(int length, int salt) => Enumerable.Range(0, length).Select(i => ((i * 37) + (salt * 101) + 11) % C.Vocab).ToArray();

    private static void AssertClose(double[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        double worst = 0;
        for (int i = 0; i < want.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(got[i] - want[i]) / (1 + Math.Abs(want[i])));
        }

        Assert.True(worst < 1e-4, $"{what}: worst scaled difference {worst}");
    }

    [Fact]
    public void ConfigurationMatchesTheGeneratedDefinition()
    {
        Assert.Equal(RoostTiny.ModelName, C.Name);
        var slots = RoostTiny.WeightSlots.ToDictionary(s => s.Name);
        Assert.Equal(1 + (C.Layers * 10) + 2, slots.Count);
        Assert.Equal($"{C.Vocab},{C.Hidden}", slots["embed"].Shape);
        Assert.Equal($"{C.Vocab},{C.Hidden}", slots["lm_head"].Shape);
        Assert.Equal($"{C.Heads * C.HeadDim},{C.Hidden}", slots["l0.q_proj"].Shape);
        Assert.Equal($"{C.KvHeads * C.HeadDim},{C.Hidden}", slots["l3.k_proj"].Shape);
        Assert.Equal($"{C.HeadDim}", slots["l2.q_norm"].Shape);
        Assert.Equal($"{2 * C.Intermediate},{C.Hidden}", slots["l1.gate_up"].Shape);
        Assert.Equal($"{C.Hidden},{C.Intermediate}", slots["l1.down"].Shape);
        Assert.False(C.TiedHead);
        Assert.All(RoostTiny.WeightSlots, s => Assert.Equal(DType.F32, s.Dtype));
        Assert.All(Enumerable.Range(0, C.Layers), l => Assert.Equal(TinyLayerType.FullAttention, C.LayerType(l)));
    }

    [Fact]
    public void WeightSlotsFollowTheQwen3CheckpointLayerByLayer()
    {
        var names = RoostTiny.WeightSlots.Select(s => s.Name).ToArray();
        Assert.Equal("embed", names[0]);
        Assert.Equal(
            ["l0.input_norm", "l0.q_proj", "l0.k_proj", "l0.v_proj", "l0.q_norm", "l0.k_norm", "l0.o_proj", "l0.post_norm", "l0.gate_up", "l0.down"],
            names.Skip(1).Take(10).ToArray());
        Assert.Equal(["final_norm", "lm_head"], names.TakeLast(2).ToArray());
        Assert.Equal("model.layers.2.self_attn.q_norm.weight", RoostTiny.WeightSlots.Single(s => s.Name == "l2.q_norm").Source);
        Assert.Equal("model.embed_tokens.weight", RoostTiny.WeightSlots[0].Source);
        Assert.Equal("lm_head.weight", RoostTiny.WeightSlots[^1].Source);
        Assert.All(RoostTiny.WeightSlots, s => Assert.False(string.IsNullOrEmpty(s.Source)));
    }

    [Fact]
    public void OneFullAttentionGroupCoversEveryLayer()
    {
        var groups = RoostTiny.CreateStateGroups();
        var g = Assert.Single(groups);
        Assert.Equal(StateGroupKind.Full, g.Kind);
        Assert.Equal(C.BlockSize, g.BlockSize);
        Assert.Equal(Enumerable.Range(0, C.Layers).Select(l => new LayerKvShape(l, C.KvHeads, C.HeadDim)).ToArray(), g.Layers.ToArray());
        Assert.Equal(2L * C.Layers * C.BlockSize * C.KvHeads * C.HeadDim * sizeof(float), g.BytesPerBlock);
        Assert.Equal(2 * C.Layers, RoostTiny.StateSlots.Length);
    }

    [Fact]
    public void GraphHasTheQwen3LayerPattern()
    {
        string[] layer = ["RmsNorm", "Gemm", "Gemm", "Gemm", "QkNorm", "Rope", "CacheWrite", "PagedAttention", "Gemm", "Residual", "RmsNorm", "Gemm", "SiluAndMul", "Gemm", "Residual"];
        var expected = new List<string> { "Embedding" };
        for (int l = 0; l < C.Layers; l++)
        {
            expected.AddRange(layer);
        }

        expected.AddRange(["RmsNorm", "LmHead"]);
        var graph = RoostTiny.BuildGraph();
        Assert.Equal(expected, graph.Nodes.Select(n => n.Op).ToArray());
        Assert.Equal("1000000", graph.Nodes.First(n => n.Op == "Rope").Attributes["arg.Theta"].Trim('(', ')'));
        RoostTiny.CheckBackend<CpuOps>();
    }

    [Fact]
    public void PrefillLogitsMatchTheIndependentReference()
    {
        int[] prompt = Prompt(20, 1);
        var reference = new TinyReference(C, _runner.Weights);
        var seq = new TinyRunner.Sequence();
        float[] got = _runner.Step((seq, prompt))[0];
        AssertClose(reference.Logits(prompt)[^1], got, "prefill of 20 tokens (two blocks)");
        Assert.Equal(2, seq.Blocks.Count);
    }

    [Fact]
    public void DecodeOneTokenAtATimeMatchesTheReferenceAtEveryPosition()
    {
        int[] tokens = Prompt(21, 2);
        var reference = new TinyReference(C, _runner.Weights);
        double[][] want = reference.Logits(tokens);
        var seq = new TinyRunner.Sequence();
        for (int p = 0; p < tokens.Length; p++)
        {
            AssertClose(want[p], _runner.Step((seq, new[] { tokens[p] }))[0], $"position {p}");
        }
    }

    [Fact]
    public void ChunkedPrefillAndMixedPrefillDecodeBatchesMatchTheReference()
    {
        int[] a = Prompt(23, 3);
        int[] b = Prompt(9, 4);
        var reference = new TinyReference(C, _runner.Weights);
        double[][] wantA = reference.Logits(a);
        double[][] wantB = reference.Logits(b);
        var sa = new TinyRunner.Sequence();
        var sb = new TinyRunner.Sequence();

        // Step 1: A's first 11 tokens (a prefill chunk) with all of B's 9 (a whole prompt) in one plan.
        float[][] r = _runner.Step((sa, a[..11]), (sb, b));
        AssertClose(wantA[10], r[0], "A chunk 1");
        AssertClose(wantB[^1], r[1], "B prompt");

        // Steps 2 and 3: A continues its prefill in chunks while B decodes the next scripted token.
        int[] bMore = Prompt(3, 5);
        double[][] wantBMore = reference.Logits([.. b, .. bMore]);
        r = _runner.Step((sa, a[11..19]), (sb, bMore[..1]));
        AssertClose(wantA[18], r[0], "A chunk 2");
        AssertClose(wantBMore[b.Length], r[1], "B decode 1");
        r = _runner.Step((sb, bMore[1..2]), (sa, a[19..]));
        AssertClose(wantBMore[b.Length + 1], r[0], "B decode 2");
        AssertClose(wantA[^1], r[1], "A chunk 3");

        // The two sequences share the pool and own disjoint blocks.
        Assert.Empty(sa.Blocks.Intersect(sb.Blocks));
        Assert.Equal(2, sa.Blocks.Count);
        Assert.Single(sb.Blocks);
    }

    [Fact]
    public void GreedyGenerationFollowsTheReferenceAndIsPinned()
    {
        int[] prompt = Prompt(18, 6);
        var reference = new TinyReference(C, _runner.Weights);
        int[] want = reference.Greedy(prompt, 8);
        int[] got = _runner.Generate(prompt, 8);
        Assert.Equal(want, got);

        // No step was a near-tie, so the equality above does not hinge on rounding.
        var seq = prompt.ToList();
        foreach (int next in want)
        {
            double[] logits = reference.Logits(seq.ToArray())[^1];
            double top = logits.Max();
            double second = logits.Where(x => x < top).Max();
            Assert.True(top - second > 1e-3, $"top two logits are {top} and {second}");
            seq.Add(next);
        }

        Assert.True(PinnedGeneration.SequenceEqual(got), string.Join(",", got));
    }

    [Fact]
    public void GenerationIsRepeatableOnAFreshRunner()
    {
        int[] prompt = Prompt(7, 7);
        int[] first = _runner.Generate(prompt, 6);
        using var other = new TinyRunner(blocks: 40);
        Assert.Equal(first, other.Generate(prompt, 6));
    }

    // Greedy ids for Prompt(18, 6) from the seed in TinyConfig.Dense; any change to the weights, the definition or CpuOps moves them.
    private static readonly int[] PinnedGeneration = [57, 4, 41, 54, 132, 98, 231, 164];
}

using Starlings.Roost.Abi;
using Starlings.Roost.ModelCompiler.Tests;
using Starlings.Roost.Ops;
using Starlings.Roost.State;

namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>
/// roost-tiny-moe (Qwen3-MoE-shaped, R-080): the generated structure, then CPU self-consistency of the generated forward on CpuOps.
/// Every CpuOps op computes each row (token) independently of the others in the call, so chunked, batched and forked runs must give
/// the same bytes as one call; the tests assert equality, not a tolerance.
/// </summary>
public sealed class RoostTinyMoeTests : IDisposable
{
    private static readonly TinyConfig C = TinyConfig.Moe;
    private readonly MoeRunner _runner = new();

    public void Dispose() => _runner.Dispose();

    private static int[] Prompt(int length, int salt) => Enumerable.Range(0, length).Select(i => ((i * 37) + (salt * 101) + 11) % C.Vocab).ToArray();

    private static void AssertSame(float[] want, float[] got, string what)
    {
        double worst = Tolerances.Worst([.. want.Select(x => (double)x)], got);
        TestContext.Current.SendDiagnosticMessage($"MEASURED self moe {what}: worst scaled {worst:E3}");
        Assert.Equal(want, got);
    }

    [Fact]
    public void ConfigurationMatchesTheGeneratedDefinition()
    {
        Assert.Equal(RoostTinyMoe.ModelName, C.Name);
        var slots = RoostTinyMoe.WeightSlots.ToDictionary(s => s.Name);
        Assert.Equal(1 + (C.Layers * 11) + 2, slots.Count);
        Assert.Equal($"{C.Vocab},{C.Hidden}", slots["embed"].Shape);
        Assert.Equal($"{C.Vocab},{C.Hidden}", slots["lm_head"].Shape);
        Assert.Equal($"{C.Heads * C.HeadDim},{C.Hidden}", slots["l0.q_proj"].Shape);
        Assert.Equal($"{C.KvHeads * C.HeadDim},{C.Hidden}", slots["l3.k_proj"].Shape);
        Assert.Equal($"{C.Experts},{C.Hidden}", slots["l1.router"].Shape);
        Assert.Equal($"{C.Experts},{2 * C.MoeIntermediate},{C.Hidden}", slots["l2.experts_gate_up"].Shape);
        Assert.Equal($"{C.Experts},{C.Hidden},{C.MoeIntermediate}", slots["l2.experts_down"].Shape);
        Assert.DoesNotContain(slots.Keys, k => k.EndsWith(".gate_up", StringComparison.Ordinal) || k.EndsWith(".down", StringComparison.Ordinal));
        Assert.All(RoostTinyMoe.WeightSlots, s => Assert.Equal(DType.BF16, s.Dtype));
        Assert.False(C.TiedHead);
        Assert.Equal(0, C.Intermediate);
        Assert.All(Enumerable.Range(0, C.Layers), l => Assert.Equal(TinyLayerType.FullAttention, C.LayerType(l)));
        Assert.Equal(TinyConfig.All.Length, TinyConfig.All.Select(c => c.Seed).Distinct().Count());
    }

    [Fact]
    public void WeightSlotsFollowTheQwen3MoeCheckpointWithStackedExperts()
    {
        var names = RoostTinyMoe.WeightSlots.Select(s => s.Name).ToArray();
        Assert.Equal(
            ["l0.input_norm", "l0.q_proj", "l0.k_proj", "l0.v_proj", "l0.q_norm", "l0.k_norm", "l0.o_proj", "l0.post_norm", "l0.router", "l0.experts_gate_up", "l0.experts_down"],
            names.Skip(1).Take(11).ToArray());
        var bySlot = RoostTinyMoe.WeightSlots.ToDictionary(s => s.Name, s => s.Source);
        Assert.Equal("model.layers.3.mlp.gate.weight", bySlot["l3.router"]);
        Assert.Equal("model.layers.1.mlp.experts.*.gate_proj.weight+model.layers.1.mlp.experts.*.up_proj.weight", bySlot["l1.experts_gate_up"]);
        Assert.Equal("model.layers.1.mlp.experts.*.down_proj.weight", bySlot["l1.experts_down"]);
    }

    [Fact]
    public void OneFullAttentionGroupCoversEveryLayer()
    {
        var g = Assert.Single(RoostTinyMoe.CreateStateGroups());
        Assert.Equal(StateGroupKind.Full, g.Kind);
        Assert.Equal(Enumerable.Range(0, C.Layers).Select(l => new LayerKvShape(l, C.KvHeads, C.HeadDim)).ToArray(), g.Layers.ToArray());
        Assert.Equal(2L * C.Layers * C.BlockSize * C.KvHeads * C.HeadDim * 2, g.BytesPerBlock);
    }

    [Fact]
    public void GraphHasTheQwen3MoeLayerPatternWithARouterAndFusedMoe()
    {
        string[] layer = ["RmsNorm", "Gemm", "Gemm", "Gemm", "QkNorm", "Rope", "CacheWrite", "PagedAttention", "Gemm", "Residual", "RmsNorm", "Gemm", "FusedMoe", "Residual"];
        var expected = new List<string> { "Embedding" };
        for (int l = 0; l < C.Layers; l++)
        {
            expected.AddRange(layer);
        }

        expected.AddRange(["RmsNorm", "LmHead"]);
        var graph = RoostTinyMoe.BuildGraph();
        Assert.Equal(expected, graph.Nodes.Select(n => n.Op).ToArray());
        var moe = graph.Nodes.First(n => n.Op == "FusedMoe");
        Assert.Equal("8", moe.Attributes["arg.Experts"].Trim('(', ')'));
        Assert.Equal("2", moe.Attributes["arg.TopK"].Trim('(', ')'));
        Assert.Equal("32", moe.Attributes["arg.Intermediate"].Trim('(', ')'));
        Assert.Equal("true", moe.Attributes["arg.Renormalize"]);
    }

    [Fact]
    public void CheckBackendNamesFusedMoeWhenABackendLacksIt()
    {
        RoostTinyMoe.CheckBackend<CpuOps>();
        var e = Assert.Throws<NotSupportedException>(RoostTinyMoe.CheckBackend<RecordingOps>);
        Assert.Contains("FusedMoe", e.Message, StringComparison.Ordinal);
        Assert.Contains("roost-tiny-moe", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RoutingSpreadsTokensOverTheExperts()
    {
        // Zero one expert's down projection in layer 0 at a time: the last token's logits change only if some token of the prompt was
        // routed to that expert (its output reaches the last token through the residual stream and the later layers' attention). A
        // router that sent every token to the same two experts would leave six of the eight without effect.
        int[] prompt = Prompt(40, 21);
        float[] baseline = _runner.Step((new MoeRunner.Sequence(), prompt))[0];
        int used = 0;
        long perExpert = (long)C.Hidden * C.MoeIntermediate;
        for (int e = 0; e < C.Experts; e++)
        {
            using var runner = new MoeRunner();
            DeviceSpan down = runner.Weight("l0.experts_down");
            unsafe
            {
                new Span<ushort>((ushort*)down.Address + (e * perExpert), (int)perExpert).Clear();
            }

            float[] got = runner.Step((new MoeRunner.Sequence(), prompt))[0];
            if (!got.SequenceEqual(baseline))
            {
                used++;
            }
        }

        TestContext.Current.SendDiagnosticMessage($"MEASURED moe layer-0 experts that move the logits of a 40-token prompt: {used} of {C.Experts}");
        Assert.True(used >= 6, $"only {used} experts of layer 0 receive tokens");
    }

    [Fact]
    public void ChunkedPrefillEqualsOneCall()
    {
        int[] prompt = Prompt(37, 10);
        int[] next = Prompt(4, 11);
        float[][] baseline = Chunked(_runner, prompt, next, [37]);
        foreach (int[] chunks in Chunkings)
        {
            using var other = new MoeRunner();
            float[][] got = Chunked(other, prompt, next, chunks);
            for (int step = 0; step < baseline.Length; step++)
            {
                AssertSame(baseline[step], got[step], $"chunking {string.Join('+', chunks)}, step {step}");
            }
        }
    }

    [Fact]
    public void SequencesInOneBatchEqualEachAlone()
    {
        int[][] prompts = [Prompt(25, 12), Prompt(7, 13), Prompt(18, 14)];
        int[][] nexts = [Prompt(3, 15), Prompt(3, 16), Prompt(3, 17)];
        var seqs = new[] { new MoeRunner.Sequence(), new MoeRunner.Sequence(), new MoeRunner.Sequence() };
        var together = new List<float[][]> { _runner.Step((seqs[0], prompts[0]), (seqs[1], prompts[1]), (seqs[2], prompts[2])) };
        for (int i = 0; i < 3; i++)
        {
            together.Add(_runner.Step((seqs[0], nexts[0][i..(i + 1)]), (seqs[1], nexts[1][i..(i + 1)]), (seqs[2], nexts[2][i..(i + 1)])));
        }

        for (int s = 0; s < 3; s++)
        {
            using var alone = new MoeRunner();
            var seq = new MoeRunner.Sequence();
            AssertSame(alone.Step((seq, prompts[s]))[0], together[0][s], $"sequence {s} prefill");
            for (int i = 0; i < 3; i++)
            {
                AssertSame(alone.Step((seq, nexts[s][i..(i + 1)]))[0], together[i + 1][s], $"sequence {s} decode {i}");
            }
        }
    }

    [Theory]
    [InlineData(21)] // one full block shared, five tokens of a partial block copied
    [InlineData(32)] // two full blocks shared, nothing copied
    [InlineData(7)] // only a partial block, copied
    public void ForkedSequencesEqualFreshOnes(int prefixLength)
    {
        int[] prefix = Prompt(prefixLength, 30);
        int[][] suffixes = [Prompt(9, 31), Prompt(4, 32)];
        var parent = new MoeRunner.Sequence();
        _runner.Step((parent, prefix));
        var children = suffixes.Select(_ => _runner.Fork(parent)).ToArray();

        // Both children continue in one batch, then decode a token each; the parent decodes alongside, so a shared block written by
        // the wrong sequence would show in one of the three.
        int[] parentNext = [5];
        int[] parentDecode = [29];
        int[][] continuations = [[17], [23]];
        float[][] prefill = _runner.Step((children[0], suffixes[0]), (children[1], suffixes[1]), (parent, parentNext));
        float[][] decode = _runner.Step((children[0], continuations[0]), (children[1], continuations[1]), (parent, parentDecode));

        for (int c = 0; c < 2; c++)
        {
            using var fresh = new MoeRunner();
            var seq = new MoeRunner.Sequence();
            AssertSame(fresh.Step((seq, prefix.Concat(suffixes[c]).ToArray()))[0], prefill[c], $"fork {c} of a {prefixLength}-token prefix");
            AssertSame(fresh.Step((seq, continuations[c]))[0], decode[c], $"fork {c} decode");
        }

        using var freshParent = new MoeRunner();
        var ps = new MoeRunner.Sequence();
        AssertSame(freshParent.Step((ps, prefix.Concat(parentNext).ToArray()))[0], prefill[2], "parent after the fork");
        AssertSame(freshParent.Step((ps, parentDecode))[0], decode[2], "parent decode");
        Assert.Equal(prefixLength / C.BlockSize, children[0].Blocks.Intersect(parent.Blocks).Count());
    }

    private static readonly int[][] Chunkings = [[20, 17], [1, 16, 1, 19], [5, 5, 5, 5, 5, 5, 5, 2], Enumerable.Repeat(1, 37).ToArray()];

    private static float[][] Chunked(MoeRunner runner, int[] prompt, int[] next, int[] chunks)
    {
        Assert.Equal(prompt.Length, chunks.Sum());
        var seq = new MoeRunner.Sequence();
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

    [Fact]
    public void GenerationIsRepeatableOnAFreshRunnerAndPinned()
    {
        int[] prompt = Prompt(18, 6);
        int[] first = _runner.Generate(prompt, 8);
        using var other = new MoeRunner(blocks: 40);
        Assert.Equal(first, other.Generate(prompt, 8));
        Assert.True(PinnedGeneration.SequenceEqual(first), string.Join(",", first));
    }

    // Greedy ids for Prompt(18, 6) from the seed in TinyConfig.Moe; any change to the weights, the definition or CpuOps moves them.
    private static readonly int[] PinnedGeneration = [97, 211, 211, 10, 110, 169, 5, 234];
}

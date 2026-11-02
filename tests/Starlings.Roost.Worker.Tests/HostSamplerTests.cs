using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;
using Starlings.Roost.Plans;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Tests.Support;

namespace Starlings.Roost.Worker.Tests;

/// <summary>The worker's CPU sampler (R-042, CPU half): plan reading with sampling on, token history, penalties, logprobs and the result.</summary>
public sealed class HostSamplerTests
{
    private const int Vocab = 8;

    private static readonly PlanLimits Limits = new(MaxTokens: 64, MaxSequences: 2, MaxKvBlocks: 8, MaxSelected: 2, Vocab: Vocab, BlockSize: 16, AttentionBlocks: 8, RecurrentSlots: 2, AttentionGroup: 0, GroupCount: 2, Sampling: true);

    private static readonly PlanSampling Greedy = new() { Temperature = 0f, TopP = 1f, RepetitionPenalty = 1f, Flags = SamplingOptions.Greedy };

    private static ParsedPlan Parse(params TestSequence[] sequences) => StepPlanReader.Parse(PlanWriter.Write(1, 2, sequences), Limits);

    private static float[] Rows(params float[][] rows) => [.. rows.SelectMany(static r => r)];

    [Fact]
    public void TheReaderAcceptsTheSamplingTableOnlyWhenTheWorkerSamples()
    {
        var hot = new PlanSampling { Temperature = 0.8f, TopP = 0.9f, TopK = 5, MinP = 0.05f, RepetitionPenalty = 1.2f, PresencePenalty = 0.5f, FrequencyPenalty = -0.5f, Logprobs = 20, Seed = 3 };
        var bytes = PlanWriter.Write(1, 2, new TestSequence(1, [1], 0, [0], 0) { Sampling = hot });
        var parsed = StepPlanReader.Parse(bytes, Limits);
        Assert.Equal(hot, parsed.Sequences[0].Sampling);

        var greedyOnly = Limits with { Sampling = false };
        Assert.Equal(PlanRejection.Unsupported, Assert.Throws<PlanRejectedException>(() => StepPlanReader.Parse(bytes, greedyOnly)).Reason);
    }

    [Theory]
    [InlineData("temperature")]
    [InlineData("top_p")]
    [InlineData("top_k")]
    [InlineData("min_p")]
    [InlineData("repetition")]
    [InlineData("presence")]
    [InlineData("logprobs")]
    public void TheReaderRefusesSamplingValuesOutOfRange(string field)
    {
        var bad = field switch
        {
            "temperature" => Greedy with { Flags = SamplingOptions.None, Temperature = float.NaN },
            "top_p" => Greedy with { TopP = 0f },
            "top_k" => Greedy with { TopK = -1 },
            "min_p" => Greedy with { MinP = 1.5f },
            "repetition" => Greedy with { RepetitionPenalty = -1f },
            "presence" => Greedy with { PresencePenalty = float.PositiveInfinity },
            _ => Greedy with { Logprobs = StepPlanReader.MaxLogprobs + 1 },
        };
        var bytes = PlanWriter.Write(1, 2, new TestSequence(1, [1], 0, [0], 0) { Sampling = bad });
        Assert.Equal(PlanRejection.OutOfRange, Assert.Throws<PlanRejectedException>(() => StepPlanReader.Parse(bytes, Limits)).Reason);
    }

    [Fact]
    public void PenaltiesCountOnlyTheGeneratedOutputAcrossSteps()
    {
        var sampler = new HostSampler(Vocab, 2);
        var penalised = Greedy with { PresencePenalty = 2f };
        float[] logits = [0f, 0f, 0f, 3f, 2f, 0f, 0f, 0f]; // token 3 leads token 4 by 1

        // Prefill of a prompt that contains token 3: presence does not see the prompt, so token 3 is chosen.
        var step1 = sampler.Sample(Parse(new TestSequence(9, [3, 3, 1], 0, [0], 0) { Sampling = penalised }), logits);
        Assert.Equal([3], step1.Tokens);

        // Decode with token 3 as input: it is now generated output, presence 2 drops it below token 4.
        var step2 = sampler.Sample(Parse(new TestSequence(9, [3], 3, [0], 0) { Sampling = penalised }), logits);
        Assert.Equal([4], step2.Tokens);

        // A repetition penalty does see the prompt: a fresh sequence whose prompt holds token 3 is penalised at once.
        var repeated = Greedy with { RepetitionPenalty = 3f };
        Assert.Equal([4], sampler.Sample(Parse(new TestSequence(10, [3, 1], 0, [0], 0) { Sampling = repeated }), logits).Tokens);
        Assert.Equal([3], sampler.Sample(Parse(new TestSequence(11, [1, 2], 0, [0], 0) { Sampling = repeated }), logits).Tokens);
    }

    [Fact]
    public void ARecomputedSequenceKeepsItsPromptBoundary()
    {
        var sampler = new HostSampler(Vocab, 1);
        var penalised = Greedy with { FrequencyPenalty = 0.6f };
        float[] logits = [0f, 0f, 0f, 3f, 2f, 0f, 0f, 0f];
        Assert.Equal([3], sampler.Sample(Parse(new TestSequence(5, [1, 2], 0, [0], 0) { Sampling = penalised }), logits).Tokens);
        Assert.Equal([3], sampler.Sample(Parse(new TestSequence(5, [3], 2, [0], 0) { Sampling = penalised }), logits).Tokens); // one 3 in the output: 2.4 > 2

        // Preempted and recomputed from position 0 with its two outputs: two 3s in the output, 3 - 1.2 = 1.8 < 2.
        Assert.Equal([4], sampler.Sample(Parse(new TestSequence(5, [1, 2, 3, 3], 0, [0], 0) { Sampling = penalised }), logits).Tokens);
    }

    [Fact]
    public void LogprobRowsCarryTheChosenTokenThenTheAlternativesAndReachTheResult()
    {
        var sampler = new HostSampler(Vocab, 2);
        float[] row = [1f, 4f, 3f, 0f, 0f, 0f, 0f, 2f];
        var plan = Parse(
            new TestSequence(1, [1], 0, [0], 0) { Sampling = Greedy with { Logprobs = 2 }, Stops = [1] },
            new TestSequence(2, [2], 0, [1], 1) { Sampling = Greedy });
        var sampled = sampler.Sample(plan, Rows(row, row));
        Assert.Equal([1, 1], sampled.Tokens);
        Assert.Equal([true, false], sampled.Stops);
        Assert.Null(sampled.Logprobs[1]);
        var lp = sampled.Logprobs[0]!;
        Assert.Equal([1, 1, 2], lp.Select(static r => r.Token));
        double z = Math.Log(row.Sum(static x => Math.Exp(x)));
        Assert.Equal(4 - z, lp[0].Logprob, 5);
        Assert.Equal(3 - z, lp[2].Logprob, 5);

        var buffer = new byte[StepResultWriter.SizeFor(2, 2, 3)];
        int bytes = StepResultWriter.Write(buffer, plan, sampled.Tokens, sampled.Logprobs, default);
        var h = MemoryMarshal.Read<StepResultHeader>(buffer);
        Assert.Equal(3u, h.Logprobs.Count);
        var seqs = MemoryMarshal.Cast<byte, ResultSequence>(buffer.AsSpan((int)h.Sequences.Offset, 2 * Unsafe.SizeOf<ResultSequence>()));
        Assert.Equal(FinishReason.Eos, seqs[0].Finish);
        Assert.Equal((0u, 3u), (seqs[0].LogprobStart, seqs[0].LogprobCount));
        Assert.Equal(0u, seqs[1].LogprobCount);
        var rows = MemoryMarshal.Cast<byte, ResultLogprob>(buffer.AsSpan((int)h.Logprobs.Offset, 3 * Unsafe.SizeOf<ResultLogprob>()));
        Assert.Equal(lp, rows.ToArray());
        Assert.True(bytes <= buffer.Length);
    }

    [Fact]
    public void TheSameSeedGivesTheSameTokensWhateverElseIsInTheStep()
    {
        var hot = new PlanSampling { Temperature = 1f, TopP = 1f, RepetitionPenalty = 1f, Seed = 77, PhiloxOffset = 4 };
        var other = hot with { Seed = 5 };
        var rng = new Random(2);
        float[] row = [.. Enumerable.Range(0, Vocab).Select(_ => (float)rng.NextDouble())];
        float[] noise = [.. Enumerable.Range(0, Vocab).Select(_ => (float)rng.NextDouble())];
        var alone = new HostSampler(Vocab, 2).Sample(Parse(new TestSequence(1, [1], 0, [0], 0) { Sampling = hot }), row);
        var second = new HostSampler(Vocab, 2).Sample(Parse(new TestSequence(2, [6], 0, [0], 0) { Sampling = other }, new TestSequence(1, [1], 0, [1], 1) { Sampling = hot }), Rows(noise, row));
        Assert.Equal(alone.Tokens[0], second.Tokens[1]);
    }

    [Fact]
    public void FinishedSequencesAreForgottenAndTheCacheIsBounded()
    {
        var sampler = new HostSampler(Vocab, 1);
        float[] logits = [0f, 9f, 0f, 0f, 0f, 0f, 0f, 0f];
        sampler.Sample(Parse(new TestSequence(1, [2], 0, [0], 0) { Stops = [1] }), logits);
        Assert.Equal(0, sampler.HistoryCount);
        sampler.Sample(Parse(new TestSequence(2, [2], 0, [0], 0) { MaxNewTokensLeft = 1 }), logits);
        Assert.Equal(0, sampler.HistoryCount);
        for (ulong h = 10; h < 60; h++)
        {
            sampler.Sample(Parse(new TestSequence(h, [2], 0, [0], 0)), logits);
        }

        Assert.Equal((4 * 1) + 16, sampler.HistoryCount);
    }

    [Fact]
    public void CudaOpsRefusesSamplingAndNamesTheVersionFourFeatures()
    {
        Assert.False(CudaOps.Supports(OpId.Sample));
        var plain = Assert.Throws<NotSupportedException>(() => CudaOps.Sample(default));
        Assert.DoesNotContain("unsupported:", plain.Message, StringComparison.Ordinal);

        var a = new SampleArgs
        {
            TokenCounts = new DeviceSpan { Address = 16, Length = 8, Dtype = DType.I32 },
            StopFlags = new DeviceSpan { Address = 32, Length = 1, Dtype = DType.I32 },
            TopLogprobs = 3,
            Math = SampleMath.Fp32,
        };
        var e = Assert.Throws<NotSupportedException>(() => CudaOps.Sample(a));
        Assert.Contains("penalties, stop flags, top logprobs, fp32 math", e.Message, StringComparison.Ordinal);

        var p = CudaOps.ToParams(a);
        Assert.Equal((uint)Unsafe.SizeOf<SampleParams>(), p.StructSize);
        Assert.Equal((uint)KernelAbi.Minor, p.AbiMinor);
        Assert.Equal(a.TokenCounts, p.TokenCounts);
        Assert.Equal(3u, p.TopLogprobs);
        Assert.Equal((uint)SampleMath.Fp32, p.Math);
    }
}

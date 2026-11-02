using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost.Tests;

/// <summary>
/// Sampling end to end on the CPU path (R-042, CPU half): a client with <see cref="RoostSampler.Cpu"/> over an executor that runs
/// the worker's <c>HostSampler</c>. The CUDA worker's client (<see cref="RoostSampler.GreedyOnly"/>) keeps refusing.
/// </summary>
public sealed class SamplingTests
{
    private static readonly ChatMessage[] Hi = [new(ChatRole.User, "Hi")];

    private static ChatOptions Options(Action<AdditionalPropertiesDictionary>? extra = null, float? temperature = null, long? seed = null, int max = 12)
    {
        var o = new ChatOptions { Temperature = temperature, Seed = seed, MaxOutputTokens = max };
        if (extra is not null)
        {
            o.AdditionalProperties = [];
            extra(o.AdditionalProperties);
        }

        return o;
    }

    [Fact]
    public async Task TheCpuSamplerMakesSamplingPenaltiesAndLogprobsUntested()
    {
        await using var cpu = await TestRig.SamplingAsync(new Script { Tokens = TestRig.Encode("ok") });
        await using var gpu = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("ok") });
        foreach (var name in new[] { RoostCapabilities.Sampling, RoostCapabilities.Penalties, RoostCapabilities.Logprobs })
        {
            Assert.Equal(RoostCapabilityStatus.Untested, cpu.Capabilities.StatusOf(name));
            Assert.Equal(RoostCapabilityStatus.Unsupported, gpu.Capabilities.StatusOf(name));
        }

        Assert.DoesNotContain(cpu.Capabilities.Statuses.Values, s => s is RoostCapabilityStatus.Supported or RoostCapabilityStatus.Experimental);
    }

    [Fact]
    public async Task EveryOptionReachesThePlanAsTheSamplingRecord()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.SamplingAsync(new Script { Tokens = TestRig.Encode("abc"), NeverStop = true }, executors);
        var options = new ChatOptions
        {
            Temperature = 0.7f,
            TopP = 0.9f,
            TopK = 40,
            Seed = 1234,
            FrequencyPenalty = 0.25f,
            PresencePenalty = -0.5f,
            MaxOutputTokens = 3,
            AdditionalProperties = new AdditionalPropertiesDictionary
            {
                [RoostChatClient.MinPKey] = 0.05,
                [RoostChatClient.RepetitionPenaltyKey] = 1.1f,
                [RoostChatClient.LogprobsKey] = true,
                [RoostChatClient.TopLogprobsKey] = 2,
            },
        };

        var response = await client.GetResponseAsync(Hi, options, TestContext.Current.CancellationToken);

        Assert.Equal(ChatFinishReason.Length, response.FinishReason);
        var records = executors.Single().Sampling.ToArray();
        Assert.Equal(3, records.Length);
        for (int i = 0; i < records.Length; i++)
        {
            var s = records[i];
            Assert.Equal((0.7f, 0.9f, 40, 0.05f, 1.1f, -0.5f, 0.25f), (s.Temperature, s.TopP, s.TopK, s.MinP, s.RepetitionPenalty, s.PresencePenalty, s.FrequencyPenalty));
            Assert.Equal(1234UL, s.Seed);
            Assert.Equal(2u, s.Logprobs);
            Assert.Equal((ulong)i, s.PhiloxOffset); // the generated-token count: the draw depends on the position only
            Assert.Equal(SamplingOptions.Seeded, s.Flags);
        }
    }

    [Fact]
    public async Task GreedyRequestsFollowTheScriptOnTheCpuSampler()
    {
        await using var client = await TestRig.SamplingAsync(new Script { Tokens = TestRig.Encode("ok") });
        var response = await client.GetResponseAsync(Hi, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal("ok", response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
    }

    [Fact]
    public async Task TheSameSeedGivesTheSameTextAndOtherSeedsDiffer()
    {
        await using var client = await TestRig.SamplingAsync(new Script { Tokens = TestRig.Encode("abcdefghijkl"), NeverStop = true });
        var ct = TestContext.Current.CancellationToken;
        var first = (await client.GetResponseAsync(Hi, Options(temperature: 1f, seed: 7), ct)).Text;
        var again = (await client.GetResponseAsync(Hi, Options(temperature: 1f, seed: 7), ct)).Text;
        Assert.Equal(first, again);

        var others = new HashSet<string>(StringComparer.Ordinal);
        for (long seed = 100; seed < 106; seed++)
        {
            others.Add((await client.GetResponseAsync(Hi, Options(temperature: 1f, seed: seed), ct)).Text);
        }

        Assert.True(others.Count > 1, "six seeds gave one text");
    }

    [Fact]
    public async Task UnseededRequestsGetDifferentSeeds()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.SamplingAsync(new Script { Tokens = TestRig.Encode("ab"), NeverStop = true }, executors);
        var ct = TestContext.Current.CancellationToken;
        await client.GetResponseAsync(Hi, Options(temperature: 1f, max: 1), ct);
        await client.GetResponseAsync(Hi, Options(temperature: 1f, max: 1), ct);
        var seeds = executors.Single().Sampling.Select(static s => s.Seed).ToArray();
        Assert.Equal(2, seeds.Length);
        Assert.NotEqual(seeds[0], seeds[1]);
    }

    [Fact]
    public async Task LogprobsArriveOnTheUpdatesWithTheRequestedAlternatives()
    {
        await using var client = await TestRig.SamplingAsync(new Script { Tokens = TestRig.Encode("abcd"), NeverStop = true });
        var options = Options(e =>
        {
            e[RoostChatClient.LogprobsKey] = true;
            e[RoostChatClient.TopLogprobsKey] = 3;
        }, max: 4);

        var entries = new List<RoostTokenLogprob>();
        await foreach (var update in client.GetStreamingResponseAsync(Hi, options, TestContext.Current.CancellationToken))
        {
            if (update.AdditionalProperties?.TryGetValue(RoostChatClient.LogprobsKey, out var value) == true)
            {
                entries.AddRange(Assert.IsAssignableFrom<IReadOnlyList<RoostTokenLogprob>>(value));
            }
        }

        // Greedy: each token is the scripted one, which leads the executor's row by Lead over a fixed spread.
        Assert.Equal(TestRig.Encode("abcd"), entries.Select(static e => e.TokenId));
        int vocab = TestRig.Limits().Vocab;
        double best = (6 * 0.25) + ScriptedExecutor.Lead;
        foreach (var e in entries)
        {
            double z = Enumerable.Range(0, vocab).Sum(v => Math.Exp(v == e.TokenId ? best : (v % 7) * 0.25));
            Assert.Equal(best - Math.Log(z), e.Logprob, 4);
            Assert.Equal(3, e.TopLogprobs.Count);
            Assert.Equal(e.TokenId, e.TopLogprobs[0].TokenId);
            Assert.Equal(e.Logprob, e.TopLogprobs[0].Logprob);
            Assert.True(e.TopLogprobs[1].Logprob >= e.TopLogprobs[2].Logprob);
        }
    }

    [Fact]
    public async Task LogprobsWithoutAlternativesCarryTheChosenTokenOnly()
    {
        await using var client = await TestRig.SamplingAsync(new Script { Tokens = TestRig.Encode("ab"), NeverStop = true });
        var response = await client.GetStreamingResponseAsync(Hi, Options(e => e[RoostChatClient.LogprobsKey] = true, max: 2), TestContext.Current.CancellationToken)
            .ToListAsync(TestContext.Current.CancellationToken);
        var entries = response.SelectMany(static u => u.AdditionalProperties?.TryGetValue(RoostChatClient.LogprobsKey, out var v) == true ? (IReadOnlyList<RoostTokenLogprob>)v! : []).ToArray();
        Assert.Equal(2, entries.Length);
        Assert.All(entries, static e => Assert.Empty(e.TopLogprobs));
    }

    public static TheoryData<string, string> OutOfRange() => new()
    {
        { "temperature", "temperature" },
        { "top_p", "top_p" },
        { "top_k", "top_k" },
        { "frequency_penalty", "frequency_penalty" },
        { "min_p", "min_p" },
        { "repetition_penalty", "repetition_penalty" },
        { "top_logprobs", "top_logprobs" },
        { "top_logprobs_alone", "top_logprobs" },
        { "logprobs_not_bool", "logprobs" },
        { "unknown", "logit_bias" },
    };

    [Theory]
    [MemberData(nameof(OutOfRange))]
    public async Task OutOfRangeValuesAreRefusedNamingTheParameter(string option, string param)
    {
        await using var client = await TestRig.SamplingAsync(new Script { Tokens = TestRig.Encode("ok") });
        var options = option switch
        {
            "temperature" => new ChatOptions { Temperature = -1f },
            "top_p" => new ChatOptions { TopP = 0f },
            "top_k" => new ChatOptions { TopK = -2 },
            "frequency_penalty" => new ChatOptions { FrequencyPenalty = 3f },
            "min_p" => Options(e => e["min_p"] = 1.5),
            "repetition_penalty" => Options(e => e["repetition_penalty"] = 0),
            "top_logprobs" => Options(e =>
            {
                e["logprobs"] = true;
                e["top_logprobs"] = 21;
            }),
            "top_logprobs_alone" => Options(e => e["top_logprobs"] = 2),
            "logprobs_not_bool" => Options(e => e["logprobs"] = "yes"),
            _ => Options(e => e["logit_bias"] = 1),
        };

        var e = Assert.Throws<RoostChatException>(() => client.GetStreamingResponseAsync(Hi, options, TestContext.Current.CancellationToken));

        Assert.Equal(RoostErrorCode.InvalidOption, e.Error.Code);
        Assert.Equal(param, e.Error.Param);
    }

    [Theory]
    [InlineData("min_p")]
    [InlineData("repetition_penalty")]
    [InlineData("logprobs")]
    [InlineData("top_logprobs")]
    public async Task TheGreedyOnlyClientRefusesTheSamplerProperties(string key)
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("ok") });
        var options = Options(e => e[key] = key == "logprobs" ? (object)true : 1);

        var e = Assert.Throws<RoostChatException>(() => client.GetStreamingResponseAsync(Hi, options, TestContext.Current.CancellationToken));

        Assert.Equal(key, e.Error.Param);
        Assert.Contains("R-042", e.Error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ThePackagedCudaWorkerCannotClaimTheCpuSampler()
    {
        var options = new RoostChatClientOptions { Model = TestRig.Profile(), Worker = new RoostWorkerOptions { Sampler = RoostSampler.Cpu } };

        var e = await Assert.ThrowsAsync<ArgumentException>(() => RoostChatClient.StartAsync(options, TestContext.Current.CancellationToken));

        Assert.Contains("CPU sampler", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SamplingRunsOverTheProcessTransport()
    {
        await using var client = await TestRig.ProcessAsync(new Script { Tokens = TestRig.Encode("abcdef"), NeverStop = true, Sample = true });
        var ct = TestContext.Current.CancellationToken;
        var a = await client.GetResponseAsync(Hi, Options(temperature: 1f, seed: 3, max: 6), ct);
        var b = await client.GetResponseAsync(Hi, Options(temperature: 1f, seed: 3, max: 6), ct);
        Assert.Equal(a.Text, b.Text);
        Assert.Equal(ChatFinishReason.Length, a.FinishReason);
    }
}

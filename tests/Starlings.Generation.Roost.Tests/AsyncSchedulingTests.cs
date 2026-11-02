using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost.Tests;

/// <summary>
/// Async scheduling end to end on the CPU path (R-032): a client with <see cref="RoostWorkerOptions.AsyncScheduling"/> keeps
/// two steps published, the executor patches each chained decode row with the worker's <c>SampledInputPatcher</c>, and the
/// caller sees exactly what a synchronous client sees: the stop id's one-step lag is hidden.
/// </summary>
public sealed class AsyncSchedulingTests
{
    private const string Reply = "Hello, world! 中文 \U0001F600 done.";

    private static readonly ChatMessage[] Hi = [new(ChatRole.User, "Hi")];

    private static CancellationToken Timeout => TestContext.Current.CancellationToken;

    private static Task<RoostChatClient> AsyncClient(Script script, List<ScriptedExecutor> executors) =>
        TestRig.InProcessAsync(script with { PatchInputs = true }, executors, o => o with { Worker = o.Worker with { AsyncScheduling = true } });

    [Fact]
    public async Task OnlyAnAsyncClientListsAsyncSchedulingAndItIsUntested()
    {
        var executors = new List<ScriptedExecutor>();
        await using var async = await AsyncClient(new Script { Tokens = TestRig.Encode("ok") }, executors);
        await using var sync = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("ok") });
        Assert.Equal(RoostCapabilityStatus.Untested, async.Capabilities.StatusOf(RoostCapabilities.AsyncScheduling));
        Assert.Equal(RoostCapabilityStatus.Unsupported, sync.Capabilities.StatusOf(RoostCapabilities.AsyncScheduling));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TheAsyncClientGivesTheSyncTextFinishAndUsage(bool sample)
    {
        var script = new Script { Tokens = TestRig.Encode(Reply), Sample = sample };
        var syncExecutors = new List<ScriptedExecutor>();
        var asyncExecutors = new List<ScriptedExecutor>();
        Func<RoostChatClientOptions, RoostChatClientOptions> sampler = o => sample ? o with { Worker = o.Worker with { Sampler = RoostSampler.Cpu } } : o;
        await using var sync = await TestRig.InProcessAsync(script, syncExecutors, sampler);
        await using var async = await TestRig.InProcessAsync(script with { PatchInputs = true }, asyncExecutors, o => sampler(o) with { Worker = sampler(o).Worker with { AsyncScheduling = true } });
        var options = new ChatOptions { Temperature = sample ? 0.8f : null, Seed = sample ? 99 : null };

        var expected = await sync.GetResponseAsync(Hi, options, Timeout);
        var actual = await async.GetResponseAsync(Hi, options, Timeout);

        Assert.Equal(sample ? expected.Text : Reply, expected.Text); // a sampled reply draws other tokens at temperature 0.8
        Assert.Equal(expected.Text, actual.Text); // the same seed and Philox offsets: the same draws
        Assert.Equal(expected.FinishReason, actual.FinishReason);
        Assert.Equal(expected.Usage!.OutputTokenCount, actual.Usage!.OutputTokenCount); // the row after the stop id is not counted
        var executor = asyncExecutors.Single();
        Assert.Empty(executor.Rejections);
        Assert.Empty(executor.PatchMismatches);
        // Every decode input is patched. The row planned after the stop id (before it returned) may still be running when the
        // response completes, since the caller is answered as soon as the stop id's step is applied.
        Assert.InRange(executor.Patched, expected.Usage.OutputTokenCount!.Value - 1, expected.Usage.OutputTokenCount!.Value);
        Assert.Equal(0, syncExecutors.Single().Patched);
    }

    [Fact]
    public async Task TheLengthLimitIsNeverOverrunAndConcurrentRequestsComplete()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await AsyncClient(new Script { Tokens = TestRig.Encode(Reply), NeverStop = true }, executors);

        var limited = await client.GetResponseAsync(Hi, new ChatOptions { MaxOutputTokens = 5 }, Timeout);
        Assert.Equal(ChatFinishReason.Length, limited.FinishReason);
        Assert.Equal(5, limited.Usage!.OutputTokenCount);

        var responses = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => client.GetResponseAsync(Hi, new ChatOptions { MaxOutputTokens = 9 }, Timeout)));
        Assert.All(responses, r => Assert.Equal(9, r.Usage!.OutputTokenCount));
        Assert.Single(responses.Select(r => r.Text).Distinct());
        Assert.Empty(executors.Single().Rejections);
        Assert.Empty(executors.Single().PatchMismatches);
        Assert.Equal(1, executors.Single().Steps.Max(s => s.Sequences)); // the test profile runs one sequence per step
    }

    [Fact]
    public async Task ARecoverableFaultFailsTheRequestAndTheWorkerRefusesTheStepChainedOnIt()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await AsyncClient(new Script { Tokens = TestRig.Encode(Reply), FaultAtStep = 2, Fault = FaultClass.Recoverable }, executors);

        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.DeviceFault, e.Error.Code);

        var next = await client.GetResponseAsync(Hi, cancellationToken: Timeout);
        Assert.Equal(Reply, next.Text);
        var executor = Assert.Single(executors);
        Assert.Single(executor.Rejections); // step 3 named the failed step 2 as its input
        Assert.Empty(executor.PatchMismatches);
    }

    [Fact]
    public async Task TheProcessTransportRunsAsyncWithAWorkerThatPatches()
    {
        var script = new Script { Tokens = TestRig.Encode(Reply), PatchInputs = true };
        await using var client = await RoostChatClient.StartAsync(new RoostChatClientOptions
        {
            Model = TestRig.Profile(),
            Worker = new RoostWorkerOptions
            {
                AsyncScheduling = true,
                Command = WorkerCommand.ForCurrentProcess(TestRig.WorkerFlag, TestRig.ScriptPrefix + script.Encode()),
                StepDeadline = TimeSpan.FromSeconds(10),
                TransportOptions = new WorkerTransportOptions { StartupTimeout = TimeSpan.FromSeconds(60) },
            },
        }, Timeout);

        var response = await client.GetResponseAsync(Hi, cancellationToken: Timeout);
        Assert.Equal(Reply, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
    }

    [Fact]
    public async Task ThePackagedCudaWorkerCannotBeAskedForAsyncScheduling()
    {
        var e = await Assert.ThrowsAsync<ArgumentException>(() => RoostChatClient.StartAsync(new RoostChatClientOptions
        {
            Model = TestRig.Profile(),
            Worker = new RoostWorkerOptions { AsyncScheduling = true },
        }, Timeout));
        Assert.Contains("stream F", e.Message, StringComparison.Ordinal);
    }
}

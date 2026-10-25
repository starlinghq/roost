using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost.Tests;

/// <summary>
/// Review of #133, finding 1 to 3: nothing that goes wrong inside a step may end the driver thread (which would end the
/// consumer's process or break the client), a request racing the driver's exit always completes, and pool exhaustion under
/// concurrency is not reported as a bad prompt.
/// </summary>
public sealed class ContainmentTests
{
    private const string Reply = "Contained, 中文.";
    private static readonly ChatMessage[] Hi = [new(ChatRole.User, "Hi")];

    private static int[] ReplyIds => TestRig.Encode(Reply);

    private static CancellationToken Timeout => TestContext.Current.CancellationToken;

    [Fact]
    public async Task AnExecutorThrowingAnArbitraryExceptionFailsTheRequestAndTheClientSurvives()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds, ThrowAtStep = 2 }, executors);

        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.DeviceFault, e.Error.Code); // the worker runtime treats an executor exception as sticky

        Assert.Equal(Reply, (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text);
        Assert.Equal(2, executors.Count);
    }

    [Fact]
    public async Task ATransportThrowingAnUnexpectedExceptionTypeFailsTheRequestAndTheNextOneRuns()
    {
        var faults = new TransportFaults();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds }, decorate: t => new FaultyTransport(t, faults));

        faults.ThrowOnTake = true;
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.Internal, e.Error.Code);
        Assert.Contains("KeyNotFoundException", e.Message, StringComparison.Ordinal);

        Assert.Equal(Reply, (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text);
        Assert.Equal(1, faults.Kills);
    }

    [Fact]
    public async Task AMalformedResultFailsTheRequestAndTheNextOneRuns()
    {
        var faults = new TransportFaults();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds }, decorate: t => new FaultyTransport(t, faults));

        faults.CorruptResult = true;
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.Internal, e.Error.Code);
        Assert.Contains("InvalidDataException", e.Message, StringComparison.Ordinal);

        Assert.Equal(Reply, (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text);
    }

    [Fact]
    public async Task UnpublishableStepFailsItsRequestsAndKeepsTheWorker()
    {
        var faults = new TransportFaults();
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds }, executors, decorate: t => new FaultyTransport(t, faults));

        faults.ThrowOnPlanBuffer = true;
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.Internal, e.Error.Code);
        Assert.Contains("OverflowException", e.Message, StringComparison.Ordinal);

        Assert.Equal(Reply, (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text);
        Assert.Single(executors);
        Assert.Equal(0, faults.Kills);
    }

    [Fact]
    public async Task ARingWithNoFreePlanBufferAndNothingInFlightReplacesTheWorker()
    {
        var faults = new TransportFaults();
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds }, executors, decorate: t => new FaultyTransport(t, faults));

        faults.NoFreeBuffer = true;
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.WorkerRestarted, e.Error.Code);
        Assert.Contains("out of step", e.Message, StringComparison.Ordinal);

        Assert.Equal(Reply, (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text);
        Assert.Equal(1, faults.Kills);
        Assert.Equal(2, executors.Count); // a new worker
    }

    [Fact]
    public async Task AKillThatThrowsReplacesTheTransportAndTheNextRequestRuns()
    {
        var faults = new TransportFaults();
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(
            new Script { Tokens = ReplyIds, FaultAtStep = 2, Fault = Starlings.Roost.Plans.FaultClass.Sticky },
            executors,
            decorate: t => new FaultyTransport(t, faults));

        faults.ThrowOnKill = true;
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.DeviceFault, e.Error.Code);

        Assert.Equal(Reply, (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text);
        Assert.Equal(2, faults.Created); // the transport whose kill threw was replaced
        Assert.Equal(2, executors.Count);
    }

    [Fact]
    public async Task RequestsRacingDisposalAllComplete()
    {
        for (var round = 0; round < 20; round++)
        {
            var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds });
            var requests = Enumerable.Range(0, 8).Select(i => Task.Run(async () =>
            {
                try
                {
                    _ = await client.GetResponseAsync(Hi, cancellationToken: Timeout);
                    return "ok";
                }
                catch (RoostChatException e)
                {
                    return e.Error.CodeName;
                }
                catch (ObjectDisposedException)
                {
                    return "disposed";
                }
            }, Timeout)).ToArray();
            var dispose = Task.Run(async () => await client.DisposeAsync(), Timeout);

            var all = Task.WhenAll(requests);
            var finished = await Task.WhenAny(all, Task.Delay(TimeSpan.FromSeconds(20), Timeout));
            Assert.Same(all, finished);
            await dispose;
            Assert.All(await all, r => Assert.True(r is "ok" or "internal" or "disposed", r));
        }
    }

    [Fact]
    public async Task ConcurrentLongRequestsThatExhaustThePoolAllFinishWithTheirFullOutput()
    {
        // Two rows per step and two recurrent slots: two requests are resident at once and together need more than the
        // 16-block pool. The planner preempts the later one (recompute), and both still produce their whole output.
        var limits = TestRig.Limits() with { MaxSequences = 2, MaxSelected = 2, RecurrentSlots = 2 };
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(
            new Script { Tokens = ReplyIds, NeverStop = true },
            executors,
            configure: o => o with { Model = o.Model with { MaxSequencesPerStep = 2, RecurrentSlots = 2 } },
            limits: limits);
        var question = string.Concat(Enumerable.Repeat("long ", 4));

        var responses = await Task.WhenAll(Enumerable.Range(0, 2).Select(_ =>
            client.GetResponseAsync([new ChatMessage(ChatRole.User, question)], new ChatOptions { MaxOutputTokens = 150 }, Timeout)));

        var expected = TestRig.Tokenizer.Decode([.. ReplyIds, .. Enumerable.Repeat(ReplyIds[^1], 150 - ReplyIds.Length)], true);
        Assert.All(responses, r =>
        {
            Assert.Equal(ChatFinishReason.Length, r.FinishReason);
            Assert.Equal(150, r.Usage!.OutputTokenCount);
            Assert.Equal(expected, r.Text);
        });
        var steps = executors.Single().Steps.ToArray();
        Assert.Contains(steps, s => s.Sequences == 2);
        Assert.Empty(executors.Single().Rejections);
    }
}

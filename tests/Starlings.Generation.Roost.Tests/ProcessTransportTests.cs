using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Driver;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost.Tests;

/// <summary>The default transport (decision 21): the worker in its own process, here this test executable re-run with a scripted executor.</summary>
public sealed class ProcessTransportTests
{
    private const string Reply = "From another process: 中文.";

    private static CancellationToken Timeout => TestContext.Current.CancellationToken;

    [Fact]
    public async Task AWorkerProcessServesStreamingAndNonStreamingRequests()
    {
        var ids = TestRig.Encode(Reply);
        await using var client = await TestRig.ProcessAsync(new Script { Tokens = ids });
        var driver = Assert.IsType<WorkerDriver>(client.Backend);
        Assert.Equal(TransportKind.Process, driver.Transport.Kind);
        var pid = Assert.IsType<ProcessWorkerTransport>(((DriverTransport)driver.Transport).Inner).ProcessId;
        Assert.NotEqual(Environment.ProcessId, pid);

        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")], cancellationToken: Timeout);
        Assert.Equal(Reply, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Equal(ids.Length + 1, response.Usage!.OutputTokenCount);

        var text = new System.Text.StringBuilder();
        await foreach (var u in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "Again")], new ChatOptions { MaxOutputTokens = 4 }, Timeout))
        {
            text.Append(u.Text);
        }

        Assert.Equal(TestRig.Tokenizer.Decode(ids.AsSpan(0, 4), true), text.ToString());
    }

    [Fact]
    public async Task AStickyFaultKillsTheWorkerProcessAndTheNextRequestSpawnsANewOne()
    {
        // The command line carries the fault, so every spawned worker faults at its second step: each request fails, on a new process.
        await using var client = await TestRig.ProcessAsync(new Script { Tokens = TestRig.Encode(Reply), FaultAtStep = 2, Fault = FaultClass.Sticky });
        var transport = Assert.IsType<ProcessWorkerTransport>(((DriverTransport)((WorkerDriver)client.Backend).Transport).Inner);
        var first = transport.ProcessId;

        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")], cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.DeviceFault, e.Error.Code);

        e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")], cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.DeviceFault, e.Error.Code);
        Assert.NotEqual(first, transport.ProcessId);
    }

    [Fact]
    public async Task AHungStepMissesItsDeadlineAndTheWorkerIsKilledAndReplaced()
    {
        await using var client = await TestRig.ProcessAsync(new Script { Tokens = TestRig.Encode(Reply), HangAtStep = 5 }, TimeSpan.FromMilliseconds(500));
        var transport = Assert.IsType<ProcessWorkerTransport>(((DriverTransport)((WorkerDriver)client.Backend).Transport).Inner);
        var first = transport.ProcessId;

        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")], cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.WorkerRestarted, e.Error.Code);
        Assert.Contains("Hang", e.Message, StringComparison.Ordinal);

        // A short request (its prefill chunks only, fewer than five steps) runs on a new worker process.
        var next = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")], new ChatOptions { MaxOutputTokens = 1 }, Timeout);
        Assert.Equal(1, next.Usage!.OutputTokenCount);
        Assert.NotEqual(first, transport.ProcessId);
    }
}

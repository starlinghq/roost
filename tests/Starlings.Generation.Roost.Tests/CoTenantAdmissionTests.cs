using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Engine.Admission;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost.Tests;

/// <summary>The driver's free-memory floor (R-044) over a settable source: refusal while a co-tenant holds the memory, never an abort.</summary>
public sealed class CoTenantAdmissionTests
{
    private static readonly ChatMessage[] Hi = [new(ChatRole.User, "Hi")];

    private static CancellationToken Timeout => TestContext.Current.CancellationToken;

    [Fact]
    public async Task ALowCeilingRefusesNewRequestsAndTheyAreAdmittedWhenMemoryReturns()
    {
        var source = new FixedDeviceMemorySource(24_000, 10_000);
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(
            new Script { Tokens = TestRig.Encode("ok") },
            executors,
            configure: o => o with { Worker = o.Worker with { DeviceMemory = source, MinFreeDeviceBytes = 4_000 } });

        Assert.Equal("ok", (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text);
        var stepsBefore = executors.Single().Steps.Count;

        source.Set(24_000, 3_999); // a co-tenant allocates
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, e.Error.Code);
        Assert.Null(e.Error.Param);
        Assert.Equal(stepsBefore, executors.Single().Steps.Count); // refused before any device work
        Assert.Single(executors); // the worker was kept

        source.MakeUnreadable(); // unreadable: the pools decide
        Assert.Equal("ok", (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text);

        source.Set(24_000, 4_000);
        Assert.Equal("ok", (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text);
        Assert.Single(executors);
    }

    [Fact]
    public async Task ARunningRequestIsNeverAbortedByAFallingCeiling()
    {
        var source = new FixedDeviceMemorySource(24_000, 10_000);
        await using var client = await TestRig.InProcessAsync(
            new Script { Tokens = TestRig.Encode("a longer reply"), DelayMs = 5 },
            configure: o => o with { Worker = o.Worker with { DeviceMemory = source, MinFreeDeviceBytes = 4_000 } });

        var text = new System.Text.StringBuilder();
        await foreach (var update in client.GetStreamingResponseAsync(Hi, cancellationToken: Timeout))
        {
            source.Set(24_000, 0); // the co-tenant takes everything mid-stream
            text.Append(update.Text);
        }

        Assert.Equal("a longer reply", text.ToString());
    }

    [Fact]
    public async Task NoSourceAndANegativeFloor()
    {
        await using (var client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("ok") }, configure: o => o with { Worker = o.Worker with { MinFreeDeviceBytes = 1L << 40 } }))
        {
            Assert.Equal("ok", (await client.GetResponseAsync(Hi, cancellationToken: Timeout)).Text); // no source: the floor decides nothing
            Assert.Equal(RoostCapabilityStatus.Untested, client.Capabilities.StatusOf(RoostCapabilities.CoTenantAdmission));
        }

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => TestRig.InProcessAsync(
            new Script { Tokens = TestRig.Encode("ok") },
            configure: o => o with { Worker = o.Worker with { MinFreeDeviceBytes = -1 } }));
    }
}

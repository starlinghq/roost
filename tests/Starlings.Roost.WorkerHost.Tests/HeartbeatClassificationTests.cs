using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Roost.WorkerHost.Tests;

/// <summary>The liveness rule behind Hang and Stalled, with the clock under the test's control.</summary>
public sealed class HeartbeatClassificationTests
{
    private const ulong Liveness = 50_000_000;

    private static Heartbeat Beat(ulong counter, ulong workerNs) => new() { Counter = counter, WorkerNs = workerNs };

    [Fact]
    public void BeatNewerThanTheHostsClockReadIsAlive()
    {
        // The worker beat between the host taking `now` and reading the record: the beat time is ahead of `now`.
        Assert.True(WorkerTransportBase.HeartbeatAdvanced(Beat(401, 1_000_500), 400, 1_000_000, Liveness));
    }

    [Fact]
    public void BeatFarAheadOfTheHostsClockIsStillAlive()
    {
        Assert.True(WorkerTransportBase.HeartbeatAdvanced(Beat(2, ulong.MaxValue), 1, 5, Liveness));
    }

    [Fact]
    public void RecentBeatWithAMovedCounterIsAlive()
    {
        Assert.True(WorkerTransportBase.HeartbeatAdvanced(Beat(9, 1_000_000), 8, 1_000_000 + Liveness, Liveness));
    }

    [Fact]
    public void BeatOlderThanTheLivenessWindowIsStalled()
    {
        Assert.False(WorkerTransportBase.HeartbeatAdvanced(Beat(9, 1_000_000), 8, 1_000_000 + Liveness + 1, Liveness));
    }

    [Fact]
    public void CounterThatDidNotMoveIsStalledHoweverRecentTheBeat()
    {
        Assert.False(WorkerTransportBase.HeartbeatAdvanced(Beat(8, 1_000_000), 8, 1_000_000, Liveness));
    }

    [Fact]
    public void FaultRulesMapTheTwoCases()
    {
        Assert.Equal(FaultClass.Hang, FaultRules.ClassifyMissedDeadline(true));
        Assert.Equal(FaultClass.Stalled, FaultRules.ClassifyMissedDeadline(false));
    }
}

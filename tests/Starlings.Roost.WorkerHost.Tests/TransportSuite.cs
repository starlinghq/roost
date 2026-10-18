using System.Diagnostics;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;
using Starlings.Roost.WorkerHost.Tests.Support;

namespace Starlings.Roost.WorkerHost.Tests;

/// <summary>
/// The tests both transports must pass (R-069: the in-process transport passes the same tests). Each subclass says how to
/// build its transport and how to make the worker's heartbeat stop.
/// </summary>
public abstract class TransportSuite
{
    protected abstract TransportKind Kind { get; }

    /// <summary>Whether a crash (the worker dying at once) can be injected: not for a thread in the test process.</summary>
    protected abstract bool CanCrash { get; }

    internal abstract Rig NewRig(WorkerTransportOptions? options = null);

    /// <summary>Freezes the worker's heartbeat; returns the op a test must publish so the step does not finish.</summary>
    internal abstract FakeOp Freeze(Rig rig);

    internal abstract void Thaw(Rig rig);

    /// <summary>Lets a hung in-process step end so the test does not leave a thread behind.</summary>
    internal virtual void ReleaseHang()
    {
    }

    [Fact]
    public async Task HandshakeReportsReadyWithDeviceAndWorkerIdentity()
    {
        await using var rig = NewRig();
        var result = await rig.StartAsync();

        Assert.Equal(HandshakeState.Ready, result.State);
        Assert.Equal(HandshakeReject.None, result.Reject);
        Assert.Equal(FakeExecutor.UuidFor(0), result.DeviceUuid.ToArray());
        Assert.True(result.WorkerProcessId > 0);
        Assert.Equal(Kind, rig.Transport.Kind);
        Assert.Equal(Kind == TransportKind.InProcess, result.WorkerProcessId == Environment.ProcessId);
    }

    [Fact]
    public async Task HandshakeRejectsWrongManifestDigest()
    {
        await using var rig = NewRig();
        var launch = Rig.DefaultLaunch() with { ManifestDigest = FakeExecutor.Digest("another manifest") };
        var result = await rig.StartAsync(launch);

        Assert.Equal(HandshakeState.Rejected, result.State);
        Assert.Equal(HandshakeReject.ManifestDigest, result.Reject);
        Assert.Equal(TransportState.Idle, rig.Transport.State);
    }

    [Fact]
    public async Task HandshakeRejectsWrongWeightsDigest()
    {
        await using var rig = NewRig();
        var launch = Rig.DefaultLaunch() with { WeightsDigest = FakeExecutor.Digest("other weights") };
        var result = await rig.StartAsync(launch);

        Assert.Equal(HandshakeState.Rejected, result.State);
        Assert.Equal(HandshakeReject.WeightsDigest, result.Reject);
    }

    [Fact]
    public async Task HandshakeRejectsIncompatibleAbi()
    {
        await using var rig = NewRig(new WorkerTransportOptions { AbiVersion = (Abi.KernelAbi.Major + 1) << 16 });
        var result = await rig.StartAsync();

        Assert.Equal(HandshakeState.Rejected, result.State);
        Assert.Equal(HandshakeReject.Abi, result.Reject);
    }

    [Fact]
    public async Task HandshakeRejectsMissingDevice()
    {
        await using var rig = NewRig();
        var result = await rig.StartAsync(Rig.DefaultLaunch(device: 7));

        Assert.Equal(HandshakeState.Rejected, result.State);
        Assert.Equal(HandshakeReject.Device, result.Reject);
    }

    [Fact]
    public async Task HandshakeRejectsBadMeshRank()
    {
        await using var rig = NewRig();
        var result = await rig.StartAsync(Rig.DefaultLaunch(mesh: (2, 2)));

        Assert.Equal(HandshakeState.Rejected, result.State);
        Assert.Equal(HandshakeReject.Mesh, result.Reject);
    }

    [Fact]
    public async Task RejectedTransportCanStartAgainWithAGoodOffer()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync(Rig.DefaultLaunch() with { ManifestDigest = FakeExecutor.Digest("wrong") });
        var again = await rig.StartAsync();

        Assert.Equal(HandshakeState.Ready, again.State);
        Assert.Equal(TransportState.Running, rig.Transport.State);
    }

    [Fact]
    public async Task StepsRoundTripIntactInOrderOneAtATime()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();

        for (ulong i = 1; i <= 500; i++)
        {
            var id = rig.Publish();
            var seen = rig.Wait(TimeSpan.FromSeconds(10));
            Assert.True(seen.HasResult);
            Assert.Equal(id, seen.StepId);
            Assert.Equal(FaultClass.None, seen.Header.Fault);
            Assert.True(seen.EchoOk, $"payload echo differs at step {id}");
            Assert.True(seen.ChecksumOk, $"checksum differs at step {id}");
        }
    }

    [Fact]
    public async Task TwoStepsInFlightReturnInOrder()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();

        ulong expected = 1;
        var inFlight = 0;
        var sawTwo = false;
        for (var round = 0; round < 2000; round++)
        {
            while (inFlight < 2 && rig.TryPublish() != 0)
            {
                inFlight++;
            }

            sawTwo |= inFlight == 2;
            var seen = rig.Wait(TimeSpan.FromSeconds(10));
            Assert.True(seen.HasResult);
            Assert.Equal(expected++, seen.StepId);
            Assert.True(seen.EchoOk && seen.ChecksumOk);
            inFlight--;
        }

        Assert.True(sawTwo, "two plans were never in flight");
    }

    [Fact]
    public async Task HeartbeatAdvancesAndReportsReady()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();

        var first = rig.Transport.ReadHeartbeat();
        await Task.Delay(50, TestContext.Current.CancellationToken);
        var second = rig.Transport.ReadHeartbeat();

        Assert.True(second.Counter > first.Counter, $"heartbeat did not advance: {first.Counter} -> {second.Counter}");
        Assert.True(second.WorkerNs >= first.WorkerNs);
        Assert.Equal(WorkerState.Ready, second.State);
    }

    [Fact]
    public async Task HeartbeatRecordsLastStep()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();
        var id = rig.Publish();
        _ = rig.Wait(TimeSpan.FromSeconds(10));
        await Task.Delay(20, TestContext.Current.CancellationToken);

        Assert.Equal(id, rig.Transport.ReadHeartbeat().LastStepId);
    }

    [Fact]
    public async Task RecoverableFaultIsReportedAndTheWorkerKeepsStepping()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();

        _ = rig.Publish(FakeOp.Recoverable);
        var faulted = rig.Wait(TimeSpan.FromSeconds(10));
        Assert.True(faulted.HasResult);
        Assert.Equal(FaultClass.Recoverable, faulted.Header.Fault);
        Assert.Equal(FakeExecutor.ErrorRecoverable, faulted.Header.ErrorCode);
        Assert.Equal(RecoveryAction.RetryStep, FaultRules.ActionFor(faulted.Fault));

        _ = rig.Publish();
        var ok = rig.Wait(TimeSpan.FromSeconds(10));
        Assert.Equal(FaultClass.None, ok.Fault);
        Assert.True(ok.EchoOk);
        Assert.Equal(FaultClass.None, rig.Transport.Poll());
    }

    [Fact]
    public async Task InvalidPlanIsARecoverableFaultNotACrash()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();
        var index = rig.Transport.NextPlanBufferIndex;
        rig.Transport.PlanBuffer(index).Clear(); // no magic
        Assert.True(rig.Transport.TryPublish(new RingSlot { StepId = 1, Kind = RingSlotKind.Step, BufferIndex = (uint)index }));
        rig.NextStepId = 2;

        var seen = rig.Wait(TimeSpan.FromSeconds(10));

        Assert.Equal(FaultClass.Recoverable, seen.Header.Fault);
        _ = rig.Publish();
        Assert.Equal(FaultClass.None, rig.Wait(TimeSpan.FromSeconds(10)).Fault);
    }

    [Fact]
    public async Task StickyFaultPublishesTheResultMarksFaultedAndTheWorkerExits()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();

        _ = rig.Publish(FakeOp.Sticky);
        var seen = rig.Wait(TimeSpan.FromSeconds(10));
        Assert.True(seen.HasResult);
        Assert.Equal(FaultClass.Sticky, seen.Header.Fault);
        Assert.Equal(FakeExecutor.ErrorSticky, seen.Header.ErrorCode);
        Assert.Equal(RecoveryAction.Respawn, FaultRules.ActionFor(seen.Fault));

        var exit = rig.WaitForPollFault(TimeSpan.FromSeconds(10));
        Assert.Equal(FaultClass.Exited, exit.Fault);
        Assert.Equal(WorkerState.Faulted, rig.Transport.ReadHeartbeat().State);
    }

    [Fact]
    public async Task ExecutorExceptionIsClassifiedSticky()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();

        _ = rig.Publish(FakeOp.Throw);
        var seen = rig.Wait(TimeSpan.FromSeconds(10));

        Assert.Equal(FaultClass.Sticky, seen.Header.Fault);
        Assert.Equal(FaultClass.Exited, rig.WaitForPollFault(TimeSpan.FromSeconds(10)).Fault);
    }

    [Fact]
    public async Task MissedDeadlineWithALiveHeartbeatIsAHang()
    {
        try
        {
            // A wide liveness window: on a loaded machine the heartbeat thread can be late, and that is not what this test is about.
            await using var rig = NewRig(new WorkerTransportOptions { HeartbeatLiveness = TimeSpan.FromSeconds(5) });
            _ = await rig.StartAsync(Rig.DefaultLaunch(deadline: TimeSpan.FromMilliseconds(200)));
            _ = rig.Publish(FakeOp.Hang);

            var (fault, after) = rig.WaitForPollFault(TimeSpan.FromSeconds(10));

            Assert.Equal(FaultClass.Hang, fault);
            Assert.InRange(after.TotalMilliseconds, 150, 1500);
            Assert.Equal(RecoveryAction.Respawn, FaultRules.ActionFor(fault));
            ReleaseHang();
            await rig.Transport.KillAsync(CancellationToken.None);
            Assert.Equal(TransportState.Idle, rig.Transport.State);
        }
        finally
        {
            ReleaseHang();
        }
    }

    [Fact]
    public async Task MissedDeadlineWithAFrozenHeartbeatIsStalled()
    {
        try
        {
            await using var rig = NewRig();
            _ = await rig.StartAsync(Rig.DefaultLaunch(deadline: TimeSpan.FromMilliseconds(200)));
            var op = Freeze(rig);
            await Task.Delay(30, TestContext.Current.CancellationToken); // let the last beat age past the liveness window
            _ = rig.Publish(op);

            var (fault, _) = rig.WaitForPollFault(TimeSpan.FromSeconds(10));

            Assert.Equal(FaultClass.Stalled, fault);
            Thaw(rig);
            ReleaseHang();
            await rig.Transport.KillAsync(CancellationToken.None);
        }
        finally
        {
            ReleaseHang();
        }
    }

    [Fact]
    public async Task ASlowStepWithinTheDeadlineIsNotAFault()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync(Rig.DefaultLaunch(deadline: TimeSpan.FromMilliseconds(500)));
        _ = rig.Publish(FakeOp.Slow, 100);

        var seen = rig.Wait(TimeSpan.FromSeconds(10));

        Assert.True(seen.HasResult);
        Assert.Equal(FaultClass.None, seen.Fault);
    }

    [Fact]
    public async Task ASecondStepIsNotChargedForTheTimeItWaitsBehindTheFirst()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync(Rig.DefaultLaunch(deadline: TimeSpan.FromMilliseconds(250)));
        _ = rig.Publish(FakeOp.Slow, 180);
        _ = rig.Publish(FakeOp.Slow, 180); // published together; finishes about 360 ms after publication

        var polls = Stopwatch.StartNew();
        var results = 0;
        while (results < 2)
        {
            if (rig.Transport.TryTakeResult(out var slot))
            {
                rig.Transport.ReleaseResult((int)slot.BufferIndex);
                results++;
            }

            Assert.Equal(FaultClass.None, rig.Transport.Poll());
            Assert.True(polls.Elapsed < TimeSpan.FromSeconds(10));
            Thread.Sleep(1);
        }
    }

    [Fact]
    public async Task ACrashedWorkerIsReportedExitedWithoutAResult()
    {
        Assert.SkipUnless(CanCrash, "a thread of the test process cannot crash on its own");
        await using var rig = NewRig();
        _ = await rig.StartAsync();

        _ = rig.Publish(FakeOp.Crash);
        var seen = rig.Wait(TimeSpan.FromSeconds(10));

        Assert.False(seen.HasResult);
        Assert.Equal(FaultClass.Exited, seen.Polled);
        Assert.Equal(RecoveryAction.Respawn, FaultRules.ActionFor(seen.Polled));
    }

    [Fact]
    public async Task KillThenStartAgainRunsAFreshWorkerFromStepOne()
    {
        await using var rig = NewRig();
        var first = await rig.StartAsync();
        _ = rig.Publish();
        Assert.True(rig.Wait(TimeSpan.FromSeconds(10)).EchoOk);

        await rig.Transport.KillAsync(CancellationToken.None);
        Assert.Equal(TransportState.Idle, rig.Transport.State);
        Assert.Equal(FaultClass.None, rig.Transport.Poll());

        rig.NextStepId = 1;
        var second = await rig.StartAsync();
        Assert.Equal(HandshakeState.Ready, second.State);
        if (Kind == TransportKind.Process)
        {
            Assert.NotEqual(first.WorkerProcessId, second.WorkerProcessId);
        }

        _ = rig.Publish();
        var seen = rig.Wait(TimeSpan.FromSeconds(10));
        Assert.Equal(1UL, seen.StepId);
        Assert.True(seen.EchoOk);
    }

    [Fact]
    public async Task ShutdownIsCleanAndLeavesNothingBehind()
    {
        var rig = NewRig();
        _ = await rig.StartAsync();
        var path = rig.Transport.SegmentPath!;
        Assert.True(File.Exists(path));
        _ = rig.Publish();
        _ = rig.Wait(TimeSpan.FromSeconds(10));

        var started = Stopwatch.StartNew();
        await rig.DisposeAsync();

        Assert.True(started.Elapsed < TimeSpan.FromSeconds(5), $"shutdown took {started.Elapsed}");
        Assert.False(File.Exists(path), "segment file left behind");
        Assert.Equal(TransportState.Disposed, rig.Transport.State);
        if (rig.Transport is ProcessWorkerTransport process)
        {
            Assert.Equal(0, process.ExitCode); // a clean shutdown slot, not a kill
        }
    }

    [Fact]
    public async Task SteadyStateAllocatesNothingOnTheHostThread()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();
        for (var i = 0; i < 3000; i++)
        {
            _ = rig.Publish();
            _ = rig.Wait(TimeSpan.FromSeconds(10));
        }

        var transport = rig.Transport;
        var before = GC.GetAllocatedBytesForCurrentThread();
        ulong done = 0;
        var inFlight = 0;
        var id = rig.NextStepId;
        while (done < 20_000)
        {
            while (inFlight < 2 && transport.CanPublish())
            {
                var index = transport.NextPlanBufferIndex;
                TestPlans.Fill(transport.PlanBuffer(index), id);
                Assert.True(transport.TryPublish(new RingSlot { StepId = id++, Kind = RingSlotKind.Step, BufferIndex = (uint)index }));
                inFlight++;
            }

            if (transport.TryTakeResult(out var slot))
            {
                transport.ReleaseResult((int)slot.BufferIndex);
                inFlight--;
                done++;
            }

            _ = transport.Poll();
            _ = transport.ReadHeartbeat();
        }

        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
        rig.NextStepId = id;
    }
}

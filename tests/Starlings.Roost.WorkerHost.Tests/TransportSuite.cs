using System.Diagnostics;
using System.Globalization;
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
            Assert.InRange(after.TotalMilliseconds, 150, 8000); // the upper bound only catches a classification that never comes
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

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public async Task MissedDeadlineWithALiveHeartbeatIsAHangAtTheDefaultLivenessWindow(int round)
    {
        try
        {
            // The default HeartbeatLiveness (50 ms), as shipped: the hang path must not depend on a widened window on a quiet machine.
            await using var rig = NewRig();
            _ = await rig.StartAsync(Rig.DefaultLaunch(deadline: TimeSpan.FromMilliseconds(200)));
            _ = rig.Publish(FakeOp.Hang);

            var (fault, _) = rig.WaitForPollFault(TimeSpan.FromSeconds(10));

            Assert.True(fault == FaultClass.Hang, $"round {round}: classified {fault}; heartbeat counter {rig.Transport.ReadHeartbeat().Counter}");
            ReleaseHang();
            await rig.Transport.KillAsync(CancellationToken.None);
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
        _ = await rig.StartAsync(Rig.DefaultLaunch(deadline: TimeSpan.FromMilliseconds(3000)));
        _ = rig.Publish(FakeOp.Slow, 100);

        var seen = rig.Wait(TimeSpan.FromSeconds(10));

        Assert.True(seen.HasResult);
        Assert.Equal(FaultClass.None, seen.Fault);
    }

    [Fact]
    public async Task ASecondStepIsNotChargedForTheTimeItWaitsBehindTheFirst()
    {
        // Each fake step sleeps StepMs; the deadline is DeadlineMs. Charged from execution start, step two takes about StepMs
        // after step one's result, well inside the deadline (DeadlineMs - StepMs of room for a late wake-up on a busy runner).
        // Charged from publication it would be due at DeadlineMs while it cannot finish before 2 * StepMs, so the fault would
        // always show. The slack on the failing side is 2 * StepMs - DeadlineMs, and a sleep never ends early, so only the
        // passing side depends on the machine.
        const int StepMs = 600;
        const int DeadlineMs = 1000;
        await using var rig = NewRig();
        _ = await rig.StartAsync(Rig.DefaultLaunch(deadline: TimeSpan.FromMilliseconds(DeadlineMs)));
        var clock = Stopwatch.StartNew();
        _ = rig.Publish(FakeOp.Slow, StepMs);
        _ = rig.Publish(FakeOp.Slow, StepMs); // published together; finishes about 2 * StepMs after publication

        var resultNs = new List<ulong>();
        var takenMs = new List<double>();
        var lastProgress = Stopwatch.StartNew();
        string Timings() =>
            $"deadline {DeadlineMs} ms, step {StepMs} ms; results taken at [{string.Join(", ", takenMs.Select(t => t.ToString("F0", CultureInfo.InvariantCulture)))}] ms after publication; "
            + (resultNs.Count == 2
                ? $"second step ran {(resultNs[1] - resultNs[0]) / 1e6:F0} ms after the first result (limit {DeadlineMs} ms)"
                : $"{resultNs.Count} of 2 results, {clock.Elapsed.TotalMilliseconds:F0} ms since publication");

        while (resultNs.Count < 2)
        {
            if (rig.Transport.TryTakeResult(out var slot))
            {
                resultNs.Add(slot.PublishedNs);
                takenMs.Add(clock.Elapsed.TotalMilliseconds);
                rig.Transport.ReleaseResult((int)slot.BufferIndex);
                lastProgress.Restart();
            }

            var fault = rig.Transport.Poll();
            Assert.True(fault == FaultClass.None, $"classified {fault}; {Timings()}");
            Assert.True(lastProgress.Elapsed < TimeSpan.FromSeconds(10), $"no progress for 10 s; {Timings()}");
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(60), $"ceiling reached; {Timings()}");
            Thread.Sleep(1);
        }

        TestContext.Current.SendDiagnosticMessage(Timings());
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

using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;
using Starlings.Roost.WorkerHost.Tests.Support;

namespace Starlings.Roost.WorkerHost.Tests;

/// <summary>The shared suite over the worker on a thread of the test process.</summary>
public sealed class InProcessTransportTests : TransportSuite, IDisposable
{
    private readonly ManualResetEventSlim _hangLatch = new(false);
    private volatile bool _pauseHeartbeat;

    public void Dispose() => _hangLatch.Dispose();

    protected override TransportKind Kind => TransportKind.InProcess;

    protected override bool CanCrash => false;

    internal override Rig NewRig(WorkerTransportOptions? options = null) =>
        new(new InProcessWorkerTransport(() => new FakeExecutor(_hangLatch), TestSegments.Options(options), () => _pauseHeartbeat));

    internal override FakeOp Freeze(Rig rig)
    {
        _pauseHeartbeat = true;
        return FakeOp.Hang; // a thread cannot be stopped, so the step itself must not finish
    }

    internal override void Thaw(Rig rig) => _pauseHeartbeat = false;

    internal override void ReleaseHang() => _hangLatch.Set();

    [Fact]
    public async Task KillingAWorkerStuckInAStepAbandonsTheThreadAndRefusesToRestart()
    {
        try
        {
            await using var rig = NewRig(new WorkerTransportOptions { KillTimeout = TimeSpan.FromMilliseconds(300) });
            _ = await rig.StartAsync(Rig.DefaultLaunch(deadline: TimeSpan.FromMilliseconds(100)));
            var path = rig.Transport.SegmentPath!;
            _ = rig.Publish(FakeOp.Hang);
            Assert.Equal(FaultClass.Hang, rig.WaitForPollFault(TimeSpan.FromSeconds(10)).Fault);

            await rig.Transport.KillAsync(CancellationToken.None);

            Assert.Equal(TransportState.Abandoned, rig.Transport.State);
            Assert.False(File.Exists(path), "the segment file of an abandoned worker was left in the segment directory");
            _ = await Assert.ThrowsAsync<InvalidOperationException>(async () => await rig.StartAsync());
        }
        finally
        {
            _hangLatch.Set();
        }
    }

    [Fact]
    public async Task OpenDeviceThrowingInProcessIsReportedAsAStartFailure()
    {
        await using var transport = new InProcessWorkerTransport(() => new FakeExecutor(failOpen: true), TestSegments.Options());

        var ex = await Assert.ThrowsAsync<WorkerStartException>(async () =>
            await transport.StartAsync(Rig.DefaultLaunch(), CancellationToken.None));

        Assert.Contains("OpenDevice threw", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void InProcessTransportIsTheOnlyOneThatRunsOnAThread() =>
        Assert.Equal(TransportKind.Process, WorkerTransports.Default);
}

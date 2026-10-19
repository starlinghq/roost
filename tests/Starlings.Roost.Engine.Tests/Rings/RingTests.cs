using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Engine.Threading;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Tests.Rings;

public sealed class RingTests : IDisposable
{
    private readonly string _path = Path.Combine(RingSegment.DefaultDirectory(), $"roost-ring-test-{Guid.NewGuid():N}");
    private readonly RingSegment _segment;

    public RingTests()
    {
        _segment = RingSegment.Create(_path, new SegmentOptions { PlanSlotCount = 8, ResultSlotCount = 8 });
    }

    // Close stops and joins every poller the test made with CreatePoller before it unmaps; false means one is still
    // running and the view was kept mapped, which is a test failure, not a crash.
    public void Dispose() => Assert.True(_segment.Close(TimeSpan.FromSeconds(30)), "a poller did not stop; the segment was left mapped");

    [Fact]
    public void PlanRingHoldsAtMostTheBufferCountUntilTheWorkerReleases()
    {
        var host = _segment.OpenHost();
        var worker = _segment.OpenWorker();
        for (var n = 0; n < 2; n++)
        {
            Assert.True(host.CanPublish());
            Assert.Equal(n, host.NextPlanBufferIndex);
            Assert.True(host.TryPublish(new RingSlot { StepId = (ulong)n, Kind = RingSlotKind.Step, BufferIndex = (uint)n }));
        }

        Assert.False(host.CanPublish());
        Assert.False(host.TryPublish(new RingSlot { StepId = 2, Kind = RingSlotKind.Step, BufferIndex = 0 }));

        Assert.True(worker.TryTakePlan(out var first));
        Assert.Equal(0UL, first.StepId);
        Assert.False(host.CanPublish()); // taken but not released: the buffer is still the worker's
        worker.ReleasePlan();
        Assert.True(host.CanPublish());
        Assert.Equal(0, host.NextPlanBufferIndex);
        Assert.True(worker.TryTakePlan(out var second));
        Assert.Equal(1UL, second.StepId);
        Assert.False(worker.TryTakePlan(out _));
    }

    [Fact]
    public void ABufferOtherThanTheNextOneIsRefused()
    {
        var host = _segment.OpenHost();
        Assert.Throws<InvalidDataException>(() => host.TryPublish(new RingSlot { StepId = 1, Kind = RingSlotKind.Step, BufferIndex = 1 }));
    }

    [Fact]
    public void ResultsAreReleasedInOrderOnly()
    {
        var host = _segment.OpenHost();
        var worker = _segment.OpenWorker();
        for (var n = 0; n < 2; n++)
        {
            Assert.True(worker.CanPublishResult());
            Assert.True(worker.TryPublishResult(new RingSlot { StepId = (ulong)n, Kind = RingSlotKind.Step, BufferIndex = (uint)n }));
        }

        Assert.False(worker.CanPublishResult());
        Assert.True(host.TryTakeResult(out _));
        Assert.True(host.TryTakeResult(out _));
        Assert.Throws<ArgumentException>(() => host.ReleaseResult(1));
        host.ReleaseResult(0);
        Assert.True(worker.CanPublishResult());
        host.ReleaseResult(1);
        Assert.Equal(0, host.ResultsOutstanding);
        Assert.Throws<InvalidOperationException>(() => host.ReleaseResult(0));
    }

    [Fact]
    public void CorruptCursorsAreReportedNotFollowed()
    {
        var host = _segment.OpenHost();
        var worker = _segment.OpenWorker();
        Assert.True(host.TryPublish(new RingSlot { StepId = 1, Kind = RingSlotKind.Step, BufferIndex = 0 }));
        PokeProducerCursor(_segment.Header.PlanRingOffset, 1000);
        Assert.Throws<InvalidDataException>(() => worker.TryTakePlan(out _));
    }

    [Fact]
    public void ShutdownAndDrainSlotsPassThrough()
    {
        var host = _segment.OpenHost();
        var worker = _segment.OpenWorker();
        Assert.True(host.TryPublish(new RingSlot { Kind = RingSlotKind.Drain }));
        Assert.True(host.TryPublish(new RingSlot { Kind = RingSlotKind.Shutdown }));
        Assert.True(worker.TryTakePlan(out var a));
        Assert.True(worker.TryTakePlan(out var b));
        Assert.Equal(RingSlotKind.Drain, a.Kind);
        Assert.Equal(RingSlotKind.Shutdown, b.Kind);
    }

    [Fact]
    public void TwoThreadsExchangeEveryStepInOrder()
    {
        const int steps = 400_000;
        var host = _segment.OpenHost();
        var worker = _segment.OpenWorker();
        string? workerFailure = null;
        var done = 0;

        // The worker is a PollingThread over the segment: take a plan, check it, publish a result, release the plan.
        // It takes a plan only when a result slot is free, so a poll never waits inside itself and a stop request is
        // seen between polls even if the host stops releasing results.
        using var workerLoop = _segment.CreatePoller("test-worker", () =>
        {
            if (done >= steps || !worker.CanPublishResult() || !worker.TryTakePlan(out var slot))
            {
                return false;
            }

            var failure = Stress.StressProtocol.VerifyPlan(worker.PlanBuffer((int)slot.BufferIndex), slot.StepId);
            if (slot.StepId != Stress.StressProtocol.FirstStepId + (ulong)done || failure is not null)
            {
                workerFailure ??= failure ?? $"step {slot.StepId} at position {done}";
            }

            Stress.StressProtocol.FillResult(worker.ResultBuffer(worker.NextResultBufferIndex), slot.StepId);
            Assert.True(worker.TryPublishResult(Stress.StressProtocol.ResultSlot((ulong)done, 2)));
            worker.ReleasePlan();
            Volatile.Write(ref done, done + 1);
            return true;
        });
        workerLoop.Start();

        ulong sent = 0;
        ulong got = 0;
        var spin = new SpinWait();
        var deadline = Environment.TickCount64 + 60_000;
        while (got < steps)
        {
            if (sent < steps && host.CanPublish())
            {
                var slot = Stress.StressProtocol.PlanSlot(sent, 2);
                Stress.StressProtocol.FillPlan(host.PlanBuffer(host.NextPlanBufferIndex), slot.StepId);
                Assert.True(host.TryPublish(slot));
                sent++;
            }

            if (host.TryTakeResult(out var result))
            {
                Assert.Null(Stress.StressProtocol.CheckSlot(result, Stress.StressProtocol.ResultSlot(got, 2), "result"));
                Assert.Null(Stress.StressProtocol.VerifyResult(host.ResultBuffer((int)result.BufferIndex), result.StepId));
                host.ReleaseResult((int)result.BufferIndex);
                got++;
                spin.Reset();
            }
            else
            {
                spin.SpinOnce(-1);
                Assert.True(Environment.TickCount64 < deadline, "exchange stalled");
            }
        }

        Assert.True(workerLoop.Stop(TimeSpan.FromSeconds(30)));
        Assert.Null(workerLoop.Fault);
        Assert.Null(workerFailure);
    }

    private unsafe void PokeProducerCursor(ulong ringOffset, ulong value)
    {
        var ring = System.Runtime.CompilerServices.Unsafe.Read<RingHeader>(_segment.Base + (long)ringOffset);
        *(ulong*)(_segment.Base + (long)ring.ProducerCursorOffset) = value;
    }
}

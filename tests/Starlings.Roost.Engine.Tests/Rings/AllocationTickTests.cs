using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Engine.Tests.Stress;
using Starlings.Roost.Engine.Threading;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Tests.Rings;

/// <summary>
/// The allocation-tick test (R-027): the hot path must allocate nothing. Each "tick" is one engine step as the rings
/// see it; the thread's allocated-byte counter is read around many ticks after a warm-up, and any difference fails.
/// A control tick that allocates proves the detector sees allocations.
/// </summary>
public sealed class AllocationTickTests
{
    private const int Warmup = 20_000;
    private const int Ticks = 200_000;

    private static long Measure(Action tick)
    {
        for (var i = 0; i < Warmup; i++)
        {
            tick();
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < Ticks; i++)
        {
            tick();
        }

        var after = GC.GetAllocatedBytesForCurrentThread();
        return after - before;
    }

    [Fact]
    public void TheDetectorSeesAnAllocation()
    {
        object? sink = null;
        var bytes = Measure(() => sink = new byte[24]);
        GC.KeepAlive(sink);
        Assert.True(bytes >= Ticks * 24L, $"control allocated only {bytes} bytes");
    }

    [Fact]
    public void AFullStepThroughBothRingsAllocatesNothing()
    {
        var path = Path.Combine(RingSegment.DefaultDirectory(), $"roost-alloc-test-{Guid.NewGuid():N}");
        using var segment = RingSegment.Create(path, StressProtocol.Options);
        var host = segment.OpenHost();
        var worker = segment.OpenWorker();
        ulong n = 0;

        Action tick = () =>
        {
            // host: write the plan, publish it
            var plan = StressProtocol.PlanSlot(n, 2);
            StressProtocol.FillPlan(host.PlanBuffer(host.NextPlanBufferIndex), plan.StepId);
            if (!host.TryPublish(plan))
            {
                throw new InvalidOperationException("plan ring full");
            }

            // worker: take it, read it, publish the result, release the plan
            if (!worker.TryTakePlan(out var seen) || StressProtocol.VerifyPlan(worker.PlanBuffer((int)seen.BufferIndex), seen.StepId) is not null)
            {
                throw new InvalidOperationException("plan lost");
            }

            StressProtocol.FillResult(worker.ResultBuffer(worker.NextResultBufferIndex), seen.StepId);
            if (!worker.TryPublishResult(StressProtocol.ResultSlot(n, 2)))
            {
                throw new InvalidOperationException("result ring full");
            }

            worker.ReleasePlan();

            // host: take the result, read it, hand the buffer back
            if (!host.TryTakeResult(out var result) || StressProtocol.VerifyResult(host.ResultBuffer((int)result.BufferIndex), result.StepId) is not null)
            {
                throw new InvalidOperationException("result lost");
            }

            host.ReleaseResult((int)result.BufferIndex);
            n++;
        };

        Assert.Equal(0, Measure(tick));
        Assert.Equal((ulong)(Warmup + Ticks), n);
    }

    [Fact]
    public void RequestQueueEnqueueAndDequeueAllocateNothing()
    {
        var queue = new MpscQueue<object>(64);
        var requests = new object[32];
        for (var i = 0; i < requests.Length; i++)
        {
            requests[i] = new object();
        }

        var at = 0;
        Action tick = () =>
        {
            for (var i = 0; i < 3; i++)
            {
                if (!queue.TryEnqueue(requests[(at + i) % requests.Length]))
                {
                    throw new InvalidOperationException("queue full");
                }
            }

            for (var i = 0; i < 3; i++)
            {
                if (!queue.TryDequeue(out _))
                {
                    throw new InvalidOperationException("queue empty");
                }
            }

            at = (at + 3) % requests.Length;
        };

        Assert.Equal(0, Measure(tick));
    }

    [Fact]
    public void ThePollingLoopAllocatesNothingOnItsOwnThread()
    {
        long polls = 0;
        long atWarm = 0;
        long atEnd = -1;
        using var finished = new ManualResetEventSlim();
        using var thread = new PollingThread("alloc-poll", () =>
        {
            polls++;
            if (polls == Warmup)
            {
                atWarm = GC.GetAllocatedBytesForCurrentThread();
            }
            else if (polls == Warmup + Ticks)
            {
                atEnd = GC.GetAllocatedBytesForCurrentThread();
                finished.Set();
            }

            return polls % 3 != 0; // alternate work and idle so the backoff path runs too
        });
        thread.Start();
        Assert.True(finished.Wait(TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken));
        Assert.True(thread.Stop(TimeSpan.FromSeconds(30)));
        Assert.Equal(atWarm, atEnd);
        Assert.Null(thread.Fault);
    }
}

using System.Runtime.CompilerServices;
using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Tests.Stress;

/// <summary>
/// The worker process of the stress test. The test executable starts itself with <c>--roost-ring-worker</c>; the
/// module initializer below runs before the xunit runner and turns that process into this worker.
/// </summary>
internal static class StressWorker
{
    public const string Flag = "--roost-ring-worker";
    private const ulong WarmupSteps = 20_000;

    [ModuleInitializer]
    internal static void EnterWorkerIfRequested()
    {
        var args = Environment.GetCommandLineArgs();
        var at = Array.IndexOf(args, Flag);
        if (at < 0)
        {
            return;
        }

        var code = 2;
        try
        {
            code = Run(args[at + 1], ulong.Parse(args[at + 2], System.Globalization.CultureInfo.InvariantCulture), ulong.Parse(args[at + 3], System.Globalization.CultureInfo.InvariantCulture));
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"worker crashed: {ex}");
        }

        Environment.Exit(code);
    }

    private static int Run(string path, ulong seed, ulong steps)
    {
        using var segment = RingSegment.Open(path);
        var worker = segment.OpenWorker();
        var rng = new StressProtocol.Rng(seed);
        var planBuffers = (int)segment.Header.PlanBufferCount;
        var resultBuffers = (int)segment.Header.ResultBufferCount;

        // Up to two plans are held at once (two steps in flight, S8): take the next while the previous is processed.
        Span<RingSlot> held = stackalloc RingSlot[2];
        var heldCount = 0;
        ulong done = 0;
        ulong planEmpty = 0;
        ulong resultFull = 0;
        ulong twoInFlight = 0;
        var allocAtWarm = 0L;
        var shutdown = false;
        string? failure = null;

        while (!shutdown && failure is null)
        {
            var progressed = false;
            while (heldCount < 2 && worker.TryTakePlan(out var slot))
            {
                if (slot.Kind == RingSlotKind.Shutdown)
                {
                    shutdown = true;
                    worker.ReleasePlan();
                    break;
                }

                held[heldCount++] = slot;
                progressed = true;
            }

            if (heldCount == 2)
            {
                twoInFlight++;
            }

            if (heldCount > 0)
            {
                ref readonly var slot = ref held[0];
                var expected = StressProtocol.PlanSlot(done, planBuffers);
                failure = StressProtocol.CheckSlot(slot, expected, "plan slot")
                    ?? StressProtocol.VerifyPlan(worker.PlanBuffer((int)slot.BufferIndex), slot.StepId);
                if (failure is not null)
                {
                    break;
                }

                rng.RandomDelay(); // the "device step"
                while (!worker.CanPublishResult())
                {
                    resultFull++;
                    Thread.SpinWait(20);
                }

                var resultIndex = worker.NextResultBufferIndex;
                StressProtocol.FillResult(worker.ResultBuffer(resultIndex), slot.StepId);
                if (!worker.TryPublishResult(StressProtocol.ResultSlot(done, resultBuffers)))
                {
                    failure = "result ring refused a slot after CanPublishResult";
                    break;
                }

                worker.ReleasePlan(); // after the plan buffer was fully read
                held[0] = held[1];
                heldCount--;
                done++;
                progressed = true;
                if (done == WarmupSteps)
                {
                    allocAtWarm = GC.GetAllocatedBytesForCurrentThread();
                }
            }

            if (!progressed)
            {
                planEmpty++;
                Thread.SpinWait(10);
            }
        }

        var allocated = done > WarmupSteps ? GC.GetAllocatedBytesForCurrentThread() - allocAtWarm : 0;
        if (failure is null && done != steps)
        {
            failure = $"worker saw shutdown after {done} steps, expected {steps}";
        }

        if (failure is not null)
        {
            Console.Out.WriteLine($"FAIL {failure}");
            return 1;
        }

        Console.Out.WriteLine($"OK steps={done} planEmpty={planEmpty} resultFull={resultFull} twoInFlight={twoInFlight} allocatedBytes={allocated}");
        return 0;
    }
}

using System.Diagnostics;
using System.Reflection;
using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Tests.Stress;

/// <summary>The host side of the stress test: creates the segment, starts the worker process, drives it, checks every result.</summary>
internal static class StressHost
{
    private const ulong WarmupSteps = 20_000;

    public sealed record Outcome(
        ulong Steps,
        ulong PublishFull,
        ulong ResultEmpty,
        long HostAllocatedBytes,
        string WorkerSummary,
        string WorkerErrors);

    public static Outcome Run(ulong steps, ulong seed, TimeSpan timeout)
    {
        var path = Path.Combine(RingSegment.DefaultDirectory(), $"roost-ring-stress-{Environment.ProcessId}-{Guid.NewGuid():N}");
        using var segment = RingSegment.Create(path, StressProtocol.Options);
        var host = segment.OpenHost();
        using var process = StartWorker(path, seed ^ 0x5DEECE66DUL, steps);
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();

        var rng = new StressProtocol.Rng(seed);
        var planBuffers = host.PlanBufferCount;
        const int resultBuffers = 2;
        ulong sent = 0;
        ulong got = 0;
        ulong publishFull = 0;
        ulong resultEmpty = 0;
        var allocAtWarm = 0L;
        string? failure = null;
        var deadline = Stopwatch.GetTimestamp() + (long)(timeout.TotalSeconds * Stopwatch.Frequency);
        var wait = new SpinWait();

        while (got < steps && failure is null)
        {
            var progressed = false;
            if (sent < steps)
            {
                if (host.CanPublish())
                {
                    var slot = StressProtocol.PlanSlot(sent, planBuffers);
                    StressProtocol.FillPlan(host.PlanBuffer(host.NextPlanBufferIndex), slot.StepId);
                    if (!host.TryPublish(slot))
                    {
                        failure = "plan ring refused a slot after CanPublish";
                        break;
                    }

                    sent++;
                    progressed = true;
                }
                else
                {
                    publishFull++;
                }
            }

            while (host.TryTakeResult(out var result))
            {
                failure = StressProtocol.CheckSlot(result, StressProtocol.ResultSlot(got, resultBuffers), "result slot")
                    ?? StressProtocol.VerifyResult(host.ResultBuffer((int)result.BufferIndex), result.StepId);
                if (failure is not null)
                {
                    break;
                }

                rng.RandomDelay(); // the host "uses" the result before handing the buffer back
                host.ReleaseResult((int)result.BufferIndex);
                got++;
                progressed = true;
                if (got == WarmupSteps)
                {
                    allocAtWarm = GC.GetAllocatedBytesForCurrentThread();
                }
            }

            if (progressed)
            {
                wait.Reset();
                rng.RandomDelay();
            }
            else
            {
                resultEmpty++;
                wait.SpinOnce(-1);
                if (Stopwatch.GetTimestamp() > deadline || process.HasExited)
                {
                    failure = $"stalled or worker exited with {sent} sent and {got} received (exited={process.HasExited})";
                }
            }
        }

        var allocated = got > WarmupSteps ? GC.GetAllocatedBytesForCurrentThread() - allocAtWarm : 0;
        if (failure is null)
        {
            var shutdown = new RingSlot { StepId = 0, Kind = RingSlotKind.Shutdown };
            while (!host.TryPublish(shutdown))
            {
                Thread.SpinWait(50);
            }
        }

        var exited = process.WaitForExit(failure is null ? 60_000 : 2_000);
        if (!exited)
        {
            process.Kill(entireProcessTree: true);
            process.WaitForExit();
            failure ??= "worker did not exit after shutdown";
        }

        var summary = stdout.GetAwaiter().GetResult().Trim();
        var errors = stderr.GetAwaiter().GetResult().Trim();
        if (failure is null && process.ExitCode != 0)
        {
            failure = $"worker exit code {process.ExitCode}: {summary} {errors}";
        }

        if (failure is not null)
        {
            throw new InvalidOperationException($"{failure}. Worker said: {summary} {errors}");
        }

        return new Outcome(got, publishFull, resultEmpty, allocated, summary, errors);
    }

    private static Process StartWorker(string path, ulong seed, ulong steps)
    {
        var host = Environment.ProcessPath ?? throw new InvalidOperationException("No process path.");
        var info = new ProcessStartInfo(host)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        if (Path.GetFileNameWithoutExtension(host).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
        {
            info.ArgumentList.Add((Assembly.GetEntryAssembly() ?? typeof(StressHost).Assembly).Location);
        }

        info.ArgumentList.Add(StressWorker.Flag);
        info.ArgumentList.Add(path);
        info.ArgumentList.Add(seed.ToString(System.Globalization.CultureInfo.InvariantCulture));
        info.ArgumentList.Add(steps.ToString(System.Globalization.CultureInfo.InvariantCulture));
        return Process.Start(info) ?? throw new InvalidOperationException("Could not start the worker process.");
    }
}

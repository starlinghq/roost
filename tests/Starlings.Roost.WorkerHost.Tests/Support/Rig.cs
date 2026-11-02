using System.Collections.Immutable;
using System.Diagnostics;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Roost.WorkerHost.Tests.Support;

/// <summary>What the host saw for one wait: a result slot (with its header and echo check), or a fault concluded by <c>Poll</c>.</summary>
internal readonly record struct Seen(bool HasResult, ulong StepId, StepResultHeader Header, bool EchoOk, bool ChecksumOk, FaultClass Polled)
{
    public FaultClass Fault => HasResult ? Header.Fault : Polled;
}

/// <summary>Drives a transport the way an engine thread would: publish plans, take results, poll.</summary>
internal sealed class Rig : IAsyncDisposable
{
    public Rig(WorkerTransportBase transport)
    {
        Transport = transport;
    }

    public WorkerTransportBase Transport { get; }

    public ulong NextStepId { get; set; } = 1;

    public static WorkerLaunch DefaultLaunch(TimeSpan? deadline = null, int device = 0, (int, int)? mesh = null) => new()
    {
        DeviceOrdinal = device,
        Mesh = mesh ?? (0, 1),
        ManifestDigest = FakeExecutor.ManifestDigest,
        WeightsDigest = FakeExecutor.WeightsDigest,
        StepDeadline = deadline ?? TimeSpan.FromSeconds(5),
        HeartbeatPeriod = TimeSpan.FromMilliseconds(1),
        PlanBufferBytes = 8192,
        ResultBufferBytes = 4096,
    };

    public async Task<HandshakeResult> StartAsync(WorkerLaunch? launch = null)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        return await Transport.StartAsync(launch ?? DefaultLaunch(), timeout.Token);
    }

    /// <summary>Publishes one step if a plan buffer is free; returns its id or 0.</summary>
    public ulong TryPublish(FakeOp op = FakeOp.Normal, uint argument = 0, ulong deadlineNs = 0)
    {
        if (!Transport.CanPublish())
        {
            return 0;
        }

        var index = Transport.NextPlanBufferIndex;
        var id = NextStepId;
        TestPlans.Fill(Transport.PlanBuffer(index), id, op, argument);
        var slot = new RingSlot { StepId = id, Kind = RingSlotKind.Step, BufferIndex = (uint)index, DeadlineNs = deadlineNs };
        if (!Transport.TryPublish(slot))
        {
            return 0;
        }

        NextStepId++;
        return id;
    }

    public ulong Publish(FakeOp op = FakeOp.Normal, uint argument = 0)
    {
        var until = Stopwatch.GetTimestamp() + (5 * Stopwatch.Frequency);
        ulong id;
        while ((id = TryPublish(op, argument)) == 0)
        {
            if (Stopwatch.GetTimestamp() > until)
            {
                throw new TimeoutException("Could not publish a plan within 5 s.");
            }

            Thread.SpinWait(100);
        }

        return id;
    }

    /// <summary>Waits for a result or for <c>Poll</c> to conclude a fault. A taken result is released.</summary>
    public Seen Wait(TimeSpan timeout)
    {
        var until = Stopwatch.GetTimestamp() + (long)(timeout.TotalSeconds * Stopwatch.Frequency);
        var wait = new SpinWait();
        while (true)
        {
            if (Transport.TryTakeResult(out var slot))
            {
                var buffer = Transport.ResultBuffer((int)slot.BufferIndex);
                var header = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<StepResultHeader>(ref System.Runtime.CompilerServices.Unsafe.AsRef(in buffer[0]));
                var ok = header.Fault == FaultClass.None;
                var echo = !ok || TestPlans.EchoMatches(buffer, slot.StepId);
                var sum = !ok || header.Reserved1 == TestPlans.ExpectedChecksum(slot.StepId);
                Transport.ReleaseResult((int)slot.BufferIndex);
                return new Seen(true, slot.StepId, header, echo && header.StepId == slot.StepId, sum, FaultClass.None);
            }

            var polled = Transport.Poll();
            if (polled != FaultClass.None)
            {
                return new Seen(false, 0, default, false, false, polled);
            }

            if (Stopwatch.GetTimestamp() > until)
            {
                throw new TimeoutException($"No result and no fault within {timeout}.");
            }

            wait.SpinOnce(-1);
        }
    }

    /// <summary>Publishes a control message (R-100) written by <paramref name="write"/> and waits for its answer, which is released.</summary>
    public ControlReply Control(RingSlotKind kind, Func<Memory<byte>, ulong, int>? write = null, TimeSpan? timeout = null)
    {
        var until = Stopwatch.GetTimestamp() + (long)((timeout ?? TimeSpan.FromSeconds(10)).TotalSeconds * Stopwatch.Frequency);
        while (!Transport.CanPublish())
        {
            if (Stopwatch.GetTimestamp() > until)
            {
                throw new TimeoutException("No plan buffer came free.");
            }

            Thread.SpinWait(100);
        }

        var index = Transport.NextPlanBufferIndex;
        var id = NextStepId++;
        var buffer = Transport.PlanBuffer(index);
        if (write is null)
        {
            _ = kind switch
            {
                RingSlotKind.Sleep => ControlMessages.WriteSleep(buffer, id, SleepLevel.Kv),
                RingSlotKind.Wake => ControlMessages.WriteWake(buffer, id, FakeExecutor.WeightsDigest.AsSpan()),
                _ => throw new ArgumentException("A replace-weights message needs a writer.", nameof(write)),
            };
        }
        else
        {
            var scratch = new byte[buffer.Length];
            _ = write(scratch, id);
            scratch.CopyTo(buffer);
        }

        Assert.True(Transport.TryPublish(new RingSlot { StepId = id, Kind = kind, BufferIndex = (uint)index }));
        var wait = new SpinWait();
        while (true)
        {
            if (Transport.TryTakeResult(out var slot))
            {
                try
                {
                    Assert.Equal(kind, slot.Kind);
                    Assert.Equal(id, slot.StepId);
                    return ControlMessages.ReadResult(Transport.ResultBuffer((int)slot.BufferIndex), slot);
                }
                finally
                {
                    Transport.ReleaseResult((int)slot.BufferIndex);
                }
            }

            Assert.Equal(FaultClass.None, Transport.Poll());
            if (Stopwatch.GetTimestamp() > until)
            {
                throw new TimeoutException("No control result.");
            }

            wait.SpinOnce(-1);
        }
    }

    /// <summary>Polls until a fault class other than none is concluded, or the timeout passes.</summary>
    public (FaultClass Fault, TimeSpan After) WaitForPollFault(TimeSpan timeout)
    {
        var start = Stopwatch.GetTimestamp();
        var until = start + (long)(timeout.TotalSeconds * Stopwatch.Frequency);
        while (Stopwatch.GetTimestamp() < until)
        {
            var polled = Transport.Poll();
            if (polled != FaultClass.None)
            {
                return (polled, Stopwatch.GetElapsedTime(start));
            }

            Thread.Sleep(1);
        }

        return (FaultClass.None, Stopwatch.GetElapsedTime(start));
    }

    public async ValueTask DisposeAsync() => await Transport.DisposeAsync();
}

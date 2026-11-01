using Starlings.Roost.Engine.Admission;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Requests;
using static Starlings.Roost.Engine.Tests.Scheduling.PlannerHarness;

namespace Starlings.Roost.Engine.Tests.Admission;

public sealed class PolledDeviceMemorySourceTests
{
    private static readonly TimeSpan Period = TimeSpan.FromSeconds(2);

    private static PolledDeviceMemorySource Make(ScriptedMemoryReader reader, ManualTimeProvider time, TimeSpan? maxAge = null) =>
        new(reader, new PolledDeviceMemorySourceOptions { Period = Period, MaxAge = maxAge }, time);

    [Fact]
    public void NothingIsReadableBeforeTheFirstTickAndTheFirstTickIsImmediate()
    {
        var time = new ManualTimeProvider();
        var reader = new ScriptedMemoryReader();
        reader.Returns(24_000, 10_000);
        using var source = Make(reader, time);

        Assert.False(source.TryRead(out _));
        Assert.Equal(DeviceMemoryReadStatus.NotYetRead, source.Current.Status);
        Assert.Equal(0, reader.Calls);

        time.RunDue();
        Assert.Equal(1, reader.Calls);
        Assert.True(source.TryRead(out var m));
        Assert.Equal(new DeviceMemory(24_000, 10_000), m);
    }

    [Fact]
    public void TheCeilingIsReReadEveryPeriodAndTryReadNeverCallsTheReader()
    {
        var time = new ManualTimeProvider();
        var reader = new ScriptedMemoryReader();
        reader.Returns(24_000, 10_000);
        using var source = Make(reader, time);
        time.RunDue();

        reader.Returns(24_000, 6_000); // a co-tenant allocates
        for (var i = 0; i < 100; i++)
        {
            Assert.True(source.TryRead(out var cached));
            Assert.Equal(10_000, cached.FreeBytes); // served from the cache until the next tick
        }

        Assert.Equal(1, reader.Calls);
        time.Advance(Period - TimeSpan.FromMilliseconds(1));
        Assert.Equal(1, reader.Calls);
        time.Advance(TimeSpan.FromMilliseconds(1));
        Assert.Equal(2, reader.Calls);
        Assert.True(source.TryRead(out var m));
        Assert.Equal(6_000, m.FreeBytes);

        time.Advance(Period * 5);
        Assert.Equal(7, reader.Calls);
        Assert.Equal(7, source.Reads);
    }

    [Theory]
    [InlineData(DeviceMemoryReadStatus.LibraryMissing)]
    [InlineData(DeviceMemoryReadStatus.DeviceNotFound)]
    [InlineData(DeviceMemoryReadStatus.DeviceLost)]
    [InlineData(DeviceMemoryReadStatus.Failed)]
    public void AnUnreadableDeviceIsTypedAndRecoversOnTheNextGoodRead(DeviceMemoryReadStatus status)
    {
        var time = new ManualTimeProvider();
        var reader = new ScriptedMemoryReader();
        reader.Returns(24_000, 10_000);
        using var source = Make(reader, time);
        time.RunDue();

        reader.Returns(DeviceMemoryReading.Unreadable(status, "status 15"));
        time.Advance(Period);
        Assert.False(source.TryRead(out var none));
        Assert.Equal(default, none);
        Assert.Equal(status, source.Current.Status);
        Assert.Equal("status 15", source.Current.Detail);
        Assert.Equal(new DeviceMemory(24_000, 10_000), source.LastGood); // kept for diagnostics only

        reader.Returns(24_000, 9_000);
        time.Advance(Period);
        Assert.True(source.TryRead(out var m));
        Assert.Equal(9_000, m.FreeBytes);
        Assert.Equal(DeviceMemoryReadStatus.Read, source.Current.Status);
    }

    [Fact]
    public void AReaderThatThrowsIsAFailedReadingNotAnException()
    {
        var time = new ManualTimeProvider();
        var reader = new ScriptedMemoryReader();
        reader.Throws(new DllNotFoundException("libnvidia-ml.so.1"));
        using var source = Make(reader, time);

        time.RunDue(); // the timer callback must not throw
        Assert.False(source.TryRead(out _));
        Assert.Equal(DeviceMemoryReadStatus.Failed, source.Current.Status);
        Assert.Contains("DllNotFoundException", source.Current.Detail, StringComparison.Ordinal);
        Assert.True(source.Refresh());
        Assert.Equal(2, reader.Calls);
    }

    [Fact]
    public void ANegativeReadingIsRejected()
    {
        var time = new ManualTimeProvider();
        var reader = new ScriptedMemoryReader();
        reader.Returns(24_000, -1);
        using var source = Make(reader, time);
        time.RunDue();
        Assert.False(source.TryRead(out _));
        Assert.Equal(DeviceMemoryReadStatus.Failed, source.Current.Status);
    }

    [Fact]
    public void AReadingOlderThanMaxAgeGoesStaleWhenThePollerStops()
    {
        var time = new ManualTimeProvider();
        var reader = new ScriptedMemoryReader();
        reader.Returns(24_000, 10_000);
        var source = Make(reader, time, maxAge: TimeSpan.FromSeconds(5));
        time.RunDue();
        source.Dispose(); // nothing re-reads any more
        Assert.Equal(0, time.ActiveTimers);

        time.Advance(TimeSpan.FromSeconds(5));
        Assert.True(source.TryRead(out _));
        time.Advance(TimeSpan.FromTicks(1));
        Assert.False(source.TryRead(out _));
        Assert.Equal(DeviceMemoryReadStatus.Stale, source.Current.Status);
        Assert.False(source.Refresh()); // disposed
        Assert.Equal(1, reader.Calls);
    }

    [Fact]
    public void TheDefaultMaxAgeIsThreePeriods()
    {
        var time = new ManualTimeProvider();
        var reader = new ScriptedMemoryReader();
        reader.Returns(24_000, 10_000);
        var source = Make(reader, time);
        time.RunDue();
        source.Dispose();
        time.Advance(Period * 3);
        Assert.True(source.TryRead(out _));
        time.Advance(TimeSpan.FromTicks(1));
        Assert.False(source.TryRead(out _));
    }

    [Fact]
    public void OverlappingReadsAreSkipped()
    {
        var time = new ManualTimeProvider();
        var reader = new ScriptedMemoryReader();
        reader.Returns(24_000, 10_000);
        using var source = Make(reader, time);
        var nested = true;
        reader.DuringRead = () => nested = source.Refresh();
        Assert.True(source.Refresh());
        Assert.False(nested);
        Assert.Equal(1, reader.Calls);
    }

    [Fact]
    public void OptionsAreValidated()
    {
        var reader = new ScriptedMemoryReader();
        Assert.Throws<ArgumentNullException>(() => new PolledDeviceMemorySource(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PolledDeviceMemorySource(reader, new PolledDeviceMemorySourceOptions { Period = TimeSpan.Zero }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PolledDeviceMemorySource(reader, new PolledDeviceMemorySourceOptions { MaxAge = TimeSpan.FromSeconds(-1) }));
        Assert.Throws<ArgumentOutOfRangeException>(() => DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.Read));
    }

    [Fact]
    public void TheSystemTimerPollsInTheBackground()
    {
        var reader = new ScriptedMemoryReader();
        reader.Returns(24_000, 10_000);
        using var source = new PolledDeviceMemorySource(reader, new PolledDeviceMemorySourceOptions { Period = TimeSpan.FromMilliseconds(20) });
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(30);
        while (reader.Calls < 2 && DateTime.UtcNow < deadline)
        {
            Thread.Sleep(5);
        }

        Assert.True(reader.Calls >= 2);
        Assert.True(source.TryRead(out _));
    }

    /// <summary>
    /// R-044's acceptance in miniature, without a GPU: a co-tenant takes memory, the polled ceiling falls below the floor,
    /// and admission turns new requests into a typed <see cref="RoostErrorCode.InsufficientDeviceMemory"/> refusal while the
    /// running request carries on with no abort; when the co-tenant lets go, the next tick lets requests in again.
    /// </summary>
    [Fact]
    public void AFallingCeilingRefusesNewRequestsWithoutAbortingAndRecoversWhenMemoryReturns()
    {
        var time = new ManualTimeProvider();
        var reader = new ScriptedMemoryReader();
        reader.Returns(24_000, 12_000);
        using var source = Make(reader, time);
        time.RunDue();

        var m = Hybrid(attentionBlocks: 100, slots: 8);
        var p = new AdmittedPlanner(
            new StepPlanner(m, new PlannerOptions { MaxTokensPerStep = 64, MaxSequencesPerStep = 8 }),
            new AdmissionController(m, new AdmissionOptions { ContextLength = 64, MinFreeDeviceBytes = 4_000 }, source));

        Assert.Null(p.TryAdd(Req(1, 4, 40)));
        p.ApplyResult(Simulate(p.Plan()!));

        // A co-tenant allocates: the engine still sees the old ceiling until the next tick, then refuses.
        reader.Returns(24_000, 3_000);
        Assert.Null(p.TryAdd(Req(2, 4, 4)));
        time.Advance(Period);
        var refusal = p.TryAdd(Req(3, 4, 4));
        Assert.NotNull(refusal);
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, refusal.Code);
        Assert.Null(refusal.Param); // retryable: not a fault in the request
        Assert.False(p.Planner.TryGetInfo(3, out _));

        // The admitted requests keep running while new ones are refused: no abort, no preemption.
        for (var step = 0; step < 5; step++)
        {
            Assert.NotNull(p.TryAdd(Req(10 + (ulong)step, 4, 4)));
            var plan = p.Plan();
            Assert.NotNull(plan);
            p.ApplyResult(Simulate(plan));
            time.Advance(Period);
        }

        Assert.DoesNotContain(p.DrainNotices(), n => n.Kind == PlannerNoticeKind.Aborted);

        // The device becomes unreadable (NVML lost): admission falls back to the pools and never throws.
        reader.Returns(DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.DeviceLost));
        time.Advance(Period);
        Assert.Null(p.TryAdd(Req(20, 4, 4)));

        // Memory comes back: the next tick admits again.
        reader.Returns(24_000, 3_000);
        time.Advance(Period);
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, p.TryAdd(Req(21, 4, 4))!.Code);
        reader.Returns(24_000, 16_000);
        time.Advance(Period);
        Assert.Null(p.TryAdd(Req(21, 4, 4)));

        for (var step = 0; step < 200 && p.Planner.SequenceCount > 0; step++)
        {
            p.ApplyResult(Simulate(p.Plan()!));
        }

        Assert.Equal(0, p.Planner.SequenceCount);
        Assert.Equal(0, p.Admission.LiveRows);
        Assert.DoesNotContain(p.DrainNotices(), n => n.Kind == PlannerNoticeKind.Aborted);
        p.Planner.CheckInvariants();
    }
}

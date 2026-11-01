using Starlings.Roost.Engine.Admission;

namespace Starlings.Roost.Engine.Tests.Admission;

/// <summary>A clock and timers that move only when <see cref="Advance"/> is called; due callbacks run on the caller's thread.</summary>
internal sealed class ManualTimeProvider : TimeProvider
{
    private readonly Lock _gate = new();
    private readonly List<ManualTimer> _timers = [];
    private long _ticks;

    public override long TimestampFrequency => TimeSpan.TicksPerSecond;

    public int ActiveTimers
    {
        get
        {
            lock (_gate)
            {
                return _timers.Count;
            }
        }
    }

    public override long GetTimestamp() => Interlocked.Read(ref _ticks);

    public override DateTimeOffset GetUtcNow() => new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero).AddTicks(GetTimestamp());

    public override ITimer CreateTimer(TimerCallback callback, object? state, TimeSpan dueTime, TimeSpan period)
    {
        var timer = new ManualTimer(this, callback, state);
        lock (_gate)
        {
            _timers.Add(timer);
        }

        _ = timer.Change(dueTime, period);
        return timer;
    }

    /// <summary>Moves the clock forward, firing every timer that falls due on the way, in order.</summary>
    public void Advance(TimeSpan by)
    {
        var target = GetTimestamp() + by.Ticks;
        while (true)
        {
            ManualTimer? next = null;
            lock (_gate)
            {
                foreach (var t in _timers)
                {
                    if (t.DueAt is { } due && due <= target && (next is null || due < next.DueAt))
                    {
                        next = t;
                    }
                }
            }

            if (next is null)
            {
                break;
            }

            Interlocked.Exchange(ref _ticks, Math.Max(GetTimestamp(), next.DueAt!.Value));
            next.Fire();
        }

        Interlocked.Exchange(ref _ticks, target);
    }

    /// <summary>Runs every timer due now without moving the clock.</summary>
    public void RunDue() => Advance(TimeSpan.Zero);

    private void Remove(ManualTimer timer)
    {
        lock (_gate)
        {
            _timers.Remove(timer);
        }
    }

    private sealed class ManualTimer(ManualTimeProvider owner, TimerCallback callback, object? state) : ITimer
    {
        private long _period;

        public long? DueAt { get; private set; }

        public bool Change(TimeSpan dueTime, TimeSpan period)
        {
            DueAt = dueTime == Timeout.InfiniteTimeSpan ? null : owner.GetTimestamp() + dueTime.Ticks;
            _period = period == Timeout.InfiniteTimeSpan ? 0 : period.Ticks;
            return true;
        }

        public void Fire()
        {
            DueAt = _period > 0 ? DueAt + _period : null;
            callback(state);
        }

        public void Dispose()
        {
            DueAt = null;
            owner.Remove(this);
        }

        public ValueTask DisposeAsync()
        {
            Dispose();
            return ValueTask.CompletedTask;
        }
    }
}

/// <summary>A reader whose next result the test sets; counts its calls.</summary>
internal sealed class ScriptedMemoryReader : IDeviceMemoryReader
{
    private Func<DeviceMemoryReading> _next = () => DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.LibraryMissing);

    public int Calls { get; private set; }

    public Action? DuringRead { get; set; }

    public void Returns(DeviceMemoryReading reading) => _next = () => reading;

    public void Returns(long total, long free) => Returns(DeviceMemoryReading.Of(total, free));

    public void Throws(Exception e) => _next = () => throw e;

    public DeviceMemoryReading Read()
    {
        Calls++;
        DuringRead?.Invoke();
        return _next();
    }
}

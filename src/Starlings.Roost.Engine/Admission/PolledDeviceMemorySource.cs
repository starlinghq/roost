namespace Starlings.Roost.Engine.Admission;

/// <summary>Why a <see cref="DeviceMemoryReading"/> does or does not carry memory.</summary>
public enum DeviceMemoryReadStatus
{
    /// <summary>The reading carries the device's total and free bytes.</summary>
    Read = 0,

    /// <summary>No read has finished yet.</summary>
    NotYetRead,

    /// <summary>The management library (NVML) is not installed or could not be loaded or initialised.</summary>
    LibraryMissing,

    /// <summary>The library loaded but the device was not found (wrong index or UUID, or not visible to this process).</summary>
    DeviceNotFound,

    /// <summary>The device fell off the bus or needs a reset.</summary>
    DeviceLost,

    /// <summary>The read failed for another reason, or the reader threw.</summary>
    Failed,

    /// <summary>The last good reading is older than <see cref="PolledDeviceMemorySourceOptions.MaxAge"/>: the poller has stopped keeping up.</summary>
    Stale,
}

/// <summary>One read of a device's memory: the memory when <see cref="Status"/> is <see cref="DeviceMemoryReadStatus.Read"/>.</summary>
/// <param name="Status">Whether the read succeeded, and why not.</param>
/// <param name="Memory">The device memory; meaningful only when <paramref name="Status"/> is <see cref="DeviceMemoryReadStatus.Read"/>.</param>
/// <param name="Detail">What the reader saw when the read failed (a library status or exception), for logs; null on success.</param>
public readonly record struct DeviceMemoryReading(DeviceMemoryReadStatus Status, DeviceMemory Memory, string? Detail = null)
{
    /// <summary>A successful reading.</summary>
    /// <param name="totalBytes">Total bytes.</param>
    /// <param name="freeBytes">Free bytes.</param>
    /// <returns>The reading.</returns>
    public static DeviceMemoryReading Of(long totalBytes, long freeBytes) =>
        new(DeviceMemoryReadStatus.Read, new DeviceMemory(totalBytes, freeBytes));

    /// <summary>A reading that carries no memory.</summary>
    /// <param name="status">Why; anything but <see cref="DeviceMemoryReadStatus.Read"/>.</param>
    /// <param name="detail">What the reader saw.</param>
    /// <returns>The reading.</returns>
    public static DeviceMemoryReading Unreadable(DeviceMemoryReadStatus status, string? detail = null)
    {
        if (status == DeviceMemoryReadStatus.Read)
        {
            throw new ArgumentOutOfRangeException(nameof(status), status, "An unreadable reading needs a failure status.");
        }

        return new DeviceMemoryReading(status, default, detail);
    }

    /// <summary>Whether the reading carries memory.</summary>
    public bool IsReadable => Status == DeviceMemoryReadStatus.Read;
}

/// <summary>
/// Reads a device's memory from the driver (R-044). An implementation may block on the driver; it is called from
/// <see cref="PolledDeviceMemorySource"/>'s timer, never from the engine thread. The NVML implementation lives with the
/// CUDA bindings (stream F), so the engine stays free of CUDA and NVML.
/// </summary>
public interface IDeviceMemoryReader
{
    /// <summary>Reads the device's current memory.</summary>
    /// <returns>The reading, with a typed status when the device cannot be read. Should not throw; a throw is recorded as <see cref="DeviceMemoryReadStatus.Failed"/>.</returns>
    DeviceMemoryReading Read();
}

/// <summary>How often a <see cref="PolledDeviceMemorySource"/> re-reads, and how old a reading may get.</summary>
public sealed record PolledDeviceMemorySourceOptions
{
    /// <summary>The period between reads. The default re-reads the ceiling every 2 seconds (build plan R-044: "every few seconds").</summary>
    public TimeSpan Period { get; init; } = TimeSpan.FromSeconds(2);

    /// <summary>
    /// The age past which the last good reading no longer counts and <see cref="IDeviceMemorySource.TryRead"/> returns false
    /// (<see cref="DeviceMemoryReadStatus.Stale"/>), so a stuck reader cannot pin an old ceiling. Null means three periods.
    /// </summary>
    public TimeSpan? MaxAge { get; init; }

    internal TimeSpan EffectiveMaxAge => MaxAge ?? (Period * 3);

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Period, TimeSpan.Zero);
        if (MaxAge is { } maxAge)
        {
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(maxAge, TimeSpan.Zero);
        }
    }
}

/// <summary>
/// An <see cref="IDeviceMemorySource"/> that re-reads an <see cref="IDeviceMemoryReader"/> on a background timer and
/// serves the cached reading (R-044). <see cref="TryRead"/> is a volatile read of the last snapshot: it never blocks,
/// never calls the driver and never throws, so the engine thread can call it on every admission.
/// </summary>
/// <remarks>
/// <para><b>Unreadable is typed.</b> A missing library, a lost device, a reader that throws, no read yet, or a reading
/// older than <see cref="PolledDeviceMemorySourceOptions.MaxAge"/> all make <see cref="TryRead"/> return false, and
/// <see cref="Current"/> says which. <see cref="AdmissionController"/> then decides from the pools alone, as R-031 specifies;
/// nothing is thrown into admission.</para>
/// <para><b>Recovery.</b> Every tick reads again, so a device that comes back or memory a co-tenant frees shows up within
/// one period. Reads do not overlap: a tick that finds the previous read still running is skipped.</para>
/// <para><b>Ownership.</b> <see cref="Dispose"/> stops the timer; it does not dispose the reader.</para>
/// </remarks>
public sealed class PolledDeviceMemorySource : IDeviceMemorySource, IDisposable
{
    private readonly IDeviceMemoryReader _reader;
    private readonly TimeProvider _time;
    private readonly TimeSpan _maxAge;
    private readonly ITimer _timer;
    private Snapshot _snapshot;
    private int _reading;
    private int _disposed;

    /// <summary>Creates the source and starts polling: the first read is queued at once on the timer.</summary>
    /// <param name="reader">Reads the device.</param>
    /// <param name="options">Period and maximum age; defaults when null.</param>
    /// <param name="timeProvider">The clock and timer; <see cref="TimeProvider.System"/> when null.</param>
    public PolledDeviceMemorySource(IDeviceMemoryReader reader, PolledDeviceMemorySourceOptions? options = null, TimeProvider? timeProvider = null)
    {
        ArgumentNullException.ThrowIfNull(reader);
        options ??= new PolledDeviceMemorySourceOptions();
        options.Validate();
        _reader = reader;
        _time = timeProvider ?? TimeProvider.System;
        _maxAge = options.EffectiveMaxAge;
        _snapshot = new Snapshot(DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.NotYetRead), 0, null, 0);
        _timer = _time.CreateTimer(static state => ((PolledDeviceMemorySource)state!).Refresh(), this, TimeSpan.Zero, options.Period);
    }

    /// <summary>The last reading, with <see cref="DeviceMemoryReadStatus.Stale"/> when the last good one is too old.</summary>
    public DeviceMemoryReading Current
    {
        get
        {
            var s = Volatile.Read(ref _snapshot);
            if (s.Reading.IsReadable && IsStale(s.ReadAt))
            {
                return DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.Stale, $"The last reading is older than {_maxAge}.");
            }

            return s.Reading;
        }
    }

    /// <summary>The last successful reading, however old, or null when none has succeeded. For diagnostics, not for admission.</summary>
    public DeviceMemory? LastGood => Volatile.Read(ref _snapshot).LastGood;

    /// <summary>Reads that have finished, successful or not.</summary>
    public long Reads => Volatile.Read(ref _snapshot).Reads;

    /// <inheritdoc />
    public bool TryRead(out DeviceMemory memory)
    {
        var s = Volatile.Read(ref _snapshot);
        if (s.Reading.IsReadable && !IsStale(s.ReadAt))
        {
            memory = s.Reading.Memory;
            return true;
        }

        memory = default;
        return false;
    }

    /// <summary>
    /// Reads the device now, on the calling thread, and publishes the result. The timer calls this; a host may call it to
    /// refresh before sizing pools. Never throws. A call that overlaps a read in progress returns without reading.
    /// </summary>
    /// <returns>False when another read was in progress or the source is disposed.</returns>
    public bool Refresh()
    {
        if (Volatile.Read(ref _disposed) != 0 || Interlocked.CompareExchange(ref _reading, 1, 0) != 0)
        {
            return false;
        }

        try
        {
            DeviceMemoryReading reading;
            try
            {
                reading = _reader.Read();
            }
#pragma warning disable CA1031 // A reader fault is a typed reading, never an exception into admission or the timer thread.
            catch (Exception e)
#pragma warning restore CA1031
            {
                reading = DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.Failed, e.GetType().Name + ": " + e.Message);
            }

            if (reading.IsReadable && (reading.Memory.TotalBytes < 0 || reading.Memory.FreeBytes < 0))
            {
                reading = DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.Failed, "The reader returned a negative byte count.");
            }

            var previous = Volatile.Read(ref _snapshot);
            var now = _time.GetTimestamp();
            var next = reading.IsReadable
                ? new Snapshot(reading, now, reading.Memory, previous.Reads + 1)
                : new Snapshot(reading, previous.ReadAt, previous.LastGood, previous.Reads + 1);
            Volatile.Write(ref _snapshot, next);
            return true;
        }
        finally
        {
            Volatile.Write(ref _reading, 0);
        }
    }

    /// <summary>Stops polling. <see cref="TryRead"/> keeps serving the last reading until it goes stale.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _timer.Dispose();
        }
    }

    private bool IsStale(long readAt) => _time.GetElapsedTime(readAt) > _maxAge;

    private sealed record Snapshot(DeviceMemoryReading Reading, long ReadAt, DeviceMemory? LastGood, long Reads);
}

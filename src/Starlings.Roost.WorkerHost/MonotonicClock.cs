using System.Diagnostics;

namespace Starlings.Roost.WorkerHost;

/// <summary>
/// The clock both sides of the C8 segment stamp with. On Linux <see cref="Stopwatch.GetTimestamp"/> is
/// <c>CLOCK_MONOTONIC</c> in nanoseconds, which is system wide, so a time written by the worker process is comparable
/// with one read by the host (S8 relied on the same clock).
/// </summary>
public static class MonotonicClock
{
    /// <summary>Nanoseconds on the shared monotonic clock.</summary>
    /// <returns>The time.</returns>
    public static ulong NowNs()
    {
        var ticks = Stopwatch.GetTimestamp();
        return Stopwatch.Frequency == 1_000_000_000L
            ? (ulong)ticks
            : (ulong)(ticks * (1_000_000_000.0 / Stopwatch.Frequency));
    }
}

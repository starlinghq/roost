using System.Diagnostics;
using System.Globalization;

namespace Starlings.Roost.Server;

/// <summary>One measured stage of <c>roost serve</c>'s start.</summary>
/// <param name="Name">The stage (<see cref="ColdStart"/> names them).</param>
/// <param name="Duration">How long it took.</param>
internal sealed record ColdStartStage(string Name, TimeSpan Duration);

/// <summary>
/// Cold-start stage timing for <c>roost serve</c> (R-046): each stage's wall-clock duration, written to the log as it ends and
/// exported on <c>GET /metrics</c> as <c>roost_cold_start_stage_seconds{stage}</c> and <c>roost_cold_start_seconds</c>. The
/// stages run one after another, so their sum is the time from process start to listening. Time to first token is not a stage:
/// the server sends no warm-up request (build plan X9 measures it from outside the process).
/// </summary>
internal sealed class ColdStart
{
    /// <summary>From process creation to <c>Main</c>: the runtime's own start (for Native AOT, loading the image).</summary>
    public const string Runtime = "runtime";

    /// <summary>Parsing and checking the command line.</summary>
    public const string Arguments = "arguments";

    /// <summary>Loading the tokenizer and the chat template and hashing the kernel manifest (<c>RoostCli.Options</c>).</summary>
    public const string Profile = "profile";

    /// <summary>
    /// Starting the worker until it is ready: spawning <c>roost-worker</c>, CUDA initialisation, loading the kernel library,
    /// streaming the weights, and the C8 handshake (<c>RoostChatClient.StartAsync</c>). Its sub-stages are inside the worker.
    /// </summary>
    public const string Worker = "worker";

    /// <summary>Building the ASP.NET Core host and binding the listener.</summary>
    public const string Listen = "listen";

    private readonly Lock _gate = new();
    private readonly List<ColdStartStage> _stages = [];
    private readonly TimeProvider _time;
    private readonly TextWriter? _log;

    /// <summary>Creates a timeline.</summary>
    /// <param name="time">The clock stages are measured with.</param>
    /// <param name="log">Where each finished stage is written, or null for nowhere.</param>
    public ColdStart(TimeProvider time, TextWriter? log)
    {
        _time = time;
        _log = log;
    }

    /// <summary>The stages finished so far, in order.</summary>
    public IReadOnlyList<ColdStartStage> Stages
    {
        get
        {
            lock (_gate)
            {
                return [.. _stages];
            }
        }
    }

    /// <summary>The sum of the finished stages.</summary>
    public TimeSpan Total => Stages.Aggregate(TimeSpan.Zero, (sum, s) => sum + s.Duration);

    /// <summary>The time since this process was created, or null where the OS does not report its start time.</summary>
    /// <returns>The elapsed time.</returns>
    public static TimeSpan? SinceProcessStart()
    {
        try
        {
            using var process = Process.GetCurrentProcess();
            var elapsed = DateTime.Now - process.StartTime;
            return elapsed >= TimeSpan.Zero ? elapsed : null;
        }
        catch (Exception e) when (e is InvalidOperationException or NotSupportedException or System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }

    /// <summary>Records a stage measured elsewhere.</summary>
    /// <param name="name">The stage.</param>
    /// <param name="duration">Its duration.</param>
    public void Record(string name, TimeSpan duration)
    {
        lock (_gate)
        {
            _stages.Add(new ColdStartStage(name, duration));
        }

        _log?.WriteLine(string.Create(CultureInfo.InvariantCulture, $"roost: cold start: {name} {duration.TotalSeconds:0.000} s"));
    }

    /// <summary>Runs a stage and records its duration, also when it throws.</summary>
    /// <typeparam name="T">The stage's result.</typeparam>
    /// <param name="name">The stage.</param>
    /// <param name="stage">The work.</param>
    /// <returns>The work's result.</returns>
    public T Measure<T>(string name, Func<T> stage)
    {
        ArgumentNullException.ThrowIfNull(stage);
        long start = _time.GetTimestamp();
        try
        {
            return stage();
        }
        finally
        {
            Record(name, _time.GetElapsedTime(start));
        }
    }

    /// <summary>Runs an asynchronous stage and records its duration, also when it throws.</summary>
    /// <typeparam name="T">The stage's result.</typeparam>
    /// <param name="name">The stage.</param>
    /// <param name="stage">The work.</param>
    /// <returns>The work's result.</returns>
    public async Task<T> MeasureAsync<T>(string name, Func<Task<T>> stage)
    {
        ArgumentNullException.ThrowIfNull(stage);
        long start = _time.GetTimestamp();
        try
        {
            return await stage().ConfigureAwait(false);
        }
        finally
        {
            Record(name, _time.GetElapsedTime(start));
        }
    }

    /// <summary>Writes the total, the line an operator compares with the cold-start budget.</summary>
    public void WriteTotal() =>
        _log?.WriteLine(string.Create(CultureInfo.InvariantCulture, $"roost: cold start: total {Total.TotalSeconds:0.000} s ({string.Join(", ", Stages.Select(s => s.Name))})"));
}

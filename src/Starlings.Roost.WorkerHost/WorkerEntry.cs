using Starlings.Roost.Engine.Rings;

namespace Starlings.Roost.WorkerHost;

/// <summary>
/// The entry point of a worker process. The host starts the worker command with <see cref="SegmentFlag"/> and the
/// segment path appended; <see cref="Run"/> opens that segment, runs a <see cref="WorkerRuntime"/> over an executor made
/// by the caller, and returns the process exit code. Packaging the worker executable (which executor, which native
/// libraries) is the Worker package's and decision 12's; this class is what that executable's <c>Main</c> calls.
/// </summary>
public static class WorkerEntry
{
    /// <summary>The argument that carries the segment path.</summary>
    public const string SegmentFlag = "--roost-worker-segment";

    /// <summary>Whether the command line asks for a worker process.</summary>
    /// <param name="args">The arguments.</param>
    /// <returns>True when <see cref="SegmentFlag"/> is present.</returns>
    public static bool IsWorkerInvocation(string[] args) => Array.IndexOf(args, SegmentFlag) >= 0;

    /// <summary>Runs the worker to the end.</summary>
    /// <param name="args">The command line, including <see cref="SegmentFlag"/> and a path.</param>
    /// <param name="executorFactory">Makes the executor; it receives the whole command line, so it can read its own options.</param>
    /// <returns>The exit code: see <see cref="WorkerRuntime.ExitCodeFor"/>; 10 when the segment cannot be opened, 1 on any other failure.</returns>
    public static int Run(string[] args, Func<string[], IStepExecutor> executorFactory)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(executorFactory);
        var at = Array.IndexOf(args, SegmentFlag);
        if (at < 0 || at + 1 >= args.Length)
        {
            Console.Error.WriteLine("roost worker: missing " + SegmentFlag + " <path>");
            return 2;
        }

        Libc.DisableCoreDumps(); // before anything can fail: a crash must not write a multi-GB core
        RingSegment segment;
        try
        {
            segment = RingSegment.Open(args[at + 1]);
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine("roost worker: cannot open the segment: " + ex.Message);
            return 10;
        }

        try
        {
            using (segment)
            {
                var executor = executorFactory(args);
                using var runtime = new WorkerRuntime(segment, executor, new WorkerRuntimeOptions { IsolatedProcess = true });
                var exit = runtime.Run();
                if (exit == WorkerExit.Shutdown)
                {
                    executor.Dispose(); // a faulted context is not torn down politely; the process ends instead
                }

                return WorkerRuntime.ExitCodeFor(exit);
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("roost worker: " + ex);
            return 1;
        }
    }
}

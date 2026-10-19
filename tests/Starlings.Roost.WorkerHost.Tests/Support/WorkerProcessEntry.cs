using System.Runtime.CompilerServices;
using Starlings.Roost.WorkerHost;

namespace Starlings.Roost.WorkerHost.Tests.Support;

/// <summary>
/// Turns the test executable into a worker process (<c>--roost-fake-worker</c>) or into a host harness that starts
/// a worker and then idles (<c>--roost-fake-host</c>, used to kill a host and watch its worker). The module
/// initializer runs before the xunit runner starts.
/// </summary>
internal static class WorkerProcessEntry
{
    public const string WorkerFlag = "--roost-fake-worker";
    public const string HostFlag = "--roost-fake-host";
    public const string OpenDelayPrefix = "--open-delay-ms=";
    public const string FailOpenFlag = "--fail-open";
    public const string SegmentDirPrefix = "--segment-dir=";

    [ModuleInitializer]
    internal static void Enter()
    {
        var args = Environment.GetCommandLineArgs();
        if (Array.IndexOf(args, WorkerFlag) >= 0)
        {
            Environment.Exit(WorkerEntry.Run(args, MakeExecutor));
        }

        if (Array.IndexOf(args, HostFlag) >= 0)
        {
            Environment.Exit(RunHostHarness());
        }
    }

    public static WorkerCommand Command(int openDelayMs = 0, bool failOpen = false)
    {
        var extra = new List<string> { WorkerFlag };
        if (openDelayMs > 0)
        {
            extra.Add(OpenDelayPrefix + openDelayMs);
        }

        if (failOpen)
        {
            extra.Add(FailOpenFlag);
        }

        return WorkerCommand.ForCurrentProcess([.. extra]);
    }

    private static FakeExecutor MakeExecutor(string[] args)
    {
        var delay = 0;
        foreach (var arg in args)
        {
            if (arg.StartsWith(OpenDelayPrefix, StringComparison.Ordinal))
            {
                delay = int.Parse(arg.AsSpan(OpenDelayPrefix.Length), System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        return new FakeExecutor(openDelayMs: delay, failOpen: Array.IndexOf(args, FailOpenFlag) >= 0);
    }

    /// <summary>
    /// Starts a worker, prints "READY pid path" and idles until killed. Its worker must notice the host is gone. Written
    /// without <c>await</c> of its own: it runs inside the module initializer, and a state machine of this module that
    /// resumes on a pool thread waits for the initializer to finish.
    /// </summary>
    private static int RunHostHarness()
    {
        var directory = Environment.GetCommandLineArgs().FirstOrDefault(a => a.StartsWith(SegmentDirPrefix, StringComparison.Ordinal))?[SegmentDirPrefix.Length..];
        var transport = new ProcessWorkerTransport(Command(), new WorkerTransportOptions { SegmentDirectory = directory });
        var result = transport.StartAsync(Rig.DefaultLaunch(), CancellationToken.None).AsTask().GetAwaiter().GetResult();
        Console.Out.WriteLine($"READY {result.WorkerProcessId} {transport.SegmentPath}");
        Console.Out.Flush();
        Thread.Sleep(Timeout.Infinite);
        return 0;
    }
}

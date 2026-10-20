using System.Diagnostics;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;
using Starlings.Roost.WorkerHost.Tests.Support;

namespace Starlings.Roost.WorkerHost.Tests;

/// <summary>The shared suite over a real worker process.</summary>
public sealed class ProcessTransportTests : TransportSuite
{
    protected override TransportKind Kind => TransportKind.Process;

    protected override bool CanCrash => true;

    internal override Rig NewRig(WorkerTransportOptions? options = null) =>
        new(new ProcessWorkerTransport(WorkerProcessEntry.Command(), TestSegments.Options(options)));

    internal override FakeOp Freeze(Rig rig)
    {
        var process = (ProcessWorkerTransport)rig.Transport;
        Assert.True(OsProcess.Freeze(process.ProcessId));
        return FakeOp.Normal; // a stopped process cannot finish any step
    }

    internal override void Thaw(Rig rig)
    {
        var process = (ProcessWorkerTransport)rig.Transport;
        _ = OsProcess.Thaw(process.ProcessId);
    }

    [Fact]
    public async Task WorkerThatExitsBeforeAnsweringFailsTheStartWithItsOutput()
    {
        await using var transport = new ProcessWorkerTransport(WorkerProcessEntry.Command(failOpen: true), TestSegments.Options());
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

        var ex = await Assert.ThrowsAsync<WorkerStartException>(async () =>
            await transport.StartAsync(Rig.DefaultLaunch(), cts.Token));

        Assert.Contains("OpenDevice threw", ex.Message, StringComparison.Ordinal);
        Assert.Equal(TransportState.Idle, transport.State);
    }

    [Fact]
    public async Task WorkerThatAnswersTooSlowlyIsKilledAtTheStartupTimeout()
    {
        var directory = TestSegments.NewDirectory();
        try
        {
            await using var transport = new ProcessWorkerTransport(
                WorkerProcessEntry.Command(openDelayMs: 30_000),
                new WorkerTransportOptions { StartupTimeout = TimeSpan.FromMilliseconds(500), SegmentDirectory = directory });
            var timer = Stopwatch.StartNew();

            var ex = await Assert.ThrowsAsync<WorkerStartException>(async () =>
                await transport.StartAsync(Rig.DefaultLaunch(), CancellationToken.None));

            Assert.Contains("did not answer", ex.Message, StringComparison.Ordinal);
            Assert.True(timer.Elapsed < TimeSpan.FromSeconds(10));
            Assert.Equal(TransportState.Idle, transport.State);
            Assert.Empty(Directory.EnumerateFiles(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task CancellingTheStartKillsTheWorker()
    {
        await using var transport = new ProcessWorkerTransport(WorkerProcessEntry.Command(openDelayMs: 30_000), TestSegments.Options());
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));

        _ = await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
            await transport.StartAsync(Rig.DefaultLaunch(), cts.Token));

        Assert.Equal(TransportState.Idle, transport.State);
        Assert.True(transport.ProcessId == 0 || !ProcessExists(transport.ProcessId));
    }

    [Fact]
    public async Task KilledHostLeavesNoWorkerAndNoSegment()
    {
        var command = WorkerCommand.ForCurrentProcess(WorkerProcessEntry.HostFlag, WorkerProcessEntry.SegmentDirPrefix + TestSegments.Root);
        var info = new ProcessStartInfo(command.FileName) { RedirectStandardOutput = true, UseShellExecute = false };
        foreach (var argument in command.Arguments)
        {
            info.ArgumentList.Add(argument);
        }

        var harness = Process.Start(info)!;
        try
        {
            var line = await harness.StandardOutput.ReadLineAsync(TestContext.Current.CancellationToken).AsTask().WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);
            Assert.NotNull(line);
            var parts = line.Split(' ');
            Assert.Equal("READY", parts[0]);
            var workerPid = int.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
            var path = parts[2];
            Assert.True(ProcessExists(workerPid));
            Assert.True(File.Exists(path));

            harness.Kill(); // SIGKILL: the host gets no chance to clean up
            await harness.WaitForExitAsync(TestContext.Current.CancellationToken);

            var until = Stopwatch.StartNew();
            while (until.Elapsed < TimeSpan.FromSeconds(10) && (ProcessExists(workerPid) || File.Exists(path)))
            {
                await Task.Delay(20, TestContext.Current.CancellationToken);
            }

            Assert.False(ProcessExists(workerPid), "the worker outlived its host");
            Assert.False(File.Exists(path), "the segment file outlived its host");
        }
        finally
        {
            if (!harness.HasExited)
            {
                harness.Kill();
            }

            harness.Dispose();
        }
    }

    /// <summary>
    /// The one test that kills the worker with a real fatal signal (<c>Environment.FailFast</c>: SIGABRT on Linux, a
    /// fail-fast exit on Windows). The kernel and systemd-coredump record such a death, and a desktop shows it as a crashed
    /// process, so it runs only on GitHub-hosted runners (<c>GITHUB_ACTIONS</c> is <c>true</c> and <c>RUNNER_ENVIRONMENT</c> is <c>github-hosted</c>; a self-hosted runner may be a developer's desktop, never by agents or
    /// developers locally). Every other crash test uses a clean exit with a nonzero code (<see cref="FakeOp.Crash"/>).
    /// </summary>
    [Fact]
    public async Task WorkerKilledByARealFatalSignalIsReportedExited()
    {
        Assert.SkipUnless(
            (Environment.GetEnvironmentVariable("GITHUB_ACTIONS") == "true" && Environment.GetEnvironmentVariable("RUNNER_ENVIRONMENT") == "github-hosted"),
            "raises a real fatal signal (SIGABRT), which is recorded as a crash by systemd-coredump; runs only on GitHub-hosted runners (GITHUB_ACTIONS=true and RUNNER_ENVIRONMENT=github-hosted)");
        await using var rig = NewRig();
        _ = await rig.StartAsync();

        _ = rig.Publish(FakeOp.CrashSignal);
        var seen = rig.Wait(TimeSpan.FromSeconds(30));

        Assert.False(seen.HasResult);
        Assert.Equal(FaultClass.Exited, seen.Polled);
        Assert.Equal(RecoveryAction.Respawn, FaultRules.ActionFor(seen.Polled));
    }

    [Fact]
    public async Task ExternalKillOfTheWorkerIsReportedExited()
    {
        await using var rig = NewRig();
        _ = await rig.StartAsync();
        var pid = ((ProcessWorkerTransport)rig.Transport).ProcessId;

        Assert.True(OsProcess.Kill(pid));
        var (fault, _) = rig.WaitForPollFault(TimeSpan.FromSeconds(10));

        Assert.Equal(FaultClass.Exited, fault);
    }

    internal static bool ProcessExists(int pid) => OsProcess.Exists(pid);
}

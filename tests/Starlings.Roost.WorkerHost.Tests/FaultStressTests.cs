using System.Diagnostics;
using System.Globalization;
using System.Text;
using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost.Tests.Support;

namespace Starlings.Roost.WorkerHost.Tests;

/// <summary>
/// A seeded, randomised sequence of episodes over a real worker process (two processes, one shared segment): each
/// episode starts a worker, runs a random burst of good steps (one or two in flight, slow steps, recoverable faults, a
/// brief SIGSTOP inside the deadline), and then ends in one of the terminal events a worker can meet. After each
/// terminal event the test checks the exact fault class the host concludes, kills the worker, and starts again on the
/// same transport object, checking that the step ids restart at 1, that the old worker is gone and that the segment file
/// is gone. The first episodes cycle through every terminal event once in a seeded order, so a short run still covers all
/// of them.
/// <para>Length: <c>ROOST_WORKER_STRESS_SECONDS</c> (default 6; the long evidence run uses several minutes).
/// <c>ROOST_WORKER_STRESS_SEED</c> fixes the seed, which is printed with the result and fully determines the sequence of
/// operations and terminal events (timing on the machine is not deterministic, so a replay with the same seed repeats the
/// sequence, not the interleaving). <c>ROOST_WORKER_STRESS_LOG</c> names a file that gets the result line and the trace.</para>
/// <para>This is not <c>compute-sanitizer racecheck</c>, which observes device code only; this item has no kernel code
/// except a spin kernel with no shared memory.</para>
/// </summary>
public sealed class FaultStressTests
{
    private enum Terminal
    {
        Sticky,
        Throw,
        Crash,
        Hang,
        ExternalKill,
        FrozenWorker,
        KillWhileHealthy,
        CleanShutdown,
        RejectedHandshake,
        CancelledStart,
    }

    [Fact]
    public async Task EveryFaultPathIsClassifiedExactlyAndTheTransportRecoversEveryTime()
    {
        var seconds = Env("ROOST_WORKER_STRESS_SECONDS", 6UL);
        var seed = Env("ROOST_WORKER_STRESS_SEED", (ulong)Environment.TickCount64);
        var directory = Path.Combine(RingSegment.DefaultDirectory(), $"roost-stress-{Guid.NewGuid():N}");
        _ = Directory.CreateDirectory(directory);
        var run = new Run(seed, directory);
        try
        {
            try
            {
                await run.ExecuteAsync(TimeSpan.FromSeconds(seconds));
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"seed={seed} {run.Summary()}\nlast trace lines:\n{run.TraceTail()}", ex);
            }
            finally
            {
                await run.DisposeAsync();
            }

            var leaked = Directory.GetFileSystemEntries(directory);
            var alive = run.Pids.Where(ProcessTransportTests.ProcessExists).ToArray();
            var line = $"seed={seed} seconds={seconds} {run.Summary()}";
            TestContext.Current.SendDiagnosticMessage(line);
            var log = Environment.GetEnvironmentVariable("ROOST_WORKER_STRESS_LOG");
            if (!string.IsNullOrEmpty(log))
            {
                File.AppendAllText(log, line + Environment.NewLine + run.TraceAll() + Environment.NewLine);
            }

            Assert.True(leaked.Length == 0, $"segment files left behind: {string.Join(", ", leaked)} ({line})");
            Assert.True(alive.Length == 0, $"worker processes left behind: {string.Join(", ", alive)} ({line})");
            Assert.True(run.Episodes >= (ulong)Enum.GetValues<Terminal>().Length, $"too few episodes to cover every terminal event: {line}");
            foreach (var terminal in Enum.GetValues<Terminal>())
            {
                Assert.True(run.Count(terminal.ToString()) > 0, $"{terminal} never ran: {line}");
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static ulong Env(string name, ulong fallback) =>
        ulong.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.None, CultureInfo.InvariantCulture, out var v) ? v : fallback;

    private sealed class Run : IAsyncDisposable
    {
        private static readonly TimeSpan Deadline = TimeSpan.FromMilliseconds(400);
        private static readonly TimeSpan Wait = TimeSpan.FromSeconds(20);

        private readonly Random _random;
        private readonly string _directory;
        private readonly Dictionary<string, ulong> _counts = [];
        private readonly List<string> _trace = [];
        private readonly HashSet<int> _pids = [];
        private readonly WorkerTransportOptions _options;
        private Rig _rig;
        private ulong _goodSteps;
        private ulong _recoverable;
        private ulong _twoInFlight;

        public Run(ulong seed, string directory)
        {
            _random = new Random(unchecked((int)(seed ^ (seed >> 32))));
            _directory = directory;
            _options = new WorkerTransportOptions
            {
                SegmentDirectory = directory,
                // Wide, so a loaded machine does not turn a hang into a stall. Stalls are produced with SIGSTOP and are
                // told apart by the counter not advancing since the publish, which no window can hide.
                HeartbeatLiveness = TimeSpan.FromSeconds(5),
                StartupTimeout = TimeSpan.FromSeconds(30),
                KillTimeout = TimeSpan.FromSeconds(10),
                ShutdownTimeout = TimeSpan.FromSeconds(10),
            };
            _rig = NewRig();
        }

        public ulong Episodes { get; private set; }

        public IEnumerable<int> Pids => _pids;

        private ProcessWorkerTransport Process => (ProcessWorkerTransport)_rig.Transport;

        public ulong Count(string name) => _counts.TryGetValue(name, out var n) ? n : 0;

        public string Summary() =>
            $"episodes={Episodes} goodSteps={_goodSteps} recoverable={_recoverable} twoInFlight={_twoInFlight} " +
            string.Join(' ', _counts.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}={p.Value}"));

        public string TraceTail() => string.Join('\n', _trace.TakeLast(40));

        public string TraceAll() => string.Join('\n', _trace);

        public async ValueTask DisposeAsync() => await _rig.DisposeAsync();

        public async Task ExecuteAsync(TimeSpan duration)
        {
            var terminals = Enum.GetValues<Terminal>().OrderBy(_ => _random.Next()).ToList();
            var clock = Stopwatch.StartNew();
            while (Episodes < (ulong)terminals.Count || clock.Elapsed < duration)
            {
                var terminal = Episodes < (ulong)terminals.Count ? terminals[(int)Episodes] : (Terminal)_random.Next(terminals.Count);
                Episodes++;
                Log($"episode {Episodes} -> {terminal}");
                Bump(terminal.ToString());
                await EpisodeAsync(terminal);
            }
        }

        private static async Task<bool> WaitGoneAsync(int pid)
        {
            var until = Stopwatch.StartNew();
            while (until.Elapsed < TimeSpan.FromSeconds(10))
            {
                if (!ProcessTransportTests.ProcessExists(pid))
                {
                    return true;
                }

                await Task.Delay(5);
            }

            return false;
        }

        private string HeartbeatNote()
        {
            var beat = _rig.Transport.ReadHeartbeat();
            return $"heartbeat counter={beat.Counter} state={beat.State} ageMs={(MonotonicClock.NowNs() - beat.WorkerNs) / 1e6:0.0} loadavg={File.ReadAllText("/proc/loadavg").Split(' ')[0]}";
        }

        private Rig NewRig() => new(new ProcessWorkerTransport(WorkerProcessEntry.Command(), _options));

        private static WorkerLaunch Launch() => Rig.DefaultLaunch(deadline: Deadline);

        private void Log(string text) => _trace.Add(text);

        private void Bump(string name) => _counts[name] = Count(name) + 1;

        private static void Check(bool condition, string what)
        {
            if (!condition)
            {
                throw new InvalidOperationException(what);
            }
        }

        private async Task<int> StartAsync(WorkerLaunch? launch = null)
        {
            _rig.NextStepId = 1;
            var result = await _rig.StartAsync(launch ?? Launch());
            Check(result.State == HandshakeState.Ready, $"handshake {result.State}/{result.Reject}");
            Check(Process.ProcessId == result.WorkerProcessId && result.WorkerProcessId > 0, "worker pid");
            _ = _pids.Add(Process.ProcessId);
            return Process.ProcessId;
        }

        private async Task EpisodeAsync(Terminal terminal)
        {
            if (terminal == Terminal.RejectedHandshake)
            {
                await RejectedStartAsync();
                return;
            }

            if (terminal == Terminal.CancelledStart)
            {
                await CancelledStartAsync();
                return;
            }

            var pid = await StartAsync();
            var segment = Process.SegmentPath!;
            Check(File.Exists(segment), "segment file missing while running");
            Burst();
            await TerminalAsync(terminal, pid);

            // Whatever happened, a kill must leave an idle transport, no worker and no file, and a restart must work.
            if (Process.State != TransportState.Disposed)
            {
                await _rig.Transport.KillAsync(CancellationToken.None);
                Check(_rig.Transport.State == TransportState.Idle, $"state after kill {_rig.Transport.State}");
                Check(await WaitGoneAsync(pid), $"worker {pid} still alive after kill");
                Check(!File.Exists(segment), "segment file left after kill");
                Check(_rig.Transport.Poll() == FaultClass.None, "poll after kill");
            }

            await RestartAndProveItWorksAsync();
        }

        private async Task RestartAndProveItWorksAsync()
        {
            var pid = await StartAsync();
            _ = _rig.Publish();
            var seen = _rig.Wait(Wait);
            Check(seen.HasResult && seen.StepId == 1 && seen.EchoOk && seen.ChecksumOk, $"first step after restart: {seen}");
            await _rig.Transport.KillAsync(CancellationToken.None);
            Check(await WaitGoneAsync(pid), "worker alive after proof kill");
        }

        /// <summary>A random run of good steps with the faults that leave the worker running.</summary>
        private void Burst()
        {
            var steps = _random.Next(0, 40);
            for (var i = 0; i < steps; i++)
            {
                var roll = _random.Next(100);
                if (roll < 10)
                {
                    Check(RunOne(FakeOp.Recoverable, 0).Fault == FaultClass.Recoverable, "recoverable not reported");
                    _recoverable++;
                }
                else if (roll < 20)
                {
                    Check(RunOne(FakeOp.Slow, (uint)_random.Next(1, 15)).Fault == FaultClass.None, "slow step failed");
                }
                else if (roll < 30)
                {
                    RunTwoInFlight();
                }
                else if (roll < 33)
                {
                    BriefFreeze();
                }
                else
                {
                    Check(RunOne(FakeOp.Normal, 0).Fault == FaultClass.None, "normal step failed");
                }
            }
        }

        private Seen RunOne(FakeOp op, uint argument)
        {
            var id = _rig.Publish(op, argument);
            var seen = _rig.Wait(Wait);
            Check(seen.HasResult && seen.StepId == id, $"result for step {id}: {seen}");
            Check(seen.Header.StepId == id, "header step id");
            if (seen.Fault == FaultClass.None)
            {
                Check(seen.EchoOk && seen.ChecksumOk, $"payload of step {id} did not round-trip");
                _goodSteps++;
            }

            return seen;
        }

        private void RunTwoInFlight()
        {
            var first = _rig.TryPublish(FakeOp.Normal);
            var second = _rig.TryPublish(_random.Next(4) == 0 ? FakeOp.Recoverable : FakeOp.Normal);
            Check(first != 0, "first plan buffer not free in a quiescent ring");
            if (second == 0)
            {
                // Both plan buffers were not free yet; the one published is still in order.
                Check(_rig.Wait(Wait).StepId == first, "single result order");
                return;
            }

            _twoInFlight++;
            var a = _rig.Wait(Wait);
            var b = _rig.Wait(Wait);
            Check(a.StepId == first && b.StepId == second, $"results out of order {a.StepId},{b.StepId} for {first},{second}");
            Check(a.EchoOk && a.ChecksumOk, "payload a");
            Check(b.Fault != FaultClass.None || (b.EchoOk && b.ChecksumOk), "payload b");
            _goodSteps += b.Fault == FaultClass.None ? 2UL : 1UL;
            _recoverable += b.Fault == FaultClass.Recoverable ? 1UL : 0UL;
        }

        /// <summary>A stop shorter than the deadline is not a fault: the step finishes after the continue.</summary>
        private void BriefFreeze()
        {
            Check(Libc.Signal(Process.ProcessId, Libc.SigStop), "SIGSTOP");
            var id = _rig.Publish();
            Thread.Sleep(_random.Next(1, 25));
            _ = Libc.Signal(Process.ProcessId, Libc.SigCont);
            var seen = _rig.Wait(Wait);
            Check(seen.HasResult && seen.StepId == id && seen.Fault == FaultClass.None && seen.EchoOk, $"step after a brief freeze: {seen}");
            _goodSteps++;
        }

        private async Task TerminalAsync(Terminal terminal, int pid)
        {
            switch (terminal)
            {
                case Terminal.Sticky:
                case Terminal.Throw:
                    {
                        // Sometimes with a second plan queued behind it, which must never run.
                        var op = terminal == Terminal.Sticky ? FakeOp.Sticky : FakeOp.Throw;
                        var id = _rig.Publish(op);
                        var queued = _random.Next(2) == 0 ? _rig.TryPublish(FakeOp.Normal) : 0UL;
                        var seen = _rig.Wait(Wait);
                        Check(seen.HasResult && seen.StepId == id && seen.Fault == FaultClass.Sticky, $"sticky result: {seen}");
                        var polled = _rig.WaitForPollFault(Wait);
                        Check(polled.Fault == FaultClass.Exited, $"after a sticky result the poll said {polled.Fault}");
                        Check(!_rig.Transport.TryTakeResult(out _), queued != 0 ? "the plan queued behind a sticky fault produced a result" : "extra result");
                        break;
                    }

                case Terminal.Crash:
                    {
                        _ = _rig.Publish(FakeOp.Crash);
                        var seen = _rig.Wait(Wait);
                        Check(!seen.HasResult && seen.Polled == FaultClass.Exited, $"crash: {seen}");
                        break;
                    }

                case Terminal.Hang:
                    {
                        _ = _rig.Publish(FakeOp.Hang);
                        var polled = _rig.WaitForPollFault(Wait);
                        Check(polled.Fault == FaultClass.Hang, $"hang was classified {polled.Fault}; {HeartbeatNote()}");
                        Check(polled.After >= Deadline - TimeSpan.FromMilliseconds(50), $"hang concluded after {polled.After}, before the deadline");
                        Check(FaultRules.ActionFor(polled.Fault) == RecoveryAction.Respawn, "hang action");
                        break;
                    }

                case Terminal.ExternalKill:
                    {
                        if (_random.Next(2) == 0)
                        {
                            _ = _rig.Publish(FakeOp.Slow, 50); // killed mid-step
                        }

                        Check(Libc.Signal(pid, Libc.SigKill), "SIGKILL");
                        var seen = DrainUntilFault();
                        Check(seen == FaultClass.Exited, $"external kill classified {seen}");
                        break;
                    }

                case Terminal.FrozenWorker:
                    {
                        Check(Libc.Signal(pid, Libc.SigStop), "SIGSTOP");
                        Thread.Sleep(_random.Next(5, 30));
                        _ = _rig.Publish();
                        var polled = _rig.WaitForPollFault(Wait);
                        Check(polled.Fault == FaultClass.Stalled, $"frozen worker classified {polled.Fault}; {HeartbeatNote()}");
                        Check(FaultRules.ActionFor(polled.Fault) == RecoveryAction.Respawn, "stall action");
                        break;
                    }

                case Terminal.KillWhileHealthy:
                    Check(_rig.Transport.Poll() == FaultClass.None, "a healthy idle worker reported a fault");
                    break;

                case Terminal.CleanShutdown:
                    {
                        var segment = Process.SegmentPath!;
                        var timer = Stopwatch.StartNew();
                        await _rig.DisposeAsync();
                        Check(timer.Elapsed < TimeSpan.FromSeconds(5), $"shutdown took {timer.Elapsed}");
                        Check(((ProcessWorkerTransport)_rig.Transport).ExitCode == 0, "exit code of a clean shutdown");
                        Check(await WaitGoneAsync(pid), "worker alive after a clean shutdown");
                        Check(!File.Exists(segment), "segment left after a clean shutdown");
                        _rig = NewRig();
                        break;
                    }

                default:
                    throw new InvalidOperationException(terminal.ToString());
            }

            Log($"  {terminal} ok");
        }

        /// <summary>Takes results until the host concludes a fault; a result that was already published may come first.</summary>
        private FaultClass DrainUntilFault()
        {
            var until = Stopwatch.StartNew();
            while (until.Elapsed < Wait)
            {
                if (_rig.Transport.TryTakeResult(out var slot))
                {
                    _rig.Transport.ReleaseResult((int)slot.BufferIndex);
                    continue;
                }

                var polled = _rig.Transport.Poll();
                if (polled != FaultClass.None)
                {
                    return polled;
                }

                Thread.Sleep(1);
            }

            return FaultClass.None;
        }

        private async Task RejectedStartAsync()
        {
            var wrongManifest = _random.Next(2) == 0;
            var launch = Launch() with
            {
                ManifestDigest = wrongManifest ? FakeExecutor.Digest("not the manifest") : FakeExecutor.ManifestDigest,
                WeightsDigest = wrongManifest ? FakeExecutor.WeightsDigest : FakeExecutor.Digest("not the weights"),
            };
            _rig.NextStepId = 1;
            var result = await _rig.StartAsync(launch);
            Check(result.State == HandshakeState.Rejected, $"expected a rejection, got {result.State}");
            Check(result.Reject == (wrongManifest ? HandshakeReject.ManifestDigest : HandshakeReject.WeightsDigest), $"reject reason {result.Reject}");
            Check(_rig.Transport.State == TransportState.Idle, "state after a rejection");
            Check(Process.ProcessId == 0 || await WaitGoneAsync(Process.ProcessId), "rejecting worker still alive");
            Check(Directory.GetFileSystemEntries(_directory).Length == 0, "segment left after a rejection");
            Log("  RejectedHandshake ok");
            await RestartAndProveItWorksAsync();
        }

        private async Task CancelledStartAsync()
        {
            var slow = new ProcessWorkerTransport(WorkerProcessEntry.Command(openDelayMs: 20_000), _options);
            await using (slow)
            {
                using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(_random.Next(30, 400)));
                try
                {
                    _ = await slow.StartAsync(Launch(), cts.Token);
                    Check(false, "a start that cannot finish completed");
                }
                catch (OperationCanceledException)
                {
                }

                var pid = slow.ProcessId;
                Check(slow.State == TransportState.Idle, $"state after a cancelled start {slow.State}");
                if (pid != 0)
                {
                    _ = _pids.Add(pid);
                    Check(await WaitGoneAsync(pid), "worker alive after a cancelled start");
                }
            }

            Check(Directory.GetFileSystemEntries(_directory).Length == 0, "segment left after a cancelled start");
            Log("  CancelledStart ok");
            await RestartAndProveItWorksAsync();
        }
    }
}

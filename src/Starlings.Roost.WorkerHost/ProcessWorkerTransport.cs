using System.Diagnostics;
using System.Text;
using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost;

/// <summary>
/// One worker process per GPU over the C8 shared-memory segment (the default transport, decision 21). The worker is
/// started from a <see cref="WorkerCommand"/> as a direct child of the host: it watches its parent and ends itself when
/// the host dies, so a host crash does not leave a worker holding a GPU. Exit is detected from the process event, not by
/// polling the process table.
/// </summary>
public sealed class ProcessWorkerTransport : WorkerTransportBase
{
    private const int TailBytes = 8192;

    private readonly WorkerCommand _command;
    private readonly StringBuilder _tail = new();
    private Process? _process;
    private int _exited;

    /// <summary>Creates the transport.</summary>
    /// <param name="command">How to start a worker.</param>
    /// <param name="options">Host settings; defaults when null.</param>
    public ProcessWorkerTransport(WorkerCommand command, WorkerTransportOptions? options = null)
        : base(options ?? new WorkerTransportOptions())
    {
        ArgumentNullException.ThrowIfNull(command);
        _command = command;
    }

    /// <inheritdoc />
    public override TransportKind Kind => TransportKind.Process;

    /// <summary>The worker process id, or 0.</summary>
    public int ProcessId { get; private set; }

    /// <inheritdoc />
    protected override bool WorkerHasExited => Volatile.Read(ref _exited) != 0;

    /// <inheritdoc />
    protected override long WorkerId => ProcessId;

    /// <summary>The exit code of the last worker, once it has exited.</summary>
    public int? ExitCode { get; private set; }

    /// <inheritdoc />
    protected override void OnReset()
    {
        Volatile.Write(ref _exited, 0);
        ExitCode = null;
        ProcessId = 0;
        lock (_tail)
        {
            _ = _tail.Clear();
        }
    }

    /// <inheritdoc />
    protected override void LaunchWorker(RingSegment segment, string path)
    {
        var process = new Process { StartInfo = _command.ToStartInfo(path), EnableRaisingEvents = true };
        process.Exited += (_, _) => Volatile.Write(ref _exited, 1);
        process.OutputDataReceived += (_, e) => Append(e.Data);
        process.ErrorDataReceived += (_, e) => Append(e.Data);
        try
        {
            _ = process.Start();
        }
        catch
        {
            process.Dispose();
            throw;
        }

        _process = process;
        ProcessId = process.Id;
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
    }

    /// <inheritdoc />
    protected override async ValueTask TerminateWorkerAsync()
    {
        var process = _process;
        if (process is null)
        {
            return;
        }

        try
        {
            if (!process.HasExited)
            {
                try
                {
                    process.Kill(entireProcessTree: false);
                }
                catch (InvalidOperationException)
                {
                    // Exited between the check and the kill.
                }
            }

            using var timeout = new CancellationTokenSource(Options.KillTimeout);
            try
            {
                await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw new InvalidOperationException($"Worker process {ProcessId} did not exit {Options.KillTimeout} after SIGKILL (uninterruptible sleep in a driver call?).");
            }

            ExitCode = process.ExitCode;
        }
        finally
        {
            _process = null;
            process.Dispose();
            Volatile.Write(ref _exited, 1);
        }
    }

    /// <inheritdoc />
    protected override string Diagnostics()
    {
        lock (_tail)
        {
            var exit = _process is { HasExited: true } p ? $" (exit code {p.ExitCode})" : string.Empty;
            return $"Worker output{exit}: {_tail}";
        }
    }

    private void Append(string? line)
    {
        if (line is null)
        {
            return;
        }

        lock (_tail)
        {
            _ = _tail.AppendLine(line);
            if (_tail.Length > TailBytes)
            {
                _ = _tail.Remove(0, _tail.Length - TailBytes);
            }
        }
    }
}

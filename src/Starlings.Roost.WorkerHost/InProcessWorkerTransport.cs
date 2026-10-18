using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost;

/// <summary>
/// The worker on a thread of the host process, over the same segment, rings, handshake and <see cref="WorkerRuntime"/>
/// as the process transport. For tests and development only: a sticky fault leaves the context failing for the life of
/// the process, a hung step cannot be cancelled (S8), and the host cannot be protected from either. It never loads ONNX
/// Runtime. <see cref="IWorkerTransport.KillAsync"/> stops a worker that is between steps; on a worker stuck inside a step it gives up
/// after <see cref="WorkerTransportOptions.KillTimeout"/>, leaves the thread running and the transport
/// <see cref="TransportState.Abandoned"/>, because the host application has to exit in that case.
/// </summary>
public sealed class InProcessWorkerTransport : WorkerTransportBase
{
    private readonly Func<IStepExecutor> _executorFactory;
    private readonly Func<bool>? _pauseHeartbeat;
    private Thread? _thread;
    private CancellationTokenSource? _stop;
    private WorkerRuntime? _runtime;
    private IStepExecutor? _executor;
    private int _ended;
    private Exception? _failure;

    /// <summary>Creates the transport.</summary>
    /// <param name="executorFactory">Makes a fresh executor for each start.</param>
    /// <param name="options">Host settings; defaults when null.</param>
    /// <param name="pauseHeartbeat">Test hook: while it returns true the worker's heartbeat does not beat.</param>
    public InProcessWorkerTransport(Func<IStepExecutor> executorFactory, WorkerTransportOptions? options = null, Func<bool>? pauseHeartbeat = null)
        : base(options ?? new WorkerTransportOptions())
    {
        ArgumentNullException.ThrowIfNull(executorFactory);
        _executorFactory = executorFactory;
        _pauseHeartbeat = pauseHeartbeat;
    }

    /// <inheritdoc />
    public override TransportKind Kind => TransportKind.InProcess;

    /// <summary>The exception that ended the worker thread, if any.</summary>
    public Exception? Failure => Volatile.Read(ref _failure);

    /// <inheritdoc />
    protected override bool WorkerHasExited => Volatile.Read(ref _ended) != 0;

    /// <inheritdoc />
    protected override long WorkerId => Environment.ProcessId;

    /// <inheritdoc />
    protected override void OnReset()
    {
        Volatile.Write(ref _ended, 0);
        Volatile.Write(ref _failure, null);
    }

    /// <inheritdoc />
    protected override void LaunchWorker(RingSegment segment, string path)
    {
        _stop = new CancellationTokenSource();
        _executor = _executorFactory();
        _runtime = new WorkerRuntime(segment, _executor, new WorkerRuntimeOptions { Stop = _stop.Token, PauseHeartbeat = _pauseHeartbeat });
        var runtime = _runtime;
        _thread = new Thread(() =>
        {
            try
            {
                _ = runtime.Run();
            }
            catch (Exception ex)
            {
                Volatile.Write(ref _failure, ex);
            }
            finally
            {
                Volatile.Write(ref _ended, 1);
            }
        })
        {
            Name = "roost-inprocess-worker",
            IsBackground = true,
        };
        _thread.Start();
    }

    /// <inheritdoc />
    protected override async ValueTask TerminateWorkerAsync()
    {
        var thread = _thread;
        if (thread is null)
        {
            return;
        }

        _stop?.Cancel();
        var give = MonotonicClock.NowNs() + (ulong)(Options.KillTimeout.TotalMilliseconds * 1_000_000);
        while (!WorkerHasExited && MonotonicClock.NowNs() < give)
        {
            await Task.Delay(1).ConfigureAwait(false);
        }

        if (!WorkerHasExited)
        {
            MarkAbandoned(); // stuck inside a step: nothing in this process can cancel it
            return;
        }

        _ = thread.Join(Options.KillTimeout);
        _runtime?.Dispose();
        _executor?.Dispose();
        _stop?.Dispose();
        _thread = null;
        _runtime = null;
        _executor = null;
        _stop = null;
    }

    /// <inheritdoc />
    protected override string Diagnostics() => Failure is { } failure ? "Worker thread failed: " + failure : string.Empty;
}

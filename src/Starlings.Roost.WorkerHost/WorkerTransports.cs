using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost;

/// <summary>Makes the transport for a <see cref="TransportKind"/>.</summary>
public static class WorkerTransports
{
    /// <summary>The default kind: a worker process (decision 21).</summary>
    public const TransportKind Default = TransportKind.Process;

    /// <summary>Creates a transport.</summary>
    /// <param name="kind">Which one.</param>
    /// <param name="command">How to start a worker process; required for <see cref="TransportKind.Process"/>.</param>
    /// <param name="inProcessExecutor">Makes the executor for <see cref="TransportKind.InProcess"/> (tests and development only).</param>
    /// <param name="options">Host settings.</param>
    /// <returns>An unstarted transport.</returns>
    public static IWorkerTransport Create(
        TransportKind kind,
        WorkerCommand? command = null,
        Func<IStepExecutor>? inProcessExecutor = null,
        WorkerTransportOptions? options = null) => kind switch
        {
            TransportKind.Process => new ProcessWorkerTransport(
                command ?? throw new ArgumentNullException(nameof(command), "The process transport needs a worker command."),
                options),
            TransportKind.InProcess => new InProcessWorkerTransport(
                inProcessExecutor ?? throw new ArgumentNullException(nameof(inProcessExecutor), "The in-process transport needs an executor factory."),
                options),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
}

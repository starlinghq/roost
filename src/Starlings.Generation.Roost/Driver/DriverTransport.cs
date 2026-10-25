using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost.Driver;

/// <summary>
/// The members of the C8 transport the driver uses: <see cref="IWorkerTransport"/> plus <see cref="State"/>,
/// <see cref="CanPublish"/> and <see cref="NextPlanBufferIndex"/>, which only <see cref="WorkerTransportBase"/> has (a
/// contract-change request to stream A). Internal, so tests can wrap a real transport and inject transport failures.
/// </summary>
internal interface IDriverTransport : IAsyncDisposable
{
    TransportKind Kind { get; }

    TransportState State { get; }

    bool CanPublish();

    int NextPlanBufferIndex { get; }

    ValueTask<HandshakeResult> StartAsync(WorkerLaunch launch, CancellationToken cancellationToken);

    Span<byte> PlanBuffer(int index);

    bool TryPublish(in RingSlot slot);

    bool TryTakeResult(out RingSlot slot);

    ReadOnlySpan<byte> ResultBuffer(int index);

    void ReleaseResult(int index);

    FaultClass Poll();

    ValueTask KillAsync(CancellationToken cancellationToken);
}

/// <summary>An <see cref="IDriverTransport"/> over a process or in-process transport.</summary>
/// <param name="inner">The transport.</param>
internal sealed class DriverTransport(WorkerTransportBase inner) : IDriverTransport
{
    public WorkerTransportBase Inner { get; } = inner;

    public TransportKind Kind => Inner.Kind;

    public TransportState State => Inner.State;

    public int NextPlanBufferIndex => Inner.NextPlanBufferIndex;

    public bool CanPublish() => Inner.CanPublish();

    public ValueTask<HandshakeResult> StartAsync(WorkerLaunch launch, CancellationToken cancellationToken) => Inner.StartAsync(launch, cancellationToken);

    public Span<byte> PlanBuffer(int index) => Inner.PlanBuffer(index);

    public bool TryPublish(in RingSlot slot) => Inner.TryPublish(slot);

    public bool TryTakeResult(out RingSlot slot) => Inner.TryTakeResult(out slot);

    public ReadOnlySpan<byte> ResultBuffer(int index) => Inner.ResultBuffer(index);

    public void ReleaseResult(int index) => Inner.ReleaseResult(index);

    public FaultClass Poll() => Inner.Poll();

    public ValueTask KillAsync(CancellationToken cancellationToken) => Inner.KillAsync(cancellationToken);

    public ValueTask DisposeAsync() => Inner.DisposeAsync();
}

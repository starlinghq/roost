using Starlings.Generation.Roost.Driver;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost.Tests.Support;

/// <summary>Transport failures to inject, each once, shared by every transport the driver creates.</summary>
internal sealed class TransportFaults
{
    private int _throwOnTake;
    private int _corruptResult;
    private int _throwOnKill;
    private int _throwOnPlanBuffer;
    private int _noFreeBuffer;

    /// <summary>The next CanPublish says no plan buffer is free, as a desynchronised ring would.</summary>
    public bool NoFreeBuffer { set => _noFreeBuffer = value ? 1 : 0; }

    internal bool TakeNoFreeBuffer() => Take(ref _noFreeBuffer);

    /// <summary>The next result take throws an exception type the driver does not expect.</summary>
    public bool ThrowOnTake { set => _throwOnTake = value ? 1 : 0; }

    /// <summary>The next result buffer read returns garbage.</summary>
    public bool CorruptResult { set => _corruptResult = value ? 1 : 0; }

    /// <summary>The next kill throws, as a worker that does not exit after SIGKILL does.</summary>
    public bool ThrowOnKill { set => _throwOnKill = value ? 1 : 0; }

    /// <summary>The next plan buffer request throws, before anything is published.</summary>
    public bool ThrowOnPlanBuffer { set => _throwOnPlanBuffer = value ? 1 : 0; }

    public int Kills;

    public int Created;

    internal static bool Take(ref int flag) => Interlocked.Exchange(ref flag, 0) == 1;

    internal bool TakeThrowOnTake() => Take(ref _throwOnTake);

    internal bool TakeCorruptResult() => Take(ref _corruptResult);

    internal bool TakeThrowOnKill() => Take(ref _throwOnKill);

    internal bool TakeThrowOnPlanBuffer() => Take(ref _throwOnPlanBuffer);
}

/// <summary>Wraps a real transport and fails as <see cref="TransportFaults"/> says.</summary>
internal sealed class FaultyTransport : IDriverTransport
{
    private static readonly byte[] Garbage = new byte[256];
    private readonly IDriverTransport _inner;
    private readonly TransportFaults _faults;

    public FaultyTransport(IDriverTransport inner, TransportFaults faults)
    {
        _inner = inner;
        _faults = faults;
        Interlocked.Increment(ref faults.Created);
    }

    public TransportKind Kind => _inner.Kind;

    public TransportState State => _inner.State;

    public int NextPlanBufferIndex => _inner.NextPlanBufferIndex;

    public bool CanPublish() => !_faults.TakeNoFreeBuffer() && _inner.CanPublish();

    public ValueTask<HandshakeResult> StartAsync(WorkerLaunch launch, CancellationToken cancellationToken) => _inner.StartAsync(launch, cancellationToken);

    public Span<byte> PlanBuffer(int index) =>
        _faults.TakeThrowOnPlanBuffer() ? throw new OverflowException("injected: plan buffer") : _inner.PlanBuffer(index);

    public bool TryPublish(in RingSlot slot) => _inner.TryPublish(slot);

    public bool TryTakeResult(out RingSlot slot)
    {
        var taken = _inner.TryTakeResult(out slot);
        if (taken && _faults.TakeThrowOnTake())
        {
            _inner.ReleaseResult((int)slot.BufferIndex);
            throw new KeyNotFoundException("injected: result take");
        }

        return taken;
    }

    public ReadOnlySpan<byte> ResultBuffer(int index) => _faults.TakeCorruptResult() ? Garbage : _inner.ResultBuffer(index);

    public void ReleaseResult(int index) => _inner.ReleaseResult(index);

    public FaultClass Poll() => _inner.Poll();

    public ValueTask KillAsync(CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _faults.Kills);
        return _faults.TakeThrowOnKill()
            ? throw new InvalidOperationException("injected: the worker did not exit after SIGKILL")
            : _inner.KillAsync(cancellationToken);
    }

    public ValueTask DisposeAsync() => _inner.DisposeAsync();
}

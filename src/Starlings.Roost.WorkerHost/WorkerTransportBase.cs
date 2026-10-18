using System.Buffers.Binary;
using System.Collections.Immutable;
using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost;

/// <summary>Where a transport is in its life.</summary>
public enum TransportState
{
    /// <summary>Not started, or stopped by <see cref="IWorkerTransport.KillAsync"/>; <see cref="IWorkerTransport.StartAsync"/> may be called.</summary>
    Idle = 0,

    /// <summary>Spawned, handshake pending.</summary>
    Starting,

    /// <summary>The handshake succeeded; steps may be published.</summary>
    Running,

    /// <summary>The worker thread could not be stopped (in-process hang); the transport cannot be restarted.</summary>
    Abandoned,

    /// <summary>Disposed.</summary>
    Disposed,
}

/// <summary>
/// The host's half of C8, shared by the process and in-process transports: creates the segment, makes the offer, waits
/// for the answer, forwards the ring calls to the engine's <see cref="HostRingEndpoint"/>, and watches the in-flight
/// step's deadline, the heartbeat and the worker. A subclass says how a worker is started, whether it has exited, and
/// how it is killed.
/// </summary>
/// <remarks>
/// <para><b>Single host thread.</b> Every member except <see cref="State"/> is for the engine thread only, as the
/// interface says. The hot members (<see cref="TryPublish"/>, <see cref="TryTakeResult"/>, <see cref="ReleaseResult"/>,
/// <see cref="Poll"/>, <see cref="ReadHeartbeat"/>) never allocate.</para>
/// <para><b>Deadline.</b> The deadline of a step is the slot's <c>DeadlineNs</c> when the caller set one. Otherwise it
/// is <c>StepDeadline</c> after the later of two times: the step was published, and the previous step's result was
/// published by the worker (its <c>PublishedNs</c>, on the shared clock). The second term matters with two steps in
/// flight: the second must not be charged for the time it waits behind the first.</para>
/// <para><b>Fault classes.</b> <see cref="Poll"/> reports the host's conclusions: <see cref="FaultClass.Exited"/> (the worker
/// is gone and left nothing to read), <see cref="FaultClass.Hang"/> (deadline missed, heartbeat alive) and
/// <see cref="FaultClass.Stalled"/> (deadline missed, heartbeat stopped). Recoverable and sticky faults arrive in the
/// result header, which is authoritative; a sticky worker publishes, marks itself faulted and exits, so the result is
/// taken first and <see cref="Poll"/> reports <see cref="FaultClass.Exited"/> afterwards. A result that is waiting always
/// wins over a deadline or an exit, so a worker that published and then died is not misreported.</para>
/// <para><b>Restart.</b> After <see cref="KillAsync"/>, <see cref="StartAsync"/> makes a new segment, endpoints and
/// worker; nothing of the old rings is reused, so no stale cursor can survive a respawn. Spans from the old segment are
/// invalid after <see cref="KillAsync"/>.</para>
/// </remarks>
public abstract class WorkerTransportBase : IWorkerTransport
{
    private const int InFlightSlots = 64;

    private readonly WorkerTransportOptions _options;
    private readonly ulong[] _publishedNs = new ulong[InFlightSlots];
    private readonly ulong[] _deadlineNs = new ulong[InFlightSlots];
    private RingSegment? _segment;
    private SegmentAccess? _access;
    private HostRingEndpoint? _host;
    private WorkerLaunch? _launch;
    private ulong _stepsPublished;
    private ulong _stepsTaken;
    private ulong _lastResultNs;
    private ulong _heartbeatBase;
    private ulong _stepDeadlineNs;
    private ulong _livenessNs;
    private int _state;

    /// <summary>Creates the base.</summary>
    /// <param name="options">Host settings.</param>
    protected WorkerTransportBase(WorkerTransportOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
    }

    /// <inheritdoc />
    public abstract TransportKind Kind { get; }

    /// <summary>Where the transport is in its life. Readable from any thread.</summary>
    public TransportState State => (TransportState)Volatile.Read(ref _state);

    /// <inheritdoc />
    public int PlanBufferCount => Host.PlanBufferCount;

    /// <summary>The segment path of the running worker, for diagnostics.</summary>
    public string? SegmentPath => _segment?.Path;

    /// <summary>Whether the worker has exited. Safe to call from any thread; used by <see cref="Poll"/>.</summary>
    protected abstract bool WorkerHasExited { get; }

    /// <summary>The worker's process id (the host's own for the in-process transport); 0 before the worker starts.</summary>
    protected abstract long WorkerId { get; }

    private HostRingEndpoint Host => _host ?? throw new InvalidOperationException("The transport has no running worker. Call StartAsync first.");

    private SegmentAccess Access => _access ?? throw new InvalidOperationException("The transport has no running worker. Call StartAsync first.");

    /// <inheritdoc />
    public async ValueTask<HandshakeResult> StartAsync(WorkerLaunch launch, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(launch);
        Validate(launch);
        if (State is not TransportState.Idle)
        {
            throw new InvalidOperationException($"StartAsync is valid only on an idle transport; this one is {State}.");
        }

        Volatile.Write(ref _state, (int)TransportState.Starting);
        Reset();
        _launch = launch;
        _stepDeadlineNs = (ulong)(launch.StepDeadline.TotalMilliseconds * 1_000_000);
        _livenessNs = (ulong)Math.Max(_options.HeartbeatLiveness.TotalMilliseconds, launch.HeartbeatPeriod.TotalMilliseconds * 20) * 1_000_000;
        var path = Path.Combine(_options.SegmentDirectory ?? RingSegment.DefaultDirectory(), $"roost-worker-{Environment.ProcessId}-{Guid.NewGuid():N}");
        var segment = RingSegment.Create(path, new SegmentOptions
        {
            PlanBufferCount = launch.PlanBufferCount,
            ResultBufferCount = launch.ResultBufferCount,
            PlanBufferBytes = checked((int)launch.PlanBufferBytes),
            ResultBufferBytes = checked((int)launch.ResultBufferBytes),
            InputRegionBytes = launch.InputRegionBytes,
        });
        _segment = segment;
        _access = new SegmentAccess(segment);
        _host = segment.OpenHost();
        WriteOffer(launch);

        try
        {
            LaunchWorker(segment, path);
            var answered = await WaitForAnswerAsync(cancellationToken).ConfigureAwait(false);
            var handshake = _access!.ReadHandshake();
            if (answered == HandshakeState.Rejected)
            {
                await TerminateWorkerAsync().ConfigureAwait(false); // a rejecting worker exits by itself; make sure
                TearDown();
                Volatile.Write(ref _state, (int)TransportState.Idle);
                return new HandshakeResult(HandshakeState.Rejected, handshake.RejectReason, (long)handshake.WorkerPid, UuidOf(handshake));
            }

            Volatile.Write(ref _state, (int)TransportState.Running);
            return new HandshakeResult(HandshakeState.Ready, HandshakeReject.None, (long)handshake.WorkerPid, UuidOf(handshake));
        }
        catch
        {
            await TerminateWorkerAsync().ConfigureAwait(false);
            TearDown();
            if (State != TransportState.Abandoned)
            {
                Volatile.Write(ref _state, (int)TransportState.Idle);
            }

            throw;
        }
    }

    /// <summary>
    /// Whether another plan can be published now, that is whether the plan buffer <see cref="NextPlanBufferIndex"/> names
    /// has been released by the worker. Not part of <see cref="IWorkerTransport"/>, which has no way to ask this before
    /// writing a buffer; reported to stream A as a contract-change request.
    /// </summary>
    /// <returns>False while every plan buffer is in flight.</returns>
    public bool CanPublish() => Host.CanPublish();

    /// <summary>The plan buffer the next published plan must use (slot number modulo buffer count). See <see cref="CanPublish"/>.</summary>
    public int NextPlanBufferIndex => Host.NextPlanBufferIndex;

    /// <inheritdoc />
    public Span<byte> PlanBuffer(int index) => Host.PlanBuffer(index);

    /// <inheritdoc />
    public bool TryPublish(in RingSlot slot)
    {
        var host = Host;
        var stamped = slot;
        var now = MonotonicClock.NowNs();
        if (stamped.PublishedNs == 0)
        {
            stamped.PublishedNs = now;
        }

        if (!host.TryPublish(stamped))
        {
            return false;
        }

        if (stamped.Kind == RingSlotKind.Step)
        {
            if (_stepsPublished == _stepsTaken)
            {
                _heartbeatBase = Access.HeartbeatCounter();
            }

            var at = (int)(_stepsPublished % InFlightSlots);
            _publishedNs[at] = stamped.PublishedNs;
            _deadlineNs[at] = stamped.DeadlineNs;
            _stepsPublished++;
        }

        return true;
    }

    /// <inheritdoc />
    public bool TryTakeResult(out RingSlot slot)
    {
        if (!Host.TryTakeResult(out slot))
        {
            return false;
        }

        if (slot.Kind == RingSlotKind.Step)
        {
            _stepsTaken++;
            _lastResultNs = slot.PublishedNs;
            if (_stepsTaken < _stepsPublished)
            {
                _heartbeatBase = Access.HeartbeatCounter(); // the next step starts now
            }
        }

        return true;
    }

    /// <inheritdoc />
    public ReadOnlySpan<byte> ResultBuffer(int index) => Host.ResultBuffer(index);

    /// <inheritdoc />
    public void ReleaseResult(int index) => Host.ReleaseResult(index);

    /// <inheritdoc />
    public Heartbeat ReadHeartbeat() => _access is null ? default : _access.ReadHeartbeat();

    /// <inheritdoc />
    public FaultClass Poll()
    {
        if (State != TransportState.Running || _host is null || _access is null)
        {
            return FaultClass.None;
        }

        var exited = WorkerHasExited; // read the exit first: anything published before the exit is visible to the next read
        if (_access.ResultsPublished() > _host.ResultsTaken)
        {
            return FaultClass.None; // a result is waiting; it carries its own fault class
        }

        if (exited)
        {
            return FaultClass.Exited;
        }

        if (_stepsPublished == _stepsTaken)
        {
            return FaultClass.None; // nothing in flight, so no deadline
        }

        var now = MonotonicClock.NowNs();
        var at = (int)(_stepsTaken % InFlightSlots);
        var deadline = _deadlineNs[at] != 0
            ? _deadlineNs[at]
            : Math.Max(_publishedNs[at], _lastResultNs) + _stepDeadlineNs;
        if (now < deadline)
        {
            return FaultClass.None;
        }

        var heartbeat = _access.ReadHeartbeat();
        var advanced = heartbeat.Counter != _heartbeatBase && now - heartbeat.WorkerNs <= _livenessNs;
        return FaultRules.ClassifyMissedDeadline(advanced);
    }

    /// <inheritdoc />
    public async ValueTask KillAsync(CancellationToken cancellationToken)
    {
        if (State is TransportState.Idle or TransportState.Disposed)
        {
            return;
        }

        await TerminateWorkerAsync().ConfigureAwait(false);
        if (State == TransportState.Abandoned)
        {
            return; // the worker thread is still running over the segment; leave the segment mapped
        }

        TearDown();
        Volatile.Write(ref _state, (int)TransportState.Idle);
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (State == TransportState.Disposed)
        {
            return;
        }

        if (State == TransportState.Running && !WorkerHasExited)
        {
            await ShutDownCleanlyAsync().ConfigureAwait(false);
        }

        await TerminateWorkerAsync().ConfigureAwait(false);
        if (State != TransportState.Abandoned)
        {
            TearDown();
        }

        Volatile.Write(ref _state, (int)TransportState.Disposed);
        GC.SuppressFinalize(this);
    }

    /// <summary>Starts the worker over a segment whose offer is already published.</summary>
    /// <param name="segment">The host's mapping of the segment.</param>
    /// <param name="path">The segment path, for a worker that opens it.</param>
    protected abstract void LaunchWorker(RingSegment segment, string path);

    /// <summary>Makes sure the worker is gone and reaped: kills it if alive, waits up to <see cref="WorkerTransportOptions.KillTimeout"/>. Idempotent.</summary>
    /// <returns>A task that completes when the worker is gone, or when the transport gave up (set <see cref="MarkAbandoned"/>).</returns>
    protected abstract ValueTask TerminateWorkerAsync();

    /// <summary>The worker's last output, for start failures.</summary>
    /// <returns>Text; may be empty.</returns>
    protected virtual string Diagnostics() => string.Empty;

    /// <summary>Called on every start before the segment is created.</summary>
    protected virtual void OnReset()
    {
    }

    /// <summary>Records that a worker could not be stopped and still runs over the segment.</summary>
    protected void MarkAbandoned() => Volatile.Write(ref _state, (int)TransportState.Abandoned);

    /// <summary>The options.</summary>
    protected WorkerTransportOptions Options => _options;

    private static ImmutableArray<byte> UuidOf(in Handshake handshake)
    {
        var uuid = new byte[16];
        BinaryPrimitives.WriteUInt64LittleEndian(uuid, handshake.DeviceUuid0);
        BinaryPrimitives.WriteUInt64LittleEndian(uuid.AsSpan(8), handshake.DeviceUuid1);
        return ImmutableCollectionsMarshalFromArray(uuid);
    }

    private static ImmutableArray<byte> ImmutableCollectionsMarshalFromArray(byte[] array) =>
        System.Runtime.InteropServices.ImmutableCollectionsMarshal.AsImmutableArray(array);

    private static void Validate(WorkerLaunch launch)
    {
        if (launch.ManifestDigest.IsDefaultOrEmpty || launch.ManifestDigest.Length != 32 || launch.WeightsDigest.IsDefaultOrEmpty || launch.WeightsDigest.Length != 32)
        {
            throw new ArgumentException("Digests are SHA-256: 32 bytes.", nameof(launch));
        }

        if (launch.StepDeadline <= TimeSpan.Zero || launch.HeartbeatPeriod <= TimeSpan.Zero)
        {
            throw new ArgumentException("The step deadline and the heartbeat period must be positive.", nameof(launch));
        }

        if (launch.PlanBufferCount < 2 || launch.ResultBufferCount < 2 || launch.PlanBufferBytes <= 0 || launch.ResultBufferBytes <= 0)
        {
            throw new ArgumentException("At least two plan and two result buffers of positive size.", nameof(launch));
        }
    }

    private void Reset()
    {
        _stepsPublished = 0;
        _stepsTaken = 0;
        _lastResultNs = 0;
        _heartbeatBase = 0;
        OnReset();
    }

    private void WriteOffer(WorkerLaunch launch)
    {
        var manifest = launch.ManifestDigest.AsSpan();
        var weights = launch.WeightsDigest.AsSpan();
        var offer = new Handshake
        {
            ContractVersion = (uint)ContractVersion.Current,
            AbiVersion = _options.AbiVersion,
            LayoutVersion = 1,
            DeviceOrdinal = launch.DeviceOrdinal,
            MeshRank = (uint)launch.Mesh.Rank,
            MeshSize = (uint)launch.Mesh.Size,
            ManifestDigest0 = BinaryPrimitives.ReadUInt64LittleEndian(manifest),
            ManifestDigest1 = BinaryPrimitives.ReadUInt64LittleEndian(manifest[8..]),
            ManifestDigest2 = BinaryPrimitives.ReadUInt64LittleEndian(manifest[16..]),
            ManifestDigest3 = BinaryPrimitives.ReadUInt64LittleEndian(manifest[24..]),
            WeightsDigest0 = BinaryPrimitives.ReadUInt64LittleEndian(weights),
            WeightsDigest1 = BinaryPrimitives.ReadUInt64LittleEndian(weights[8..]),
            WeightsDigest2 = BinaryPrimitives.ReadUInt64LittleEndian(weights[16..]),
            WeightsDigest3 = BinaryPrimitives.ReadUInt64LittleEndian(weights[24..]),
            HostPid = (ulong)Environment.ProcessId,
            StepDeadlineNs = _stepDeadlineNs,
            HeartbeatPeriodNs = (ulong)(launch.HeartbeatPeriod.TotalMilliseconds * 1_000_000),
        };
        _access!.WriteOffer(offer); // the state is published last, with release
    }

    private async ValueTask<HandshakeState> WaitForAnswerAsync(CancellationToken cancellationToken)
    {
        var access = _access!;
        var give = MonotonicClock.NowNs() + (ulong)(_options.StartupTimeout.TotalMilliseconds * 1_000_000);
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var exited = WorkerHasExited;
            var state = access.ReadState(); // after the exit read: a worker that answered and then exited is seen answering
            if (state is HandshakeState.Ready or HandshakeState.Rejected)
            {
                return state;
            }

            if (exited)
            {
                throw new WorkerStartException($"The worker exited before it answered the handshake. {Diagnostics()}".TrimEnd());
            }

            if (MonotonicClock.NowNs() > give)
            {
                throw new WorkerStartException($"The worker did not answer the handshake within {_options.StartupTimeout}. {Diagnostics()}".TrimEnd());
            }

            await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        }
    }

    private async ValueTask ShutDownCleanlyAsync()
    {
        var host = Host;
        var shutdown = new RingSlot { Kind = RingSlotKind.Shutdown, PublishedNs = MonotonicClock.NowNs() };
        var give = MonotonicClock.NowNs() + (ulong)(_options.ShutdownTimeout.TotalMilliseconds * 1_000_000);
        while (!host.TryPublish(shutdown))
        {
            if (WorkerHasExited || MonotonicClock.NowNs() > give)
            {
                return;
            }

            await Task.Delay(1).ConfigureAwait(false);
        }

        while (!WorkerHasExited && MonotonicClock.NowNs() < give)
        {
            await Task.Delay(1).ConfigureAwait(false);
        }
    }

    private void TearDown()
    {
        _host = null;
        _access = null;
        var segment = _segment;
        _segment = null;
        segment?.Dispose(); // the creator removes the file
    }
}

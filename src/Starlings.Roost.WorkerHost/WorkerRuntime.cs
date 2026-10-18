using System.Buffers.Binary;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost;

/// <summary>Why a worker runtime ended; the worker process exits with <see cref="WorkerRuntime.ExitCodeFor"/> of it.</summary>
public enum WorkerExit
{
    /// <summary>Still running.</summary>
    Running = 0,

    /// <summary>The host published a shutdown slot.</summary>
    Shutdown,

    /// <summary>The handshake was rejected; the reason is in the segment.</summary>
    Rejected,

    /// <summary>A sticky fault: the result was published, the heartbeat says faulted, and the worker left.</summary>
    Sticky,

    /// <summary>The host process is gone.</summary>
    HostGone,

    /// <summary>Stopped from outside (in-process transport kill).</summary>
    Stopped,

    /// <summary>The ring protocol was violated (corrupt cursor or slot).</summary>
    ProtocolError,
}

/// <summary>How a <see cref="WorkerRuntime"/> is embedded.</summary>
public sealed record WorkerRuntimeOptions
{
    /// <summary>
    /// True in a worker process: disables core dumps, and ends the process when the parent process is no longer the host
    /// (an orphaned worker would hold a GPU and a segment file forever).
    /// </summary>
    public bool IsolatedProcess { get; init; }

    /// <summary>Asks the runtime to end (in-process kill). Checked once per poll.</summary>
    public CancellationToken Stop { get; init; }

    /// <summary>
    /// Test hook for the in-process transport: while true, the heartbeat thread does not beat, which presents as a
    /// stalled worker. Never set in a worker process.
    /// </summary>
    public Func<bool>? PauseHeartbeat { get; init; }
}

/// <summary>
/// The worker side of C8, one instance per worker: the handshake, the heartbeat thread, and the step loop that takes
/// plans from the plan ring, runs them on an <see cref="IStepExecutor"/> and publishes results. A worker process runs it
/// with <see cref="Run"/>; the in-process transport calls <see cref="Poll"/> from a polling thread. Both use this one
/// implementation, which is what lets the same tests run against both transports.
/// </summary>
/// <remarks>
/// <para><b>Handshake</b> (S8 conditions 2 and 4). The host fills every field, then publishes
/// <see cref="HandshakeState.Offered"/> with a release write. The worker reads the state with acquire, checks the offer in
/// a fixed order that costs nothing before it costs anything (layout, contract, ABI, manifest digest, weights digest,
/// mesh), and only then opens the device, registers the segment and loads the weights. It writes its answer fields, then
/// <see cref="HandshakeState.Ready"/> or <see cref="HandshakeState.Rejected"/> last. Digests are compared with the ones
/// recorded in the worker library's manifest; weights are never hashed.</para>
/// <para><b>Steps.</b> A plan slot is taken, a result buffer is awaited, the plan is validated and executed, the plan
/// slot is released (the executor has finished reading it), and the result slot is published with the worker's clock in
/// <c>PublishedNs</c>. A recoverable fault is published and the loop continues. A sticky fault, or an exception from the
/// executor, is published, the heartbeat state becomes <see cref="WorkerState.Faulted"/>, and the runtime ends.</para>
/// <para>The step loop does not allocate.</para>
/// </remarks>
public sealed unsafe class WorkerRuntime : IDisposable
{
    private const uint MagicPlan = 0x4C505352u;
    private const uint MagicResult = 0x53525352u;
    private const uint LayoutVersion = 1;
    private const int ErrorInvalidPlan = -22;
    private const int ErrorExecutor = -1;

    private readonly SegmentAccess _access;
    private readonly WorkerRingEndpoint _ring;
    private readonly IStepExecutor _executor;
    private readonly WorkerRuntimeOptions _options;
    private readonly Thread _heartbeat;
    private readonly ulong _hostPid;
    private int _stopHeartbeat;
    private int _exit;
    private bool _handshakeDone;
    private bool _haveSlot;
    private RingSlot _slot;
    private bool _started;

    /// <summary>Creates a runtime over a mapped segment. Nothing runs until <see cref="Start"/>.</summary>
    /// <param name="segment">The segment, opened by the worker process or shared with the host (in-process).</param>
    /// <param name="executor">The step executor.</param>
    /// <param name="options">How the runtime is embedded.</param>
    public WorkerRuntime(RingSegment segment, IStepExecutor executor, WorkerRuntimeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(segment);
        ArgumentNullException.ThrowIfNull(executor);
        _options = options ?? new WorkerRuntimeOptions();
        _access = new SegmentAccess(segment);
        _ring = segment.OpenWorker();
        _executor = executor;
        _hostPid = _access.HandshakeRecord->HostPid;
        _heartbeat = new Thread(HeartbeatLoop) { Name = "roost-worker-heartbeat", IsBackground = true };
    }

    /// <summary>Why the runtime ended, or <see cref="WorkerExit.Running"/>.</summary>
    public WorkerExit Exit => (WorkerExit)Volatile.Read(ref _exit);

    /// <summary>The process exit code for an exit reason.</summary>
    /// <param name="exit">The reason.</param>
    /// <returns>0 for a clean shutdown; distinct non-zero codes otherwise.</returns>
    public static int ExitCodeFor(WorkerExit exit) => exit switch
    {
        WorkerExit.Shutdown => 0,
        WorkerExit.Rejected => 20,
        WorkerExit.Sticky => 21,
        WorkerExit.HostGone => 22,
        WorkerExit.Stopped => 23,
        WorkerExit.ProtocolError => 24,
        _ => 1,
    };

    /// <summary>Starts the heartbeat thread. In a worker process, first disables core dumps.</summary>
    public void Start()
    {
        if (_started)
        {
            return;
        }

        _started = true;
        if (_options.IsolatedProcess)
        {
            Libc.DisableCoreDumps();
        }

        _access.Beat();
        _heartbeat.Start();
    }

    /// <summary>Runs until the runtime ends (worker process entry). Spins, then yields; never sleeps while a plan may arrive.</summary>
    /// <returns>The reason it ended.</returns>
    public WorkerExit Run()
    {
        Start();
        var wait = new SpinWait();
        while (Exit == WorkerExit.Running)
        {
            if (Poll())
            {
                wait.Reset();
            }
            else
            {
                wait.SpinOnce(sleep1Threshold: -1);
            }
        }

        return Exit;
    }

    /// <summary>Does one unit of work. Returns true when it made progress, false when it found nothing to do (the caller waits).</summary>
    /// <returns>Whether progress was made.</returns>
    public bool Poll()
    {
        if (Exit != WorkerExit.Running)
        {
            return false;
        }

        if (_options.Stop.IsCancellationRequested)
        {
            Finish(WorkerExit.Stopped, WorkerState.Exiting);
            return true;
        }

        try
        {
            return _handshakeDone ? PollStep() : PollHandshake();
        }
        catch (InvalidDataException)
        {
            Finish(WorkerExit.ProtocolError, WorkerState.Faulted);
            return true;
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Volatile.Write(ref _stopHeartbeat, 1);
        if (_started)
        {
            _ = _heartbeat.Join(TimeSpan.FromSeconds(2));
        }
    }

    private static bool AbiCompatible(uint offered, uint library) =>
        (library >> 16) == (offered >> 16) && (library & 0xFFFFu) >= (offered & 0xFFFFu);

    private static bool DigestEquals(ImmutableArray<byte> digest, ulong d0, ulong d1, ulong d2, ulong d3)
    {
        if (digest.IsDefault || digest.Length != 32)
        {
            return false;
        }

        var span = digest.AsSpan();
        return BinaryPrimitives.ReadUInt64LittleEndian(span) == d0
            && BinaryPrimitives.ReadUInt64LittleEndian(span[8..]) == d1
            && BinaryPrimitives.ReadUInt64LittleEndian(span[16..]) == d2
            && BinaryPrimitives.ReadUInt64LittleEndian(span[24..]) == d3;
    }

    private bool PollHandshake()
    {
        if (_access.ReadState() != HandshakeState.Offered)
        {
            return false;
        }

        var offer = *_access.HandshakeRecord; // acquire above: every field the host wrote first is visible
        var identity = _executor.Identity;
        var reject = HandshakeReject.None;
        if (offer.LayoutVersion != LayoutVersion)
        {
            reject = HandshakeReject.Layout;
        }
        else if (offer.ContractVersion != (uint)ContractVersion.Current)
        {
            reject = HandshakeReject.Contract;
        }
        else if (!AbiCompatible(offer.AbiVersion, identity.AbiVersion))
        {
            reject = HandshakeReject.Abi;
        }
        else if (!DigestEquals(identity.ManifestDigest, offer.ManifestDigest0, offer.ManifestDigest1, offer.ManifestDigest2, offer.ManifestDigest3))
        {
            reject = HandshakeReject.ManifestDigest;
        }
        else if (!DigestEquals(identity.WeightsDigest, offer.WeightsDigest0, offer.WeightsDigest1, offer.WeightsDigest2, offer.WeightsDigest3))
        {
            reject = HandshakeReject.WeightsDigest;
        }
        else if (offer.MeshSize == 0 || offer.MeshRank >= offer.MeshSize)
        {
            reject = HandshakeReject.Mesh;
        }
        else if (_access.Segment.Header.ResultBufferBytes < (ulong)sizeof(StepResultHeader))
        {
            reject = HandshakeReject.Layout;
        }

        Span<byte> uuid = stackalloc byte[16];
        if (reject == HandshakeReject.None)
        {
            reject = _executor.OpenDevice(offer.DeviceOrdinal, uuid);
        }

        if (reject == HandshakeReject.None)
        {
            _executor.RegisterSegment((nint)_access.Base, _access.Segment.Bytes);
            reject = _executor.LoadWeights();
        }

        var record = _access.HandshakeRecord;
        record->WorkerPid = (ulong)Environment.ProcessId;
        if (reject != HandshakeReject.None)
        {
            record->RejectReason = reject;
            _access.WriteState(HandshakeState.Rejected);
            Finish(WorkerExit.Rejected, WorkerState.Exiting);
            return true;
        }

        record->DeviceUuid0 = BinaryPrimitives.ReadUInt64LittleEndian(uuid);
        record->DeviceUuid1 = BinaryPrimitives.ReadUInt64LittleEndian(uuid[8..]);
        _handshakeDone = true;
        _access.WriteWorkerState(WorkerState.Ready);
        _access.WriteState(HandshakeState.Ready); // last: the answer fields above are visible to a host that sees Ready
        return true;
    }

    private bool PollStep()
    {
        if (!_haveSlot)
        {
            if (!_ring.TryTakePlan(out _slot))
            {
                return false;
            }

            switch (_slot.Kind)
            {
                case RingSlotKind.Shutdown:
                    _ring.ReleasePlan();
                    Finish(WorkerExit.Shutdown, WorkerState.Exiting);
                    return true;
                case RingSlotKind.Drain:
                    _ring.ReleasePlan(); // steps run to completion one at a time, so nothing is in flight when a drain is seen
                    return true;
                case RingSlotKind.Step:
                    _haveSlot = true;
                    break;
                default:
                    Finish(WorkerExit.ProtocolError, WorkerState.Faulted);
                    return true;
            }
        }

        if (!_ring.CanPublishResult())
        {
            return false; // the host still holds every result buffer
        }

        _haveSlot = false;
        RunStep(_slot);
        return true;
    }

    private void RunStep(in RingSlot slot)
    {
        var resultIndex = _ring.NextResultBufferIndex;
        var result = _ring.ResultBuffer(resultIndex);
        var plan = _ring.PlanBuffer((int)slot.BufferIndex);
        _access.WriteWorkerState(WorkerState.Stepping);

        StepOutcome outcome;
        if (!PlanIsValid(plan, slot.StepId))
        {
            outcome = StepOutcome.Recoverable(ErrorInvalidPlan);
        }
        else
        {
            try
            {
                outcome = _executor.Execute(plan, result);
                outcome = Vet(outcome, result, slot.StepId);
            }
            catch (Exception)
            {
                outcome = StepOutcome.Sticky(ErrorExecutor); // the context cannot be vouched for
            }
        }

        if (outcome.Fault != FaultClass.None)
        {
            WriteFaultResult(result, slot.StepId, outcome);
        }

        _ring.ReleasePlan(); // the executor is done with the plan buffer
        var published = new RingSlot
        {
            StepId = slot.StepId,
            Kind = RingSlotKind.Step,
            BufferIndex = (uint)resultIndex,
            PublishedNs = MonotonicClock.NowNs(),
        };
        if (!_ring.TryPublishResult(published))
        {
            throw new InvalidDataException("The result ring refused a slot after CanPublishResult.");
        }

        _access.WriteLastStep(slot.StepId);
        if (outcome.Fault == FaultClass.Sticky)
        {
            Finish(WorkerExit.Sticky, WorkerState.Faulted);
        }
        else
        {
            _access.WriteWorkerState(WorkerState.Ready);
        }
    }

    private static bool PlanIsValid(ReadOnlySpan<byte> plan, ulong stepId)
    {
        if (plan.Length < sizeof(StepPlanHeader))
        {
            return false;
        }

        var header = Unsafe.ReadUnaligned<StepPlanHeader>(ref Unsafe.AsRef(in plan[0]));
        return header.Magic == MagicPlan
            && header.LayoutVersion == LayoutVersion
            && header.HeaderSize >= (uint)sizeof(StepPlanHeader)
            && header.TotalBytes >= header.HeaderSize
            && header.TotalBytes <= (ulong)plan.Length
            && header.StepId == stepId;
    }

    private static StepOutcome Vet(StepOutcome outcome, ReadOnlySpan<byte> result, ulong stepId)
    {
        if (outcome.Fault is FaultClass.Recoverable or FaultClass.Sticky)
        {
            return outcome;
        }

        if (outcome.Fault != FaultClass.None || outcome.ResultBytes < sizeof(StepResultHeader) || outcome.ResultBytes > result.Length)
        {
            return StepOutcome.Sticky(ErrorExecutor);
        }

        var header = Unsafe.ReadUnaligned<StepResultHeader>(ref Unsafe.AsRef(in result[0]));
        return header.Magic == MagicResult && header.StepId == stepId && header.Fault == FaultClass.None
            ? outcome
            : StepOutcome.Sticky(ErrorExecutor);
    }

    private static void WriteFaultResult(Span<byte> result, ulong stepId, in StepOutcome outcome)
    {
        var header = new StepResultHeader
        {
            Magic = MagicResult,
            LayoutVersion = LayoutVersion,
            HeaderSize = (uint)sizeof(StepResultHeader),
            Fault = outcome.Fault,
            StepId = stepId,
            TotalBytes = (ulong)sizeof(StepResultHeader),
            ErrorCode = outcome.ErrorCode,
        };
        Unsafe.WriteUnaligned(ref result[0], header);
    }

    private void Finish(WorkerExit exit, WorkerState state)
    {
        _access.WriteWorkerState(state);
        Volatile.Write(ref _exit, (int)exit);
    }

    private void HeartbeatLoop()
    {
        var periodNs = _access.HandshakeRecord->HeartbeatPeriodNs;
        var period = periodNs == 0 ? 1 : (int)Math.Max(1, periodNs / 1_000_000);
        var fineTimer = OperatingSystem.IsWindows() && period < 16;
        if (fineTimer && OperatingSystem.IsWindows())
        {
            Win32.BeginFineTimer(); // else Thread.Sleep(1) sleeps a 15.6 ms tick and the beat is later than the liveness window allows
        }

        // On Windows the handle pins the host's process object, so a reused process id cannot pass for the host.
        var hostProcess = _options.IsolatedProcess && _hostPid != 0 && OperatingSystem.IsWindows()
            ? Win32.OpenProcess((int)_hostPid)
            : null;
        try
        {
            BeatUntilStopped(period, hostProcess);
        }
        finally
        {
            hostProcess?.Dispose();
            if (fineTimer && OperatingSystem.IsWindows())
            {
                Win32.EndFineTimer();
            }
        }
    }

    private bool HostIsGone(System.Diagnostics.Process? hostProcess)
    {
        if (_hostPid == 0)
        {
            return false;
        }

        if (OperatingSystem.IsWindows())
        {
            return hostProcess is null || Win32.HasEnded(hostProcess);
        }

        return OperatingSystem.IsLinux() && (ulong)Libc.ParentPid() != _hostPid;
    }

    private static void ExitProcessNow(int code)
    {
        if (OperatingSystem.IsWindows())
        {
            Win32.ExitNow(code);
        }
        else
        {
            Libc.Exit(code);
        }
    }

    private void BeatUntilStopped(int period, System.Diagnostics.Process? hostProcess)
    {
        var sinceParentCheck = 0;
        while (Volatile.Read(ref _stopHeartbeat) == 0)
        {
            if (_options.PauseHeartbeat?.Invoke() != true)
            {
                _access.Beat();
            }

            if (_options.IsolatedProcess && ++sinceParentCheck >= Math.Max(1, 10 / period))
            {
                sinceParentCheck = 0;
                if (HostIsGone(hostProcess))
                {
                    // The host is gone (on Linux also: it is not this process's parent, so the worker must be started
                    // directly by the host, not through a wrapper). Remove the segment file it can no longer remove, and
                    // leave without running managed shutdown (the main thread may be inside a driver call that never returns).
                    try
                    {
                        File.Delete(_access.Segment.Path);
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                    {
                        // Best effort.
                    }

                    ExitProcessNow(ExitCodeFor(WorkerExit.HostGone));
                }
            }

            Thread.Sleep(period);
        }
    }
}

using System.Collections.Immutable;
using System.Diagnostics;
using System.Globalization;
using System.Threading.Channels;
using Starlings.Roost.Engine.Admission;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.State;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;
using StepOutcome = Starlings.Roost.Engine.Scheduling.StepOutcome;

namespace Starlings.Generation.Roost.Driver;

/// <summary>
/// The M0 backend: one host thread that owns the engine's <see cref="StepPlanner"/> and the worker transport (C8), plans a
/// step, writes it to a plan buffer, waits for its result and turns the result into C6 events. One step is in flight at a
/// time, as the eager worker runs synchronously (R-025), unless <see cref="RoostWorkerOptions.AsyncScheduling"/> is set; the
/// engine's own loop replaces this class from M1.
/// </summary>
/// <remarks>
/// <para><b>Threading.</b> <see cref="Submit"/> and the cancel callback may be called from any thread: they only enqueue a
/// command, under the same lock the driver takes when it stops, so a request is either taken by the driver or failed. Everything
/// that touches the planner or the transport runs on the driver thread (the transport is single host thread by contract).</para>
/// <para><b>Faults.</b> A recoverable step fault fails the requests of that step with
/// <see cref="RoostErrorCode.DeviceFault"/> and keeps the worker. A sticky fault, a hang, a stall or an exited worker fails
/// every live request (<see cref="RoostErrorCode.DeviceFault"/> for sticky, <see cref="RoostErrorCode.WorkerRestarted"/>
/// otherwise), kills the worker, and the next request starts a new one. Resuming requests on the new worker with
/// <c>Recovered</c> set is R-070.</para>
/// <para><b>Async scheduling (R-032).</b> With <see cref="RoostWorkerOptions.AsyncScheduling"/> the planner chains decode rows
/// onto the step in flight and the driver keeps two steps published (the plan ring is double buffered): it plans and publishes
/// step N+1, then waits for step N, applies it, and plans again. Results are applied in step order. A sequence that sampled a
/// stop id in step N finishes when N is applied; its row in N+1 is ignored and its token never reaches the channel, so the
/// one-step stop lag is not visible. A recoverable fault in step N fails N's requests; step N+1, already published, names N as
/// its input step and is refused by the worker, so its requests fail too.</para>
/// <para><b>Containment.</b> The driver thread never ends while the client is alive: an unexpected exception anywhere in a
/// step (planner, plan writer, transport, a malformed result, a kill that fails) fails the requests it affects with
/// <see cref="RoostErrorCode.Internal"/>, discards the planner state and the worker (replacing the transport when it cannot
/// be killed), and the loop goes on. An exception escaping a dedicated thread would end the consumer's process, which is
/// what the worker process exists to prevent (decision 21).</para>
/// <para><b>Sleep, wake and weight update (R-100).</b> <see cref="ControlAsync"/> queues a C8 control message; the driver thread
/// sends it once no step is in flight and waits for the answer (up to <see cref="RoostWorkerOptions.ControlTimeout"/>). Each is
/// refused with <see cref="RoostErrorCode.InvalidOption"/> while requests are live, a sleep while asleep and a wake while awake.
/// While asleep, new requests are refused with <see cref="RoostErrorCode.InsufficientDeviceMemory"/> (the worker released its
/// memory). A sleep also replaces the planner, so no block it remembers outlives the state the worker released. The driver
/// keeps the digest of the weights the worker serves (the profile's at start, a replacement's after an update) and sends it with
/// every wake, so a worker that would come back with other weights is refused. A replacement is sent again to every worker the
/// driver starts after it, so a respawned worker serves the replaced weights too; a worker that is lost while asleep is replaced
/// by one that is awake.</para>
/// </remarks>
internal sealed class WorkerDriver : IGenerationBackend
{
    private const int MinResultBufferBytes = 64 * 1024;

    // Per row of a step result: the sequence record, one token and (1 + 20) logprob rows of 8 bytes, with room to spare.
    private const long ResultBytesPerRow = 64 + ((1 + 20) * 8);

    // Control message ids have the top bit set, so a control slot's step_id never equals a planned step's.
    private const ulong ControlIdBit = 1UL << 63;

    private readonly RoostModelProfile _profile;
    private readonly IDeviceMemorySource? _memory;
    private readonly long _minFreeDeviceBytes;
    private readonly WorkerLaunch _launch;
    private readonly Func<IDriverTransport> _createTransport;
    private readonly object _gate = new();
    private readonly Queue<Command> _commands = new();
    private readonly SemaphoreSlim _wake = new(0);
    private readonly Dictionary<ulong, Live> _live = [];
    private readonly Queue<PlannedStep> _published = new();
    private readonly Queue<Control> _controls = new();
    private readonly TimeSpan _controlTimeout;
    private readonly bool _async;
    private readonly Thread _thread;
    private readonly TaskCompletionSource _exited = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private IDriverTransport _transport;
    private StepPlanner _planner;
    private BlockManager _blocks;
    private double _kvCacheUsage;
    private volatile bool _stopping;
    private volatile bool _asleep;
    private ImmutableArray<byte> _weightsDigest;
    private WeightSource? _replacement;
    private ulong _controlIds;
    private long _lastWakeTicks = -1;
    private int _disposed;

    private WorkerDriver(RoostModelProfile profile, RoostWorkerOptions worker, Func<IDriverTransport> createTransport)
    {
        _profile = profile;
        _memory = worker.DeviceMemory;
        _minFreeDeviceBytes = worker.MinFreeDeviceBytes;
        _createTransport = createTransport;
        _async = worker.AsyncScheduling;
        _controlTimeout = worker.ControlTimeout;
        _weightsDigest = profile.WeightsDigest;
        _transport = createTransport();
        _planner = CreatePlanner(profile, _async, out _blocks);
        long planBytes = PlanBuffer.SizeFor(profile.MaxSequencesPerStep, profile.MaxTokensPerStep, profile.StateGroups.Length, profile.AttentionBlocks, profile.StopTokenIds.Length);
        _launch = new WorkerLaunch
        {
            DeviceOrdinal = worker.DeviceOrdinal,
            Mesh = (0, 1),
            ManifestDigest = profile.ManifestDigest,
            WeightsDigest = profile.WeightsDigest,
            StepDeadline = worker.StepDeadline,
            HeartbeatPeriod = worker.HeartbeatPeriod,
            PlanBufferBytes = planBytes,
            ResultBufferBytes = Math.Max(MinResultBufferBytes, 4096 + (profile.MaxSequencesPerStep * ResultBytesPerRow)),
        };
        _thread = new Thread(Run) { Name = "roost-chat-driver", IsBackground = true };
    }

    /// <summary>The offer made to the worker (tests read the buffer sizes).</summary>
    public WorkerLaunch Launch => _launch;

    /// <summary>The current transport (tests read its state and kind). Replaced when a worker cannot be killed.</summary>
    public IDriverTransport Transport => _transport;

    /// <inheritdoc />
    public bool IsSleeping => _asleep;

    /// <inheritdoc />
    public TimeSpan? LastWakeDuration
    {
        get
        {
            long ticks = Interlocked.Read(ref _lastWakeTicks);
            return ticks < 0 ? null : TimeSpan.FromTicks(ticks);
        }
    }

    /// <summary>Creates the transport, starts the worker and the driver thread.</summary>
    /// <param name="profile">The model profile, validated.</param>
    /// <param name="worker">How the worker is started.</param>
    /// <param name="cancellationToken">Cancels the start.</param>
    /// <param name="decorate">Test hook: wraps each transport the driver creates.</param>
    /// <returns>The running driver.</returns>
    /// <exception cref="RoostChatException">The worker rejected the handshake or did not start.</exception>
    public static async Task<WorkerDriver> StartAsync(RoostModelProfile profile, RoostWorkerOptions worker, CancellationToken cancellationToken, Func<IDriverTransport, IDriverTransport>? decorate = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(worker.MinFreeDeviceBytes);
        if (worker.Sampler == RoostSampler.Cpu && worker.Transport == TransportKind.Process && worker.Command is null)
        {
            throw new ArgumentException("The packaged roost-worker is the CUDA worker, which has no CPU sampler; RoostSampler.Cpu needs an in-process executor or a custom Command that runs one.", nameof(worker));
        }

        if (worker.ControlTimeout <= TimeSpan.Zero)
        {
            throw new ArgumentException("ControlTimeout must be positive.", nameof(worker));
        }

        if (worker.AsyncScheduling && worker.Transport == TransportKind.Process && worker.Command is null)
        {
            throw new ArgumentException("The packaged roost-worker is the CUDA worker, which does not patch sampled inputs yet (stream F, R-032); AsyncScheduling needs an in-process executor or a custom Command whose worker does.", nameof(worker));
        }

        var command = worker.Transport switch
        {
            TransportKind.Process => worker.Command ?? RoostWorkerRuntime.CommandFor(profile, worker),
            TransportKind.InProcess => worker.InProcessExecutor is null ? throw new ArgumentException("The in-process transport needs RoostWorkerOptions.InProcessExecutor.", nameof(worker)) : null,
            _ => throw new ArgumentOutOfRangeException(nameof(worker), worker.Transport, "Unknown transport kind."),
        };
        IDriverTransport Create()
        {
            WorkerTransportBase inner = command is not null
                ? new ProcessWorkerTransport(command, worker.TransportOptions)
                : new InProcessWorkerTransport(worker.InProcessExecutor!, worker.TransportOptions);
            var transport = new DriverTransport(inner);
            return decorate?.Invoke(transport) ?? transport;
        }

        var driver = new WorkerDriver(profile, worker, Create);
        try
        {
            await driver.StartWorkerAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            await driver._transport.DisposeAsync().ConfigureAwait(false);
            driver._wake.Dispose();
            throw;
        }

        driver._thread.Start();
        return driver;
    }

    /// <inheritdoc />
    public IRequestChannel Submit(Request request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var live = new Live(request.Id, request.PromptTokenIds.Length);
        if (!TryEnqueue(new Command(request, live, 0)))
        {
            live.Fail(new RoostError(RoostErrorCode.Internal, "The chat client is disposed."));
        }

        return new RequestChannel(request.Id, live.Channel.Reader, id => TryEnqueue(new Command(null, null, id.Value)));
    }

    /// <inheritdoc />
    public double? KvCacheUsage => Volatile.Read(ref _kvCacheUsage);

    /// <inheritdoc />
    public Task<RoostWorkerControlResult> ControlAsync(RoostWorkerOperation operation, RoostSleepLevel level, WeightSource? source, CancellationToken cancellationToken)
    {
        if (operation == RoostWorkerOperation.UpdateWeights)
        {
            ArgumentNullException.ThrowIfNull(source);
            source.Validate();
        }

        if (operation == RoostWorkerOperation.Sleep && level is not (RoostSleepLevel.KvCache or RoostSleepLevel.KvCacheAndWeights))
        {
            throw new ArgumentOutOfRangeException(nameof(level), level, "Unknown sleep level.");
        }

        var control = new Control(operation, level, source, cancellationToken);
        if (!TryEnqueue(new Command(null, null, 0, control)))
        {
            control.Fail(new RoostError(RoostErrorCode.Internal, "The chat client is disposed."));
        }

        return control.Task;
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _stopping = true;
        Wake();
        if (_thread.IsAlive)
        {
            await _exited.Task.ConfigureAwait(false);
        }

        try
        {
            await _transport.DisposeAsync().ConfigureAwait(false);
        }
        finally
        {
            _wake.Dispose();
        }
    }

    private static StepPlanner CreatePlanner(RoostModelProfile profile, bool asyncScheduling, out BlockManager blocks)
    {
        var groups = profile.StateGroups;
        var capacities = groups
            .Select(g => new GroupCapacity(g.Index, g.Kind == StateGroupKind.Recurrent ? profile.RecurrentSlots : profile.AttentionBlocks, 0))
            .ToImmutableArray();
        blocks = new BlockManager(groups, capacities);
        return new StepPlanner(blocks, new PlannerOptions
        {
            MaxTokensPerStep = profile.MaxTokensPerStep,
            MaxSequencesPerStep = profile.MaxSequencesPerStep,
            ModelStopTokenIds = profile.StopTokenIds,
            AsyncScheduling = asyncScheduling,
        });
    }

    private static string Describe(Exception e) => e.GetType().Name + ": " + e.Message;

    private bool TryEnqueue(Command command)
    {
        lock (_gate)
        {
            if (_stopping)
            {
                return false;
            }

            _commands.Enqueue(command);
        }

        Wake();
        return true;
    }

    private void Wake()
    {
        try
        {
            _ = _wake.Release();
        }
        catch (ObjectDisposedException)
        {
            // Disposed: the driver thread is gone and every channel is complete.
        }
    }

    private async Task StartWorkerAsync(CancellationToken cancellationToken)
    {
        HandshakeResult result;
        try
        {
            result = await _transport.StartAsync(_launch, cancellationToken).ConfigureAwait(false);
        }
        catch (WorkerStartException e)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.Internal, "The worker did not start: " + e.Message), e);
        }

        if (result.State != HandshakeState.Ready)
        {
            var code = result.Reject switch
            {
                HandshakeReject.ManifestDigest or HandshakeReject.WeightsDigest or HandshakeReject.Abi or HandshakeReject.Contract or HandshakeReject.Layout => RoostErrorCode.AssetMismatch,
                HandshakeReject.Device or HandshakeReject.Mesh => RoostErrorCode.ProfileUnsupported,
                _ => RoostErrorCode.Internal,
            };
            throw new RoostChatException(new RoostError(code, $"The worker rejected the handshake: {result.Reject}."));
        }
    }

    private void Run()
    {
        try
        {
            while (!_stopping)
            {
                try
                {
                    DrainCommands();
                    PublishKvCacheUsage();

                    // Control messages go only to a worker with no step in flight: finish the published steps first.
                    if (_controls.Count > 0)
                    {
                        if (_published.Count > 0)
                        {
                            AwaitOldest();
                        }
                        else
                        {
                            RunControl(_controls.Dequeue());
                        }

                        continue;
                    }

                    // Plan while a plan buffer is free: always with nothing in flight (a full ring is then a broken worker,
                    // which Publish reports), and with async scheduling once more while one step runs.
                    var depth = _async ? 2 : 1;
                    if (_live.Count > 0 && _published.Count < depth && (_published.Count == 0 || _transport.CanPublish()))
                    {
                        var step = NextPlan();
                        if (step is not null)
                        {
                            Publish(step);
                            continue;
                        }
                    }

                    if (_published.Count > 0)
                    {
                        AwaitOldest();
                        continue;
                    }

                    _ = _wake.Wait(_live.Count > 0 ? 10 : Timeout.Infinite);
                }
                catch (Exception e)
                {
                    // Containment: see the class remarks. Nothing may escape this thread.
                    LoseWorker(new RoostError(RoostErrorCode.Internal, $"The chat driver failed ({Describe(e)}); the requests in flight were failed and the worker is restarted."));
                }
            }
        }
        finally
        {
            var disposed = new RoostError(RoostErrorCode.Internal, "The chat client is disposed.");
            lock (_gate)
            {
                _stopping = true;
                while (_commands.TryDequeue(out var c))
                {
                    c.Live?.Fail(disposed);
                    c.Control?.Fail(disposed);
                }
            }

            while (_controls.TryDequeue(out var control))
            {
                control.Fail(disposed);
            }

            FailAll(disposed);
            _exited.TrySetResult();
        }
    }

    private void DrainCommands()
    {
        while (true)
        {
            Command command;
            lock (_gate)
            {
                if (!_commands.TryDequeue(out command))
                {
                    return;
                }
            }

            if (command.Control is not null)
            {
                _controls.Enqueue(command.Control);
                continue;
            }

            if (command.Request is null)
            {
                if (_live.Remove(command.CancelId, out var cancelled))
                {
                    _ = _planner.Cancel(command.CancelId);
                    cancelled.Complete();
                }

                continue;
            }

            var live = command.Live!;
            if (live.Channel.Reader.Completion.IsCompleted)
            {
                continue;
            }

            if (_asleep)
            {
                live.Fail(new RoostError(RoostErrorCode.InsufficientDeviceMemory, "The worker is asleep and has released its memory; wake it (WakeUpAsync, POST /wake_up) and retry."));
                continue;
            }

            if (!FloorHolds(out var refusal))
            {
                live.Fail(refusal!);
                continue;
            }

            if (!EnsureWorker(out var unavailable))
            {
                live.Fail(unavailable!);
                continue;
            }

            AddResult added;
            try
            {
                added = _planner.Add(command.Request);
            }
            catch (Exception e)
            {
                live.Fail(new RoostError(RoostErrorCode.Internal, "The planner refused the request: " + Describe(e)));
                continue;
            }

            if (added == AddResult.Added)
            {
                _live.Add(command.Request.Id.Value, live);
                continue;
            }

            live.Fail(added switch
            {
                AddResult.PromptTooLong => new RoostError(RoostErrorCode.PromptTooLong, $"The prompt does not fit the worker's {_profile.ContextTokens}-token context.", "messages"),
                AddResult.Unsupported => new RoostError(RoostErrorCode.ProfileUnsupported, "The request uses a feature the planner does not support yet."),
                AddResult.DuplicateId => new RoostError(RoostErrorCode.Internal, "Duplicate request id."),
                _ => new RoostError(RoostErrorCode.InvalidOption, "The prompt is empty or the output limit is not positive.", "messages"),
            });
        }
    }

    // Co-tenant admission (R-044): a cached, non-blocking read of the device's free memory. Below the floor a new request is
    // refused before it reaches the planner or the worker; running requests are untouched. An unreadable source decides nothing.
    private bool FloorHolds(out RoostError? error)
    {
        error = null;
        if (_memory is null || _minFreeDeviceBytes <= 0)
        {
            return true;
        }

        DeviceMemory memory;
        try
        {
            if (!_memory.TryRead(out memory) || memory.FreeBytes >= _minFreeDeviceBytes)
            {
                return true;
            }
        }
        catch (Exception)
        {
            return true; // a source must not throw; if one does, the pools decide, as when it cannot be read
        }

        error = new RoostError(
            RoostErrorCode.InsufficientDeviceMemory,
            string.Create(CultureInfo.InvariantCulture, $"{memory.FreeBytes} bytes of device memory are free; at least {_minFreeDeviceBytes} are needed to admit a request. Retry later."));
        return false;
    }

    private bool EnsureWorker(out RoostError? error)
    {
        error = null;
        try
        {
            if (_transport.State == TransportState.Running)
            {
                return true;
            }

            if (_transport.State != TransportState.Idle)
            {
                ReplaceTransport(); // abandoned (an in-process worker stuck in a step) or otherwise unusable
            }

            StartWorkerAsync(CancellationToken.None).GetAwaiter().GetResult();
            return ReapplyReplacement(out error);
        }
        catch (RoostChatException e)
        {
            error = e.Error;
            return false;
        }
        catch (Exception e)
        {
            error = new RoostError(RoostErrorCode.WorkerRestarted, "The worker could not be started: " + Describe(e));
            return false;
        }
    }

    // A new worker loads the weights its command line or executor names; after an update it must serve the replacement.
    private bool ReapplyReplacement(out RoostError? error)
    {
        error = null;
        if (_replacement is null)
        {
            return true;
        }

        ControlReply reply;
        try
        {
            reply = Exchange(RingSlotKind.ReplaceWeights, SleepLevel.None, _replacement);
        }
        catch (Exception e)
        {
            error = new RoostError(RoostErrorCode.WorkerRestarted, "The restarted worker did not take the updated weights: " + Describe(e));
            LoseWorker(error);
            return false;
        }

        if (reply.Status != ControlStatus.Ok)
        {
            error = ControlError(RoostWorkerOperation.UpdateWeights, reply);
            LoseWorker(error);
            return false;
        }

        return true;
    }

    private void RunControl(Control control)
    {
        if (control.Task.IsCompleted)
        {
            return; // cancelled before it was sent
        }

        var operation = control.Operation;
        string? refusal = _live.Count > 0
            ? $"{operation} needs a worker with no requests running; {_live.Count} are."
            : operation switch
            {
                RoostWorkerOperation.Sleep when _asleep => "The worker is already asleep.",
                RoostWorkerOperation.WakeUp when !_asleep => "The worker is not asleep.",
                _ => null,
            };
        if (refusal is not null)
        {
            control.Fail(new RoostError(RoostErrorCode.InvalidOption, refusal));
            return;
        }

        if (!_asleep && !EnsureWorker(out var unavailable))
        {
            control.Fail(unavailable!);
            return;
        }

        var started = Stopwatch.GetTimestamp();
        ControlReply reply;
        try
        {
            reply = operation switch
            {
                RoostWorkerOperation.Sleep => Exchange(RingSlotKind.Sleep, control.Level == RoostSleepLevel.KvCacheAndWeights ? SleepLevel.Weights : SleepLevel.Kv, null),
                RoostWorkerOperation.WakeUp => Exchange(RingSlotKind.Wake, SleepLevel.None, null),
                _ => Exchange(RingSlotKind.ReplaceWeights, SleepLevel.None, control.Source),
            };
        }
        catch (Exception e)
        {
            var lost = new RoostError(RoostErrorCode.WorkerRestarted, $"The worker did not answer {operation} ({Describe(e)}); it was killed and the next request starts a new one.");
            LoseWorker(lost);
            control.Fail(lost);
            return;
        }

        var duration = Stopwatch.GetElapsedTime(started);
        if (reply.Status != ControlStatus.Ok)
        {
            control.Fail(ControlError(operation, reply));
            return;
        }

        switch (operation)
        {
            case RoostWorkerOperation.Sleep:
                _asleep = true;
                _planner = CreatePlanner(_profile, _async, out var released); // no block the planner remembers may outlive the released state
                _blocks = released;
                break;
            case RoostWorkerOperation.WakeUp:
                _asleep = false;
                Interlocked.Exchange(ref _lastWakeTicks, duration.Ticks);
                break;
            default:
                _replacement = control.Source;
                _weightsDigest = control.Source!.Sha256;
                break;
        }

        control.Complete(new RoostWorkerControlResult(operation, duration, reply.WorkerDuration, reply.Bytes, _asleep, reply.WeightsDigest));
    }

    private static RoostError ControlError(RoostWorkerOperation operation, ControlReply reply)
    {
        var code = reply.Status switch
        {
            ControlStatus.RefusedState => RoostErrorCode.InvalidOption,
            ControlStatus.WeightsDigest => RoostErrorCode.AssetMismatch,
            ControlStatus.Unsupported => RoostErrorCode.ProfileUnsupported,
            _ => RoostErrorCode.Internal,
        };
        var detail = string.IsNullOrEmpty(reply.ErrorText) ? reply.Status.ToString() : reply.ErrorText;
        return new RoostError(code, $"The worker refused {operation} ({reply.Status}): {detail}");
    }

    /// <summary>Sends one control message to a worker with nothing in flight and waits for its answer. Throws when the worker is lost or does not answer in time.</summary>
    private ControlReply Exchange(RingSlotKind kind, SleepLevel level, WeightSource? source)
    {
        if (!_transport.CanPublish())
        {
            throw new InvalidOperationException("No plan buffer is free with no step in flight.");
        }

        int index = _transport.NextPlanBufferIndex;
        var buffer = _transport.PlanBuffer(index);
        ulong id = ControlIdBit | ++_controlIds;
        _ = kind switch
        {
            RingSlotKind.Sleep => ControlMessages.WriteSleep(buffer, id, level),
            RingSlotKind.Wake => ControlMessages.WriteWake(buffer, id, _weightsDigest.AsSpan()),
            _ => ControlMessages.WriteReplaceWeights(buffer, id, source!),
        };
        if (!_transport.TryPublish(new RingSlot { StepId = id, Kind = kind, BufferIndex = (uint)index }))
        {
            throw new InvalidOperationException("The plan ring is full with no step in flight.");
        }

        var give = Stopwatch.GetTimestamp() + (long)(_controlTimeout.TotalSeconds * Stopwatch.Frequency);
        var spin = default(SpinWait);
        while (true)
        {
            if (_transport.TryTakeResult(out var slot))
            {
                try
                {
                    if (slot.Kind != kind || slot.StepId != id)
                    {
                        throw new InvalidDataException($"Expected the answer to control message {id} ({kind}); the worker published {slot.Kind} {slot.StepId}.");
                    }

                    return ControlMessages.ReadResult(_transport.ResultBuffer((int)slot.BufferIndex), slot);
                }
                finally
                {
                    _transport.ReleaseResult((int)slot.BufferIndex);
                }
            }

            var fault = _transport.Poll();
            if (fault != FaultClass.None)
            {
                throw new InvalidOperationException($"The worker was lost ({fault}).");
            }

            if (Stopwatch.GetTimestamp() > give)
            {
                throw new TimeoutException($"No answer within {_controlTimeout}.");
            }

            spin.SpinOnce();
        }
    }

    private PlannedStep? NextPlan()
    {
        var step = _planner.Plan();
        foreach (var notice in _planner.DrainNotices())
        {
            if (notice.Kind == PlannerNoticeKind.Aborted && _live.Remove(notice.Handle, out var live))
            {
                // The planner aborts a sequence that cannot get its blocks even after preempting every later one. The client caps
                // each request's output at the context left, so a lone request always fits; this happens only when the pools are
                // exhausted by other live requests. That is a shortage of device memory (C6 InsufficientDeviceMemory, 503), not a
                // fault in the request (PromptTooLong, 400) and not an admission refusal (QueueFull: the request had been admitted).
                live.Fail(new RoostError(RoostErrorCode.InsufficientDeviceMemory, "The worker's KV and state pools were exhausted by the requests running at the same time; retry later."));
            }
        }

        return step;
    }

    private void Publish(PlannedStep step)
    {
        if (!_transport.CanPublish())
        {
            // With no step in flight every plan buffer should be free: the host's and the worker's view of the rings disagree.
            // The worker cannot be trusted with the next step, so replace it.
            LoseWorker(new RoostError(RoostErrorCode.WorkerRestarted, "The plan ring is out of step with the worker (no plan buffer free with no step in flight); the worker was killed and the request cannot be resumed (R-070)."));
            return;
        }

        try
        {
            int index = _transport.NextPlanBufferIndex;
            _ = PlanBuffer.Write(_transport.PlanBuffer(index), step);
            if (!_transport.TryPublish(new RingSlot { StepId = step.StepId, Kind = RingSlotKind.Step, BufferIndex = (uint)index }))
            {
                throw new InvalidOperationException("The plan ring is full with no step in flight.");
            }
        }
        catch (Exception e)
        {
            // Nothing reached the worker: fail this step's requests and keep the worker.
            FailStep(step, new RoostError(RoostErrorCode.Internal, "The step could not be published: " + Describe(e)));
            return;
        }

        _published.Enqueue(step);
    }

    /// <summary>Waits for the oldest published step's result and applies it (results come back in step order).</summary>
    private void AwaitOldest()
    {
        var step = _published.Peek();
        StepOutcome? outcome = null;
        IReadOnlyDictionary<ulong, TokenLogprob[]>? logprobs = null;
        StepResultHeader header = default;
        try
        {
            var spin = default(SpinWait);
            while (true)
            {
                if (_transport.TryTakeResult(out var slot))
                {
                    try
                    {
                        if (slot.Kind != RingSlotKind.Step)
                        {
                            continue;
                        }

                        var buffer = _transport.ResultBuffer((int)slot.BufferIndex);
                        header = PlanBuffer.ReadHeader(buffer);
                        if (header.Fault == FaultClass.None)
                        {
                            outcome = PlanBuffer.ReadResult(buffer, step.StepId);
                            logprobs = PlanBuffer.ReadLogprobs(buffer, header);
                        }
                    }
                    finally
                    {
                        _transport.ReleaseResult((int)slot.BufferIndex);
                    }

                    _ = _published.Dequeue();
                    break;
                }

                var fault = _transport.Poll();
                if (fault != FaultClass.None)
                {
                    LoseWorker(new RoostError(RoostErrorCode.WorkerRestarted, $"The worker was lost ({fault}); it was killed and the request cannot be resumed (R-070)."));
                    return;
                }

                spin.SpinOnce();
            }
        }
        catch (Exception e)
        {
            // The worker's answer cannot be trusted (malformed result, a transport that throws): discard the worker.
            LoseWorker(new RoostError(RoostErrorCode.Internal, $"The step result could not be read ({Describe(e)}); the worker was killed and the request cannot be resumed (R-070)."));
            return;
        }

        if (outcome is not null)
        {
            Apply(outcome, logprobs);
        }
        else if (header.Fault == FaultClass.Recoverable)
        {
            FailStep(step, RecoverableError(header.ErrorCode));
        }
        else
        {
            LoseWorker(header.Fault == FaultClass.Sticky
                ? new RoostError(RoostErrorCode.DeviceFault, string.Create(CultureInfo.InvariantCulture, $"The GPU faulted (sticky, status {header.ErrorCode}); the worker was killed and the request cannot be resumed (R-070)."))
                : new RoostError(RoostErrorCode.WorkerRestarted, $"The worker reported {header.Fault}; it was killed and the request cannot be resumed (R-070)."));
        }
    }

    private static RoostError RecoverableError(int errorCode) => errorCode switch
    {
        // The eager worker reports a plan it refused as a recoverable fault with its PlanRejection (1000-1003): the profile's
        // capacities then disagree with the worker's (the C8 handshake does not carry them).
        >= 1000 and <= 1003 => new RoostError(
            RoostErrorCode.DeviceFault,
            string.Create(CultureInfo.InvariantCulture, $"The worker refused the step plan (rejection {errorCode}); the profile's pool and step capacities may not match the worker's.")),
        _ => new RoostError(RoostErrorCode.DeviceFault, string.Create(CultureInfo.InvariantCulture, $"The worker failed the step (recoverable fault, status {errorCode}); the worker is kept.")),
    };

    private void Apply(StepOutcome outcome, IReadOnlyDictionary<ulong, TokenLogprob[]>? logprobs)
    {
        var finished = _planner.ApplyResult(outcome);
        foreach (var row in outcome.Sequences)
        {
            if (!row.Tokens.IsDefaultOrEmpty && _live.TryGetValue(row.Handle, out var live))
            {
                live.Tokens(row.Tokens, logprobs is not null && logprobs.TryGetValue(row.Handle, out var lp) ? lp : null);
            }
        }

        foreach (var f in finished)
        {
            if (_live.Remove(f.Handle, out var live))
            {
                live.Finish(f.Reason);
            }
        }
    }

    private void FailStep(PlannedStep step, RoostError error)
    {
        foreach (var s in step.Sequences)
        {
            if (_live.Remove(s.SequenceHandle, out var live))
            {
                live.Fail(error);
            }
        }

        try
        {
            var rows = step.Sequences.Select(static s => new SequenceResult(s.SequenceHandle, [], FinishReason.Error)).ToImmutableArray();
            _ = _planner.ApplyResult(new StepOutcome(step.StepId, rows));
        }
        catch (Exception e)
        {
            LoseWorker(new RoostError(RoostErrorCode.Internal, "The planner could not drop a failed step: " + Describe(e)));
        }
    }

    /// <summary>Fails every live request, discards the planner state and the worker. Never throws.</summary>
    private void LoseWorker(RoostError error)
    {
        FailAll(error);
        _published.Clear(); // the worker is killed: none of them returns
        _asleep = false; // the next worker starts awake
        try
        {
            _planner = CreatePlanner(_profile, _async, out var blocks); // every sequence it held has just been failed
            _blocks = blocks;
        }
        catch (Exception)
        {
            // CreatePlanner succeeded at construction with the same profile; keep the old planner if it ever does not.
        }

        try
        {
            _transport.KillAsync(CancellationToken.None).AsTask().GetAwaiter().GetResult();
            if (_transport.State is not (TransportState.Idle or TransportState.Running))
            {
                ReplaceTransport();
            }
        }
        catch (Exception)
        {
            ReplaceTransport();
        }
    }

    /// <summary>Abandons a transport that cannot be killed or restarted and makes a fresh one. Never throws.</summary>
    private void ReplaceTransport()
    {
        var old = _transport;
        try
        {
            _transport = _createTransport();
        }
        catch (Exception)
        {
            return; // keep the old one; EnsureWorker reports the failure to the next request
        }

        try
        {
            old.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
        catch (Exception)
        {
            // Best effort: a process worker watches its parent and ends itself; the segment file is the transport's.
        }
    }

    // R-038: the share of attention (non-recurrent) blocks held, read on the driver thread after each plan and published for the
    // metrics gauge. Recurrent slots are one per sequence and are not KV cache.
    private void PublishKvCacheUsage()
    {
        long used = 0;
        long capacity = 0;
        foreach (var group in _blocks.Groups)
        {
            if (group.Kind == StateGroupKind.Recurrent)
            {
                continue;
            }

            var pool = _blocks.GetPool(group.Index);
            used += pool.Capacity - pool.FreeCount;
            capacity += pool.Capacity;
        }

        Volatile.Write(ref _kvCacheUsage, capacity == 0 ? 0 : (double)used / capacity);
    }

    private void FailAll(RoostError error)
    {
        foreach (var live in _live.Values)
        {
            live.Fail(error);
        }

        _live.Clear();
    }

    private readonly record struct Command(Request? Request, Live? Live, ulong CancelId, Control? Control = null);

    /// <summary>A queued control operation and the task its caller awaits.</summary>
    private sealed class Control
    {
        private readonly TaskCompletionSource<RoostWorkerControlResult> _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly CancellationTokenRegistration _registration;

        public Control(RoostWorkerOperation operation, RoostSleepLevel level, WeightSource? source, CancellationToken cancellationToken)
        {
            Operation = operation;
            Level = level;
            Source = source;
            if (cancellationToken.CanBeCanceled)
            {
                // Cancels the wait; an operation already sent to the worker still completes there.
                _registration = cancellationToken.Register(() => _completion.TrySetCanceled(cancellationToken));
            }
        }

        public RoostWorkerOperation Operation { get; }

        public RoostSleepLevel Level { get; }

        public WeightSource? Source { get; }

        public Task<RoostWorkerControlResult> Task => _completion.Task;

        public void Complete(RoostWorkerControlResult result)
        {
            _registration.Dispose();
            _ = _completion.TrySetResult(result);
        }

        public void Fail(RoostError error)
        {
            _registration.Dispose();
            _ = _completion.TrySetException(new RoostChatException(error));
        }
    }

    private sealed class Live(RequestId id, int promptTokens)
    {
        private int _completion;

        public Channel<TokenEvent> Channel { get; } = System.Threading.Channels.Channel.CreateUnbounded<TokenEvent>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

        public void Tokens(ImmutableArray<int> tokens, TokenLogprob[]? logprobs)
        {
            _completion += tokens.Length;
            _ = Channel.Writer.TryWrite(new TokenEvent
            {
                RequestId = id,
                Kind = TokenEventKind.Tokens,
                TokenIds = tokens.ToArray(),
                Logprobs = logprobs,
                Usage = (promptTokens, _completion),
            });
        }

        public void Finish(FinishReason reason)
        {
            _ = Channel.Writer.TryWrite(new TokenEvent { RequestId = id, Kind = TokenEventKind.Finished, Finish = reason, Usage = (promptTokens, _completion) });
            _ = Channel.Writer.TryComplete();
        }

        public void Fail(RoostError error)
        {
            _ = Channel.Writer.TryWrite(new TokenEvent { RequestId = id, Kind = TokenEventKind.Failed, Error = error, Usage = (promptTokens, _completion) });
            _ = Channel.Writer.TryComplete();
        }

        public void Complete() => _ = Channel.Writer.TryComplete();
    }
}

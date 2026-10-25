using System.Collections.Immutable;
using System.Globalization;
using System.Threading.Channels;
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
/// time, as the eager worker runs synchronously (R-025); async scheduling (R-032) and the engine's own loop replace this
/// class from M1.
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
/// <para><b>Containment.</b> The driver thread never ends while the client is alive: an unexpected exception anywhere in a
/// step (planner, plan writer, transport, a malformed result, a kill that fails) fails the requests it affects with
/// <see cref="RoostErrorCode.Internal"/>, discards the planner state and the worker (replacing the transport when it cannot
/// be killed), and the loop goes on. An exception escaping a dedicated thread would end the consumer's process, which is
/// what the worker process exists to prevent (decision 21).</para>
/// </remarks>
internal sealed class WorkerDriver : IGenerationBackend
{
    private const int MinResultBufferBytes = 64 * 1024;

    private readonly RoostModelProfile _profile;
    private readonly WorkerLaunch _launch;
    private readonly Func<IDriverTransport> _createTransport;
    private readonly object _gate = new();
    private readonly Queue<Command> _commands = new();
    private readonly SemaphoreSlim _wake = new(0);
    private readonly Dictionary<ulong, Live> _live = [];
    private readonly Thread _thread;
    private readonly TaskCompletionSource _exited = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private IDriverTransport _transport;
    private StepPlanner _planner;
    private volatile bool _stopping;
    private int _disposed;

    private WorkerDriver(RoostModelProfile profile, RoostWorkerOptions worker, Func<IDriverTransport> createTransport)
    {
        _profile = profile;
        _createTransport = createTransport;
        _transport = createTransport();
        _planner = CreatePlanner(profile);
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
            ResultBufferBytes = Math.Max(MinResultBufferBytes, 4096 + (profile.MaxSequencesPerStep * 64L)),
        };
        _thread = new Thread(Run) { Name = "roost-chat-driver", IsBackground = true };
    }

    /// <summary>The offer made to the worker (tests read the buffer sizes).</summary>
    public WorkerLaunch Launch => _launch;

    /// <summary>The current transport (tests read its state and kind). Replaced when a worker cannot be killed.</summary>
    public IDriverTransport Transport => _transport;

    /// <summary>Creates the transport, starts the worker and the driver thread.</summary>
    /// <param name="profile">The model profile, validated.</param>
    /// <param name="worker">How the worker is started.</param>
    /// <param name="cancellationToken">Cancels the start.</param>
    /// <param name="decorate">Test hook: wraps each transport the driver creates.</param>
    /// <returns>The running driver.</returns>
    /// <exception cref="RoostChatException">The worker rejected the handshake or did not start.</exception>
    public static async Task<WorkerDriver> StartAsync(RoostModelProfile profile, RoostWorkerOptions worker, CancellationToken cancellationToken, Func<IDriverTransport, IDriverTransport>? decorate = null)
    {
        var command = worker.Transport switch
        {
            TransportKind.Process => worker.Command ?? throw new ArgumentException("The process transport (the default, decision 21) needs RoostWorkerOptions.Command.", nameof(worker)),
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

    private static StepPlanner CreatePlanner(RoostModelProfile profile)
    {
        var groups = profile.StateGroups;
        var capacities = groups
            .Select(g => new GroupCapacity(g.Index, g.Kind == StateGroupKind.Recurrent ? profile.RecurrentSlots : profile.AttentionBlocks, 0))
            .ToImmutableArray();
        return new StepPlanner(new BlockManager(groups, capacities), new PlannerOptions
        {
            MaxTokensPerStep = profile.MaxTokensPerStep,
            MaxSequencesPerStep = profile.MaxSequencesPerStep,
            ModelStopTokenIds = profile.StopTokenIds,
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
                    var step = _live.Count > 0 ? NextPlan() : null;
                    if (step is null)
                    {
                        _ = _wake.Wait(_live.Count > 0 ? 10 : Timeout.Infinite);
                        continue;
                    }

                    RunStep(step);
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
                }
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
            return true;
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

    private void RunStep(PlannedStep step)
    {
        try
        {
            if (!_transport.CanPublish())
            {
                throw new InvalidOperationException("No plan buffer is free with no step in flight.");
            }

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

        StepOutcome? outcome = null;
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
                        }
                    }
                    finally
                    {
                        _transport.ReleaseResult((int)slot.BufferIndex);
                    }

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
            Apply(outcome);
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

    private void Apply(StepOutcome outcome)
    {
        var finished = _planner.ApplyResult(outcome);
        foreach (var row in outcome.Sequences)
        {
            if (!row.Tokens.IsDefaultOrEmpty && _live.TryGetValue(row.Handle, out var live))
            {
                live.Tokens(row.Tokens);
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
        try
        {
            _planner = CreatePlanner(_profile); // every sequence it held has just been failed
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

    private void FailAll(RoostError error)
    {
        foreach (var live in _live.Values)
        {
            live.Fail(error);
        }

        _live.Clear();
    }

    private readonly record struct Command(Request? Request, Live? Live, ulong CancelId);

    private sealed class Live(RequestId id, int promptTokens)
    {
        private int _completion;

        public Channel<TokenEvent> Channel { get; } = System.Threading.Channels.Channel.CreateUnbounded<TokenEvent>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

        public void Tokens(ImmutableArray<int> tokens)
        {
            _completion += tokens.Length;
            _ = Channel.Writer.TryWrite(new TokenEvent
            {
                RequestId = id,
                Kind = TokenEventKind.Tokens,
                TokenIds = tokens.ToArray(),
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

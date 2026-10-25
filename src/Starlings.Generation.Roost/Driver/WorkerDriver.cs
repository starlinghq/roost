using System.Collections.Concurrent;
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
/// command. Everything that touches the planner or the transport runs on the driver thread (the transport is single
/// host thread by contract).</para>
/// <para><b>Faults.</b> A recoverable step fault fails the requests of that step with
/// <see cref="RoostErrorCode.DeviceFault"/> and keeps the worker. A sticky fault, a hang, a stall or an exited worker fails
/// every live request (<see cref="RoostErrorCode.DeviceFault"/> for sticky, <see cref="RoostErrorCode.WorkerRestarted"/>
/// otherwise), kills the worker, and the next request starts a new one. Resuming requests on the new worker with
/// <c>Recovered</c> set is R-070.</para>
/// </remarks>
internal sealed class WorkerDriver : IGenerationBackend
{
    private const int MinResultBufferBytes = 64 * 1024;

    private readonly RoostModelProfile _profile;
    private readonly WorkerLaunch _launch;
    private readonly WorkerTransportBase _transport;
    private readonly StepPlanner _planner;
    private readonly ConcurrentQueue<Command> _commands = new();
    private readonly SemaphoreSlim _wake = new(0);
    private readonly Dictionary<ulong, Live> _live = [];
    private readonly Thread _thread;
    private readonly TaskCompletionSource _exited = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private volatile bool _stopping;
    private int _disposed;

    private WorkerDriver(RoostModelProfile profile, RoostWorkerOptions worker, WorkerTransportBase transport)
    {
        _profile = profile;
        _transport = transport;
        var groups = profile.StateGroups;
        var capacities = groups
            .Select(g => new GroupCapacity(g.Index, g.Kind == StateGroupKind.Recurrent ? profile.RecurrentSlots : profile.AttentionBlocks, 0))
            .ToImmutableArray();
        _planner = new StepPlanner(new BlockManager(groups, capacities), new PlannerOptions
        {
            MaxTokensPerStep = profile.MaxTokensPerStep,
            MaxSequencesPerStep = profile.MaxSequencesPerStep,
            ModelStopTokenIds = profile.StopTokenIds,
        });
        long planBytes = PlanBuffer.SizeFor(profile.MaxSequencesPerStep, profile.MaxTokensPerStep, groups.Length, profile.AttentionBlocks, profile.StopTokenIds.Length);
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

    /// <summary>The transport (tests read its state and kind).</summary>
    public WorkerTransportBase Transport => _transport;

    /// <summary>Creates the transport, starts the worker and the driver thread.</summary>
    /// <param name="profile">The model profile, validated.</param>
    /// <param name="worker">How the worker is started.</param>
    /// <param name="cancellationToken">Cancels the start.</param>
    /// <returns>The running driver.</returns>
    /// <exception cref="RoostChatException">The worker rejected the handshake or did not start.</exception>
    public static async Task<WorkerDriver> StartAsync(RoostModelProfile profile, RoostWorkerOptions worker, CancellationToken cancellationToken)
    {
        WorkerTransportBase transport = worker.Transport switch
        {
            TransportKind.Process => new ProcessWorkerTransport(
                worker.Command ?? throw new ArgumentException("The process transport (the default, decision 21) needs RoostWorkerOptions.Command.", nameof(worker)),
                worker.TransportOptions),
            TransportKind.InProcess => new InProcessWorkerTransport(
                worker.InProcessExecutor ?? throw new ArgumentException("The in-process transport needs RoostWorkerOptions.InProcessExecutor.", nameof(worker)),
                worker.TransportOptions),
            _ => throw new ArgumentOutOfRangeException(nameof(worker), worker.Transport, "Unknown transport kind."),
        };
        var driver = new WorkerDriver(profile, worker, transport);
        try
        {
            await driver.StartWorkerAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            await transport.DisposeAsync().ConfigureAwait(false);
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
        if (_stopping)
        {
            live.Fail(new RoostError(RoostErrorCode.Internal, "The chat client is disposed."));
        }
        else
        {
            _commands.Enqueue(new Command(request, live, 0));
            Wake();
        }

        return new RequestChannel(request.Id, live.Channel.Reader, id =>
        {
            _commands.Enqueue(new Command(null, null, id.Value));
            Wake();
        });
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

        await _transport.DisposeAsync().ConfigureAwait(false);
        _wake.Dispose();
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
        var reason = new RoostError(RoostErrorCode.Internal, "The chat client is disposed.");
        try
        {
            while (!_stopping)
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
        }
        catch (Exception e) when (e is InvalidOperationException or InvalidDataException or ArgumentException or RoostChatException or IOException)
        {
            reason = new RoostError(RoostErrorCode.Internal, "The chat driver failed: " + e.Message);
        }
        finally
        {
            _stopping = true;
            FailAll(reason);
            while (_commands.TryDequeue(out var c))
            {
                c.Live?.Fail(reason);
            }

            _exited.TrySetResult();
        }
    }

    private void DrainCommands()
    {
        while (_commands.TryDequeue(out var command))
        {
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

            var added = _planner.Add(command.Request);
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
        switch (_transport.State)
        {
            case TransportState.Running:
                return true;
            case TransportState.Idle:
                try
                {
                    StartWorkerAsync(CancellationToken.None).GetAwaiter().GetResult();
                    return true;
                }
                catch (RoostChatException e)
                {
                    error = e.Error;
                    return false;
                }

            default:
                error = new RoostError(RoostErrorCode.WorkerRestarted, $"The worker is {_transport.State} and cannot be restarted in this process.");
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
                live.Fail(new RoostError(RoostErrorCode.PromptTooLong, "The sequence no longer fits the worker's pools.", "messages"));
            }
        }

        return step;
    }

    private void RunStep(PlannedStep step)
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

        var spin = default(SpinWait);
        while (true)
        {
            if (_transport.TryTakeResult(out var slot))
            {
                if (slot.Kind != RingSlotKind.Step)
                {
                    continue;
                }

                StepResultHeader header;
                StepOutcome? outcome = null;
                try
                {
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

                if (outcome is not null)
                {
                    Apply(outcome);
                }
                else if (header.Fault == FaultClass.Recoverable)
                {
                    FailStep(step, header.ErrorCode);
                }
                else
                {
                    LoseWorker(header.Fault, header.ErrorCode);
                }

                return;
            }

            var fault = _transport.Poll();
            if (fault != FaultClass.None)
            {
                LoseWorker(fault, 0);
                return;
            }

            spin.SpinOnce();
        }
    }

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

    private void FailStep(PlannedStep step, int errorCode)
    {
        var rows = step.Sequences.Select(static s => new SequenceResult(s.SequenceHandle, [], FinishReason.Error)).ToImmutableArray();
        _ = _planner.ApplyResult(new StepOutcome(step.StepId, rows));
        foreach (var s in step.Sequences)
        {
            if (_live.Remove(s.SequenceHandle, out var live))
            {
                live.Fail(new RoostError(RoostErrorCode.DeviceFault, string.Create(CultureInfo.InvariantCulture, $"The worker failed the step (recoverable fault, status {errorCode}); the worker is kept.")));
            }
        }
    }

    private void LoseWorker(FaultClass fault, int errorCode)
    {
        _planner.OnWorkerRestarted();
        var error = fault == FaultClass.Sticky
            ? new RoostError(RoostErrorCode.DeviceFault, string.Create(CultureInfo.InvariantCulture, $"The GPU faulted (sticky, status {errorCode}); the worker was killed and the request cannot be resumed (R-070)."))
            : new RoostError(RoostErrorCode.WorkerRestarted, $"The worker was lost ({fault}); it was killed and the request cannot be resumed (R-070).");
        foreach (var handle in _live.Keys)
        {
            _ = _planner.Cancel(handle);
        }

        FailAll(error);
        _transport.KillAsync(CancellationToken.None).AsTask().GetAwaiter().GetResult();
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
        public Channel<TokenEvent> Channel { get; } = System.Threading.Channels.Channel.CreateUnbounded<TokenEvent>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

        private int _completion;

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

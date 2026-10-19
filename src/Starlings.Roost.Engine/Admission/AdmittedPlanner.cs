using System.Collections.Immutable;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Engine.Admission;

/// <summary>
/// A <see cref="StepPlanner"/> behind an <see cref="AdmissionController"/> (R-031): requests and forks are admitted
/// first, so a refusal is a <see cref="RoostError"/> returned before the planner, the pools or the worker see anything,
/// and every way a row ends gives its reservation back.
/// </summary>
/// <remarks>
/// Use this type's <see cref="Cancel"/>, <see cref="ApplyResult"/> and <see cref="DrainNotices"/> instead of the
/// planner's, which this type calls; going around it leaves reservations held. Engine thread only.
/// </remarks>
public sealed class AdmittedPlanner
{
    private readonly StepPlanner _planner;
    private readonly AdmissionController _admission;

    /// <summary>Creates the pair.</summary>
    /// <param name="planner">The planner. Add and fork only through this type.</param>
    /// <param name="admission">The controller, built over the planner's block manager.</param>
    public AdmittedPlanner(StepPlanner planner, AdmissionController admission)
    {
        ArgumentNullException.ThrowIfNull(planner);
        ArgumentNullException.ThrowIfNull(admission);
        _planner = planner;
        _admission = admission;
    }

    /// <summary>The planner, for read-only members such as <see cref="StepPlanner.TryGetInfo"/> and <see cref="StepPlanner.CheckInvariants"/>.</summary>
    public StepPlanner Planner => _planner;

    /// <summary>The controller.</summary>
    public AdmissionController Admission => _admission;

    /// <summary>Admits and queues a request.</summary>
    /// <param name="request">The request.</param>
    /// <returns>Null when it is queued; otherwise the typed refusal, with nothing changed.</returns>
    public RoostError? TryAdd(Request request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (!_admission.TryAdmit(request, out var error))
        {
            return error;
        }

        var result = _planner.Add(request);
        if (result == AddResult.Added)
        {
            return null;
        }

        _admission.Release(request.Id.Value);
        return result switch
        {
            AddResult.DuplicateId => new RoostError(RoostErrorCode.InvalidOption, "A request with this id is already live.", "id"),
            AddResult.InvalidOption => new RoostError(RoostErrorCode.InvalidOption, "The prompt is empty or max_tokens is not positive."),
            AddResult.PromptTooLong => new RoostError(RoostErrorCode.PromptTooLong, "The prompt needs more blocks than a pool has.", "prompt"),
            _ => new RoostError(RoostErrorCode.ProfileUnsupported, "The request uses a feature this profile does not support yet."),
        };
    }

    /// <summary>Admits and creates a branch row of a live sequence.</summary>
    /// <param name="parent">The parent's handle.</param>
    /// <param name="branch">The branch's request.</param>
    /// <returns>Null when the branch exists; otherwise the typed refusal, with nothing changed.</returns>
    public RoostError? TryFork(ulong parent, Request branch)
    {
        ArgumentNullException.ThrowIfNull(branch);

        // Check forkability first: a refusal after admission would hand back a row that n, best_of or the beam width reserved.
        if (!_planner.TryGetInfo(parent, out var info))
        {
            return new RoostError(RoostErrorCode.InvalidOption, "The parent is not live.", "parent");
        }

        if (!info.Resident || info.ComputedTokens >= info.TokenCount)
        {
            return new RoostError(RoostErrorCode.InvalidOption, "The parent holds no state yet, or waits for a sampled token.", "parent");
        }

        if (!_admission.TryAdmitFork(parent, branch.Id.Value, out var error))
        {
            return error;
        }

        var result = _planner.Fork(parent, branch);
        if (result == ForkResult.Forked)
        {
            return null;
        }

        _admission.Release(branch.Id.Value);
        return result switch
        {
            ForkResult.NoCapacity => new RoostError(RoostErrorCode.InsufficientDeviceMemory, "The recurrent pool has no free slot for the branch; retry when other sequences finish."),
            ForkResult.NotForkable => new RoostError(RoostErrorCode.InvalidOption, "The parent holds no state yet, or waits for a sampled token.", "parent"),
            _ => new RoostError(RoostErrorCode.InvalidOption, "The parent is not live.", "parent"),
        };
    }

    /// <summary>Drops a sequence and returns its reservation.</summary>
    /// <param name="handle">The handle.</param>
    /// <returns>False when no such sequence is live.</returns>
    public bool Cancel(ulong handle)
    {
        var cancelled = _planner.Cancel(handle);
        _admission.Release(handle);
        return cancelled;
    }

    /// <summary>Builds the next plan.</summary>
    /// <returns>See <see cref="StepPlanner.Plan"/>.</returns>
    public PlannedStep? Plan() => _planner.Plan();

    /// <summary>Applies a result and returns the reservations of the sequences that finished.</summary>
    /// <param name="outcome">The result of a plan.</param>
    /// <returns>The finished sequences.</returns>
    public ImmutableArray<SequenceFinished> ApplyResult(StepOutcome outcome)
    {
        var finished = _planner.ApplyResult(outcome);
        foreach (var f in finished)
        {
            _admission.Release(f.Handle);
        }

        return finished;
    }

    /// <summary>Returns the notices since the last call, releasing the reservation of every aborted sequence.</summary>
    /// <returns>The notices in order.</returns>
    public ImmutableArray<PlannerNotice> DrainNotices()
    {
        var notices = _planner.DrainNotices();
        foreach (var n in notices)
        {
            if (n.Kind == PlannerNoticeKind.Aborted)
            {
                _admission.Release(n.Handle);
            }
        }

        return notices;
    }

    /// <summary>The worker restarted; sequences recompute and keep their reservations.</summary>
    public void OnWorkerRestarted() => _planner.OnWorkerRestarted();
}

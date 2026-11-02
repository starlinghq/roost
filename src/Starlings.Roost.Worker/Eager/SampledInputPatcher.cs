using Starlings.Roost.Plans;

namespace Starlings.Roost.Worker.Eager;

/// <summary>
/// The host-side half of async scheduling (R-032) for a worker that runs its forward on the CPU or reads its inputs from host
/// memory: it keeps the ids sampled by the step the worker ran last and writes them into the
/// <see cref="ParsedSequence.SampledInput"/> rows of the next plan before the forward. A worker that uses it reports
/// <see cref="PlanLimits.SampledInputs"/>. The CUDA eager worker does not: its inputs are uploaded before the previous step's
/// ids reach the host, so it needs device-side patching (stream F) and refuses such plans until then.
/// </summary>
/// <remarks>
/// A plan names the step it takes its inputs from (<see cref="ParsedPlan.InputStepId"/>). The patcher refuses the plan, with
/// nothing patched, unless that is the last step it recorded and every marked row's sequence sampled in it: after a fault, a
/// refused plan or a plan that never reached the worker, a stale id is never used. One instance per worker, on the step thread.
/// </remarks>
public sealed class SampledInputPatcher
{
    private readonly Dictionary<ulong, int> _sampled = [];
    private readonly int _vocab;
    private ulong _step;

    /// <summary>Creates the patcher.</summary>
    /// <param name="vocab">Token ids are in [0, vocab).</param>
    public SampledInputPatcher(int vocab)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocab);
        _vocab = vocab;
    }

    /// <summary>The step whose sampled ids are held, or 0.</summary>
    public ulong LastStep => _step;

    /// <summary>Writes the held ids into the plan's sampled-input rows.</summary>
    /// <param name="plan">A plan checked with <see cref="PlanLimits.SampledInputs"/> set.</param>
    /// <exception cref="PlanRejectedException">The plan takes its inputs from a step other than the last one recorded, or a marked sequence did not sample in it.</exception>
    public void Patch(ParsedPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        if (plan.InputStepId == 0)
        {
            return;
        }

        if (plan.InputStepId != _step)
        {
            throw new PlanRejectedException(PlanRejection.OutOfRange, $"plan {plan.StepId} takes its inputs from step {plan.InputStepId}; the worker holds the ids of step {_step}");
        }

        // Check every row before writing any, so a refused plan leaves its token table as it arrived.
        foreach (var s in plan.Sequences)
        {
            if (s.SampledInput && !_sampled.ContainsKey(s.Handle))
            {
                throw new PlanRejectedException(PlanRejection.OutOfRange, $"plan {plan.StepId}: a sampled-input row's sequence did not sample in step {_step}");
            }
        }

        foreach (var s in plan.Sequences)
        {
            if (s.SampledInput)
            {
                plan.TokenIds[s.TokenStart + s.TokenCount - 1] = _sampled[s.Handle];
            }
        }
    }

    /// <summary>Records the ids a step sampled, replacing those of the step before.</summary>
    /// <param name="plan">The step that ran.</param>
    /// <param name="sampled">The token of each sampling sequence, in plan order.</param>
    /// <exception cref="ArgumentException">The count does not match the plan's sampling sequences, or an id is outside the vocabulary.</exception>
    public void Record(ParsedPlan plan, ReadOnlySpan<int> sampled)
    {
        ArgumentNullException.ThrowIfNull(plan);
        if (sampled.Length != plan.Selected)
        {
            throw new ArgumentException($"{sampled.Length} sampled ids for {plan.Selected} sampling sequences", nameof(sampled));
        }

        _sampled.Clear();
        _step = 0;
        int row = 0;
        foreach (var s in plan.Sequences)
        {
            if (!s.Sample)
            {
                continue;
            }

            int id = sampled[row++];
            if ((uint)id >= (uint)_vocab)
            {
                _sampled.Clear();
                throw new ArgumentException($"sampled id {id} is outside [0, {_vocab})", nameof(sampled));
            }

            _sampled[s.Handle] = id;
        }

        _step = plan.StepId;
    }

    /// <summary>Drops the held ids, after a step that failed or a plan that was refused: the next plan that takes inputs is refused.</summary>
    public void Forget()
    {
        _sampled.Clear();
        _step = 0;
    }
}

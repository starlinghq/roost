using System.Collections.Immutable;
using System.Globalization;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Engine.Scheduling;

/// <summary>One call into the planner. A recorded run is a list of these; replaying it on a fresh planner must give the same plans.</summary>
public abstract record PlannerInput
{
    /// <summary><see cref="StepPlanner.Add"/>.</summary>
    /// <param name="Request">The request.</param>
    public sealed record Add(Request Request) : PlannerInput;

    /// <summary><see cref="StepPlanner.Fork"/>.</summary>
    /// <param name="Parent">Parent handle.</param>
    /// <param name="Branch">Branch request.</param>
    public sealed record Fork(ulong Parent, Request Branch) : PlannerInput;

    /// <summary><see cref="StepPlanner.Cancel"/>.</summary>
    /// <param name="Handle">The handle.</param>
    public sealed record Cancel(ulong Handle) : PlannerInput;

    /// <summary><see cref="StepPlanner.Plan"/>.</summary>
    public sealed record Plan : PlannerInput;

    /// <summary><see cref="StepPlanner.ApplyResult"/>.</summary>
    /// <param name="Outcome">The recorded result.</param>
    public sealed record Result(StepOutcome Outcome) : PlannerInput;

    /// <summary><see cref="StepPlanner.OnWorkerRestarted"/>.</summary>
    public sealed record Restart : PlannerInput;
}

/// <summary>Applies recorded <see cref="PlannerInput"/>s and writes a transcript that two runs can compare line by line.</summary>
public sealed class PlannerReplay
{
    private readonly StepPlanner _planner;
    private readonly List<string> _lines = [];

    /// <summary>Creates a replay over a fresh planner.</summary>
    /// <param name="planner">The planner to drive.</param>
    public PlannerReplay(StepPlanner planner)
    {
        ArgumentNullException.ThrowIfNull(planner);
        _planner = planner;
    }

    /// <summary>The transcript so far.</summary>
    public ImmutableArray<string> Transcript => [.. _lines];

    /// <summary>Applies one input and appends what the planner did to the transcript.</summary>
    /// <param name="input">The input.</param>
    /// <returns>The plan when the input was a <see cref="PlannerInput.Plan"/> that produced one.</returns>
    public PlannedStep? Apply(PlannerInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        PlannedStep? plan = null;
        switch (input)
        {
            case PlannerInput.Add a:
                _lines.Add(string.Create(CultureInfo.InvariantCulture, $"add {a.Request.Id.Value} {_planner.Add(a.Request)}"));
                break;
            case PlannerInput.Fork f:
                _lines.Add(string.Create(CultureInfo.InvariantCulture, $"fork {f.Parent} {f.Branch.Id.Value} {_planner.Fork(f.Parent, f.Branch)}"));
                break;
            case PlannerInput.Cancel c:
                _lines.Add(string.Create(CultureInfo.InvariantCulture, $"cancel {c.Handle} {_planner.Cancel(c.Handle)}"));
                break;
            case PlannerInput.Plan:
                plan = _planner.Plan();
                _lines.Add(plan is null
                    ? "plan none"
                    : string.Create(CultureInfo.InvariantCulture, $"plan {plan.StepId} rows={plan.Sequences.Length} tokens={plan.TokenIds.Length} ops={plan.CacheOps.Length} {plan.Digest}"));
                break;
            case PlannerInput.Result r:
                var finished = _planner.ApplyResult(r.Outcome);
                _lines.Add(string.Create(CultureInfo.InvariantCulture, $"result {r.Outcome.StepId} finished=[{string.Join(',', finished.Select(f => $"{f.Handle}:{f.Reason}"))}]"));
                break;
            case PlannerInput.Restart:
                _planner.OnWorkerRestarted();
                _lines.Add("restart");
                break;
            default:
                throw new ArgumentException("Unknown input.", nameof(input));
        }

        foreach (var n in _planner.DrainNotices())
        {
            _lines.Add(string.Create(CultureInfo.InvariantCulture, $"  {n.Kind} {n.Handle}"));
        }

        _planner.CheckInvariants();
        return plan;
    }

    /// <summary>Applies every input in order.</summary>
    /// <param name="inputs">The recorded inputs.</param>
    /// <returns>The transcript.</returns>
    public ImmutableArray<string> Run(IEnumerable<PlannerInput> inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        foreach (var i in inputs)
        {
            Apply(i);
        }

        return Transcript;
    }
}

using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Onboarding;

/// <summary>
/// Collects the ops a model needs, in first-use order, and classifies each against C3: whether <see cref="OpId"/> names it,
/// whether <see cref="CpuOps"/> implements it (<see cref="CpuOps.Supports(OpId)"/>, read live so the report follows the
/// reference as ops land), and whether the config asks something of it its arguments cannot express.
/// </summary>
internal sealed class OpLedger
{
    private readonly List<Entry> _entries = [];
    private readonly Dictionary<string, Entry> _byName = new(StringComparer.Ordinal);

    /// <summary>Records that <paramref name="usedBy"/> needs the C3 op <paramref name="op"/>.</summary>
    public void Need(OpId op, string usedBy) => Get(op.ToString(), op).AddUse(usedBy);

    /// <summary>Records that <paramref name="usedBy"/> needs <paramref name="op"/> in a form its arguments cannot express.</summary>
    public void Gap(OpId op, string usedBy, string gap)
    {
        var e = Get(op.ToString(), op);
        e.AddUse(usedBy);
        e.AddGap(gap);
    }

    /// <summary>Records an op C3 does not have.</summary>
    public void Missing(string name, string usedBy, string reason)
    {
        var e = Get(name, null);
        e.AddUse(usedBy);
        e.AddGap(reason);
    }

    public IReadOnlyList<OpRequirement> Build() => _entries.Select(e => e.ToRequirement()).ToArray();

    private Entry Get(string name, OpId? op)
    {
        if (!_byName.TryGetValue(name, out var e))
        {
            e = new Entry(name, op);
            _byName.Add(name, e);
            _entries.Add(e);
        }

        return e;
    }

    private sealed class Entry(string name, OpId? op)
    {
        private readonly List<string> _usedBy = [];
        private readonly List<string> _gaps = [];

        public void AddUse(string usedBy)
        {
            if (!_usedBy.Contains(usedBy))
            {
                _usedBy.Add(usedBy);
            }
        }

        public void AddGap(string gap)
        {
            if (!_gaps.Contains(gap))
            {
                _gaps.Add(gap);
            }
        }

        public OpRequirement ToRequirement()
        {
            OpRequirementStatus status = op is not { } id || OpCatalog.PhaseOf(id) == OpPhase.None
                ? OpRequirementStatus.Missing
                : _gaps.Count > 0
                    ? OpRequirementStatus.Gap
                    : CpuOps.Supports(id) ? OpRequirementStatus.Available : OpRequirementStatus.Declared;
            return new OpRequirement
            {
                Op = name,
                Phase = op is { } o && OpCatalog.PhaseOf(o) != OpPhase.None ? OpCatalog.PhaseOf(o).ToString() : null,
                Status = status,
                UsedBy = _usedBy.ToArray(),
                Gaps = _gaps.ToArray(),
            };
        }
    }
}

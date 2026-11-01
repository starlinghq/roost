using Starlings.Roost.Abi;

namespace Starlings.Roost.Kernels;

/// <summary>How a selection was made.</summary>
public enum SelectionSource
{
    /// <summary>One qualified variant; nothing to choose between.</summary>
    OnlyCandidate,

    /// <summary>Every qualified variant has an autotune measurement for the bucket; the fastest was chosen.</summary>
    Autotuned,

    /// <summary>
    /// At least one qualified variant has no measurement; the deterministic default order was used and
    /// <see cref="VariantSelection.Unmeasured"/> lists what to measure.
    /// </summary>
    Default,
}

/// <summary>A candidate the selector did not consider, and why.</summary>
/// <param name="Candidate">The candidate.</param>
/// <param name="Reason">Why it was not qualified.</param>
public sealed record RejectedVariant(VariantCandidate Candidate, string Reason);

/// <summary>The result of selecting a variant for one requirement.</summary>
/// <param name="Sm">The SM.</param>
/// <param name="Bucket">The shape bucket.</param>
/// <param name="Selected">The chosen variant.</param>
/// <param name="Source">How it was chosen.</param>
/// <param name="Qualified">Every qualified candidate, in selection order (best first).</param>
/// <param name="Unmeasured">Qualified candidates with no autotune measurement for the bucket, in default order.</param>
/// <param name="Rejected">Admitted candidates that are not conformance-qualified.</param>
public sealed record VariantSelection(
    int Sm,
    ShapeBucket Bucket,
    VariantCandidate Selected,
    SelectionSource Source,
    IReadOnlyList<VariantCandidate> Qualified,
    IReadOnlyList<VariantCandidate> Unmeasured,
    IReadOnlyList<RejectedVariant> Rejected)
{
    /// <summary>
    /// Whether this selection runs different code from <paramref name="previous"/> for the same SM and bucket: a different
    /// variant or digest. A captured CUDA graph for the bucket must be recaptured when this is true (R-033).
    /// </summary>
    /// <param name="previous">The earlier selection, or null.</param>
    /// <returns>True when there was no earlier selection or it chose other code.</returns>
    public bool ChangedFrom(VariantSelection? previous) =>
        previous is null
        || previous.Sm != Sm
        || previous.Bucket != Bucket
        || previous.Selected.VariantId != Selected.VariantId
        || previous.Selected.Digest != Selected.Digest;
}

/// <summary>
/// The variant selector of R-072: for a requirement at an SM, chooses among the registered variants that admit it
/// (<see cref="KernelRegistry.Find"/>) and are conformance-qualified for that SM and shape bucket
/// (<see cref="VariantQualification"/>). When every qualified variant has an autotune measurement for the bucket the
/// fastest median wins; otherwise the default order is used. Ties, and the default order, are broken by variant id and
/// then digest (ordinal), so the choice does not depend on library load order or dictionary order. When no qualified
/// variant exists the selector refuses with <see cref="ProfileUnsupportedException"/>, never falling back to an
/// unqualified variant (bet X11).
/// </summary>
public sealed class VariantSelector
{
    private readonly VariantQualification _qualification;
    private readonly AutotuneCache? _cache;

    /// <summary>Creates a selector.</summary>
    /// <param name="qualification">The conformance-passed variants.</param>
    /// <param name="cache">The autotune cache for this device, or null to always use the default order.</param>
    public VariantSelector(VariantQualification qualification, AutotuneCache? cache = null)
    {
        ArgumentNullException.ThrowIfNull(qualification);
        _qualification = qualification;
        _cache = cache;
    }

    /// <summary>Selects a variant from a registry.</summary>
    /// <param name="registry">The loaded libraries.</param>
    /// <param name="requirement">What is needed.</param>
    /// <returns>The selection.</returns>
    /// <exception cref="ProfileUnsupportedException">No admitted variant is conformance-qualified.</exception>
    public VariantSelection Select(KernelRegistry registry, KernelRequirement requirement)
    {
        ArgumentNullException.ThrowIfNull(registry);
        ArgumentNullException.ThrowIfNull(requirement);
        return Select(requirement, registry.Find(requirement).Select(VariantCandidate.FromKernel).ToList());
    }

    /// <summary>Selects among candidates that already admit the requirement (the registry's constraint check has run).</summary>
    /// <param name="requirement">What is needed.</param>
    /// <param name="admitted">The admitted candidates, in any order.</param>
    /// <returns>The selection.</returns>
    /// <exception cref="ProfileUnsupportedException">No candidate is conformance-qualified.</exception>
    /// <exception cref="ArgumentException">A candidate is for another op, or a variant id appears twice.</exception>
    public VariantSelection Select(KernelRequirement requirement, IReadOnlyList<VariantCandidate> admitted)
    {
        ArgumentNullException.ThrowIfNull(requirement);
        ArgumentNullException.ThrowIfNull(admitted);
        var bucket = ShapeBucket.For(requirement);
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var candidate in admitted)
        {
            ArgumentNullException.ThrowIfNull(candidate, nameof(admitted));
            if (candidate.Op != requirement.Op)
            {
                throw new ArgumentException($"candidate {candidate} is for {candidate.Op}, the requirement for {requirement.Op}", nameof(admitted));
            }

            if (!ids.Add(candidate.VariantId))
            {
                throw new ArgumentException($"variant {candidate.VariantId} of {candidate.Op} appears twice", nameof(admitted));
            }
        }

        var ordered = admitted.OrderBy(c => c.VariantId, StringComparer.Ordinal).ThenBy(c => c.Digest, StringComparer.Ordinal).ToList();
        var qualified = new List<VariantCandidate>();
        var rejected = new List<RejectedVariant>();
        foreach (var candidate in ordered)
        {
            if (_qualification.IsQualified(candidate, requirement.Sm, bucket, out var reason))
            {
                qualified.Add(candidate);
            }
            else
            {
                rejected.Add(new RejectedVariant(candidate, reason!));
            }
        }

        if (qualified.Count == 0)
        {
            throw new ProfileUnsupportedException(requirement.Op, requirement.Sm, Explain(requirement, bucket, rejected));
        }

        // A cache opened for another SM holds nothing for this one; it is ignored rather than consulted.
        var cache = _cache is { } c && c.Sm == requirement.Sm ? c : null;
        var measured = new List<(VariantCandidate Candidate, AutotuneMeasurement Measurement)>();
        var unmeasured = new List<VariantCandidate>();
        foreach (var candidate in qualified)
        {
            if (cache is not null && cache.TryGet(bucket, candidate, out var m))
            {
                measured.Add((candidate, m));
            }
            else
            {
                unmeasured.Add(candidate);
            }
        }

        if (qualified.Count == 1)
        {
            return new VariantSelection(requirement.Sm, bucket, qualified[0], SelectionSource.OnlyCandidate, qualified, unmeasured, rejected);
        }

        if (unmeasured.Count == 0)
        {
            // Stable sort over the ordinal default order: equal medians keep id-then-digest order.
            var fastest = measured.OrderBy(x => x.Measurement.MedianMicroseconds).Select(x => x.Candidate).ToList();
            return new VariantSelection(requirement.Sm, bucket, fastest[0], SelectionSource.Autotuned, fastest, unmeasured, rejected);
        }

        return new VariantSelection(requirement.Sm, bucket, qualified[0], SelectionSource.Default, qualified, unmeasured, rejected);
    }

    private static string Explain(KernelRequirement requirement, ShapeBucket bucket, List<RejectedVariant> rejected)
    {
        if (rejected.Count == 0)
        {
            return $"no registered variant admits [{requirement}]";
        }

        var reasons = rejected.Take(4).Select(r => $"{r.Candidate}: {r.Reason}");
        return $"{rejected.Count} registered variant(s) admit [{requirement}], none is conformance-qualified for bucket {bucket.Key}: {string.Join("; ", reasons)}";
    }
}

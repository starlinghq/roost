using Starlings.Roost.Abi;

namespace Starlings.Roost.Kernels;

/// <summary>
/// One conformance pass of a variant (R-072): the variant digest that passed, at one SM, for one shape bucket or for every
/// bucket of the op. Producing these records (running conformance against <c>CpuOps</c>) is not part of this type; it only
/// holds what passed.
/// </summary>
/// <param name="Op">The op.</param>
/// <param name="VariantId">The variant id.</param>
/// <param name="Digest">The <see cref="VariantCandidate.Digest"/> that passed.</param>
/// <param name="Sm">The SM it passed on, as major * 10 + minor.</param>
/// <param name="BucketKey">The <see cref="ShapeBucket.Key"/> it passed for, or null for every bucket of the op.</param>
/// <param name="Evidence">Where the pass is recorded (an evidence path or run id), for messages and profiles.</param>
public sealed record QualifiedVariant(OpId Op, string VariantId, string Digest, int Sm, string? BucketKey, string Evidence);

/// <summary>
/// The set of conformance-passed variants. A candidate qualifies only when a record names its op, variant id and exact
/// digest at the requested SM, for the requested bucket or for every bucket; a rebuilt variant (new digest) does not
/// inherit its predecessor's pass. An empty set qualifies nothing, so refusal is the default (bet X11).
/// </summary>
public sealed class VariantQualification
{
    private readonly Dictionary<(OpId Op, string VariantId, int Sm), List<QualifiedVariant>> _passes = [];

    /// <summary>An empty set.</summary>
    public VariantQualification()
    {
    }

    /// <summary>A set holding <paramref name="passes"/>.</summary>
    /// <param name="passes">The passes.</param>
    public VariantQualification(IEnumerable<QualifiedVariant> passes)
    {
        ArgumentNullException.ThrowIfNull(passes);
        foreach (var pass in passes)
        {
            Add(pass);
        }
    }

    /// <summary>The number of recorded passes.</summary>
    public int Count => _passes.Values.Sum(l => l.Count);

    /// <summary>Records a pass.</summary>
    /// <param name="pass">The pass.</param>
    /// <exception cref="ArgumentException">The record has no op, an invalid variant id, digest or bucket key, a non-positive SM or no evidence.</exception>
    public void Add(QualifiedVariant pass)
    {
        ArgumentNullException.ThrowIfNull(pass);
        if (pass.Op == OpId.None || !VariantCandidate.IsValidVariantId(pass.VariantId) || !VariantCandidate.IsDigest(pass.Digest))
        {
            throw new ArgumentException("a pass names an op, a valid variant id and a 64-digit lower-case hex digest", nameof(pass));
        }

        if (pass.Sm <= 0 || (pass.BucketKey is not null && !ShapeBucket.IsValidKey(pass.BucketKey)) || string.IsNullOrWhiteSpace(pass.Evidence))
        {
            throw new ArgumentException("a pass names a positive SM, a valid bucket key or null, and its evidence", nameof(pass));
        }

        var key = (pass.Op, pass.VariantId, pass.Sm);
        if (!_passes.TryGetValue(key, out var list))
        {
            _passes[key] = list = [];
        }

        if (!list.Contains(pass))
        {
            list.Add(pass);
        }
    }

    /// <summary>Whether a candidate is qualified at an SM for a bucket.</summary>
    /// <param name="candidate">The candidate.</param>
    /// <param name="sm">The SM.</param>
    /// <param name="bucket">The bucket.</param>
    /// <param name="reason">Why it is not qualified, or null.</param>
    /// <returns>True when a pass covers it.</returns>
    public bool IsQualified(VariantCandidate candidate, int sm, ShapeBucket bucket, out string? reason)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        ArgumentNullException.ThrowIfNull(bucket);
        if (!_passes.TryGetValue((candidate.Op, candidate.VariantId, sm), out var list))
        {
            reason = $"no conformance pass at SM {sm}";
            return false;
        }

        var sameDigest = list.Where(p => p.Digest == candidate.Digest).ToList();
        if (sameDigest.Count == 0)
        {
            reason = $"the conformance pass at SM {sm} is for another digest ({list[0].Digest[..12]}), not {candidate.Digest[..12]}";
            return false;
        }

        if (sameDigest.Any(p => p.BucketKey is null || p.BucketKey == bucket.Key))
        {
            reason = null;
            return true;
        }

        reason = $"conformance passed at SM {sm} only for other shape buckets";
        return false;
    }
}

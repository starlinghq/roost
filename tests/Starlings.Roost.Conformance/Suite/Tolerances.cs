using System.Security.Cryptography;
using System.Text.Json;

namespace Starlings.Roost.Conformance;

/// <summary>
/// One op class's comparison rule: an element passes when <c>|got - ref| &lt;= AtolRel * scale + Rtol * |ref|</c>, where
/// <c>scale</c> is the reference's RMS (or, for attention, the largest |V| of the pool). An exact class compares bytes.
/// </summary>
/// <param name="Name">Class name, as it appears in results.</param>
/// <param name="Rtol">Relative part.</param>
/// <param name="AtolRel">Absolute part as a fraction of the scale.</param>
/// <param name="Exact">Bit-for-bit comparison; the other fields are ignored.</param>
/// <param name="Scale">"rms" or "max-abs-v".</param>
public sealed record ToleranceClass(string Name, double Rtol, double AtolRel, bool Exact, string Scale)
{
    /// <summary>The scale is the largest absolute value of the value pool (attention).</summary>
    public const string MaxAbsV = "max-abs-v";

    /// <summary>The scale is the RMS of the reference.</summary>
    public const string Rms = "rms";
}

/// <summary>
/// The per-op-class comparison rules the suite applies. They are read from a file: the proposal that ships with the suite
/// (<c>op-class-tolerances.proposal.json</c>), or, once a maintainer approves one, the protected file passed with
/// <c>--tolerances</c>. The suite never writes either.
/// </summary>
public sealed class ToleranceTable
{
    /// <summary>The file name of the proposal that ships next to the suite.</summary>
    public const string ProposalFileName = "op-class-tolerances.proposal.json";

    private readonly Dictionary<string, ToleranceClass> _classes;

    private ToleranceTable(Dictionary<string, ToleranceClass> classes, string source, string sha256, string status)
    {
        _classes = classes;
        Source = source;
        Sha256 = sha256;
        Status = status;
    }

    /// <summary>Where the table was read from.</summary>
    public string Source { get; }

    /// <summary>SHA-256 of that file.</summary>
    public string Sha256 { get; }

    /// <summary>"proposal" for the shipped file; whatever the file says otherwise.</summary>
    public string Status { get; }

    /// <summary>The classes by name.</summary>
    public IReadOnlyDictionary<string, ToleranceClass> Classes => _classes;

    /// <summary>The class of a name.</summary>
    /// <param name="name">Class name.</param>
    /// <returns>The class.</returns>
    /// <exception cref="KeyNotFoundException">The table has no such class.</exception>
    public ToleranceClass this[string name] => _classes.TryGetValue(name, out var c)
        ? c
        : throw new KeyNotFoundException($"The tolerance file {Source} has no class '{name}'.");

    /// <summary>The proposal that ships with the suite, from the directory of this assembly.</summary>
    /// <returns>The table.</returns>
    public static ToleranceTable LoadProposal() => Load(Path.Combine(AppContext.BaseDirectory, ProposalFileName));

    /// <summary>Reads a tolerance file.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The table.</returns>
    public static ToleranceTable Load(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        using var doc = JsonDocument.Parse(bytes);
        JsonElement root = doc.RootElement;
        string status = root.TryGetProperty("status", out var st) ? st.GetString() ?? "unknown" : "unknown";
        var classes = new Dictionary<string, ToleranceClass>(StringComparer.Ordinal);
        foreach (JsonProperty p in root.GetProperty("classes").EnumerateObject())
        {
            JsonElement v = p.Value;
            bool exact = v.TryGetProperty("exact", out var e) && e.GetBoolean();
            double rtol = v.TryGetProperty("rtol", out var r) ? r.GetDouble() : 0;
            double atol = v.TryGetProperty("atol_rel", out var a) ? a.GetDouble() : 0;
            string scale = v.TryGetProperty("scale", out var s) ? s.GetString() ?? ToleranceClass.Rms : ToleranceClass.Rms;
            classes[p.Name] = new ToleranceClass(p.Name, rtol, atol, exact, scale);
        }

        return new ToleranceTable(classes, path, Convert.ToHexStringLower(SHA256.HashData(bytes)), status);
    }
}

/// <summary>Statistics of one comparison.</summary>
/// <param name="Count">Elements compared.</param>
/// <param name="MaxAbs">Largest absolute difference.</param>
/// <param name="MaxUtilisation">Largest difference as a fraction of what the class allows (1 is the limit).</param>
/// <param name="Violations">Elements outside the class.</param>
/// <param name="NonFinite">Candidate elements that are NaN or infinite.</param>
/// <param name="RefRms">RMS of the reference.</param>
/// <param name="Scale">The scale the absolute part was multiplied by.</param>
public sealed record CompareStat(long Count, double MaxAbs, double MaxUtilisation, long Violations, long NonFinite, double RefRms, double Scale)
{
    /// <summary>No violations and no non-finite element.</summary>
    public bool Pass => Violations == 0 && NonFinite == 0;
}

/// <summary>Comparison of a candidate buffer with a reference under a tolerance class.</summary>
public static class Compare
{
    /// <summary>Compares two buffers of equal element count.</summary>
    /// <param name="got">The candidate's output.</param>
    /// <param name="want">The reference.</param>
    /// <param name="tol">The class.</param>
    /// <param name="maxAbsV">The pool's largest |V|, for classes whose scale is <see cref="ToleranceClass.MaxAbsV"/>.</param>
    /// <returns>The statistics.</returns>
    public static CompareStat Close(HostBuf got, HostBuf want, ToleranceClass tol, double maxAbsV = 0)
    {
        if (got.Count != want.Count)
        {
            return new CompareStat(want.Count, double.PositiveInfinity, double.PositiveInfinity, Math.Max(1, Math.Abs(got.Count - want.Count)), 0, 0, 0);
        }

        if (tol.Exact)
        {
            return Bits(got, want);
        }

        double ss = 0;
        for (long i = 0; i < want.Count; i++)
        {
            double r = want.Get(i);
            ss += r * r;
        }

        double rms = want.Count > 0 ? Math.Sqrt(ss / want.Count) : 0;
        double scale = tol.Scale == ToleranceClass.MaxAbsV ? maxAbsV : rms;
        double atol = tol.AtolRel * scale;
        double maxAbs = 0, maxUtil = 0;
        long bad = 0, nonFinite = 0;
        for (long i = 0; i < want.Count; i++)
        {
            double g = got.Get(i), r = want.Get(i);
            if (!double.IsFinite(g))
            {
                nonFinite++;
                continue;
            }

            double d = Math.Abs(g - r);
            double allowed = atol + (tol.Rtol * Math.Abs(r));
            maxAbs = Math.Max(maxAbs, d);
            double util = allowed > 0 ? d / allowed : (d == 0 ? 0 : double.PositiveInfinity);
            maxUtil = Math.Max(maxUtil, util);
            if (d > allowed)
            {
                bad++;
            }
        }

        return new CompareStat(want.Count, maxAbs, maxUtil, bad, nonFinite, rms, scale);
    }

    /// <summary>Bitwise comparison.</summary>
    /// <param name="got">The candidate's output.</param>
    /// <param name="want">The reference.</param>
    /// <returns>The statistics; violations count differing elements.</returns>
    public static CompareStat Bits(HostBuf got, HostBuf want)
    {
        if (got.Bytes != want.Bytes || got.Dtype != want.Dtype)
        {
            return new CompareStat(want.Count, double.PositiveInfinity, double.PositiveInfinity, Math.Max(1, want.Count), 0, 0, 0);
        }

        int es = Elem.Size(want.Dtype);
        long bad = 0;
        double maxAbs = 0;
        ReadOnlySpan<byte> a = got.Raw(), b = want.Raw();
        for (long i = 0; i < want.Count; i++)
        {
            if (!a.Slice((int)(i * es), es).SequenceEqual(b.Slice((int)(i * es), es)))
            {
                bad++;
                double d = Math.Abs(got.Get(i) - want.Get(i));
                maxAbs = Math.Max(maxAbs, double.IsNaN(d) ? double.PositiveInfinity : d);
            }
        }

        return new CompareStat(want.Count, maxAbs, bad == 0 ? 0 : double.PositiveInfinity, bad, 0, 0, 0);
    }

    /// <summary>Whether two buffers are byte for byte equal.</summary>
    /// <param name="a">One buffer.</param>
    /// <param name="b">The other.</param>
    /// <returns>True when sizes and bytes match.</returns>
    public static bool BytesEqual(HostBuf a, HostBuf b) => a.Bytes == b.Bytes && a.Raw().SequenceEqual(b.Raw());
}

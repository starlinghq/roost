using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

/// <summary>A backend that forwards to another and lets a test change what comes back.</summary>
internal sealed class DelegatingBackend : IKernelBackend
{
    private readonly IKernelBackend _inner;
    private readonly Func<string, OpCase, ExecResult, ExecResult>? _after;
    private readonly Func<string, OpCase, ExecResult?>? _replace;

    public DelegatingBackend(IKernelBackend inner, Func<string, OpCase, ExecResult, ExecResult>? after = null, Func<string, OpCase, ExecResult?>? replace = null, VariantTraits? flags = null)
    {
        _inner = inner;
        _after = after;
        _replace = replace;
        Variants = flags is null
            ? inner.Variants
            : [.. inner.Variants.Select(v => v with { Flags = flags.Value })];
    }

    public int Calls { get; private set; }

    public string Name => "Delegating(" + _inner.Name + ")";

    public int Sm => _inner.Sm;

    public IReadOnlyList<VariantInfo> Variants { get; }

    public ExecResult Execute(string variant, OpCase c)
    {
        Calls++;
        ExecResult? replaced = _replace?.Invoke(variant, c);
        if (replaced is not null)
        {
            return replaced;
        }

        ExecResult r = _inner.Execute(variant, c);
        return _after is null ? r : _after(variant, c, r);
    }

    public int? ProbeShortStruct(string variant, OpCase c) => _inner.ProbeShortStruct(variant, c);

    public void Dispose() => _inner.Dispose();
}

internal static class Fixtures
{
    public static RunOptions Options(CaseProfile? profile = null, GoldenSet? goldens = null, ISet<string>? variants = null, bool determinism = true, bool invariance = true, ISet<OpId>? ops = null)
        => new()
        {
            Profile = profile ?? CaseProfile.Ci,
            Tolerances = ToleranceTable.LoadProposal(),
            Goldens = goldens,
            Variants = variants,
            Ops = ops,
            Determinism = determinism,
            Invariance = invariance,
        };

    public static ConformanceReport RunOn(IKernelBackend b, RunOptions? o = null) => ConformanceRunner.Run(b, o ?? Options());

    /// <summary>Applies a function to every element of a float output, in place.</summary>
    public static void Map(HostBuf b, Func<long, double, double> f)
    {
        for (long i = 0; i < b.Count; i++)
        {
            b.Set(i, f(i, b.Get(i)));
        }
    }
}

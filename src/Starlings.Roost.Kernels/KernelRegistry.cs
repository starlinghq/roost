using System.Globalization;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels;

/// <summary>
/// What a profile needs from the kernel libraries for one op at one SM. A null field is not checked. The startup check
/// (<see cref="KernelRegistry.RequireAll"/>) asks for every requirement of a model's forward before the first step.
/// </summary>
public sealed record KernelRequirement
{
    /// <summary>The op.</summary>
    public required OpId Op { get; init; }

    /// <summary>The device's SM as major * 10 + minor.</summary>
    public required int Sm { get; init; }

    /// <summary>Activation dtype.</summary>
    public DType? Activation { get; init; }

    /// <summary>Weight dtype.</summary>
    public DType? Weight { get; init; }

    /// <summary>KV cache dtype.</summary>
    public DType? Kv { get; init; }

    /// <summary>Attention head dimension (a multiple of 16).</summary>
    public int? HeadDim { get; init; }

    /// <summary>Buffer layout.</summary>
    public TensorLayout? Layout { get; init; }

    /// <summary>Page (block) size, a power of two.</summary>
    public int? PageSize { get; init; }

    /// <summary>Query heads per KV head.</summary>
    public int? GroupSize { get; init; }

    /// <summary>Sequences per call.</summary>
    public int? Batch { get; init; }

    /// <summary>Tokens per call.</summary>
    public long? Tokens { get; init; }

    /// <summary>Only variants flagged with all of these (for example <see cref="VariantTraits.GraphCapturable"/>).</summary>
    public VariantTraits RequiredFlags { get; init; }

    /// <summary>A short description for messages.</summary>
    public override string ToString()
    {
        var parts = new List<string> { $"{Op}" };
        void Add(string name, object? value)
        {
            if (value is not null)
            {
                parts.Add(string.Create(CultureInfo.InvariantCulture, $"{name} {value}"));
            }
        }

        Add("act", Activation);
        Add("weight", Weight);
        Add("kv", Kv);
        Add("head_dim", HeadDim);
        Add("layout", Layout);
        Add("page_size", PageSize);
        Add("group_size", GroupSize);
        Add("batch", Batch);
        Add("tokens", Tokens);
        if (RequiredFlags != VariantTraits.None)
        {
            parts.Add($"flags {RequiredFlags}");
        }

        return string.Join(", ", parts);
    }
}

/// <summary>
/// The registration tables of every loaded kernel library. More than one library may register (the base library, the
/// persistent kernels for a profile, a user's library); a variant id is unique per op across them (contract C1), so a
/// second library registering a variant that is already registered is refused rather than silently replacing it.
/// Choosing among the variants that satisfy a requirement is the selector's job (R-072); this registry returns them in
/// the order the libraries were added.
/// </summary>
public sealed class KernelRegistry
{
    private readonly List<KernelLibrary> _libraries = [];
    private readonly Dictionary<(OpId Op, string Variant), RegisteredKernel> _byVariant = [];

    /// <summary>The libraries, in the order added.</summary>
    public IReadOnlyList<KernelLibrary> Libraries => _libraries;

    /// <summary>Every registered kernel, in the order added.</summary>
    public IEnumerable<RegisteredKernel> Kernels => _libraries.SelectMany(l => l.Kernels);

    /// <summary>Adds a loaded library's table.</summary>
    /// <param name="library">The library.</param>
    /// <exception cref="KernelLoadException">A variant id of an op is already registered by another library (<see cref="RoostErrorCode.AssetMismatch"/>).</exception>
    public void Add(KernelLibrary library)
    {
        ArgumentNullException.ThrowIfNull(library);
        foreach (var kernel in library.Kernels)
        {
            if (_byVariant.TryGetValue((kernel.Op, kernel.VariantId), out var existing))
            {
                throw new KernelLoadException(
                    RoostErrorCode.AssetMismatch,
                    $"variant {kernel.VariantId} of {kernel.Op} is registered by both {existing.Library.Name} and {library.Name}");
            }
        }

        foreach (var kernel in library.Kernels)
        {
            _byVariant[(kernel.Op, kernel.VariantId)] = kernel;
        }

        _libraries.Add(library);
    }

    /// <summary>The registered kernels that admit a requirement, in registration order.</summary>
    /// <param name="requirement">What is needed.</param>
    /// <returns>Zero or more kernels.</returns>
    public IReadOnlyList<RegisteredKernel> Find(KernelRequirement requirement)
    {
        ArgumentNullException.ThrowIfNull(requirement);
        return Kernels.Where(k => k.Op == requirement.Op && Mismatch(k, requirement) is null).ToList();
    }

    /// <summary>Returns the first kernel that admits a requirement, or throws the typed startup failure.</summary>
    /// <param name="requirement">What is needed.</param>
    /// <returns>The first matching kernel.</returns>
    /// <exception cref="ProfileUnsupportedException">No loaded library has a kernel for the op at the SM that admits it.</exception>
    public RegisteredKernel Require(KernelRequirement requirement)
    {
        var found = Find(requirement);
        if (found.Count > 0)
        {
            return found[0];
        }

        throw new ProfileUnsupportedException(requirement.Op, requirement.Sm, Explain(requirement));
    }

    /// <summary>Checks every requirement of a profile; the first unmet one throws, naming its op and SM.</summary>
    /// <param name="requirements">All requirements of the model's forward.</param>
    /// <returns>One kernel per requirement, in order.</returns>
    /// <exception cref="ProfileUnsupportedException">A requirement is unmet.</exception>
    public IReadOnlyList<RegisteredKernel> RequireAll(IEnumerable<KernelRequirement> requirements)
    {
        ArgumentNullException.ThrowIfNull(requirements);
        return requirements.Select(Require).ToList();
    }

    private string Explain(KernelRequirement requirement)
    {
        var forOp = Kernels.Where(k => k.Op == requirement.Op).ToList();
        if (forOp.Count == 0)
        {
            var listed = _libraries
                .SelectMany(l => l.Manifest.Variants.Where(v => v.Op == requirement.Op))
                .SelectMany(v => v.Binaries.Select(b => b.Arch))
                .Distinct()
                .Order()
                .ToList();
            var built = listed.Count == 0
                ? "the manifest lists no variant of the op"
                : $"the manifest builds the op only for SM {string.Join(", ", listed)}";
            return $"no variant of the op is registered by the loaded libraries ({built}); needed: {requirement}";
        }

        var reasons = forOp.Select(k => $"{k.VariantId}: {Mismatch(k, requirement)}").Take(4);
        return $"{forOp.Count} registered variant(s), none admits [{requirement}]: {string.Join("; ", reasons)}";
    }

    /// <summary>The first constraint of <paramref name="kernel"/> that the requirement violates, or null when it is admitted.</summary>
    internal static string? Mismatch(RegisteredKernel kernel, KernelRequirement r)
    {
        var c = kernel.Constraints;
        if ((kernel.Flags & r.RequiredFlags) != r.RequiredFlags)
        {
            return $"lacks flags {r.RequiredFlags & ~kernel.Flags}";
        }

        if (r.Sm < c.SmMin || (c.SmMax != 0 && r.Sm > c.SmMax))
        {
            return $"valid for SM {c.SmMin}..{(c.SmMax == 0 ? "up" : c.SmMax.ToString(CultureInfo.InvariantCulture))}";
        }

        if (r.Activation is { } act && !DTypeAdmitted(c.ActDtypeMask, act))
        {
            return $"does not take activation {act}";
        }

        if (r.Weight is { } w && !DTypeAdmitted(c.WeightDtypeMask, w))
        {
            return $"does not take weight {w}";
        }

        if (r.Kv is { } kv && !DTypeAdmitted(c.KvDtypeMask, kv))
        {
            return $"does not take KV {kv}";
        }

        if (r.HeadDim is { } hd && c.HeadDimMask != 0 && !(hd is >= 16 and <= 1024 && hd % 16 == 0 && Bit(c.HeadDimMask, (hd / 16) - 1)))
        {
            return $"does not take head_dim {hd}";
        }

        if (r.Layout is { } layout && c.LayoutMask != 0 && !Bit(c.LayoutMask, (int)layout))
        {
            return $"does not take layout {layout}";
        }

        if (r.PageSize is { } ps && c.PageSizeMask != 0 && !(ps > 0 && (ps & (ps - 1)) == 0 && Bit(c.PageSizeMask, int.Log2(ps))))
        {
            return $"does not take page_size {ps}";
        }

        if (r.GroupSize is { } gs && c.GroupSizeMask != 0 && !(gs is >= 1 and <= 32 && Bit(c.GroupSizeMask, gs - 1)))
        {
            return $"does not take group_size {gs}";
        }

        if (r.Batch is { } b && c.MaxBatch != 0 && b > c.MaxBatch)
        {
            return $"takes at most {c.MaxBatch} sequences";
        }

        if (r.Tokens is { } t && c.MaxTokens != 0 && (ulong)t > c.MaxTokens)
        {
            return $"takes at most {c.MaxTokens} tokens";
        }

        return null;
    }

    private static bool DTypeAdmitted(ulong mask, DType dtype) => mask == 0 || Bit(mask, (int)dtype);

    private static bool Bit(ulong mask, int index) => index is >= 0 and < 64 && (mask >> index & 1) != 0;
}

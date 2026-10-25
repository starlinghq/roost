using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>One op invocation with its inputs, as every backend receives it. Buffers are always logical (compact); strides are parameters.</summary>
public sealed class OpCase : IDisposable
{
    /// <summary>Stable id inside one op: family, dtype and shape.</summary>
    public required string Id { get; init; }

    /// <summary>The op.</summary>
    public required OpId Op { get; init; }

    /// <summary>The activation element type, which selects the tolerance class and the variants that apply.</summary>
    public required DType Dtype { get; init; }

    /// <summary>Tolerance class name, a key of the tolerance file.</summary>
    public required string ToleranceClass { get; init; }

    /// <summary>"fixed" (a shape of the M0 model), "random" (drawn from <see cref="Seed"/>) or "golden" (from an upstream golden file).</summary>
    public required string Source { get; init; }

    /// <summary>The seed the buffers were drawn from; 0 for golden cases.</summary>
    public ulong Seed { get; init; }

    /// <summary>Integer and floating-point parameters by name (shapes, strides, flags, scalars).</summary>
    public Dictionary<string, double> Num { get; } = new(StringComparer.Ordinal);

    /// <summary>Named buffers: inputs, and the initial contents of outputs.</summary>
    public Dictionary<string, HostBuf> Buffers { get; } = new(StringComparer.Ordinal);

    /// <summary>Names of the buffers a backend must return and the suite compares.</summary>
    public List<string> Outputs { get; } = [];

    /// <summary>
    /// Tolerance classes of outputs that do not use <see cref="ToleranceClass"/>: by output name, or by <c>output@variant</c> for a
    /// class that applies to one variant only (the chunked GatedDeltaRule). See <see cref="ClassOf"/>.
    /// </summary>
    public Dictionary<string, string> OutputClasses { get; } = new(StringComparer.Ordinal);

    /// <summary>The tolerance class of one output when run on one variant: <c>output@variant</c>, then <c>output</c>, then the case's class.</summary>
    /// <param name="output">The output name.</param>
    /// <param name="variant">The variant id.</param>
    /// <returns>The class name.</returns>
    public string ClassOf(string output, string variant) =>
        OutputClasses.TryGetValue(output + "@" + variant, out string? v) ? v
        : OutputClasses.TryGetValue(output, out string? o) ? o
        : ToleranceClass;

    /// <summary>Name of the buffer whose largest |x| scales the absolute tolerance (attention's value pool), or null.</summary>
    public string? ScaleBuffer { get; init; }

    /// <summary>Expected outputs from an upstream golden, by output name; null for cases checked against CpuOps.</summary>
    public Dictionary<string, HostBuf>? Expected { get; set; }

    /// <summary>For ops whose rows are independent: the same call on one row (token), keeping the plan bucket. Null when the op has no such form.</summary>
    public Func<int, OpCase>? SingleRow { get; init; }

    /// <summary>Rows (tokens) a <see cref="SingleRow"/> can select from.</summary>
    public int Rows { get; init; }

    /// <summary>Row-wise outputs and the width of one row of each, for <see cref="SingleRow"/>.</summary>
    public Dictionary<string, int> RowWidth { get; } = new(StringComparer.Ordinal);

    /// <summary>An integer parameter.</summary>
    /// <param name="name">The name.</param>
    /// <returns>The value.</returns>
    public int I(string name) => (int)Num[name];

    /// <summary>A parameter as float.</summary>
    /// <param name="name">The name.</param>
    /// <returns>The value.</returns>
    public float F(string name) => (float)Num[name];

    /// <summary>A parameter that may be absent.</summary>
    /// <param name="name">The name.</param>
    /// <param name="fallback">Value when absent.</param>
    /// <returns>The value.</returns>
    public int IOr(string name, int fallback) => Num.TryGetValue(name, out double v) ? (int)v : fallback;

    /// <summary>Copies of all buffers; a backend works on these so the case stays pristine.</summary>
    /// <returns>Fresh buffers by name.</returns>
    public Dictionary<string, HostBuf> CloneBuffers() => Buffers.ToDictionary(kv => kv.Key, kv => kv.Value.Clone(), StringComparer.Ordinal);

    /// <summary>Releases all buffers.</summary>
    public void Dispose()
    {
        foreach (HostBuf b in Buffers.Values)
        {
            b.Dispose();
        }

        if (Expected is not null)
        {
            foreach (HostBuf b in Expected.Values)
            {
                b.Dispose();
            }
        }
    }
}

/// <summary>A registered variant: what a library's table says about one entry.</summary>
/// <param name="Op">The op.</param>
/// <param name="Id">Variant id.</param>
/// <param name="Flags">Registration flags.</param>
/// <param name="Constraints">Constraints (all zero means none).</param>
public sealed record VariantInfo(OpId Op, string Id, VariantTraits Flags, KernelConstraints Constraints)
{
    /// <summary>Whether a case lies inside this variant's constraints for a device of compute capability <paramref name="sm"/> (0 = unknown).</summary>
    /// <param name="c">The case.</param>
    /// <param name="sm">Compute capability as major * 10 + minor.</param>
    /// <returns>True when the variant is expected to run the case.</returns>
    public bool Accepts(OpCase c, int sm)
    {
        if (c.Op != Op)
        {
            return false;
        }

        KernelConstraints k = Constraints;
        if (sm > 0 && ((k.SmMin != 0 && sm < k.SmMin) || (k.SmMax != 0 && sm > k.SmMax)))
        {
            return false;
        }

        if (k.ActDtypeMask != 0 && (k.ActDtypeMask & Elem.Bit(c.Dtype)) == 0)
        {
            return false;
        }

        if (k.HeadDimMask != 0 && c.Num.TryGetValue("head_dim", out double hd))
        {
            int bit = ((int)hd / 16) - 1;
            if ((int)hd % 16 != 0 || bit < 0 || bit > 63 || (k.HeadDimMask & (1UL << bit)) == 0)
            {
                return false;
            }
        }

        if (k.PageSizeMask != 0 && c.Num.TryGetValue("page_size", out double ps))
        {
            int bit = (int)Math.Round(Math.Log2(ps));
            if ((1 << bit) != (int)ps || bit > 31 || (k.PageSizeMask & (1u << bit)) == 0)
            {
                return false;
            }
        }

        if (k.LayoutMask != 0 && c.Num.TryGetValue("kv_layout", out double lay) && (k.LayoutMask & (1u << (int)lay)) == 0)
        {
            return false;
        }

        if (k.GroupSizeMask != 0 && c.Num.TryGetValue("q_heads", out double qh) && c.Num.TryGetValue("kv_heads", out double kh))
        {
            int g = (int)qh / (int)kh;
            if (g < 1 || g > 32 || (k.GroupSizeMask & (1u << (g - 1))) == 0)
            {
                return false;
            }
        }

        return true;
    }
}

/// <summary>A named pass or fail a backend observed while executing a case (output padding, status of a refused call).</summary>
/// <param name="Name">Check name.</param>
/// <param name="Ok">Whether it held.</param>
/// <param name="Detail">Free text.</param>
public sealed record BackendCheck(string Name, bool Ok, string Detail = "");

/// <summary>What one execution returned.</summary>
public sealed class ExecResult : IDisposable
{
    /// <summary>Status of the call (0 is OK).</summary>
    public int Status { get; init; }

    /// <summary>Message for a non-zero status.</summary>
    public string Message { get; init; } = string.Empty;

    /// <summary>The case's outputs by name, logical (compact) layout.</summary>
    public Dictionary<string, HostBuf> Outputs { get; init; } = new(StringComparer.Ordinal);

    /// <summary>Checks the backend made itself.</summary>
    public List<BackendCheck> Checks { get; } = [];

    /// <summary>Releases the outputs.</summary>
    public void Dispose()
    {
        foreach (HostBuf b in Outputs.Values)
        {
            b.Dispose();
        }
    }
}

/// <summary>Something that runs ops: <c>CpuOps</c> in the CI half, a kernel library loaded through C1 in the GPU half.</summary>
public interface IKernelBackend : IDisposable
{
    /// <summary>Backend name for results ("CpuOps" or the library name).</summary>
    string Name { get; }

    /// <summary>Compute capability of the device (major * 10 + minor), 0 for the CPU.</summary>
    int Sm { get; }

    /// <summary>The registered variants.</summary>
    IReadOnlyList<VariantInfo> Variants { get; }

    /// <summary>Runs one case on one variant. Never mutates <paramref name="c"/>.</summary>
    /// <param name="variant">Variant id.</param>
    /// <param name="c">The case.</param>
    /// <returns>Status and outputs.</returns>
    ExecResult Execute(string variant, OpCase c);

    /// <summary>
    /// Calls the variant with a params struct whose <c>struct_size</c> is too small and returns the status, or null when the
    /// backend has no params structs. C1 requires <c>ABI_MISMATCH</c> before any launch.
    /// </summary>
    /// <param name="variant">Variant id.</param>
    /// <param name="c">A case the variant accepts.</param>
    /// <returns>The status.</returns>
    int? ProbeShortStruct(string variant, OpCase c);
}

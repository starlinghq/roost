using System.Globalization;
using System.Numerics;
using System.Text;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Kernels;

/// <summary>
/// A shape bucket for variant selection (R-072): the op and a canonical key over the requirement fields a kernel's speed
/// depends on. Batch and token counts are rounded up to the next power of two, the same bucketing <c>CudaOps</c> uses for
/// GEMM plans, so one autotune measurement covers every call in the bucket. The SM is not part of the key; the selector
/// and the autotune cache key it separately.
/// </summary>
public sealed record ShapeBucket
{
    /// <summary>The longest key accepted, in characters.</summary>
    public const int MaxKeyLength = 256;

    /// <summary>Creates a bucket from an explicit key.</summary>
    /// <param name="op">The op.</param>
    /// <param name="key">Printable ASCII without spaces, at most <see cref="MaxKeyLength"/> characters; <c>*</c> for an op with one bucket.</param>
    /// <exception cref="ArgumentException">The op is <see cref="OpId.None"/> or the key is empty, too long or not printable ASCII.</exception>
    public ShapeBucket(OpId op, string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (op == OpId.None)
        {
            throw new ArgumentException("a shape bucket needs an op", nameof(op));
        }

        if (!IsValidKey(key))
        {
            throw new ArgumentException($"a shape bucket key is 1..{MaxKeyLength} printable ASCII characters without spaces", nameof(key));
        }

        Op = op;
        Key = key;
    }

    /// <summary>The op.</summary>
    public OpId Op { get; }

    /// <summary>The canonical key.</summary>
    public string Key { get; }

    /// <summary>
    /// The bucket a requirement falls in. Fields left null in the requirement are left out of the key, in a fixed order, so
    /// equal requirements always give equal keys. The required flags are part of the key because they narrow the set of
    /// variants the bucket is tuned over.
    /// </summary>
    /// <param name="requirement">The requirement.</param>
    /// <returns>The bucket.</returns>
    public static ShapeBucket For(KernelRequirement requirement)
    {
        ArgumentNullException.ThrowIfNull(requirement);
        var key = new StringBuilder();
        void Add(string name, string? value)
        {
            if (value is null)
            {
                return;
            }

            if (key.Length > 0)
            {
                key.Append(';');
            }

            key.Append(name).Append('=').Append(value);
        }

        static string? Int(long? value) => value?.ToString(CultureInfo.InvariantCulture);

        Add("act", requirement.Activation?.ToString());
        Add("weight", requirement.Weight?.ToString());
        Add("kv", requirement.Kv?.ToString());
        Add("head_dim", Int(requirement.HeadDim));
        Add("layout", requirement.Layout?.ToString());
        Add("page_size", Int(requirement.PageSize));
        Add("group_size", Int(requirement.GroupSize));
        Add("batch", Int(requirement.Batch is { } b ? RoundUp(b) : null));
        Add("tokens", Int(requirement.Tokens is { } t ? RoundUp(t) : null));
        if (requirement.RequiredFlags != VariantTraits.None)
        {
            Add("flags", ((uint)requirement.RequiredFlags).ToString("x", CultureInfo.InvariantCulture));
        }

        return new ShapeBucket(requirement.Op, key.Length == 0 ? "*" : key.ToString());
    }

    /// <summary>The op and key, for messages.</summary>
    public override string ToString() => $"{Op}[{Key}]";

    internal static bool IsValidKey(string? key) =>
        !string.IsNullOrEmpty(key) && key.Length <= MaxKeyLength && key.All(static c => c is > ' ' and < (char)0x7F);

    private static long RoundUp(long value) => value switch
    {
        <= 1 => Math.Max(value, 0),
        > 1L << 62 => 1L << 62,
        _ => (long)BitOperations.RoundUpToPowerOf2((ulong)value),
    };
}

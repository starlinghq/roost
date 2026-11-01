using System.Security.Cryptography;
using System.Text;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels;

/// <summary>
/// A registered variant as the selector (R-072) sees it: the op, the variant id, its flags and a digest that changes
/// whenever the code behind the variant could have changed. Conformance qualification and autotune measurements are
/// pinned to the digest, so a rebuilt library invalidates both.
/// </summary>
public sealed record VariantCandidate
{
    /// <summary>Creates a candidate.</summary>
    /// <param name="op">The op.</param>
    /// <param name="variantId">The variant id (printable ASCII, as C1 registers it).</param>
    /// <param name="digest">Lower-case hex SHA-256 identifying the variant's code.</param>
    /// <param name="flags">The registration flags.</param>
    /// <exception cref="ArgumentException">The op is <see cref="OpId.None"/>, the id is not printable ASCII or the digest is not 64 lower-case hex digits.</exception>
    public VariantCandidate(OpId op, string variantId, string digest, VariantTraits flags = VariantTraits.None)
    {
        ArgumentNullException.ThrowIfNull(variantId);
        ArgumentNullException.ThrowIfNull(digest);
        if (op == OpId.None)
        {
            throw new ArgumentException("a variant needs an op", nameof(op));
        }

        if (!IsValidVariantId(variantId))
        {
            throw new ArgumentException("a variant id is 1..128 printable ASCII characters without spaces", nameof(variantId));
        }

        if (!IsDigest(digest))
        {
            throw new ArgumentException("a variant digest is 64 lower-case hex digits", nameof(digest));
        }

        Op = op;
        VariantId = variantId;
        Digest = digest;
        Flags = flags;
    }

    /// <summary>The op.</summary>
    public OpId Op { get; }

    /// <summary>The variant id.</summary>
    public string VariantId { get; }

    /// <summary>Lower-case hex SHA-256 identifying the variant's code.</summary>
    public string Digest { get; }

    /// <summary>The registration flags.</summary>
    public VariantTraits Flags { get; }

    /// <summary>
    /// The candidate for a registered kernel. Its digest is <see cref="DigestOf"/> over the op, the variant id and the
    /// SHA-256 the manifest records for the library that holds it at the library's SM (the loader has already checked the
    /// file against that SHA-256), so two variants in one library have different digests and a rebuilt library changes all of
    /// them.
    /// </summary>
    /// <param name="kernel">The registered kernel.</param>
    /// <returns>The candidate.</returns>
    /// <exception cref="KernelLoadException">The manifest has no digest for the variant at the library's SM (<see cref="RoostErrorCode.AssetMismatch"/>).</exception>
    public static VariantCandidate FromKernel(RegisteredKernel kernel)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        var library = kernel.Library;
        var binary = library.Manifest.Variants
            .Where(v => v.Op == kernel.Op && v.Id == kernel.VariantId)
            .SelectMany(v => v.Binaries)
            .FirstOrDefault(b => b.Arch == library.Sm)
            ?? throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"the manifest records no library digest for {kernel.VariantId} ({kernel.Op}) at SM {library.Sm}");
        return new VariantCandidate(kernel.Op, kernel.VariantId, DigestOf(kernel.Op, kernel.VariantId, binary.Sha256), kernel.Flags);
    }

    /// <summary>
    /// The variant digest: lower-case hex SHA-256 of the UTF-8 text <c>roost-variant/1</c>, the numeric op id, the variant id
    /// and the code digest, separated by line feeds.
    /// </summary>
    /// <param name="op">The op.</param>
    /// <param name="variantId">The variant id.</param>
    /// <param name="codeSha256">Lower-case hex SHA-256 of the code that holds the variant (a library file, or a compiled cubin).</param>
    /// <returns>The digest.</returns>
    public static string DigestOf(OpId op, string variantId, string codeSha256)
    {
        ArgumentNullException.ThrowIfNull(variantId);
        ArgumentNullException.ThrowIfNull(codeSha256);
        var text = $"roost-variant/1\n{(uint)op}\n{variantId}\n{codeSha256.ToLowerInvariant()}";
        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)));
    }

    /// <summary>The variant id and the first digest digits, for messages.</summary>
    public override string ToString() => $"{VariantId}@{Digest[..12]}";

    internal static bool IsValidVariantId(string? id) =>
        !string.IsNullOrEmpty(id) && id.Length <= 128 && id.All(static c => c is > ' ' and < (char)0x7F);

    internal static bool IsDigest(string? digest) =>
        digest is { Length: 64 } && digest.All(static c => c is (>= '0' and <= '9') or (>= 'a' and <= 'f'));
}

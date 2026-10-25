using System.Globalization;
using System.Security.Cryptography;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;
using Tomlyn;
using Tomlyn.Model;

namespace Starlings.Roost.Kernels;

/// <summary>Validation status of one SM in the manifest (contract C5; AGENTS.md: an SM without hardware evidence is untested).</summary>
public enum SmStatus
{
    /// <summary>No hardware evidence.</summary>
    Untested,

    /// <summary>Ran on hardware; not yet admitted by the fidelity gates.</summary>
    Experimental,

    /// <summary>Admitted.</summary>
    Supported,

    /// <summary>Refused.</summary>
    Unsupported,
}

/// <summary>One <c>[[sm]]</c> entry.</summary>
/// <param name="Arch">The SM as major * 10 + minor.</param>
/// <param name="Status">The validation status.</param>
public sealed record ManifestSm(int Arch, SmStatus Status);

/// <summary>One <c>[[ops.variants.binaries]]</c> entry: the digest of the per-SM library that holds the variant.</summary>
/// <param name="Arch">The SM.</param>
/// <param name="Sha256">Lower-case hex SHA-256 of the library file.</param>
/// <param name="Bytes">The file size, or 0 when the manifest does not record it.</param>
public sealed record ManifestBinary(int Arch, string Sha256, long Bytes);

/// <summary>One <c>[[ops.variants]]</c> entry.</summary>
/// <param name="Op">The op.</param>
/// <param name="Id">The variant id, as the library registers it.</param>
/// <param name="Flags">The registration flags.</param>
/// <param name="SmMin">The constraints' <c>sm_min</c>.</param>
/// <param name="Binaries">The per-SM library digests.</param>
public sealed record ManifestVariant(OpId Op, string Id, VariantTraits Flags, int SmMin, IReadOnlyList<ManifestBinary> Binaries);

/// <summary>
/// The parts of <c>kernels/manifest.toml</c> (contract C5) the loader reads: the ABI, the toolkit, the SM list with
/// statuses, and each op variant's flags and per-SM library digests. The model compiler never reads it.
/// </summary>
public sealed class KernelManifest
{
    private static readonly char[] DotSeparator = ['.'];

    private KernelManifest(
        int abiMajor,
        int abiMinor,
        string cuda,
        string minDriver,
        IReadOnlyList<ManifestSm> sms,
        IReadOnlyList<ManifestVariant> variants,
        string sha256)
    {
        AbiMajor = abiMajor;
        AbiMinor = abiMinor;
        Cuda = cuda;
        MinDriver = minDriver;
        Sms = sms;
        Variants = variants;
        Sha256 = sha256;
    }

    /// <summary>The C1 ABI major the manifest's libraries were built for.</summary>
    public int AbiMajor { get; }

    /// <summary>The C1 ABI minor.</summary>
    public int AbiMinor { get; }

    /// <summary>The toolkit version, <c>toolkit.cuda</c> (for example <c>13.3</c>).</summary>
    public string Cuda { get; }

    /// <summary>The CUDA major, taken from <see cref="Cuda"/>: what the per-OS library resolver uses instead of a constant.</summary>
    public int CudaMajor => int.Parse(Cuda.Split(DotSeparator)[0], CultureInfo.InvariantCulture);

    /// <summary>The minimum driver, <c>toolkit.min_driver</c>.</summary>
    public string MinDriver { get; }

    /// <summary>The SM build list with a status per SM.</summary>
    public IReadOnlyList<ManifestSm> Sms { get; }

    /// <summary>Every op variant of every op.</summary>
    public IReadOnlyList<ManifestVariant> Variants { get; }

    /// <summary>Lower-case hex SHA-256 of the manifest file's bytes: the value C8's handshake and a profile pin.</summary>
    public string Sha256 { get; }

    /// <summary>Reads a manifest file and, when given, refuses one whose digest differs from the pinned one.</summary>
    /// <param name="path">The <c>manifest.toml</c> path.</param>
    /// <param name="expectedSha256">The digest the profile or handshake pins, or null to skip the check.</param>
    /// <returns>The manifest.</returns>
    /// <exception cref="KernelLoadException">The file is unreadable or malformed, or the digest differs (<see cref="RoostErrorCode.AssetMismatch"/>).</exception>
    public static KernelManifest Load(string path, string? expectedSha256 = null)
    {
        byte[] bytes;
        try
        {
            bytes = File.ReadAllBytes(path);
        }
        catch (IOException e)
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"cannot read the kernel manifest {path}: {e.Message}", e);
        }
        catch (UnauthorizedAccessException e)
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"cannot read the kernel manifest {path}: {e.Message}", e);
        }

        return Parse(bytes, expectedSha256);
    }

    /// <summary>
    /// The manifest this assembly was built with (<c>kernels/manifest.toml</c>, embedded at build time). In a released
    /// package its digests are those of the release's kernel library assets, which the release workflow checks (R-060).
    /// </summary>
    /// <returns>The manifest.</returns>
    /// <exception cref="KernelLoadException">The resource is missing or malformed (<see cref="RoostErrorCode.AssetMismatch"/>).</exception>
    public static KernelManifest Embedded()
    {
        using var stream = typeof(KernelManifest).Assembly.GetManifestResourceStream(EmbeddedResourceName)
            ?? throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"the assembly embeds no kernel manifest ({EmbeddedResourceName})");
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        return Parse(buffer.GetBuffer().AsSpan(0, (int)buffer.Length));
    }

    /// <summary>The logical name of the embedded manifest resource.</summary>
    internal const string EmbeddedResourceName = "Starlings.Roost.Kernels.manifest.toml";

    /// <summary>Parses manifest bytes (UTF-8 TOML) and, when given, checks their digest.</summary>
    /// <param name="utf8">The manifest bytes.</param>
    /// <param name="expectedSha256">The pinned digest, or null to skip the check.</param>
    /// <returns>The manifest.</returns>
    /// <exception cref="KernelLoadException">The digest differs, or the manifest is malformed or from another contract.</exception>
    public static KernelManifest Parse(ReadOnlySpan<byte> utf8, string? expectedSha256 = null)
    {
        var digest = Convert.ToHexStringLower(SHA256.HashData(utf8));
        if (expectedSha256 is not null && !string.Equals(expectedSha256, digest, StringComparison.OrdinalIgnoreCase))
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"kernel manifest digest mismatch: pinned {expectedSha256}, file {digest}");
        }

        TomlTable root;
        try
        {
            root = TomlSerializer.Deserialize(System.Text.Encoding.UTF8.GetString(utf8), ManifestTomlContext.Default.TomlTable)
                ?? throw new FormatException("empty manifest");
        }
        catch (Exception e) when (e is FormatException or TomlException or InvalidOperationException)
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"kernel manifest is not valid TOML: {e.Message}", e);
        }

        try
        {
            return Build(root, digest);
        }
        catch (Exception e) when (e is KeyNotFoundException or InvalidCastException or FormatException or OverflowException or ArgumentException)
        {
            throw new KernelLoadException(RoostErrorCode.AssetMismatch, $"kernel manifest does not follow the C5 schema: {e.Message}", e);
        }
    }

    private static KernelManifest Build(TomlTable root, string digest)
    {
        var abi = Table(root, "abi");
        var toolkit = Table(root, "toolkit");
        var sms = Tables(root, "sm")
            .Select(t => new ManifestSm(Int(t, "arch"), ParseStatus(Str(t, "status"))))
            .ToList();
        var variants = new List<ManifestVariant>();
        foreach (var op in Tables(root, "ops"))
        {
            var opName = Str(op, "op");
            if (!Enum.TryParse<OpId>(opName, ignoreCase: false, out var opId) || !Enum.IsDefined(opId))
            {
                throw new FormatException($"op {opName} is not in the C1 op catalogue of this build");
            }

            foreach (var v in Tables(op, "variants"))
            {
                var constraints = v.TryGetValue("constraints", out var c) && c is TomlTable ct ? ct : new TomlTable();
                var binaries = Tables(v, "binaries")
                    .Select(b => new ManifestBinary(
                        Int(b, "arch"),
                        Str(b, "sha256").ToLowerInvariant(),
                        b.TryGetValue("bytes", out var n) ? Convert.ToInt64(n, CultureInfo.InvariantCulture) : 0))
                    .ToList();
                variants.Add(new ManifestVariant(
                    opId,
                    Str(v, "id"),
                    ParseFlags(v),
                    constraints.TryGetValue("sm_min", out var smMin) ? Convert.ToInt32(smMin, CultureInfo.InvariantCulture) : 0,
                    binaries));
            }
        }

        return new KernelManifest(Int(abi, "major"), Int(abi, "minor"), Str(toolkit, "cuda"), Str(toolkit, "min_driver"), sms, variants, digest);
    }

    private static VariantTraits ParseFlags(TomlTable variant)
    {
        var flags = VariantTraits.None;
        if (!variant.TryGetValue("flags", out var raw) || raw is not TomlArray array)
        {
            return flags;
        }

        foreach (var item in array)
        {
            flags |= Convert.ToString(item, CultureInfo.InvariantCulture) switch
            {
                "graph_capturable" => VariantTraits.GraphCapturable,
                "deterministic" => VariantTraits.Deterministic,
                "batch_invariant" => VariantTraits.BatchInvariant,
                "needs_plan" => VariantTraits.NeedsPlan,
                "plan_per_shape" => VariantTraits.PlanPerShape,
                var other => throw new FormatException($"unknown variant flag {other}"),
            };
        }

        return flags;
    }

    private static SmStatus ParseStatus(string status) => status switch
    {
        "untested" => SmStatus.Untested,
        "experimental" => SmStatus.Experimental,
        "supported" => SmStatus.Supported,
        "unsupported" => SmStatus.Unsupported,
        _ => throw new FormatException($"unknown SM status {status}"),
    };

    private static TomlTable Table(TomlTable parent, string key) =>
        (TomlTable)(parent[key] ?? throw new KeyNotFoundException(key));

    private static IEnumerable<TomlTable> Tables(TomlTable parent, string key)
    {
        if (!parent.TryGetValue(key, out var value) || value is null)
        {
            return [];
        }

        return value switch
        {
            TomlTableArray tables => tables,
            TomlArray array => array.Select(x => (TomlTable)x!),
            _ => throw new InvalidCastException($"{key} is not an array of tables"),
        };
    }

    private static string Str(TomlTable t, string key) =>
        t.TryGetValue(key, out var v) && v is string s ? s : throw new KeyNotFoundException(key);

    private static int Int(TomlTable t, string key) =>
        t.TryGetValue(key, out var v) && v is not null ? Convert.ToInt32(v, CultureInfo.InvariantCulture) : throw new KeyNotFoundException(key);
}

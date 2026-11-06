using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Tomlyn;
using Tomlyn.Model;
using Tomlyn.Serialization;

namespace Starlings.Roost.Server.Doctor;

/// <summary>
/// The parts of the kernel manifest (contract C5, <c>kernels/manifest.toml</c>) that <c>roost doctor</c> reports: the ABI,
/// the toolkit's minimum driver, each SM's status and each SM's library digest. A host package names no worker-side namespace
/// (decision 21, <c>tools/check-roost-boundaries.py</c>), so this reads the TOML itself; the worker's loader stays the
/// authority on what is admitted, and this reader only reports. <see cref="RoostDoctor"/> compares the ABI read here with
/// <c>KernelAbi</c>, which is Abstractions and not worker code.
/// </summary>
internal sealed class DoctorManifest
{
    private DoctorManifest(string sha256, int abiMajor, int abiMinor, string cuda, string minDriver, IReadOnlyDictionary<int, string> sms, IReadOnlyDictionary<int, IReadOnlySet<string>> libraries)
    {
        Sha256 = sha256;
        AbiMajor = abiMajor;
        AbiMinor = abiMinor;
        Cuda = cuda;
        MinDriver = minDriver;
        SmStatuses = sms;
        LibraryDigests = libraries;
    }

    /// <summary>Lower-case hex SHA-256 of the file: the value a profile pins and the C8 handshake compares.</summary>
    public string Sha256 { get; }

    /// <summary>The C1 ABI major.</summary>
    public int AbiMajor { get; }

    /// <summary>The C1 ABI minor.</summary>
    public int AbiMinor { get; }

    /// <summary><c>toolkit.cuda</c>.</summary>
    public string Cuda { get; }

    /// <summary><c>toolkit.min_driver</c>.</summary>
    public string MinDriver { get; }

    /// <summary>Each <c>[[sm]]</c> entry's status (<c>untested</c>, <c>experimental</c>, <c>supported</c>, <c>unsupported</c>).</summary>
    public IReadOnlyDictionary<int, string> SmStatuses { get; }

    /// <summary>The library digests the manifest records per SM, over every variant (one library per SM holds every variant).</summary>
    public IReadOnlyDictionary<int, IReadOnlySet<string>> LibraryDigests { get; }

    /// <summary>Parses manifest bytes.</summary>
    /// <param name="utf8">The file's bytes.</param>
    /// <returns>The manifest.</returns>
    /// <exception cref="FormatException">The bytes are not TOML, or not the C5 shape.</exception>
    public static DoctorManifest Parse(byte[] utf8)
    {
        ArgumentNullException.ThrowIfNull(utf8);
        string sha = Convert.ToHexStringLower(SHA256.HashData(utf8));
        TomlTable root;
        try
        {
            root = TomlSerializer.Deserialize(Encoding.UTF8.GetString(utf8), DoctorTomlContext.Default.TomlTable)
                ?? throw new FormatException("the manifest is empty");
        }
        catch (Exception e) when (e is TomlException or InvalidOperationException)
        {
            throw new FormatException($"not valid TOML: {e.Message}", e);
        }

        try
        {
            var abi = (TomlTable)root["abi"];
            var toolkit = (TomlTable)root["toolkit"];
            var sms = new SortedDictionary<int, string>();
            foreach (var sm in Tables(root, "sm"))
            {
                sms[Int(sm, "arch")] = Str(sm, "status");
            }

            var libraries = new SortedDictionary<int, HashSet<string>>();
            foreach (var op in Tables(root, "ops"))
            {
                foreach (var variant in Tables(op, "variants"))
                {
                    foreach (var binary in Tables(variant, "binaries"))
                    {
                        int arch = Int(binary, "arch");
                        if (!libraries.TryGetValue(arch, out var set))
                        {
                            libraries[arch] = set = new HashSet<string>(StringComparer.Ordinal);
                        }

                        set.Add(Str(binary, "sha256").ToLowerInvariant());
                    }
                }
            }

            return new DoctorManifest(
                sha,
                Int(abi, "major"),
                Int(abi, "minor"),
                Str(toolkit, "cuda"),
                Str(toolkit, "min_driver"),
                sms,
                libraries.ToDictionary(p => p.Key, p => (IReadOnlySet<string>)p.Value));
        }
        catch (Exception e) when (e is KeyNotFoundException or InvalidCastException or OverflowException)
        {
            throw new FormatException($"does not follow the C5 schema: {e.Message}", e);
        }
    }

    private static IEnumerable<TomlTable> Tables(TomlTable parent, string key) =>
        !parent.TryGetValue(key, out var value) || value is null
            ? []
            : value switch
            {
                TomlTableArray tables => tables,
                TomlArray array => array.Select(x => (TomlTable)x!),
                _ => throw new InvalidCastException($"{key} is not an array of tables"),
            };

    private static string Str(TomlTable t, string key) =>
        t.TryGetValue(key, out var v) && v is string s ? s : throw new KeyNotFoundException(key);

    private static int Int(TomlTable t, string key) =>
        t.TryGetValue(key, out var v) && v is not null ? Convert.ToInt32(v, CultureInfo.InvariantCulture) : throw new KeyNotFoundException(key);
}

/// <summary>Source-generated Tomlyn metadata for the untyped manifest model, so reading it is trim- and AOT-safe.</summary>
[TomlSerializable(typeof(TomlTable))]
internal sealed partial class DoctorTomlContext : TomlSerializerContext
{
}

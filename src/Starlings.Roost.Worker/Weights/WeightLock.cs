using System.Globalization;
using Tomlyn;
using Tomlyn.Model;
using Tomlyn.Serialization;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Worker.Weights;

/// <summary>Source-generated Tomlyn metadata for the untyped lock model, so reading it is trim- and AOT-safe.</summary>
[TomlSerializable(typeof(TomlTable))]
internal sealed partial class WeightLockTomlContext : TomlSerializerContext
{
}

/// <summary>One file of an asset lock: a path relative to the checkpoint directory, its byte size and its SHA-256.</summary>
/// <param name="Path">The path as the lock records it.</param>
/// <param name="Size">Exact size in bytes.</param>
/// <param name="Sha256">Lower-case hexadecimal SHA-256 of the whole file.</param>
public sealed record LockedFile(string Path, long Size, string Sha256);

/// <summary>
/// The files of an asset lock under <c>evidence/registry/*.lock.toml</c> (R-013): <c>[[files]]</c> entries carrying
/// <c>path</c>, <c>size</c> and <c>sha256</c>, plus the source revision.
/// </summary>
public sealed class WeightLock
{
    private readonly Dictionary<string, LockedFile> _files;

    private WeightLock(string repo, string revision, Dictionary<string, LockedFile> files)
    {
        Repo = repo;
        Revision = revision;
        _files = files;
    }

    /// <summary>The source repository the lock was recorded from.</summary>
    public string Repo { get; }

    /// <summary>The full commit SHA the lock was recorded at.</summary>
    public string Revision { get; }

    /// <summary>All locked files.</summary>
    public IReadOnlyCollection<LockedFile> Files => _files.Values;

    /// <summary>Reads a lock file.</summary>
    /// <param name="path">Path of the <c>.lock.toml</c> file.</param>
    /// <returns>The parsed lock.</returns>
    /// <exception cref="WeightLoadException">The file is unreadable or malformed (<see cref="RoostErrorCode.AssetMismatch"/>).</exception>
    public static WeightLock Load(string path)
    {
        try
        {
            return Parse(File.ReadAllBytes(path));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            throw new WeightLoadException(RoostErrorCode.AssetMismatch, $"cannot read the asset lock {System.IO.Path.GetFileName(path)}: {e.Message}", e);
        }
    }

    /// <summary>Parses lock text.</summary>
    /// <param name="utf8">The TOML bytes.</param>
    /// <returns>The parsed lock.</returns>
    /// <exception cref="WeightLoadException">The text is malformed or a digest is not 64 hexadecimal digits.</exception>
    public static WeightLock Parse(ReadOnlySpan<byte> utf8)
    {
        try
        {
            var root = TomlSerializer.Deserialize(System.Text.Encoding.UTF8.GetString(utf8), WeightLockTomlContext.Default.TomlTable)
                ?? throw new FormatException("empty lock");
            var source = (TomlTable)(root["source"] ?? throw new FormatException("no [source] table"));
            var files = new Dictionary<string, LockedFile>(StringComparer.Ordinal);
            if (root["files"] is not TomlTableArray array)
            {
                throw new FormatException("no [[files]] entries");
            }

            foreach (var t in array)
            {
                var file = new LockedFile(
                    (string)(t["path"] ?? throw new FormatException("a file has no path")),
                    Convert.ToInt64(t["size"] ?? throw new FormatException("a file has no size"), CultureInfo.InvariantCulture),
                    (string)(t["sha256"] ?? throw new FormatException("a file has no sha256")));
                if (file.Size < 0 || !IsHexDigest(file.Sha256) || !files.TryAdd(file.Path, file))
                {
                    throw new FormatException($"bad or duplicate entry for {file.Path}");
                }
            }

            return new WeightLock((string)(source["repo"] ?? string.Empty), (string)(source["revision"] ?? string.Empty), files);
        }
        catch (Exception e) when (e is FormatException or TomlException or InvalidCastException or InvalidOperationException or KeyNotFoundException)
        {
            throw new WeightLoadException(RoostErrorCode.AssetMismatch, $"the asset lock is malformed: {e.Message}", e);
        }
    }

    /// <summary>Finds the entry for a file.</summary>
    /// <param name="path">The path as the lock records it, for example <c>model.safetensors-00001-of-00001.safetensors</c>.</param>
    /// <returns>The entry.</returns>
    /// <exception cref="WeightLoadException">The lock has no such file.</exception>
    public LockedFile Get(string path) =>
        _files.TryGetValue(path, out var f)
            ? f
            : throw new WeightLoadException(RoostErrorCode.AssetMismatch, $"the asset lock has no entry for {path}");

    private static bool IsHexDigest(string s) =>
        s.Length == 64 && s.All(static c => c is (>= '0' and <= '9') or (>= 'a' and <= 'f'));
}

using System.Text.Json;
using System.Text.Json.Serialization;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Kernels;

/// <summary>One autotune measurement of a variant in a shape bucket.</summary>
/// <param name="MedianMicroseconds">Median time of one call, in microseconds; finite and positive.</param>
/// <param name="Samples">How many timed calls the median is over; positive.</param>
public readonly record struct AutotuneMeasurement(double MedianMicroseconds, int Samples)
{
    internal bool IsValid => double.IsFinite(MedianMicroseconds) && MedianMicroseconds > 0 && Samples > 0;
}

/// <summary>One entry of the cache, with its full key.</summary>
/// <param name="Sm">The SM.</param>
/// <param name="DriverVersion">The driver version.</param>
/// <param name="Bucket">The shape bucket.</param>
/// <param name="VariantId">The variant id.</param>
/// <param name="Digest">The variant digest the measurement is for.</param>
/// <param name="Measurement">The measurement.</param>
public sealed record AutotuneEntry(int Sm, string DriverVersion, ShapeBucket Bucket, string VariantId, string Digest, AutotuneMeasurement Measurement);

/// <summary>What opening a cache file found.</summary>
public enum AutotuneLoadOutcome
{
    /// <summary>No file; the cache starts empty.</summary>
    Missing,

    /// <summary>The file was read; entries that failed validation, if any, were dropped (<see cref="AutotuneCache.Discarded"/>).</summary>
    Loaded,

    /// <summary>The file was not valid JSON for the format, or could not be read; the cache starts empty and the next save replaces it.</summary>
    Corrupt,

    /// <summary>The file has another format version; the cache starts empty and the next save replaces it.</summary>
    VersionMismatch,

    /// <summary>The file is for another SM or driver version; every entry is invalid and the cache starts empty.</summary>
    DeviceChanged,
}

/// <summary>
/// The persisted autotune cache of R-072 (bet X11): a median time per variant, keyed by SM, driver version, shape bucket
/// and variant digest. A cache is opened for one device (SM and driver), so a file written for another SM or driver is
/// invalidated whole; an entry whose stored digest differs from the candidate's is invalidated on lookup, so a rebuilt
/// variant is measured again. Saves are atomic (write to a temporary file in the same directory, flush to disk, rename
/// over), and a file that cannot be read or parsed is treated as empty rather than failing the worker. Format: JSON,
/// <c>"format": "roost-autotune"</c>, <c>"version": 1</c> (<see cref="FormatVersion"/>). Thread-safe.
/// </summary>
public sealed class AutotuneCache
{
    /// <summary>The on-disk format version this build reads and writes.</summary>
    public const int FormatVersion = 1;

    /// <summary>The <c>format</c> field of the file.</summary>
    public const string FormatName = "roost-autotune";

    /// <summary>The largest file read, in bytes; a larger file is treated as corrupt.</summary>
    public const long MaxFileBytes = 16L << 20;

    private readonly Lock _lock = new();
    private readonly Dictionary<(OpId Op, string Bucket, string VariantId), (string Digest, AutotuneMeasurement Measurement)> _entries = [];

    private AutotuneCache(int sm, string driverVersion, string? path)
    {
        if (sm <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sm), sm, "the SM is major * 10 + minor");
        }

        if (!ShapeBucket.IsValidKey(driverVersion))
        {
            throw new ArgumentException("the driver version is printable ASCII without spaces", nameof(driverVersion));
        }

        Sm = sm;
        DriverVersion = driverVersion;
        Path = path;
    }

    /// <summary>The SM the cache is for.</summary>
    public int Sm { get; }

    /// <summary>The driver version the cache is for (for example <c>580.65.06</c>).</summary>
    public string DriverVersion { get; }

    /// <summary>The file, or null for an in-memory cache.</summary>
    public string? Path { get; }

    /// <summary>What opening the file found.</summary>
    public AutotuneLoadOutcome LoadOutcome { get; private set; } = AutotuneLoadOutcome.Missing;

    /// <summary>Entries dropped at load (invalid or duplicate) or invalidated by a digest mismatch since.</summary>
    public int Discarded { get; private set; }

    /// <summary>The number of entries.</summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _entries.Count;
            }
        }
    }

    /// <summary>Every entry, in key order (op, bucket, variant id; ordinal).</summary>
    public IReadOnlyList<AutotuneEntry> Entries
    {
        get
        {
            lock (_lock)
            {
                return Ordered().Select(e => new AutotuneEntry(Sm, DriverVersion, new ShapeBucket(e.Key.Op, e.Key.Bucket), e.Key.VariantId, e.Value.Digest, e.Value.Measurement)).ToList();
            }
        }
    }

    /// <summary>An empty cache that is never saved to disk.</summary>
    /// <param name="sm">The device's SM.</param>
    /// <param name="driverVersion">The driver version.</param>
    /// <returns>The cache.</returns>
    public static AutotuneCache InMemory(int sm, string driverVersion) => new(sm, driverVersion, null);

    /// <summary>Opens the cache file for a device; a missing, unreadable, corrupt, other-version or other-device file gives an empty cache.</summary>
    /// <param name="path">The file.</param>
    /// <param name="sm">The device's SM.</param>
    /// <param name="driverVersion">The driver version.</param>
    /// <returns>The cache, with <see cref="LoadOutcome"/> set.</returns>
    public static AutotuneCache Open(string path, int sm, string driverVersion)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var cache = new AutotuneCache(sm, driverVersion, System.IO.Path.GetFullPath(path));
        cache.LoadOutcome = cache.Load();
        return cache;
    }

    /// <summary>The measurement of a candidate in a bucket. An entry stored for another digest of the same variant is removed.</summary>
    /// <param name="bucket">The bucket.</param>
    /// <param name="candidate">The candidate.</param>
    /// <param name="measurement">The measurement, when found.</param>
    /// <returns>True when a measurement for the candidate's digest exists.</returns>
    public bool TryGet(ShapeBucket bucket, VariantCandidate candidate, out AutotuneMeasurement measurement)
    {
        ArgumentNullException.ThrowIfNull(bucket);
        ArgumentNullException.ThrowIfNull(candidate);
        CheckOp(bucket, candidate);
        lock (_lock)
        {
            var key = (bucket.Op, bucket.Key, candidate.VariantId);
            if (_entries.TryGetValue(key, out var entry))
            {
                if (entry.Digest == candidate.Digest)
                {
                    measurement = entry.Measurement;
                    return true;
                }

                _entries.Remove(key);
                Discarded++;
            }

            measurement = default;
            return false;
        }
    }

    /// <summary>Records (or replaces) the measurement of a candidate in a bucket.</summary>
    /// <param name="bucket">The bucket.</param>
    /// <param name="candidate">The candidate.</param>
    /// <param name="measurement">The measurement.</param>
    /// <exception cref="ArgumentException">The measurement is not finite and positive, or the bucket's op is not the candidate's.</exception>
    public void Record(ShapeBucket bucket, VariantCandidate candidate, AutotuneMeasurement measurement)
    {
        ArgumentNullException.ThrowIfNull(bucket);
        ArgumentNullException.ThrowIfNull(candidate);
        CheckOp(bucket, candidate);
        if (!measurement.IsValid)
        {
            throw new ArgumentException("a measurement has a finite positive median and a positive sample count", nameof(measurement));
        }

        lock (_lock)
        {
            _entries[(bucket.Op, bucket.Key, candidate.VariantId)] = (candidate.Digest, measurement);
        }
    }

    /// <summary>Removes every entry of a bucket, for example after the selection for it is reset.</summary>
    /// <param name="bucket">The bucket.</param>
    /// <returns>How many entries were removed.</returns>
    public int Invalidate(ShapeBucket bucket)
    {
        ArgumentNullException.ThrowIfNull(bucket);
        lock (_lock)
        {
            var keys = _entries.Keys.Where(k => k.Op == bucket.Op && k.Bucket == bucket.Key).ToList();
            foreach (var key in keys)
            {
                _entries.Remove(key);
            }

            return keys.Count;
        }
    }

    /// <summary>
    /// Writes the cache to <see cref="Path"/> atomically: a reader sees the old file or the new one, never a partial write.
    /// Entries are written in key order, so equal caches give byte-identical files.
    /// </summary>
    /// <exception cref="InvalidOperationException">The cache is in memory only.</exception>
    /// <exception cref="IOException">The file could not be written; the previous file, if any, is unchanged.</exception>
    public void Save()
    {
        var path = Path ?? throw new InvalidOperationException("an in-memory autotune cache has no file");
        AutotuneCacheFile file;
        lock (_lock)
        {
            file = new AutotuneCacheFile
            {
                Format = FormatName,
                Version = FormatVersion,
                Sm = Sm,
                Driver = DriverVersion,
                Entries = Ordered().Select(e => new AutotuneCacheFileEntry
                {
                    Op = (uint)e.Key.Op,
                    Bucket = e.Key.Bucket,
                    Variant = e.Key.VariantId,
                    Digest = e.Value.Digest,
                    MedianUs = e.Value.Measurement.MedianMicroseconds,
                    Samples = e.Value.Measurement.Samples,
                }).ToList<AutotuneCacheFileEntry?>(),
            };
        }

        var directory = System.IO.Path.GetDirectoryName(path)!;
        Directory.CreateDirectory(directory);
        var temp = System.IO.Path.Combine(directory, $".{System.IO.Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            using (var stream = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.None))
            {
                JsonSerializer.Serialize(stream, file, AutotuneJsonContext.Default.AutotuneCacheFile);
                stream.Flush(flushToDisk: true);
            }

            File.Move(temp, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(temp))
            {
                File.Delete(temp);
            }
        }
    }

    private static void CheckOp(ShapeBucket bucket, VariantCandidate candidate)
    {
        if (bucket.Op != candidate.Op)
        {
            throw new ArgumentException($"bucket {bucket} is for {bucket.Op}, the candidate for {candidate.Op}", nameof(candidate));
        }
    }

    private IEnumerable<KeyValuePair<(OpId Op, string Bucket, string VariantId), (string Digest, AutotuneMeasurement Measurement)>> Ordered() =>
        _entries
            .OrderBy(e => (uint)e.Key.Op)
            .ThenBy(e => e.Key.Bucket, StringComparer.Ordinal)
            .ThenBy(e => e.Key.VariantId, StringComparer.Ordinal);

    private AutotuneLoadOutcome Load()
    {
        AutotuneCacheFile? file;
        try
        {
            var info = new FileInfo(Path!);
            if (!info.Exists)
            {
                return AutotuneLoadOutcome.Missing;
            }

            if (info.Length > MaxFileBytes)
            {
                return AutotuneLoadOutcome.Corrupt;
            }

            using var stream = new FileStream(Path!, FileMode.Open, FileAccess.Read, FileShare.Read);
            file = JsonSerializer.Deserialize(stream, AutotuneJsonContext.Default.AutotuneCacheFile);
        }
        catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException or NotSupportedException)
        {
            return AutotuneLoadOutcome.Corrupt;
        }

        if (file is null || file.Format != FormatName)
        {
            return AutotuneLoadOutcome.Corrupt;
        }

        if (file.Version != FormatVersion)
        {
            return AutotuneLoadOutcome.VersionMismatch;
        }

        if (file.Sm != Sm || file.Driver != DriverVersion)
        {
            return AutotuneLoadOutcome.DeviceChanged;
        }

        var seen = new HashSet<(OpId, string, string)>();
        foreach (var entry in file.Entries ?? [])
        {
            var op = (OpId)(entry?.Op ?? 0);
            var measurement = new AutotuneMeasurement(entry?.MedianUs ?? 0, entry?.Samples ?? 0);
            if (entry is null || op == OpId.None || !Enum.IsDefined(op) || !ShapeBucket.IsValidKey(entry.Bucket)
                || !VariantCandidate.IsValidVariantId(entry.Variant) || !VariantCandidate.IsDigest(entry.Digest) || !measurement.IsValid
                || !seen.Add((op, entry.Bucket!, entry.Variant!)))
            {
                Discarded++;
                continue;
            }

            _entries[(op, entry.Bucket!, entry.Variant!)] = (entry.Digest!, measurement);
        }

        return AutotuneLoadOutcome.Loaded;
    }
}

/// <summary>The on-disk document (format version 1).</summary>
internal sealed class AutotuneCacheFile
{
    public string? Format { get; set; }

    public int Version { get; set; }

    public int Sm { get; set; }

    public string? Driver { get; set; }

    public List<AutotuneCacheFileEntry?>? Entries { get; set; }
}

/// <summary>One entry on disk.</summary>
internal sealed class AutotuneCacheFileEntry
{
    public uint Op { get; set; }

    public string? Bucket { get; set; }

    public string? Variant { get; set; }

    public string? Digest { get; set; }

    public double MedianUs { get; set; }

    public int Samples { get; set; }
}

/// <summary>The source-generated serializer for the cache file (no reflection; trim- and AOT-safe).</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true)]
[JsonSerializable(typeof(AutotuneCacheFile))]
internal sealed partial class AutotuneJsonContext : JsonSerializerContext;

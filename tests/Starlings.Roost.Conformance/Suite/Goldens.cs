using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Win32.SafeHandles;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>A safetensors file read without any native library: an 8-byte header length, a JSON header, then raw tensor bytes.</summary>
public sealed class SafeTensorsFile : IDisposable
{
    private readonly SafeFileHandle _handle;
    private readonly long _dataStart;

    private SafeTensorsFile(SafeFileHandle handle, long dataStart, Dictionary<string, TensorEntry> tensors, Dictionary<string, string> metadata)
    {
        _handle = handle;
        _dataStart = dataStart;
        Tensors = tensors;
        Metadata = metadata;
    }

    /// <summary>Where a tensor lives in the file.</summary>
    /// <param name="Dtype">safetensors dtype name ("BF16", "F16", "F32", "F64", "I32", "I64", "U8").</param>
    /// <param name="Shape">Dimensions.</param>
    /// <param name="Begin">First byte, relative to the data section.</param>
    /// <param name="End">One past the last byte.</param>
    public sealed record TensorEntry(string Dtype, long[] Shape, long Begin, long End);

    /// <summary>Tensors by name.</summary>
    public IReadOnlyDictionary<string, TensorEntry> Tensors { get; }

    /// <summary>The header's <c>__metadata__</c> (string values).</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>Opens a file and parses its header.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The open file.</returns>
    /// <exception cref="InvalidDataException">The header is malformed.</exception>
    public static SafeTensorsFile Open(string path)
    {
        SafeFileHandle h = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
        try
        {
            long length = RandomAccess.GetLength(h);
            Span<byte> lenBytes = stackalloc byte[8];
            if (RandomAccess.Read(h, lenBytes, 0) != 8)
            {
                throw new InvalidDataException("safetensors: file shorter than 8 bytes.");
            }

            ulong headerLength = BitConverter.ToUInt64(lenBytes);
            if (headerLength == 0 || headerLength > 64UL * 1024 * 1024 || 8 + (long)headerLength > length)
            {
                throw new InvalidDataException($"safetensors: implausible header length {headerLength}.");
            }

            var header = new byte[(int)headerLength];
            if (RandomAccess.Read(h, header, 8) != header.Length)
            {
                throw new InvalidDataException("safetensors: truncated header.");
            }

            long dataStart = 8 + (long)headerLength;
            var tensors = new Dictionary<string, TensorEntry>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
            using var doc = JsonDocument.Parse(header);
            foreach (JsonProperty p in doc.RootElement.EnumerateObject())
            {
                if (p.Name == "__metadata__")
                {
                    foreach (JsonProperty m in p.Value.EnumerateObject())
                    {
                        metadata[m.Name] = m.Value.GetString() ?? string.Empty;
                    }

                    continue;
                }

                long[] shape = [.. p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64())];
                JsonElement offs = p.Value.GetProperty("data_offsets");
                long b = offs[0].GetInt64(), e2 = offs[1].GetInt64();
                if (b < 0 || e2 < b || dataStart + e2 > length)
                {
                    throw new InvalidDataException($"safetensors: tensor '{p.Name}' lies outside the file.");
                }

                tensors[p.Name] = new TensorEntry(p.Value.GetProperty("dtype").GetString() ?? string.Empty, shape, b, e2);
            }

            return new SafeTensorsFile(h, dataStart, tensors, metadata);
        }
        catch
        {
            h.Dispose();
            throw;
        }
    }

    /// <summary>The bytes of one tensor.</summary>
    /// <param name="name">Tensor name.</param>
    /// <returns>A new array.</returns>
    public byte[] ReadBytes(string name)
    {
        TensorEntry t = Tensors[name];
        var bytes = new byte[t.End - t.Begin];
        int got = 0;
        while (got < bytes.Length)
        {
            int n = RandomAccess.Read(_handle, bytes.AsSpan(got), _dataStart + t.Begin + got);
            if (n <= 0)
            {
                throw new InvalidDataException($"safetensors: short read of '{name}'.");
            }

            got += n;
        }

        return bytes;
    }

    /// <summary>Releases the file.</summary>
    public void Dispose() => _handle.Dispose();

    /// <summary>The suite's element type for a safetensors dtype.</summary>
    /// <param name="dtype">The name.</param>
    /// <returns>The type.</returns>
    public static DType ToDType(string dtype) => dtype switch
    {
        "BF16" => DType.BF16,
        "F16" => DType.F16,
        "F32" => DType.F32,
        "F64" => DType.F64,
        "I32" => DType.I32,
        "I64" => DType.I64,
        "U8" => DType.U8,
        _ => throw new InvalidDataException($"safetensors dtype {dtype} is not used by the goldens."),
    };
}

/// <summary>What the lock file pins about the goldens asset.</summary>
/// <param name="Asset">File name of the release asset.</param>
/// <param name="Sha256">Digest of the asset.</param>
/// <param name="SizeBytes">Size of the asset.</param>
/// <param name="GeneratorSha256">Digest of the generator script that produced it.</param>
/// <param name="CaseCount">Number of cases in it.</param>
public sealed record GoldenLock(string Asset, string Sha256, long SizeBytes, string GeneratorSha256, int CaseCount)
{
    /// <summary>File name of the committed lock.</summary>
    public const string FileName = "goldens.lock.json";

    /// <summary>Reads a lock file.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The lock.</returns>
    public static GoldenLock Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        JsonElement r = doc.RootElement;
        return new GoldenLock(
            r.GetProperty("asset").GetProperty("name").GetString()!,
            r.GetProperty("asset").GetProperty("sha256").GetString()!,
            r.GetProperty("asset").GetProperty("size_bytes").GetInt64(),
            r.GetProperty("generator").GetProperty("sha256").GetString()!,
            r.GetProperty("case_count").GetInt32());
    }
}

/// <summary>The upstream golden cases of one goldens file.</summary>
public sealed class GoldenSet : IDisposable
{
    private readonly SafeTensorsFile _file;

    private GoldenSet(SafeTensorsFile file, string path, string sha256)
    {
        _file = file;
        Path = path;
        Sha256 = sha256;
    }

    /// <summary>The file.</summary>
    public string Path { get; }

    /// <summary>Its SHA-256.</summary>
    public string Sha256 { get; }

    /// <summary>Descriptions recorded by the generator (versions, seed).</summary>
    public IReadOnlyDictionary<string, string> Metadata => _file.Metadata;

    /// <summary>Opens a goldens file, checking it against the lock when one is given.</summary>
    /// <param name="path">The goldens file (a release asset, never committed).</param>
    /// <param name="lockPath">The committed lock, or null to skip the digest check.</param>
    /// <returns>The set.</returns>
    /// <exception cref="InvalidDataException">The digest or the format does not match.</exception>
    public static GoldenSet Open(string path, string? lockPath)
    {
        string sha;
        using (FileStream fs = File.OpenRead(path))
        {
            sha = Convert.ToHexStringLower(SHA256.HashData(fs));
        }

        if (lockPath is not null)
        {
            GoldenLock l = GoldenLock.Load(lockPath);
            if (!string.Equals(l.Sha256, sha, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"The goldens file {path} has sha256 {sha}, the lock {lockPath} pins {l.Sha256}.");
            }
        }

        var file = SafeTensorsFile.Open(path);
        if (!file.Metadata.TryGetValue("format", out string? f) || f != "roost.conformance-goldens")
        {
            file.Dispose();
            throw new InvalidDataException($"{path} is not a Roost conformance goldens file.");
        }

        return new GoldenSet(file, path, sha);
    }

    /// <summary>Builds the golden cases of an op.</summary>
    /// <param name="op">The op.</param>
    /// <returns>Cases whose <see cref="OpCase.Expected"/> holds the upstream result; the caller disposes them.</returns>
    public IEnumerable<OpCase> Cases(OpId op)
    {
        using var doc = JsonDocument.Parse(_file.Metadata["cases"]);
        foreach (JsonElement d in doc.RootElement.EnumerateArray())
        {
            if (!Enum.TryParse(d.GetProperty("op").GetString(), out OpId o) || o != op)
            {
                continue;
            }

            string id = d.GetProperty("id").GetString()!;
            var c = new OpCase
            {
                Id = "golden-" + id,
                Op = o,
                Dtype = SafeTensorsFile.ToDType(d.GetProperty("dtype").GetString()!),
                ToleranceClass = d.GetProperty("class").GetString()!,
                Source = "golden",
                Seed = d.TryGetProperty("seed", out var s) ? s.GetUInt64() : 0,
                ScaleBuffer = d.TryGetProperty("scale_buffer", out var sb) && sb.ValueKind == JsonValueKind.String ? sb.GetString() : null,
            };
            foreach (JsonProperty p in d.GetProperty("params").EnumerateObject())
            {
                c.Num[p.Name] = p.Value.GetDouble();
            }

            foreach (JsonElement name in d.GetProperty("inputs").EnumerateArray())
            {
                string n = name.GetString()!;
                string key = id + "/" + n;
                c.Buffers[n] = HostBuf.FromBytes(SafeTensorsFile.ToDType(_file.Tensors[key].Dtype), _file.ReadBytes(key));
            }

            c.Expected = new Dictionary<string, HostBuf>(StringComparer.Ordinal);
            foreach (JsonElement outEl in d.GetProperty("outputs").EnumerateArray())
            {
                string n = outEl.GetProperty("name").GetString()!;
                DType target = SafeTensorsFile.ToDType(outEl.GetProperty("dtype").GetString()!);
                string key = id + "/expected/" + n;
                SafeTensorsFile.TensorEntry t = _file.Tensors[key];
                using HostBuf raw = HostBuf.FromBytes(SafeTensorsFile.ToDType(t.Dtype), _file.ReadBytes(key));
                HostBuf expected;
                if (raw.Dtype == target)
                {
                    expected = raw.Clone();
                }
                else
                {
                    // The reference is stored in double and rounded once to the output type here, as CpuOps rounds.
                    expected = new HostBuf(target, raw.Count);
                    for (long i = 0; i < raw.Count; i++)
                    {
                        expected.Set(i, raw.Get(i));
                    }
                }

                c.Expected[n] = expected;
                c.Outputs.Add(n);
                if (!c.Buffers.ContainsKey(n))
                {
                    c.Buffers[n] = outEl.TryGetProperty("initial", out var init) && init.ValueKind == JsonValueKind.Number
                        ? new HostBuf(target, expected.Count).FillBytes((byte)init.GetInt32())
                        : new HostBuf(target, expected.Count);
                }
            }

            yield return c;
        }
    }

    /// <summary>Releases the file.</summary>
    public void Dispose() => _file.Dispose();
}

using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Starlings.Roost.Models.Qwen35.Tests.Parity;

/// <summary>
/// Reads the metadata of, and writes, a per-layer capture in the R-011 layout (<c>evidence/pipeline/src/roost_evidence/layer_captures.py</c>):
/// safetensors with float32 tensors only, a <c>__metadata__</c> block of strings, the header padded with spaces to a multiple of 8
/// bytes and the tensor bytes in ordinal name order, as <c>write_capture</c> lays them out. Test-time only.
/// </summary>
internal static class LayerCaptureFile
{
    /// <summary>The <c>__metadata__</c> block of a capture.</summary>
    public static Dictionary<string, string> ReadMetadata(string path)
    {
        using var stream = File.OpenRead(path);
        Span<byte> prefix = stackalloc byte[8];
        stream.ReadExactly(prefix);
        long size = (long)BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (size <= 0 || size > 64 * 1024 * 1024)
        {
            throw new InvalidDataException($"{path}: header length {size}");
        }

        var header = new byte[size];
        stream.ReadExactly(header);
        using var doc = JsonDocument.Parse(header);
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var e in doc.RootElement.GetProperty("__metadata__").EnumerateObject())
        {
            metadata[e.Name] = e.Value.GetString()!;
        }

        return metadata;
    }

    /// <summary>Writes the capture and returns its SHA-256 (lower-case hex).</summary>
    public static string Write(string path, IReadOnlyDictionary<string, string> metadata, IReadOnlyDictionary<string, (int[] Shape, float[] Values)> tensors)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentNullException.ThrowIfNull(tensors);
        var names = tensors.Keys.ToArray();
        Array.Sort(names, StringComparer.Ordinal);
        using var headerStream = new MemoryStream();
        using (var w = new Utf8JsonWriter(headerStream))
        {
            w.WriteStartObject();
            w.WriteStartObject("__metadata__");
            foreach (var key in metadata.Keys.Order(StringComparer.Ordinal))
            {
                w.WriteString(key, metadata[key]);
            }

            w.WriteEndObject();
            long offset = 0;
            foreach (string name in names)
            {
                var (shape, values) = tensors[name];
                long elements = shape.Aggregate(1L, (p, d) => p * d);
                if (elements != values.LongLength)
                {
                    throw new ArgumentException($"tensor {name}: {values.LongLength} values for shape [{string.Join(",", shape)}]");
                }

                w.WriteStartObject(name);
                w.WriteString("dtype", "F32");
                w.WriteStartArray("shape");
                foreach (int d in shape)
                {
                    w.WriteNumberValue(d);
                }

                w.WriteEndArray();
                w.WriteStartArray("data_offsets");
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset + (4 * elements));
                w.WriteEndArray();
                w.WriteEndObject();
                offset += 4 * elements;
            }

            w.WriteEndObject();
        }

        var header = headerStream.ToArray().ToList();
        while (header.Count % 8 != 0)
        {
            header.Add((byte)' ');
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using (var file = File.Create(path))
        {
            Span<byte> prefix = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)header.Count);
            file.Write(prefix);
            file.Write(header.ToArray());
            foreach (string name in names)
            {
                WriteFloats(file, tensors[name].Values);
            }
        }

        using var read = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(read));
    }

    /// <summary>Writes little-endian float32 values (the step-0 sidecar layout of <c>stage_results.write_float32</c>).</summary>
    public static void WriteFloats(Stream stream, ReadOnlySpan<float> values)
    {
        ArgumentNullException.ThrowIfNull(stream);
        var buffer = new byte[64 * 1024];
        int at = 0;
        foreach (float v in values)
        {
            BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(at), v);
            at += 4;
            if (at == buffer.Length)
            {
                stream.Write(buffer, 0, at);
                at = 0;
            }
        }

        stream.Write(buffer, 0, at);
    }

    /// <summary>Reads a JSON list of integers from a metadata string.</summary>
    public static int[] Ints(string json) => JsonSerializer.Deserialize<int[]>(json) ?? [];

    /// <summary>Reads a JSON list of strings from a metadata string.</summary>
    public static string[] Strings(string json) => JsonSerializer.Deserialize<string[]>(json) ?? [];

    /// <summary>UTF-8 without a byte-order mark, for the text files a run writes.</summary>
    public static Encoding Utf8 { get; } = new UTF8Encoding(false);
}

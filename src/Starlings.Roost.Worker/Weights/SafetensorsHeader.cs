using System.Buffers.Binary;
using System.Globalization;
using System.Text.Json;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Worker.Weights;

/// <summary>Safetensors dtype names and sizes.</summary>
internal static class SafetensorsDType
{
    /// <summary>Maps a safetensors dtype name to the Roost dtype, or <see cref="DType.Undefined"/> when it has none.</summary>
    public static DType Parse(string name) => name switch
    {
        "F32" => DType.F32,
        "F16" => DType.F16,
        "BF16" => DType.BF16,
        "F64" => DType.F64,
        "F8_E4M3" => DType.F8E4M3,
        "F8_E5M2" => DType.F8E5M2,
        "I8" => DType.I8,
        "U8" => DType.U8,
        "I16" => DType.I16,
        "I32" => DType.I32,
        "U32" => DType.U32,
        "I64" => DType.I64,
        "U64" => DType.U64,
        _ => DType.Undefined,
    };

    /// <summary>Bytes of one element of whole-byte types; 0 for sub-byte or unknown types.</summary>
    public static int ElementSize(DType d) => d switch
    {
        DType.F32 or DType.I32 or DType.U32 => 4,
        DType.F16 or DType.BF16 or DType.I16 => 2,
        DType.F64 or DType.I64 or DType.U64 => 8,
        DType.F8E4M3 or DType.F8E5M2 or DType.I8 or DType.U8 => 1,
        _ => 0,
    };

    /// <summary>Bytes of <paramref name="elements"/> elements.</summary>
    public static long ByteLength(DType d, long elements) => checked(elements * ElementSize(d));
}

/// <summary>One tensor of a safetensors header.</summary>
/// <param name="Name">Tensor name.</param>
/// <param name="Dtype">Roost dtype.</param>
/// <param name="Elements">Element count (product of the shape).</param>
/// <param name="Begin">First byte, relative to the end of the header.</param>
/// <param name="End">One past the last byte, relative to the end of the header.</param>
internal readonly record struct SafetensorsTensor(string Name, DType Dtype, long Elements, long Begin, long End);

/// <summary>
/// A parsed and validated safetensors header: an 8-byte little-endian length, then that many bytes of JSON, then the
/// data. Validation is strict because a substituted file is exactly the input this must reject before any device memory
/// is allocated: the length is capped, every tensor has a known dtype and a byte range that matches its shape, and the
/// ranges tile the data region with no gap, overlap or trailing byte.
/// </summary>
internal sealed class SafetensorsHeader
{
    /// <summary>The largest header accepted (the format's own reference reader caps it at 100 MB).</summary>
    public const long MaxHeaderBytes = 100L * 1024 * 1024;

    private SafetensorsHeader(long headerBytes, long fileSize, SafetensorsTensor[] tensors)
    {
        HeaderBytes = headerBytes;
        DataStart = 8 + headerBytes;
        FileSize = fileSize;
        Tensors = tensors;
        ByName = tensors.ToDictionary(static t => t.Name, StringComparer.Ordinal);
    }

    /// <summary>Length of the JSON part.</summary>
    public long HeaderBytes { get; }

    /// <summary>Absolute offset of the first data byte.</summary>
    public long DataStart { get; }

    /// <summary>Total file size.</summary>
    public long FileSize { get; }

    /// <summary>Tensors sorted by <see cref="SafetensorsTensor.Begin"/>.</summary>
    public SafetensorsTensor[] Tensors { get; }

    /// <summary>Tensors by name.</summary>
    public Dictionary<string, SafetensorsTensor> ByName { get; }

    /// <summary>Reads the 8-byte length prefix and checks it against the file size.</summary>
    public static long ReadLength(ReadOnlySpan<byte> prefix, long fileSize)
    {
        var n = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (n == 0 || n > (ulong)MaxHeaderBytes || (long)n > fileSize - 8)
        {
            throw Mismatch($"the safetensors header length {n} is not valid for a file of {fileSize} bytes");
        }

        return (long)n;
    }

    /// <summary>Parses the JSON part.</summary>
    public static SafetensorsHeader Parse(ReadOnlySpan<byte> json, long fileSize)
    {
        var headerBytes = json.Length;
        var list = new List<SafetensorsTensor>();
        try
        {
            using var doc = JsonDocument.Parse(json.ToArray(), new JsonDocumentOptions { AllowTrailingCommas = false });
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw Mismatch("the safetensors header is not a JSON object");
            }

            foreach (var p in doc.RootElement.EnumerateObject())
            {
                if (p.Name == "__metadata__")
                {
                    continue;
                }

                var dtype = SafetensorsDType.Parse(p.Value.GetProperty("dtype").GetString() ?? string.Empty);
                if (SafetensorsDType.ElementSize(dtype) == 0)
                {
                    throw Mismatch($"tensor {p.Name} has an unsupported dtype");
                }

                long elements = 1;
                foreach (var d in p.Value.GetProperty("shape").EnumerateArray())
                {
                    elements = checked(elements * d.GetInt64());
                }

                var offsets = p.Value.GetProperty("data_offsets");
                if (offsets.GetArrayLength() != 2)
                {
                    throw Mismatch($"tensor {p.Name} has malformed data_offsets");
                }

                var begin = offsets[0].GetInt64();
                var end = offsets[1].GetInt64();
                if (elements < 0 || begin < 0 || end < begin || end - begin != SafetensorsDType.ByteLength(dtype, elements))
                {
                    throw Mismatch($"tensor {p.Name} has a byte range that does not match its shape and dtype");
                }

                list.Add(new SafetensorsTensor(p.Name, dtype, elements, begin, end));
            }
        }
        catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException or FormatException or OverflowException)
        {
            throw Mismatch($"the safetensors header is malformed: {e.Message}", e);
        }

        list.Sort(static (a, b) => a.Begin != b.Begin ? a.Begin.CompareTo(b.Begin) : a.End.CompareTo(b.End));
        long cursor = 0;
        foreach (var t in list)
        {
            if (t.Begin != cursor)
            {
                throw Mismatch($"tensor {t.Name} starts at {t.Begin} but the previous data ends at {cursor}");
            }

            cursor = t.End;
        }

        var dataLength = fileSize - 8 - headerBytes;
        if (cursor != dataLength)
        {
            throw Mismatch($"the tensors cover {cursor} data bytes but the file holds {dataLength}");
        }

        return new SafetensorsHeader(headerBytes, fileSize, [.. list]);
    }

    private static WeightLoadException Mismatch(string message, Exception? inner = null) =>
        new(RoostErrorCode.AssetMismatch, message, inner);

    /// <summary>Formats a tensor for messages.</summary>
    public static string Describe(in SafetensorsTensor t) =>
        string.Create(CultureInfo.InvariantCulture, $"{t.Name} ({t.Dtype}, {t.Elements} elements)");
}

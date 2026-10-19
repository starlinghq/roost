using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Starlings.Roost.Abi;
using Starlings.Roost.Worker.Weights;

namespace Starlings.Roost.Worker.Tests;

/// <summary>A tensor of a synthetic checkpoint.</summary>
internal sealed record SyntheticTensor(string Name, string Dtype, long[] Shape, byte[] Data);

/// <summary>
/// Writes a small safetensors file at test time (nothing is committed): tensors in the order given, which tests choose
/// to differ from slot order, as the real checkpoint's file order differs from layer order.
/// </summary>
internal static class SyntheticCheckpoint
{
    public static byte[] Build(IReadOnlyList<SyntheticTensor> tensors, Func<string, (long Begin, long End), (long Begin, long End)>? tamperRange = null)
    {
        using var json = new MemoryStream();
        using (var w = new Utf8JsonWriter(json))
        {
            w.WriteStartObject();
            w.WriteStartObject("__metadata__");
            w.WriteString("format", "pt");
            w.WriteEndObject();
            long cursor = 0;
            foreach (var t in tensors)
            {
                var range = (Begin: cursor, End: cursor + t.Data.Length);
                cursor = range.End;
                if (tamperRange is not null)
                {
                    range = tamperRange(t.Name, range);
                }

                w.WriteStartObject(t.Name);
                w.WriteString("dtype", t.Dtype);
                w.WriteStartArray("shape");
                foreach (var d in t.Shape)
                {
                    w.WriteNumberValue(d);
                }

                w.WriteEndArray();
                w.WriteStartArray("data_offsets");
                w.WriteNumberValue(range.Begin);
                w.WriteNumberValue(range.End);
                w.WriteEndArray();
                w.WriteEndObject();
            }

            w.WriteEndObject();
        }

        var header = json.ToArray();
        var file = new byte[8 + header.Length + tensors.Sum(static t => t.Data.Length)];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)header.Length);
        header.CopyTo(file, 8);
        var at = 8 + header.Length;
        foreach (var t in tensors)
        {
            t.Data.CopyTo(file, at);
            at += t.Data.Length;
        }

        return file;
    }

    /// <summary>Deterministic bytes, different per seed.</summary>
    public static byte[] Bytes(int seed, int count)
    {
        var rng = new Random(seed);
        var b = new byte[count];
        rng.NextBytes(b);
        return b;
    }

    public static SyntheticTensor Bf16(string name, int seed, params long[] shape) =>
        new(name, "BF16", shape, Bytes(seed, checked((int)(shape.Aggregate(1L, static (a, b) => a * b) * 2))));

    public static SyntheticTensor F32(string name, int seed, params long[] shape) =>
        new(name, "F32", shape, Bytes(seed, checked((int)(shape.Aggregate(1L, static (a, b) => a * b) * 4))));

    public static string Sha256(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    public static string LockText(string path, long size, string sha) =>
        $"""
        lock_version = 1

        [source]
        repo = "test/synthetic"
        revision = "0123456789abcdef0123456789abcdef01234567"

        [[files]]
        path = "{path}"
        size = {size}
        sha256 = "{sha}"
        """;

    public static LockedFile Lock(byte[] file, string path = "model.safetensors") => new(path, file.Length, Sha256(file));

    public static string Write(DirectoryInfo dir, byte[] file, string name = "model.safetensors")
    {
        var p = Path.Combine(dir.FullName, name);
        File.WriteAllBytes(p, file);
        return p;
    }

    public static WeightSlotSpec Slot(string name, string source, DType dtype, long elements) => new(name, source, dtype, elements);

    public static string Utf8(byte[] b) => Encoding.UTF8.GetString(b);
}

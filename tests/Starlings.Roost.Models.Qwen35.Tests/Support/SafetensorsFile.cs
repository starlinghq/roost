using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Text.Json;

namespace Starlings.Roost.Models.Qwen35.Tests;

/// <summary>A read-only memory map of a safetensors file: the header, and each tensor's bytes in place. Test-time only.</summary>
internal sealed unsafe class SafetensorsFile : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _base;
    private readonly long _dataStart;
    private readonly Dictionary<string, (string Dtype, long[] Shape, long Begin, long End)> _tensors = new(StringComparer.Ordinal);

    public SafetensorsFile(string path)
    {
        _file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _view = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* p = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        _base = p + _view.PointerOffset;
        long headerLength = (long)BitConverter.ToUInt64(new ReadOnlySpan<byte>(_base, 8));
        _dataStart = 8 + headerLength;
        using var doc = JsonDocument.Parse(new ReadOnlySpan<byte>(_base + 8, (int)headerLength).ToArray());
        foreach (var e in doc.RootElement.EnumerateObject())
        {
            if (e.Name == "__metadata__")
            {
                continue;
            }

            var offsets = e.Value.GetProperty("data_offsets");
            _tensors[e.Name] = (
                e.Value.GetProperty("dtype").GetString()!,
                e.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray(),
                offsets[0].GetInt64(),
                offsets[1].GetInt64());
        }
    }

    public bool Contains(string name) => _tensors.ContainsKey(name);

    public (string Dtype, long[] Shape) Info(string name) => (_tensors[name].Dtype, _tensors[name].Shape);

    public ReadOnlySpan<byte> Bytes(string name)
    {
        var t = _tensors[name];
        return new ReadOnlySpan<byte>(_base + _dataStart + t.Begin, checked((int)(t.End - t.Begin)));
    }

    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"{_tensors.Count} tensors");

    public void Dispose()
    {
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _file.Dispose();
    }
}

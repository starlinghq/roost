using Microsoft.Win32.SafeHandles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Requests;
using Starlings.Roost.Worker.Eager;

namespace Starlings.Roost.Worker.Weights;

/// <summary>The weights of a model on the device, verified against the lock before any address was handed out.</summary>
public sealed class LoadedWeights : IDisposable
{
    internal LoadedWeights(DeviceBuffer arena, DeviceSpan[] spans, WeightLayout verification)
    {
        Arena = arena;
        Spans = spans;
        Verification = verification;
    }

    /// <summary>The device arena.</summary>
    public DeviceBuffer Arena { get; }

    /// <summary>One span per model slot, in slot order.</summary>
    public DeviceSpan[] Spans { get; }

    /// <summary>The raw-tensor stream's layout: the verified SHA-256 of the whole file and the bytes read.</summary>
    public WeightLayout Verification { get; }

    /// <summary>Frees the arena.</summary>
    public void Dispose() => Arena.Dispose();
}

/// <summary>Weights in host memory in the model's layout (the CPU reference backend's "device").</summary>
public sealed unsafe class HostModelWeights : IDisposable
{
    private readonly List<DeviceSpan> _spans = [];
    private byte* _base;

    internal HostModelWeights(byte* memory, long bytes, WeightLayout verification)
    {
        _base = memory;
        Bytes = bytes;
        Verification = verification;
    }

    /// <summary>The arena size.</summary>
    public long Bytes { get; }

    /// <summary>The verification of the raw stream.</summary>
    public WeightLayout Verification { get; }

    /// <summary>One span per model slot, in slot order.</summary>
    public IReadOnlyList<DeviceSpan> Spans => _spans;

    internal byte* Base => _base;

    internal void Add(DeviceSpan span) => _spans.Add(span);

    /// <summary>Frees the memory.</summary>
    public void Dispose()
    {
        if (_base != null)
        {
            NativeMemory.AlignedFree(_base);
            _base = null;
        }
    }
}

/// <summary>
/// Loads a model's weights (R-025): the locked file is streamed and hashed by <see cref="WeightStreamer"/> into host
/// memory tensor by tensor (R-024), and only after the digest matched are the model's slots assembled (norm offsets, row
/// splits, concatenations) and copied to one device arena. A substituted file therefore fails before the device sees a
/// byte of it. The host copy needs the size of the tensors the model reads (1.7 GB for Qwen3.5-0.8B).
/// </summary>
public static unsafe class ModelWeightLoader
{
    /// <summary>Streams, verifies, assembles and uploads.</summary>
    /// <param name="path">The safetensors file.</param>
    /// <param name="locked">Its lock entry.</param>
    /// <param name="slots">The model's slots.</param>
    /// <param name="allocator">The device allocator.</param>
    /// <returns>The device weights.</returns>
    /// <exception cref="WeightLoadException">The file does not match the lock or the slots.</exception>
    public static LoadedWeights Load(string path, LockedFile locked, IReadOnlyList<EagerWeightSlot> slots, DeviceAllocator allocator)
    {
        ArgumentNullException.ThrowIfNull(slots);
        ArgumentNullException.ThrowIfNull(allocator);
        var (host, verification, rawIndex) = StreamRaw(path, locked, slots);
        using var hostOwner = host;

        // The device arena, in model slot order.
        var specs = slots.Select(static s => new WeightSlotSpec(s.Name, s.Name, s.Dtype, s.Elements)).ToList();
        var (placements, total) = WeightLayout.Plan(specs);
        var arena = allocator.Allocate(total, "weights");
        try
        {
            allocator.Select();
            nuint stagingBytes = (nuint)Math.Max(1, slots.Where(static s => s.Assembler is not null).Select(static s => s.Bytes).DefaultIfEmpty(1).Max());
            void* staging = NativeMemory.Alloc(stagingBytes);
            try
            {
                for (int i = 0; i < slots.Count; i++)
                {
                    var slot = slots[i];
                    var place = placements[i];
                    var sources = slot.Sources.Select(s =>
                    {
                        var bytes = host.Bytes(verification.Slots[rawIndex[s]]);
                        return new WeightBytes((nint)Unsafe.AsPointer(in MemoryMarshal.GetReference(bytes)), bytes.Length);
                    }).ToList();
                    void* from;
                    if (slot.Assembler is null)
                    {
                        CheckPlainCopy(slot, sources);
                        from = (void*)sources[0].Address;
                    }
                    else
                    {
                        slot.Assembler(sources, new Span<byte>(staging, checked((int)slot.Bytes)));
                        from = staging;
                    }

                    CudaCheck.Runtime(CudaRuntime.cudaMemcpy((void*)(arena.Address + (nint)place.Offset), from, (nuint)slot.Bytes, CudaRuntime.cudaMemcpyHostToDevice), "cudaMemcpy(weights)");
                }
            }
            finally
            {
                NativeMemory.Free(staging);
            }

            var spans = placements.Select(p => arena.Span(p.Dtype, p.Elements, p.Offset)).ToArray();
            return new LoadedWeights(arena, spans, verification);
        }
        catch
        {
            arena.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Loads the weights into host memory in the model's layout, for the CPU reference backend, where "device" addresses are
    /// host addresses. Same verification and assembly as <see cref="Load"/>.
    /// </summary>
    /// <param name="path">The safetensors file.</param>
    /// <param name="locked">Its lock entry.</param>
    /// <param name="slots">The model's slots.</param>
    /// <returns>The host weights.</returns>
    /// <exception cref="WeightLoadException">The file does not match the lock or the slots.</exception>
    public static HostModelWeights LoadHost(string path, LockedFile locked, IReadOnlyList<EagerWeightSlot> slots)
    {
        ArgumentNullException.ThrowIfNull(slots);
        var (raw, verification, rawIndex) = StreamRaw(path, locked, slots);
        using var rawOwner = raw;
        var specs = slots.Select(static s => new WeightSlotSpec(s.Name, s.Name, s.Dtype, s.Elements)).ToList();
        var (placements, total) = WeightLayout.Plan(specs);
        var weights = new HostModelWeights((byte*)NativeMemory.AlignedAlloc((nuint)Math.Max(total, 1), (nuint)WeightLayout.Alignment), total, verification);
        for (int i = 0; i < slots.Count; i++)
        {
            var slot = slots[i];
            var sources = slot.Sources.Select(s =>
            {
                var bytes = raw.Bytes(verification.Slots[rawIndex[s]]);
                return new WeightBytes((nint)Unsafe.AsPointer(in MemoryMarshal.GetReference(bytes)), bytes.Length);
            }).ToList();
            var destination = new Span<byte>(weights.Base + placements[i].Offset, checked((int)slot.Bytes));
            Assemble(slot, sources, destination);

            weights.Add(new DeviceSpan { Address = (nint)(weights.Base + placements[i].Offset), Length = (ulong)slot.Elements, Dtype = slot.Dtype });
        }

        return weights;
    }

    private static void Assemble(EagerWeightSlot slot, List<WeightBytes> sources, Span<byte> destination)
    {
        if (slot.Assembler is null)
        {
            CheckPlainCopy(slot, sources);
            sources[0].Span.CopyTo(destination);
        }
        else
        {
            slot.Assembler(sources, destination);
        }
    }

    private static void CheckPlainCopy(EagerWeightSlot slot, List<WeightBytes> sources)
    {
        if (sources.Count != 1 || sources[0].Length != slot.Bytes)
        {
            throw new WeightLoadException(RoostErrorCode.AssetMismatch, $"slot {slot.Name} is a plain copy but its source is {sources.Sum(static s => s.Length)} bytes, not {slot.Bytes}");
        }
    }

    private static (HostWeightArena Host, WeightLayout Verification, Dictionary<string, int> Index) StreamRaw(string path, LockedFile locked, IReadOnlyList<EagerWeightSlot> slots)
    {
        var header = ReadHeader(path);
        var raw = new List<WeightSlotSpec>();
        var rawIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var slot in slots)
        {
            foreach (string source in slot.Sources)
            {
                if (rawIndex.ContainsKey(source))
                {
                    continue;
                }

                if (!header.ByName.TryGetValue(source, out var t))
                {
                    throw new WeightLoadException(RoostErrorCode.AssetMismatch, $"the checkpoint has no tensor {source} for slot {slot.Name}");
                }

                rawIndex[source] = raw.Count;
                raw.Add(new WeightSlotSpec(source, source, t.Dtype, t.Elements));
            }
        }

        var host = new HostWeightArena();
        try
        {
            return (host, WeightStreamer.Load(path, locked, raw, host), rawIndex);
        }
        catch
        {
            host.Dispose();
            throw;
        }
    }

    private static SafetensorsHeader ReadHeader(string path)
    {
        try
        {
            using SafeFileHandle handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            long size = RandomAccess.GetLength(handle);
            var prefix = new byte[8];
            if (RandomAccess.Read(handle, prefix, 0) != 8)
            {
                throw new WeightLoadException(RoostErrorCode.AssetMismatch, "the file ended inside the safetensors header");
            }

            long headerBytes = SafetensorsHeader.ReadLength(prefix, size);
            var json = new byte[headerBytes];
            long read = 0;
            while (read < headerBytes)
            {
                int n = RandomAccess.Read(handle, json.AsSpan((int)read), 8 + read);
                if (n == 0)
                {
                    throw new WeightLoadException(RoostErrorCode.AssetMismatch, "the file ended inside the safetensors header");
                }

                read += n;
            }

            return SafetensorsHeader.Parse(json, size);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            throw new WeightLoadException(RoostErrorCode.AssetMismatch, $"cannot read {Path.GetFileName(path)}: {e.Message}", e);
        }
    }
}

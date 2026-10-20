namespace Starlings.Roost.Engine.Admission;

/// <summary>Total and free bytes of one device, as a <see cref="IDeviceMemorySource"/> read them.</summary>
/// <param name="TotalBytes">Device memory in bytes.</param>
/// <param name="FreeBytes">Bytes not in use by any process, including co-tenants.</param>
public readonly record struct DeviceMemory(long TotalBytes, long FreeBytes);

/// <summary>
/// Where admission control learns how much device memory is free (R-031). The seam exists so the engine stays free of
/// CUDA and NVML: R-044 supplies the NVML implementation, which re-reads the ceiling every few seconds and sees
/// co-tenants; tests use <see cref="FixedDeviceMemorySource"/>.
/// </summary>
/// <remarks>Called on the engine thread, so an implementation should return a cached reading and never block on the driver.</remarks>
public interface IDeviceMemorySource
{
    /// <summary>Reads the current memory of the device.</summary>
    /// <param name="memory">The reading when this returns true.</param>
    /// <returns>False when the device cannot be read (no NVML, device lost); admission then decides from the pools alone.</returns>
    bool TryRead(out DeviceMemory memory);
}

/// <summary>A source with settable values, for tests and for hosts without NVML.</summary>
public sealed class FixedDeviceMemorySource : IDeviceMemorySource
{
    private DeviceMemory _memory;
    private bool _readable;

    /// <summary>Creates a readable source.</summary>
    /// <param name="totalBytes">Total bytes.</param>
    /// <param name="freeBytes">Free bytes.</param>
    public FixedDeviceMemorySource(long totalBytes, long freeBytes) => Set(totalBytes, freeBytes);

    /// <summary>Replaces the reading.</summary>
    /// <param name="totalBytes">Total bytes.</param>
    /// <param name="freeBytes">Free bytes.</param>
    public void Set(long totalBytes, long freeBytes)
    {
        _memory = new DeviceMemory(totalBytes, freeBytes);
        _readable = true;
    }

    /// <summary>Makes <see cref="TryRead"/> return false, as when NVML is unavailable.</summary>
    public void MakeUnreadable() => _readable = false;

    /// <inheritdoc />
    public bool TryRead(out DeviceMemory memory)
    {
        memory = _memory;
        return _readable;
    }
}

/// <summary>Sizes LoRA adapters for admission (M3, R-050). An adapter pages from the KV budget as an <c>Adapter</c> state group.</summary>
public interface IAdapterSizeSource
{
    /// <summary>Bytes the adapter occupies once resident.</summary>
    /// <param name="adapter">The handle of the request.</param>
    /// <param name="bytes">The size when this returns true.</param>
    /// <returns>False when the adapter is unknown.</returns>
    bool TryGetBytes(Requests.LoraHandle adapter, out long bytes);
}

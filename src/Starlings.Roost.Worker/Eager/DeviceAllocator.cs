using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Worker.Eager;

/// <summary>A device allocation failed because the device cannot hold it (<see cref="RoostErrorCode.InsufficientDeviceMemory"/>).</summary>
public sealed class DeviceMemoryException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="message">What was requested.</param>
    public DeviceMemoryException(string message)
        : base(message)
    {
        Error = new RoostError(RoostErrorCode.InsufficientDeviceMemory, message);
    }

    /// <summary>The typed error.</summary>
    public RoostError Error { get; }
}

/// <summary>
/// The worker's device allocator (R-025): every device and pinned-host allocation of the eager worker (activation arena,
/// state pools, plan region, kernel workspaces) goes through it, so a budget is enforced in one place and the total is
/// reported. It does not sub-allocate: the eager worker allocates a handful of large regions once at start-up and the
/// kernel workspaces on first use of a shape, and frees nothing inside a step. A sub-allocator with reuse belongs to
/// graph capture (R-033), which needs stable addresses.
/// </summary>
public sealed unsafe class DeviceAllocator : IDisposable
{
    /// <summary>Every device allocation starts on this many bytes (cuBLASLt and the 16-byte vector loads want 256).</summary>
    public const long Alignment = 256;

    private readonly List<DeviceBuffer> _live = [];
    private readonly long _budgetBytes;
    private bool _disposed;

    /// <summary>Creates an allocator for one device.</summary>
    /// <param name="device">CUDA device ordinal (after <c>CUDA_VISIBLE_DEVICES</c>).</param>
    /// <param name="budgetBytes">The most device bytes this allocator may hand out; 0 means the device's capacity is the limit.</param>
    public DeviceAllocator(int device, long budgetBytes = 0)
    {
        Device = device;
        _budgetBytes = budgetBytes;
    }

    /// <summary>The CUDA device ordinal.</summary>
    public int Device { get; }

    /// <summary>Device bytes currently allocated through this allocator.</summary>
    public long AllocatedBytes { get; private set; }

    /// <summary>Pinned host bytes currently allocated through this allocator.</summary>
    public long PinnedBytes { get; private set; }

    /// <summary>Makes this allocator's device current on the calling thread.</summary>
    public void Select() => CudaCheck.Runtime(CudaRuntime.cudaSetDevice(Device), "cudaSetDevice");

    /// <summary>Allocates device memory.</summary>
    /// <param name="bytes">Size; rounded up to <see cref="Alignment"/>.</param>
    /// <param name="purpose">A name for messages and the allocation report.</param>
    /// <param name="zero">Whether to clear the memory.</param>
    /// <returns>The allocation; dispose it to free it.</returns>
    /// <exception cref="DeviceMemoryException">The budget or the device cannot hold it.</exception>
    public DeviceBuffer Allocate(long bytes, string purpose, bool zero = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        long rounded = Math.Max(Alignment, (bytes + Alignment - 1) / Alignment * Alignment);
        if (_budgetBytes > 0 && AllocatedBytes + rounded > _budgetBytes)
        {
            throw new DeviceMemoryException($"{purpose}: {rounded} bytes would exceed the allocator budget of {_budgetBytes} bytes ({AllocatedBytes} already allocated)");
        }

        Select();
        void* p;
        int status = CudaRuntime.cudaMalloc(&p, (nuint)rounded);
        if (status == CudaRuntime.cudaErrorMemoryAllocation)
        {
            _ = CudaRuntime.cudaGetLastError();
            throw new DeviceMemoryException($"{purpose}: cannot allocate {rounded} bytes of device memory ({AllocatedBytes} already allocated)");
        }

        CudaCheck.Runtime(status, "cudaMalloc");
        if (zero)
        {
            CudaCheck.Runtime(CudaRuntime.cudaMemset(p, 0, (nuint)rounded), "cudaMemset");
        }

        var buffer = new DeviceBuffer(this, (nint)p, rounded, purpose);
        AllocatedBytes += rounded;
        _live.Add(buffer);
        return buffer;
    }

    /// <summary>Allocates page-locked host memory, zeroed.</summary>
    /// <param name="bytes">Size.</param>
    /// <param name="purpose">A name for messages.</param>
    /// <returns>The allocation; dispose it to free it.</returns>
    public PinnedBuffer AllocatePinned(long bytes, string purpose)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        long rounded = Math.Max(Alignment, (bytes + Alignment - 1) / Alignment * Alignment);
        Select();
        void* p;
        int status = CudaRuntime.cudaMallocHost(&p, (nuint)rounded);
        if (status == CudaRuntime.cudaErrorMemoryAllocation)
        {
            _ = CudaRuntime.cudaGetLastError();
            throw new DeviceMemoryException($"{purpose}: cannot pin {rounded} bytes of host memory");
        }

        CudaCheck.Runtime(status, "cudaMallocHost");
        NativeMemory.Clear(p, (nuint)rounded);
        PinnedBytes += rounded;
        return new PinnedBuffer(this, (nint)p, rounded);
    }

    /// <summary>The live device allocations, for the allocation report.</summary>
    /// <returns>Purpose and bytes of each.</returns>
    public IReadOnlyList<(string Purpose, long Bytes)> Report() => [.. _live.Select(static b => (b.Purpose, b.Bytes))];

    /// <summary>Frees every allocation still live.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var b in _live.ToArray())
        {
            b.Dispose();
        }
    }

    internal void Release(DeviceBuffer buffer)
    {
        if (_live.Remove(buffer))
        {
            AllocatedBytes -= buffer.Bytes;
            _ = CudaRuntime.cudaSetDevice(Device);
            _ = CudaRuntime.cudaFree((void*)buffer.Address);
        }
    }

    internal void ReleasePinned(PinnedBuffer buffer)
    {
        PinnedBytes -= buffer.Bytes;
        _ = CudaRuntime.cudaFreeHost((void*)buffer.Address);
    }
}

/// <summary>A device allocation made by <see cref="DeviceAllocator"/>.</summary>
public sealed class DeviceBuffer : IDisposable
{
    private readonly DeviceAllocator _owner;
    private bool _freed;

    internal DeviceBuffer(DeviceAllocator owner, nint address, long bytes, string purpose)
    {
        _owner = owner;
        Address = address;
        Bytes = bytes;
        Purpose = purpose;
    }

    /// <summary>The device address.</summary>
    public nint Address { get; }

    /// <summary>The size, rounded to the allocator's alignment.</summary>
    public long Bytes { get; }

    /// <summary>What it is for.</summary>
    public string Purpose { get; }

    /// <summary>The allocation as a typed span.</summary>
    /// <param name="dtype">Element type.</param>
    /// <param name="elements">Element count.</param>
    /// <param name="byteOffset">Offset into the allocation.</param>
    /// <returns>The span.</returns>
    public DeviceSpan Span(DType dtype, long elements, long byteOffset = 0) =>
        new() { Address = Address + (nint)byteOffset, Length = (ulong)elements, Dtype = dtype };

    /// <summary>Frees the allocation.</summary>
    public void Dispose()
    {
        if (!_freed)
        {
            _freed = true;
            _owner.Release(this);
        }
    }
}

/// <summary>Page-locked host memory made by <see cref="DeviceAllocator"/>.</summary>
public sealed class PinnedBuffer : IDisposable
{
    private readonly DeviceAllocator _owner;
    private bool _freed;

    internal PinnedBuffer(DeviceAllocator owner, nint address, long bytes)
    {
        _owner = owner;
        Address = address;
        Bytes = bytes;
    }

    /// <summary>The host address.</summary>
    public nint Address { get; }

    /// <summary>The size.</summary>
    public long Bytes { get; }

    /// <summary>Frees the memory.</summary>
    public void Dispose()
    {
        if (!_freed)
        {
            _freed = true;
            _owner.ReleasePinned(this);
        }
    }
}

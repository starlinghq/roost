using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Worker.Weights;

/// <summary>
/// A weight arena in device memory: one <c>cudaMalloc</c>, filled by asynchronous host-to-device copies on a private
/// stream through a ring of pinned staging buffers, so reading and hashing the next chunk overlaps the copy of the
/// previous one. The device address is available only after <see cref="Commit"/>.
/// </summary>
/// <remarks>Call every member from one thread; the CUDA device is selected on it at each call.</remarks>
public sealed unsafe class CudaWeightArena : IWeightSink
{
    private readonly int _device;
    private readonly nuint _stagingBytes;
    private readonly nint[] _staging;
    private readonly nint[] _events;
    private readonly bool[] _pending;
    private byte* _base;
    private long _bytes;
    private nint _stream;
    private int _ring;
    private bool _committed;
    private bool _created;

    /// <summary>Creates an arena on a device.</summary>
    /// <param name="device">CUDA device ordinal (after <c>CUDA_VISIBLE_DEVICES</c>).</param>
    /// <param name="stagingBuffers">Pinned buffers in the ring; at least 2.</param>
    /// <param name="stagingBytes">Size of each pinned buffer.</param>
    public CudaWeightArena(int device = 0, int stagingBuffers = 3, int stagingBytes = WeightStreamer.DefaultChunkBytes)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(stagingBuffers, 2);
        ArgumentOutOfRangeException.ThrowIfLessThan(stagingBytes, 4096);
        _device = device;
        _stagingBytes = (nuint)stagingBytes;
        _staging = new nint[stagingBuffers];
        _events = new nint[stagingBuffers];
        _pending = new bool[stagingBuffers];
    }

    /// <inheritdoc/>
    public void Allocate(long bytes)
    {
        if (_base != null)
        {
            throw new InvalidOperationException("the arena is already allocated");
        }

        Select();
        void* p;
        var status = CudaRuntime.cudaMalloc(&p, (nuint)Math.Max(bytes, 1));
        if (status == CudaRuntime.cudaErrorMemoryAllocation)
        {
            _ = CudaRuntime.cudaGetLastError();
            throw new WeightLoadException(RoostErrorCode.InsufficientDeviceMemory, $"cannot allocate {bytes} bytes of device memory for the weights");
        }

        CudaCheck.Runtime(status, "cudaMalloc");
        _base = (byte*)p;
        _bytes = bytes;
        _created = true;
        nint stream;
        CudaCheck.Runtime(CudaRuntime.cudaStreamCreateWithFlags(&stream, CudaRuntime.cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        _stream = stream;
        for (var i = 0; i < _staging.Length; i++)
        {
            void* host;
            CudaCheck.Runtime(CudaRuntime.cudaMallocHost(&host, _stagingBytes), "cudaMallocHost");
            _staging[i] = (nint)host;
            nint ev;
            CudaCheck.Runtime(CudaRuntime.cudaEventCreateWithFlags(&ev, CudaRuntime.cudaEventDisableTiming), "cudaEventCreateWithFlags");
            _events[i] = ev;
        }
    }

    /// <inheritdoc/>
    public void Write(long offset, ReadOnlySpan<byte> data)
    {
        if (_base == null || offset < 0 || offset + data.Length > _bytes)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        Select();
        while (!data.IsEmpty)
        {
            var slot = _ring;
            _ring = (_ring + 1) % _staging.Length;
            if (_pending[slot])
            {
                // The previous copy out of this pinned buffer must have finished before it is overwritten.
                CudaCheck.Runtime(CudaRuntime.cudaEventSynchronize(_events[slot]), "cudaEventSynchronize");
            }

            var n = (int)Math.Min((nuint)data.Length, _stagingBytes);
            data[..n].CopyTo(new Span<byte>((void*)_staging[slot], n));
            CudaCheck.Runtime(
                CudaRuntime.cudaMemcpyAsync(_base + offset, (void*)_staging[slot], (nuint)n, CudaRuntime.cudaMemcpyHostToDevice, _stream),
                "cudaMemcpyAsync");
            CudaCheck.Runtime(CudaRuntime.cudaEventRecord(_events[slot], _stream), "cudaEventRecord");
            _pending[slot] = true;
            data = data[n..];
            offset += n;
        }
    }

    /// <inheritdoc/>
    public void Commit()
    {
        if (_base == null)
        {
            throw new InvalidOperationException("nothing was allocated");
        }

        Select();
        CudaCheck.Runtime(CudaRuntime.cudaStreamSynchronize(_stream), "cudaStreamSynchronize");
        ReleaseStaging();
        _committed = true;
    }

    /// <inheritdoc/>
    public void Discard() => Free();

    /// <inheritdoc/>
    public void Dispose() => Free();

    /// <summary>The device span of a verified slot, as the kernels take it.</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>Device address, element count and dtype; valid until the arena is disposed.</returns>
    /// <exception cref="InvalidOperationException">The stream has not been committed.</exception>
    public DeviceSpan Span(SlotPlacement slot) =>
        new() { Address = (nint)(Base() + slot.Offset), Length = (ulong)slot.Elements, Dtype = slot.Dtype };

    /// <summary>Copies a verified slot back to host memory (tests and diagnostics).</summary>
    /// <param name="slot">The slot.</param>
    /// <param name="destination">At least <see cref="SlotPlacement.Bytes"/> bytes.</param>
    public void CopyToHost(SlotPlacement slot, Span<byte> destination)
    {
        var src = Base() + slot.Offset;
        Select();
        fixed (byte* dst = destination)
        {
            CudaCheck.Runtime(CudaRuntime.cudaMemcpy(dst, src, (nuint)slot.Bytes, CudaRuntime.cudaMemcpyDeviceToHost), "cudaMemcpy");
        }
    }

    private byte* Base() =>
        _committed && _base != null ? _base : throw new InvalidOperationException("the weights are not verified yet");

    private void Select() => CudaCheck.Runtime(CudaRuntime.cudaSetDevice(_device), "cudaSetDevice");

    private void ReleaseStaging()
    {
        for (var i = 0; i < _staging.Length; i++)
        {
            if (_staging[i] != 0)
            {
                _ = CudaRuntime.cudaFreeHost((void*)_staging[i]);
                _staging[i] = 0;
            }

            if (_events[i] != 0)
            {
                _ = CudaRuntime.cudaEventDestroy(_events[i]);
                _events[i] = 0;
            }

            _pending[i] = false;
        }
    }

    private void Free()
    {
        _committed = false;
        if (!_created)
        {
            return;
        }

        _created = false;
        _ = CudaRuntime.cudaSetDevice(_device);
        if (_stream != 0)
        {
            // In-flight copies read the pinned buffers and write the arena: let them finish before either is freed.
            _ = CudaRuntime.cudaStreamSynchronize(_stream);
            ReleaseStaging();
            _ = CudaRuntime.cudaStreamDestroy(_stream);
            _stream = 0;
        }

        if (_base != null)
        {
            _ = CudaRuntime.cudaFree(_base);
            _base = null;
        }
    }
}

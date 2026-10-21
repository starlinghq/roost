using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Worker.Weights;

/// <summary>
/// A weight arena in host memory, for CPU tests and for the CPU reference backend, where "device" addresses are host
/// addresses (<see cref="DeviceSpan"/>). Same contract as the CUDA arena: no address before
/// <see cref="Commit"/>.
/// </summary>
public sealed unsafe class HostWeightArena : IWeightSink
{
    private byte* _base;
    private long _bytes;
    private bool _committed;

    /// <inheritdoc/>
    public void Allocate(long bytes)
    {
        if (_base != null)
        {
            throw new InvalidOperationException("the arena is already allocated");
        }

        try
        {
            _base = (byte*)NativeMemory.AlignedAlloc((nuint)Math.Max(bytes, 1), (nuint)WeightLayout.Alignment);
        }
        catch (OutOfMemoryException e)
        {
            throw new WeightLoadException(RoostErrorCode.InsufficientDeviceMemory, $"cannot allocate {bytes} bytes for the weights", e);
        }

        _bytes = bytes;
    }

    /// <inheritdoc/>
    public void Write(long offset, ReadOnlySpan<byte> data)
    {
        if (_base == null || offset < 0 || offset + data.Length > _bytes)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        data.CopyTo(new Span<byte>(_base + offset, data.Length));
    }

    /// <inheritdoc/>
    public void Commit() => _committed = _base != null ? true : throw new InvalidOperationException("nothing was allocated");

    /// <inheritdoc/>
    public void Discard() => Free();

    /// <inheritdoc/>
    public void Dispose() => Free();

    /// <summary>The bytes of a verified slot.</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>A view over the arena, valid until the arena is disposed.</returns>
    /// <exception cref="InvalidOperationException">The stream has not been committed.</exception>
    public ReadOnlySpan<byte> Bytes(SlotPlacement slot) =>
        new(Base() + slot.Offset, checked((int)slot.Bytes));

    /// <summary>The span of a verified slot as the kernels take it (<see cref="DeviceSpan"/>).</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>Address, element count and dtype.</returns>
    /// <exception cref="InvalidOperationException">The stream has not been committed.</exception>
    public DeviceSpan Span(SlotPlacement slot) =>
        new() { Address = (nint)(Base() + slot.Offset), Length = (ulong)slot.Elements, Dtype = slot.Dtype };

    private byte* Base() =>
        _committed && _base != null ? _base : throw new InvalidOperationException("the weights are not verified yet");

    private void Free()
    {
        _committed = false;
        if (_base != null)
        {
            NativeMemory.AlignedFree(_base);
            _base = null;
        }
    }
}

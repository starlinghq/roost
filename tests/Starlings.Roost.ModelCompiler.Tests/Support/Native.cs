using System.Runtime.InteropServices;
using Starlings.Roost.Abi;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>Zeroed native buffers that live as long as the object; host addresses stand in for device addresses.</summary>
public sealed unsafe class Native : IDisposable
{
    private readonly List<nint> _blocks = [];

    public DeviceSpan Alloc(long elements, DType dtype)
    {
        long bytes = Math.Max(elements * Size(dtype), 64);
        void* p = NativeMemory.AlignedAlloc((nuint)bytes, 64);
        new Span<byte>(p, (int)bytes).Clear();
        _blocks.Add((nint)p);
        return new DeviceSpan { Address = (nint)p, Length = (ulong)elements, Dtype = dtype };
    }

    public DeviceSpan AllocBytes(long bytes) => Alloc(bytes, DType.U8);

    public static int Size(DType t) => t switch
    {
        DType.F64 or DType.I64 or DType.U64 => 8,
        DType.F32 or DType.I32 or DType.U32 => 4,
        DType.F16 or DType.BF16 or DType.I16 => 2,
        _ => 1,
    };

    public void Dispose()
    {
        foreach (nint p in _blocks)
        {
            NativeMemory.AlignedFree((void*)p);
        }

        _blocks.Clear();
    }
}

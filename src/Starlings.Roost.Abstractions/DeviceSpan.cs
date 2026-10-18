using System.Runtime.InteropServices;
using Starlings.Roost.Abi;

namespace Starlings.Roost;

/// <summary>A buffer: device (or, in CPU tests, host) address, element count and element type. Mirrors C3 DeviceSpan. Worker-side only: never placed in shared memory between processes.</summary>
/// <remarks>Mirror of <c>RoostDeviceSpan</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostDeviceSpan")]
[StructLayout(LayoutKind.Sequential)]
public struct DeviceSpan
{
    /// <summary>Device address (host address in CPU tests).</summary>
    public nint Address;

    /// <summary>Elements, not bytes; for packed 4-bit types, logical elements</summary>
    public ulong Length;

    /// <summary>ROOST_DTYPE_*</summary>
    public DType Dtype;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

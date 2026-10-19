using System.Runtime.InteropServices;
using Starlings.Roost;
using Starlings.Roost.Abi;

namespace Roost.HybridConformance;

// Mirrors of kernels/src/ops/hybrid_params.h (provisional C structs for ops that roost_kernels.h does not declare yet).
// The registration entry carries sizeof of the C struct, and Program checks it against Marshal.SizeOf of these.

[StructLayout(LayoutKind.Sequential)]
internal struct CausalConv1dParams
{
    public uint StructSize;
    public uint AbiMinor;
    public DeviceSpan Input;
    public DeviceSpan Weight;
    public DeviceSpan Bias;
    public DeviceSpan Output;
    public DeviceSpan ConvState;
    public DeviceSpan StateSlots;
    public DeviceSpan CuSeqlens;
    public uint Batch;
    public uint Tokens;
    public uint Channels;
    public uint Width;
    public uint Silu;
    public uint Reserved0;
}

[StructLayout(LayoutKind.Sequential)]
internal struct RecurrentStateCopyParams
{
    public uint StructSize;
    public uint AbiMinor;
    public DeviceSpan Pool;
    public DeviceSpan Pairs;
    public ulong SlotBytes;
    public uint Count;
    public uint Reserved0;
}

[StructLayout(LayoutKind.Sequential)]
internal struct RecurrentStateSnapshotParams
{
    public uint StructSize;
    public uint AbiMinor;
    public DeviceSpan Pool;
    public DeviceSpan Snapshots;
    public DeviceSpan Pairs;
    public ulong SlotBytes;
    public uint Count;
    public uint Restore;
}

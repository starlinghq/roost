using System.Runtime.InteropServices;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Transport;

// Mirrors of kernels/include/roost_worker_shm.h, field for field. These records live in shared memory between the
// host and worker processes, so no field is a pointer: offsets, indices and handles only (README.md, Contracts).

/// <summary>At offset 0 of the segment. Every _offset is a byte offset from the segment start, aligned to ROOST_SHM_ALIGNMENT. The worker registers the whole segment with cudaHostRegister.</summary>
/// <remarks>Mirror of <c>RoostSegmentHeader</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostSegmentHeader")]
[StructLayout(LayoutKind.Sequential)]
public struct SegmentHeader
{
    /// <summary>ROOST_SHM_MAGIC_SEGMENT</summary>
    public uint Magic;

    /// <summary>C field <c>layout_version</c>, documented in the header.</summary>
    public uint LayoutVersion;

    /// <summary>C field <c>header_size</c>, documented in the header.</summary>
    public uint HeaderSize;

    /// <summary>Starlings.Roost.ContractVersion.Current</summary>
    public uint ContractVersion;

    /// <summary>C field <c>segment_bytes</c>, documented in the header.</summary>
    public ulong SegmentBytes;

    /// <summary>C field <c>handshake_offset</c>, documented in the header.</summary>
    public ulong HandshakeOffset;

    /// <summary>C field <c>heartbeat_offset</c>, documented in the header.</summary>
    public ulong HeartbeatOffset;

    /// <summary>RoostRingHeader</summary>
    public ulong PlanRingOffset;

    /// <summary>RoostRingHeader</summary>
    public ulong ResultRingOffset;

    /// <summary>C field <c>plan_buffers_offset</c>, documented in the header.</summary>
    public ulong PlanBuffersOffset;

    /// <summary>C field <c>plan_buffer_bytes</c>, documented in the header.</summary>
    public ulong PlanBufferBytes;

    /// <summary>C field <c>result_buffers_offset</c>, documented in the header.</summary>
    public ulong ResultBuffersOffset;

    /// <summary>C field <c>result_buffer_bytes</c>, documented in the header.</summary>
    public ulong ResultBufferBytes;

    /// <summary>At least 2: plans are double buffered (S8: two in flight)</summary>
    public uint PlanBufferCount;

    /// <summary>C field <c>result_buffer_count</c>, documented in the header.</summary>
    public uint ResultBufferCount;

    /// <summary>Region for embedding and encoder inputs, addressed by buffer_handle</summary>
    public ulong InputRegionOffset;

    /// <summary>C field <c>input_region_bytes</c>, documented in the header.</summary>
    public ulong InputRegionBytes;

    /// <summary>Reserved; zero.</summary>
    public ulong Reserved0;
}

/// <summary>Single-producer single-consumer ring. Cursors are monotonically increasing uint64 slot counts, each alone on its own cache line at the given offsets; slot i lives at slots_offset + (i % slot_count) slot_bytes.</summary>
/// <remarks>Mirror of <c>RoostRingHeader</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostRingHeader")]
[StructLayout(LayoutKind.Sequential)]
public struct RingHeader
{
    /// <summary>ROOST_SHM_MAGIC_RING</summary>
    public uint Magic;

    /// <summary>C field <c>layout_version</c>, documented in the header.</summary>
    public uint LayoutVersion;

    /// <summary>C field <c>slot_count</c>, documented in the header.</summary>
    public uint SlotCount;

    /// <summary>C field <c>slot_bytes</c>, documented in the header.</summary>
    public uint SlotBytes;

    /// <summary>C field <c>slots_offset</c>, documented in the header.</summary>
    public ulong SlotsOffset;

    /// <summary>C field <c>producer_cursor_offset</c>, documented in the header.</summary>
    public ulong ProducerCursorOffset;

    /// <summary>C field <c>consumer_cursor_offset</c>, documented in the header.</summary>
    public ulong ConsumerCursorOffset;

    /// <summary>Reserved; zero.</summary>
    public ulong Reserved0;
}

/// <summary>One ring slot: names a plan or result buffer by index.</summary>
/// <remarks>Mirror of <c>RoostRingSlot</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostRingSlot")]
[StructLayout(LayoutKind.Sequential)]
public struct RingSlot
{
    /// <summary>C field <c>step_id</c>, documented in the header.</summary>
    public ulong StepId;

    /// <summary>ROOST_RING_SLOT_*</summary>
    public RingSlotKind Kind;

    /// <summary>Plan or result buffer</summary>
    public uint BufferIndex;

    /// <summary>Host monotonic clock; 0 means none</summary>
    public ulong DeadlineNs;

    /// <summary>C field <c>published_ns</c>, documented in the header.</summary>
    public ulong PublishedNs;
}

/// <summary>Digests are SHA-256 in four little-endian words. The weights digest is the one recorded in the manifest and profile lock; a worker never hashes weights at spawn (S8: 9.4 s for 3.44 GB).</summary>
/// <remarks>Mirror of <c>RoostHandshake</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostHandshake")]
[StructLayout(LayoutKind.Sequential)]
public struct Handshake
{
    /// <summary>ROOST_HANDSHAKE_*</summary>
    public HandshakeState State;

    /// <summary>ROOST_REJECT_*</summary>
    public HandshakeReject RejectReason;

    /// <summary>C field <c>contract_version</c>, documented in the header.</summary>
    public uint ContractVersion;

    /// <summary>C1 (major &lt;&lt; 16) | minor</summary>
    public uint AbiVersion;

    /// <summary>ROOST_SHM_LAYOUT_VERSION</summary>
    public uint LayoutVersion;

    /// <summary>C field <c>device_ordinal</c>, documented in the header.</summary>
    public int DeviceOrdinal;

    /// <summary>C field <c>mesh_rank</c>, documented in the header.</summary>
    public uint MeshRank;

    /// <summary>C field <c>mesh_size</c>, documented in the header.</summary>
    public uint MeshSize;

    /// <summary>C field <c>manifest_digest0</c>, documented in the header.</summary>
    public ulong ManifestDigest0;

    /// <summary>C field <c>manifest_digest1</c>, documented in the header.</summary>
    public ulong ManifestDigest1;

    /// <summary>C field <c>manifest_digest2</c>, documented in the header.</summary>
    public ulong ManifestDigest2;

    /// <summary>C field <c>manifest_digest3</c>, documented in the header.</summary>
    public ulong ManifestDigest3;

    /// <summary>C field <c>weights_digest0</c>, documented in the header.</summary>
    public ulong WeightsDigest0;

    /// <summary>C field <c>weights_digest1</c>, documented in the header.</summary>
    public ulong WeightsDigest1;

    /// <summary>C field <c>weights_digest2</c>, documented in the header.</summary>
    public ulong WeightsDigest2;

    /// <summary>C field <c>weights_digest3</c>, documented in the header.</summary>
    public ulong WeightsDigest3;

    /// <summary>Written by the worker</summary>
    public ulong DeviceUuid0;

    /// <summary>C field <c>device_uuid1</c>, documented in the header.</summary>
    public ulong DeviceUuid1;

    /// <summary>C field <c>host_pid</c>, documented in the header.</summary>
    public ulong HostPid;

    /// <summary>C field <c>worker_pid</c>, documented in the header.</summary>
    public ulong WorkerPid;

    /// <summary>Per-step deadline the host enforces</summary>
    public ulong StepDeadlineNs;

    /// <summary>C field <c>heartbeat_period_ns</c>, documented in the header.</summary>
    public ulong HeartbeatPeriodNs;
}

/// <summary>Written by the worker's heartbeat thread; the host reads it to tell a hang from a stall.</summary>
/// <remarks>Mirror of <c>RoostHeartbeat</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostHeartbeat")]
[StructLayout(LayoutKind.Sequential)]
public struct Heartbeat
{
    /// <summary>C field <c>counter</c>, documented in the header.</summary>
    public ulong Counter;

    /// <summary>Worker monotonic clock at the last beat</summary>
    public ulong WorkerNs;

    /// <summary>Last step whose result was published</summary>
    public ulong LastStepId;

    /// <summary>ROOST_WORKER_*</summary>
    public WorkerState State;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

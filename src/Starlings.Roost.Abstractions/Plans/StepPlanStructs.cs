using System.Runtime.InteropServices;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Plans;

// Mirrors of kernels/include/roost_worker_shm.h, field for field. These records live in shared memory between the
// host and worker processes, so no field is a pointer: offsets, indices and handles only (README.md, Contracts).

/// <summary>A table in the same buffer: byte offset from the buffer start, entry count, bytes per entry.</summary>
/// <remarks>Mirror of <c>RoostTableRef</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostTableRef")]
[StructLayout(LayoutKind.Sequential)]
public struct TableRef
{
    /// <summary>C field <c>offset</c>, documented in the header.</summary>
    public ulong Offset;

    /// <summary>C field <c>count</c>, documented in the header.</summary>
    public uint Count;

    /// <summary>C field <c>stride</c>, documented in the header.</summary>
    public uint Stride;
}

/// <summary>At least one sequence is prefilling</summary>
/// <remarks>Mirror of <c>RoostStepPlanHeader</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostStepPlanHeader")]
[StructLayout(LayoutKind.Sequential)]
public struct StepPlanHeader
{
    /// <summary>ROOST_SHM_MAGIC_PLAN</summary>
    public uint Magic;

    /// <summary>ROOST_SHM_LAYOUT_VERSION</summary>
    public uint LayoutVersion;

    /// <summary>C field <c>header_size</c>, documented in the header.</summary>
    public uint HeaderSize;

    /// <summary>ROOST_PLAN_FLAG_*</summary>
    public PlanOptions Flags;

    /// <summary>C field <c>step_id</c>, documented in the header.</summary>
    public ulong StepId;

    /// <summary>Bytes of this buffer in use, header included</summary>
    public ulong TotalBytes;

    /// <summary>C field <c>shape_bucket</c>, documented in the header.</summary>
    public uint ShapeBucket;

    /// <summary>ROOST_MATH_*</summary>
    public MathMode MathMode;

    /// <summary>C field <c>sequence_count</c>, documented in the header.</summary>
    public uint SequenceCount;

    /// <summary>C field <c>token_count</c>, documented in the header.</summary>
    public uint TokenCount;

    /// <summary>State groups in the profile (C4)</summary>
    public uint GroupCount;

    /// <summary>Reserved (C9): this plan's mesh rank, 0 on a single rank</summary>
    public uint Rank;

    /// <summary>Reserved (C9)</summary>
    public uint MeshId;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;

    /// <summary>RoostPlanSequence</summary>
    public TableRef Sequences;

    /// <summary>Int32, token_count</summary>
    public TableRef TokenIds;

    /// <summary>Int32, token_count</summary>
    public TableRef Positions;

    /// <summary>RoostPlanGroupBlocks, sequence_count * group_count, sequence-major</summary>
    public TableRef GroupBlocks;

    /// <summary>Int32 block ids, referenced by group_blocks</summary>
    public TableRef BlockIds;

    /// <summary>RoostPlanSampling</summary>
    public TableRef Sampling;

    /// <summary>RoostPlanCacheOp, executed before the forward</summary>
    public TableRef CacheOps;

    /// <summary>RoostPlanEmbeddingInput</summary>
    public TableRef EmbeddingInputs;

    /// <summary>Int32, referenced by RoostPlanSequence.stop_start and stop_count</summary>
    public TableRef StopTokenIds;

    /// <summary>Int32 parent index per draft node, -1 for the root (M2 speculation trees)</summary>
    public TableRef SpecTreeParents;

    /// <summary>Uint64 attention mask words per draft node</summary>
    public TableRef SpecTreeMasks;

    /// <summary>Int32 tokens per expert (M2)</summary>
    public TableRef MoeExpertCounts;

    /// <summary>Int32 per expert: local device slot, or -1 for host experts (M2, bet X6)</summary>
    public TableRef MoeExpertPlacement;

    /// <summary>RoostPlanKvTransfer (M4)</summary>
    public TableRef KvTransfers;

    /// <summary>RoostPlanHeadRow: rows that end in a head instead of the sampler</summary>
    public TableRef HeadRows;

    /// <summary>RoostPlanEncoderOutput (M5 multimodal)</summary>
    public TableRef EncoderOutputs;

    /// <summary>Reserved; zero.</summary>
    public ulong Reserved1;

    /// <summary>Reserved; zero.</summary>
    public ulong Reserved2;
}

/// <summary>Reserved: ends in head_rows, not the sampler</summary>
/// <remarks>Mirror of <c>RoostPlanSequence</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostPlanSequence")]
[StructLayout(LayoutKind.Sequential)]
public struct PlanSequence
{
    /// <summary>Engine's id for the sequence; echoed in the result</summary>
    public ulong SequenceHandle;

    /// <summary>Index into token_ids and positions</summary>
    public uint TokenStart;

    /// <summary>C field <c>token_count</c>, documented in the header.</summary>
    public uint TokenCount;

    /// <summary>Position of the first token, i.e. tokens already in the cache</summary>
    public uint PositionOffset;

    /// <summary>Index into group_blocks; group_count consecutive entries</summary>
    public uint GroupBlocksStart;

    /// <summary>Recurrent-state pool slot (C4), -1 for none</summary>
    public int RecurrentSlot;

    /// <summary>-1 for the base model</summary>
    public int LoraSlot;

    /// <summary>Token-mask slot, -1 for none</summary>
    public int GrammarSlot;

    /// <summary>Index into embedding_inputs, -1 for none</summary>
    public int EmbeddingSlot;

    /// <summary>Index into sampling</summary>
    public uint SamplingIndex;

    /// <summary>C field <c>draft_count</c>, documented in the header.</summary>
    public uint DraftCount;

    /// <summary>ROOST_SEQ_FLAG_*</summary>
    public SequenceOptions Flags;

    /// <summary>Index into stop_token_ids: the model's and the request's stop ids</summary>
    public uint StopStart;

    /// <summary>C field <c>stop_count</c>, documented in the header.</summary>
    public uint StopCount;

    /// <summary>Tokens the request may still emit; 0 means no limit</summary>
    public uint MaxNewTokensLeft;

    /// <summary>Reserved: index into spec_tree_parents</summary>
    public uint SpecTreeStart;

    /// <summary>Reserved</summary>
    public uint SpecTreeCount;

    /// <summary>Reserved: index into head_rows, -1 for none</summary>
    public int HeadRow;

    /// <summary>Reserved: index into encoder_outputs, -1 for none</summary>
    public int EncoderOutput;
}

/// <summary>Mirror of <c>RoostPlanGroupBlocks</c>.</summary>
/// <remarks>Mirror of <c>RoostPlanGroupBlocks</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostPlanGroupBlocks")]
[StructLayout(LayoutKind.Sequential)]
public struct PlanGroupBlocks
{
    /// <summary>Index into block_ids</summary>
    public uint BlockStart;

    /// <summary>C field <c>block_count</c>, documented in the header.</summary>
    public uint BlockCount;
}

/// <summary>Mirror of <c>RoostPlanSampling</c>.</summary>
/// <remarks>Mirror of <c>RoostPlanSampling</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostPlanSampling")]
[StructLayout(LayoutKind.Sequential)]
public struct PlanSampling
{
    /// <summary>C field <c>temperature</c>, documented in the header.</summary>
    public float Temperature;

    /// <summary>C field <c>top_p</c>, documented in the header.</summary>
    public float TopP;

    /// <summary>0 disables</summary>
    public int TopK;

    /// <summary>C field <c>min_p</c>, documented in the header.</summary>
    public float MinP;

    /// <summary>1 disables</summary>
    public float RepetitionPenalty;

    /// <summary>C field <c>presence_penalty</c>, documented in the header.</summary>
    public float PresencePenalty;

    /// <summary>C field <c>frequency_penalty</c>, documented in the header.</summary>
    public float FrequencyPenalty;

    /// <summary>Top-n logprobs to return, 0 for none</summary>
    public uint Logprobs;

    /// <summary>C field <c>seed</c>, documented in the header.</summary>
    public ulong Seed;

    /// <summary>C field <c>philox_offset</c>, documented in the header.</summary>
    public ulong PhiloxOffset;

    /// <summary>ROOST_SAMPLE_FLAG_*</summary>
    public SamplingOptions Flags;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Mirror of <c>RoostPlanCacheOp</c>.</summary>
/// <remarks>Mirror of <c>RoostPlanCacheOp</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostPlanCacheOp")]
[StructLayout(LayoutKind.Sequential)]
public struct PlanCacheOp
{
    /// <summary>ROOST_CACHE_OP_*</summary>
    public CacheOpKind Kind;

    /// <summary>State group index (C4)</summary>
    public uint Group;

    /// <summary>First block or slot</summary>
    public uint Src;

    /// <summary>C field <c>dst</c>, documented in the header.</summary>
    public uint Dst;

    /// <summary>C field <c>count</c>, documented in the header.</summary>
    public uint Count;

    /// <summary>ROOST_TIER_*</summary>
    public BlockTier SrcTier;

    /// <summary>C field <c>dst_tier</c>, documented in the header.</summary>
    public BlockTier DstTier;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Precomputed input embeddings that replace token embeddings for a token range.</summary>
/// <remarks>Mirror of <c>RoostPlanEmbeddingInput</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostPlanEmbeddingInput")]
[StructLayout(LayoutKind.Sequential)]
public struct PlanEmbeddingInput
{
    /// <summary>C field <c>sequence_index</c>, documented in the header.</summary>
    public uint SequenceIndex;

    /// <summary>Index into token_ids</summary>
    public uint TokenStart;

    /// <summary>C field <c>token_count</c>, documented in the header.</summary>
    public uint TokenCount;

    /// <summary>ROOST_DTYPE_*</summary>
    public DType Dtype;

    /// <summary>Registered input region (C8), resolved by the worker</summary>
    public ulong BufferHandle;

    /// <summary>Within that region</summary>
    public ulong ByteOffset;
}

/// <summary>Reserved (M4): KV blocks sent to or received from a peer.</summary>
/// <remarks>Mirror of <c>RoostPlanKvTransfer</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostPlanKvTransfer")]
[StructLayout(LayoutKind.Sequential)]
public struct PlanKvTransfer
{
    /// <summary>ROOST_KV_TRANSFER_*</summary>
    public KvTransferKind Kind;

    /// <summary>C field <c>peer_rank</c>, documented in the header.</summary>
    public uint PeerRank;

    /// <summary>C field <c>group</c>, documented in the header.</summary>
    public uint Group;

    /// <summary>Index into block_ids</summary>
    public uint BlockStart;

    /// <summary>C field <c>block_count</c>, documented in the header.</summary>
    public uint BlockCount;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;

    /// <summary>C field <c>transfer_id</c>, documented in the header.</summary>
    public ulong TransferId;
}

/// <summary>Reserved: pooling and decision rows.</summary>
/// <remarks>Mirror of <c>RoostPlanHeadRow</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostPlanHeadRow")]
[StructLayout(LayoutKind.Sequential)]
public struct PlanHeadRow
{
    /// <summary>C field <c>sequence_index</c>, documented in the header.</summary>
    public uint SequenceIndex;

    /// <summary>ROOST_HEAD_*</summary>
    public HeadKind Head;

    /// <summary>Index into token_ids of the first row the head reads</summary>
    public uint TokenStart;

    /// <summary>C field <c>token_count</c>, documented in the header.</summary>
    public uint TokenCount;

    /// <summary>Byte offset into the result's head_outputs region</summary>
    public ulong OutputOffset;

    /// <summary>C field <c>output_bytes</c>, documented in the header.</summary>
    public uint OutputBytes;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Reserved (M5): encoder outputs (vision, audio) consumed by the decoder.</summary>
/// <remarks>Mirror of <c>RoostPlanEncoderOutput</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostPlanEncoderOutput")]
[StructLayout(LayoutKind.Sequential)]
public struct PlanEncoderOutput
{
    /// <summary>C field <c>sequence_index</c>, documented in the header.</summary>
    public uint SequenceIndex;

    /// <summary>C field <c>token_start</c>, documented in the header.</summary>
    public uint TokenStart;

    /// <summary>C field <c>token_count</c>, documented in the header.</summary>
    public uint TokenCount;

    /// <summary>Encoder-output state group (C4)</summary>
    public uint Group;

    /// <summary>C field <c>buffer_handle</c>, documented in the header.</summary>
    public ulong BufferHandle;

    /// <summary>C field <c>byte_offset</c>, documented in the header.</summary>
    public ulong ByteOffset;
}

/// <summary>Mirror of <c>RoostStepResultHeader</c>.</summary>
/// <remarks>Mirror of <c>RoostStepResultHeader</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostStepResultHeader")]
[StructLayout(LayoutKind.Sequential)]
public struct StepResultHeader
{
    /// <summary>ROOST_SHM_MAGIC_RESULT</summary>
    public uint Magic;

    /// <summary>C field <c>layout_version</c>, documented in the header.</summary>
    public uint LayoutVersion;

    /// <summary>C field <c>header_size</c>, documented in the header.</summary>
    public uint HeaderSize;

    /// <summary>ROOST_FAULT_*</summary>
    public FaultClass Fault;

    /// <summary>The plan's step_id</summary>
    public ulong StepId;

    /// <summary>C field <c>total_bytes</c>, documented in the header.</summary>
    public ulong TotalBytes;

    /// <summary>CUDA or C1 status behind a fault, 0 otherwise</summary>
    public int ErrorCode;

    /// <summary>C field <c>sequence_count</c>, documented in the header.</summary>
    public uint SequenceCount;

    /// <summary>C field <c>token_count</c>, documented in the header.</summary>
    public uint TokenCount;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;

    /// <summary>RoostResultSequence, plan order</summary>
    public TableRef Sequences;

    /// <summary>Int32 emitted tokens</summary>
    public TableRef TokenIds;

    /// <summary>RoostResultLogprob</summary>
    public TableRef Logprobs;

    /// <summary>Bytes, laid out by the plan's head_rows</summary>
    public TableRef HeadOutputs;

    /// <summary>One RoostDeviceTimings</summary>
    public TableRef Timings;

    /// <summary>Reserved: int32 tokens routed per expert (M2)</summary>
    public TableRef ExpertLoad;

    /// <summary>UTF-8 bytes of the fault message, may be empty</summary>
    public TableRef ErrorText;

    /// <summary>Reserved; zero.</summary>
    public ulong Reserved1;
}

/// <summary>Mirror of <c>RoostResultSequence</c>.</summary>
/// <remarks>Mirror of <c>RoostResultSequence</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostResultSequence")]
[StructLayout(LayoutKind.Sequential)]
public struct ResultSequence
{
    /// <summary>C field <c>sequence_handle</c>, documented in the header.</summary>
    public ulong SequenceHandle;

    /// <summary>Index into token_ids</summary>
    public uint TokenStart;

    /// <summary>Emitted: accepted drafts plus one</summary>
    public uint TokenCount;

    /// <summary>C field <c>accepted_drafts</c>, documented in the header.</summary>
    public uint AcceptedDrafts;

    /// <summary>ROOST_FINISH_*</summary>
    public FinishReason Finish;

    /// <summary>Index into logprobs; (1 + top-n) rows per emitted token</summary>
    public uint LogprobStart;

    /// <summary>C field <c>logprob_count</c>, documented in the header.</summary>
    public uint LogprobCount;
}

/// <summary>Mirror of <c>RoostResultLogprob</c>.</summary>
/// <remarks>Mirror of <c>RoostResultLogprob</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostResultLogprob")]
[StructLayout(LayoutKind.Sequential)]
public struct ResultLogprob
{
    /// <summary>C field <c>token</c>, documented in the header.</summary>
    public int Token;

    /// <summary>C field <c>logprob</c>, documented in the header.</summary>
    public float Logprob;
}

/// <summary>Mirror of <c>RoostDeviceTimings</c>.</summary>
/// <remarks>Mirror of <c>RoostDeviceTimings</c> in <c>kernels/include/roost_worker_shm.h</c>.</remarks>
[NativeStruct("RoostDeviceTimings")]
[StructLayout(LayoutKind.Sequential)]
public struct DeviceTimings
{
    /// <summary>Worker: plan seen to result published</summary>
    public ulong StepNs;

    /// <summary>Plan tables to device</summary>
    public ulong UploadNs;

    /// <summary>GPU, from events</summary>
    public ulong ForwardNs;

    /// <summary>C field <c>sample_ns</c>, documented in the header.</summary>
    public ulong SampleNs;

    /// <summary>C field <c>download_ns</c>, documented in the header.</summary>
    public ulong DownloadNs;

    /// <summary>C field <c>graph_replayed</c>, documented in the header.</summary>
    public uint GraphReplayed;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

using System.Runtime.InteropServices;

namespace Starlings.Roost.Abi;

// Mirrors of the C1 structs in kernels/include/roost_kernels.h, field for field. The layout tests compare them with
// tests/Starlings.Roost.Abstractions.Tests/Layout/abi-layout.golden.txt, which tools/abi-layout.py builds with gcc.

/// <summary>Sizes a call needs, from an entry's workspace query. Zero means none.</summary>
/// <remarks>Mirror of <c>RoostWorkspaceSizes</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostWorkspaceSizes")]
[StructLayout(LayoutKind.Sequential)]
public struct WorkspaceSizes
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>Device scratch</summary>
    public ulong DeviceBytes;

    /// <summary>Page-locked host staging used by plan()</summary>
    public ulong PinnedHostBytes;

    /// <summary>Host plan record written by plan() and read by run()</summary>
    public ulong PlanBytes;

    /// <summary>Minimum alignment of the device workspace in bytes</summary>
    public uint DeviceAlignment;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Caller-owned workspaces, sized from RoostWorkspaceSizes.</summary>
/// <remarks>Mirror of <c>RoostWorkspace</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostWorkspace")]
[StructLayout(LayoutKind.Sequential)]
public struct Workspace
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>C field <c>device</c>, documented in the header.</summary>
    public nint Device;

    /// <summary>C field <c>device_bytes</c>, documented in the header.</summary>
    public ulong DeviceBytes;

    /// <summary>C field <c>pinned_host</c>, documented in the header.</summary>
    public nint PinnedHost;

    /// <summary>C field <c>pinned_host_bytes</c>, documented in the header.</summary>
    public ulong PinnedHostBytes;

    /// <summary>C field <c>plan</c>, documented in the header.</summary>
    public nint Plan;

    /// <summary>C field <c>plan_bytes</c>, documented in the header.</summary>
    public ulong PlanBytes;
}

/// <summary>Constraints under which a variant is valid. A zero mask or zero bound means "no constraint" except where noted.</summary>
/// <remarks>Mirror of <c>RoostConstraints</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostConstraints")]
[StructLayout(LayoutKind.Sequential)]
public struct KernelConstraints
{
    /// <summary>Compute capability as major * 10 + minor, e.g. 86; required</summary>
    public uint SmMin;

    /// <summary>Inclusive; 0 means no upper bound</summary>
    public uint SmMax;

    /// <summary>Bit ROOST_DTYPE_x: activation types accepted</summary>
    public ulong ActDtypeMask;

    /// <summary>Bit ROOST_DTYPE_x: weight types accepted</summary>
    public ulong WeightDtypeMask;

    /// <summary>Bit ROOST_DTYPE_x: KV cache types accepted</summary>
    public ulong KvDtypeMask;

    /// <summary>Bit i: head dimension 16 * (i + 1) accepted</summary>
    public ulong HeadDimMask;

    /// <summary>Bit ROOST_LAYOUT_x</summary>
    public uint LayoutMask;

    /// <summary>Bit i: page (block) size 2^i accepted</summary>
    public uint PageSizeMask;

    /// <summary>Minimum pointer alignment in bytes of every buffer argument</summary>
    public uint MinAlignment;

    /// <summary>Sequences per call</summary>
    public uint MaxBatch;

    /// <summary>Tokens per call</summary>
    public ulong MaxTokens;

    /// <summary>Bit i: query heads per KV head of i + 1 accepted (attention)</summary>
    public uint GroupSizeMask;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Mirror of <c>RoostContextCreateParams</c>.</summary>
/// <remarks>Mirror of <c>RoostContextCreateParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostContextCreateParams")]
[StructLayout(LayoutKind.Sequential)]
public struct ContextCreateParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>CUDA device ordinal, current on the calling thread</summary>
    public int Device;

    /// <summary>ROOST_CONTEXT_*</summary>
    public ContextOptions Flags;

    /// <summary>Stream for setup work, may be NULL for the legacy default stream</summary>
    public nint Stream;

    /// <summary>Reserved; zero.</summary>
    public ulong Reserved0;
}

/// <summary>Mirror of <c>RoostOpEntry</c>.</summary>
/// <remarks>Mirror of <c>RoostOpEntry</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostOpEntry")]
[StructLayout(LayoutKind.Sequential)]
public struct OpEntry
{
    /// <summary>ROOST_OP_*</summary>
    public OpId Op;

    /// <summary>ROOST_FLAG_*</summary>
    public VariantTraits Flags;

    /// <summary>Stable ASCII id, unique per op across loaded libraries; conflicts resolve by it</summary>
    public nint VariantId;

    /// <summary>Sizeof the op's params struct this library was built against</summary>
    public uint ParamsStructSize;

    /// <summary>The minor of that struct</summary>
    public uint ParamsAbiMinor;

    /// <summary>C field <c>constraints</c>, documented in the header.</summary>
    public KernelConstraints Constraints;

    /// <summary>Required</summary>
    public nint Run;

    /// <summary>Required when ROOST_FLAG_NEEDS_PLAN, else NULL</summary>
    public nint Plan;

    /// <summary>NULL means no workspace</summary>
    public nint Workspace;

    /// <summary>Reserved; zero.</summary>
    public ulong Reserved0;
}

/// <summary>Filled by roost_register. Entries are static data owned by the library and live until it unloads.</summary>
/// <remarks>Mirror of <c>RoostOpTable</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostOpTable")]
[StructLayout(LayoutKind.Sequential)]
public struct OpTable
{
    /// <summary>Set by the caller</summary>
    public uint StructSize;

    /// <summary>Set by the caller</summary>
    public uint AbiMinor;

    /// <summary>Set by the library: sizeof(RoostOpEntry) it was built with</summary>
    public uint EntrySize;

    /// <summary>Set by the library</summary>
    public uint EntryCount;

    /// <summary>C field <c>entries</c>, documented in the header.</summary>
    public nint Entries;

    /// <summary>ASCII, for diagnostics and evidence</summary>
    public nint LibraryName;

    /// <summary>C field <c>library_abi_major</c>, documented in the header.</summary>
    public uint LibraryAbiMajor;

    /// <summary>C field <c>library_abi_minor</c>, documented in the header.</summary>
    public uint LibraryAbiMinor;
}

/// <summary>Out = x rsqrt(mean(x^2) + eps) weight, per row of width dim.</summary>
/// <remarks>Mirror of <c>RoostRmsNormParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostRmsNormParams")]
[StructLayout(LayoutKind.Sequential)]
public struct RmsNormParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[rows, dim]</summary>
    public DeviceSpan Input;

    /// <summary>[dim]</summary>
    public DeviceSpan Weight;

    /// <summary>[rows, dim]</summary>
    public DeviceSpan Output;

    /// <summary>C field <c>rows</c>, documented in the header.</summary>
    public uint Rows;

    /// <summary>C field <c>dim</c>, documented in the header.</summary>
    public uint Dim;

    /// <summary>Elements between rows; 0 means dim</summary>
    public uint InputStride;

    /// <summary>0 means dim</summary>
    public uint OutputStride;

    /// <summary>C field <c>eps</c>, documented in the header.</summary>
    public float Eps;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Residual += x; x = RmsNorm(residual). Fused form of Residual then RmsNorm.</summary>
/// <remarks>Mirror of <c>RoostAddRmsNormParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostAddRmsNormParams")]
[StructLayout(LayoutKind.Sequential)]
public struct AddRmsNormParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[rows, dim], overwritten with the normalised output</summary>
    public DeviceSpan Input;

    /// <summary>[rows, dim], overwritten with input + residual</summary>
    public DeviceSpan Residual;

    /// <summary>[dim]</summary>
    public DeviceSpan Weight;

    /// <summary>C field <c>rows</c>, documented in the header.</summary>
    public uint Rows;

    /// <summary>C field <c>dim</c>, documented in the header.</summary>
    public uint Dim;

    /// <summary>C field <c>eps</c>, documented in the header.</summary>
    public float Eps;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Per-head RmsNorm of query and key heads, in place.</summary>
/// <remarks>Mirror of <c>RoostQkNormParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostQkNormParams")]
[StructLayout(LayoutKind.Sequential)]
public struct QkNormParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[tokens, q_heads, head_dim] at q_stride</summary>
    public DeviceSpan Q;

    /// <summary>[tokens, kv_heads, head_dim] at k_stride</summary>
    public DeviceSpan K;

    /// <summary>[head_dim]</summary>
    public DeviceSpan QWeight;

    /// <summary>[head_dim]</summary>
    public DeviceSpan KWeight;

    /// <summary>C field <c>tokens</c>, documented in the header.</summary>
    public uint Tokens;

    /// <summary>C field <c>q_heads</c>, documented in the header.</summary>
    public uint QHeads;

    /// <summary>C field <c>kv_heads</c>, documented in the header.</summary>
    public uint KvHeads;

    /// <summary>C field <c>head_dim</c>, documented in the header.</summary>
    public uint HeadDim;

    /// <summary>Elements between tokens; 0 means q_heads * head_dim</summary>
    public uint QStride;

    /// <summary>0 means kv_heads * head_dim</summary>
    public uint KStride;

    /// <summary>C field <c>eps</c>, documented in the header.</summary>
    public float Eps;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Rotary position embedding of q and k, in place, NeoX (half-split) style unless interleaved is set.</summary>
/// <remarks>Mirror of <c>RoostRopeParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostRopeParams")]
[StructLayout(LayoutKind.Sequential)]
public struct RopeParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[tokens, q_heads, head_dim]</summary>
    public DeviceSpan Q;

    /// <summary>[tokens, kv_heads, head_dim]</summary>
    public DeviceSpan K;

    /// <summary>[tokens] int32</summary>
    public DeviceSpan Positions;

    /// <summary>C field <c>tokens</c>, documented in the header.</summary>
    public uint Tokens;

    /// <summary>C field <c>q_heads</c>, documented in the header.</summary>
    public uint QHeads;

    /// <summary>C field <c>kv_heads</c>, documented in the header.</summary>
    public uint KvHeads;

    /// <summary>C field <c>head_dim</c>, documented in the header.</summary>
    public uint HeadDim;

    /// <summary>0 means head_dim</summary>
    public uint RotaryDim;

    /// <summary>ROOST_ROPE_*</summary>
    public RopeKind Kind;

    /// <summary>C field <c>interleaved</c>, documented in the header.</summary>
    public uint Interleaved;

    /// <summary>C field <c>theta</c>, documented in the header.</summary>
    public float Theta;

    /// <summary>C field <c>llama3_factor</c>, documented in the header.</summary>
    public float Llama3Factor;

    /// <summary>C field <c>llama3_low_freq_factor</c>, documented in the header.</summary>
    public float Llama3LowFreqFactor;

    /// <summary>C field <c>llama3_high_freq_factor</c>, documented in the header.</summary>
    public float Llama3HighFreqFactor;

    /// <summary>C field <c>llama3_original_max_position</c>, documented in the header.</summary>
    public uint Llama3OriginalMaxPosition;
}

/// <summary>Paged attention over a state group's KV pools. Batch layout tables are device copies; the _host fields are host copies of the same tables read by plan() only (S2).</summary>
/// <remarks>Mirror of <c>RoostPagedAttentionParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostPagedAttentionParams")]
[StructLayout(LayoutKind.Sequential)]
public struct PagedAttentionParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[tokens, q_heads, head_dim]</summary>
    public DeviceSpan Q;

    /// <summary>[tokens, q_heads, head_dim]</summary>
    public DeviceSpan Output;

    /// <summary>Optional [tokens, q_heads] f32; pointer NULL when unused</summary>
    public DeviceSpan Lse;

    /// <summary>Pool, layout as kv_layout</summary>
    public DeviceSpan KCache;

    /// <summary>C field <c>v_cache</c>, documented in the header.</summary>
    public DeviceSpan VCache;

    /// <summary>[batch + 1] int32, token offsets per sequence</summary>
    public DeviceSpan QIndptr;

    /// <summary>[batch + 1] int32, offsets into kv_indices</summary>
    public DeviceSpan KvIndptr;

    /// <summary>Page ids</summary>
    public DeviceSpan KvIndices;

    /// <summary>[batch] int32, 1..page_size</summary>
    public DeviceSpan KvLastPageLen;

    /// <summary>[batch + 1], plan() only</summary>
    public nint QIndptrHost;

    /// <summary>[batch + 1], plan() only</summary>
    public nint KvIndptrHost;

    /// <summary>[batch], plan() only</summary>
    public nint KvLastPageLenHost;

    /// <summary>ROOST_ATTENTION_*</summary>
    public AttentionMode Mode;

    /// <summary>C field <c>batch</c>, documented in the header.</summary>
    public uint Batch;

    /// <summary>C field <c>tokens</c>, documented in the header.</summary>
    public uint Tokens;

    /// <summary>C field <c>q_heads</c>, documented in the header.</summary>
    public uint QHeads;

    /// <summary>C field <c>kv_heads</c>, documented in the header.</summary>
    public uint KvHeads;

    /// <summary>C field <c>head_dim</c>, documented in the header.</summary>
    public uint HeadDim;

    /// <summary>C field <c>page_size</c>, documented in the header.</summary>
    public uint PageSize;

    /// <summary>ROOST_LAYOUT_PAGED_*</summary>
    public TensorLayout KvLayout;

    /// <summary>0 means 1 / sqrt(head_dim)</summary>
    public float SmScale;

    /// <summary>0 disables</summary>
    public float LogitsSoftCap;

    /// <summary>-1 disables the sliding window</summary>
    public int WindowLeft;

    /// <summary>Prefill only</summary>
    public uint Causal;
}

/// <summary>Mirror of <c>RoostActAndMulParams</c>.</summary>
/// <remarks>Mirror of <c>RoostActAndMulParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostActAndMulParams")]
[StructLayout(LayoutKind.Sequential)]
public struct ActAndMulParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[rows, 2 * dim]</summary>
    public DeviceSpan Input;

    /// <summary>[rows, dim]</summary>
    public DeviceSpan Output;

    /// <summary>C field <c>rows</c>, documented in the header.</summary>
    public uint Rows;

    /// <summary>C field <c>dim</c>, documented in the header.</summary>
    public uint Dim;

    /// <summary>ROOST_GELU_*, GeluAndMul only</summary>
    public GeluApproximation GeluApproximation;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>D = alpha a @ b (+ beta c), a [m, k], b [k, n] or its transpose, fp32 accumulation.</summary>
/// <remarks>Mirror of <c>RoostGemmParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostGemmParams")]
[StructLayout(LayoutKind.Sequential)]
public struct GemmParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>C field <c>a</c>, documented in the header.</summary>
    public DeviceSpan A;

    /// <summary>C field <c>b</c>, documented in the header.</summary>
    public DeviceSpan B;

    /// <summary>Optional; pointer NULL when beta is 0</summary>
    public DeviceSpan C;

    /// <summary>C field <c>d</c>, documented in the header.</summary>
    public DeviceSpan D;

    /// <summary>Optional [n]</summary>
    public DeviceSpan Bias;

    /// <summary>C field <c>m</c>, documented in the header.</summary>
    public ulong M;

    /// <summary>C field <c>n</c>, documented in the header.</summary>
    public ulong N;

    /// <summary>C field <c>k</c>, documented in the header.</summary>
    public ulong K;

    /// <summary>C field <c>lda</c>, documented in the header.</summary>
    public ulong Lda;

    /// <summary>C field <c>ldb</c>, documented in the header.</summary>
    public ulong Ldb;

    /// <summary>C field <c>ldd</c>, documented in the header.</summary>
    public ulong Ldd;

    /// <summary>C field <c>transpose_a</c>, documented in the header.</summary>
    public uint TransposeA;

    /// <summary>C field <c>transpose_b</c>, documented in the header.</summary>
    public uint TransposeB;

    /// <summary>C field <c>alpha</c>, documented in the header.</summary>
    public float Alpha;

    /// <summary>C field <c>beta</c>, documented in the header.</summary>
    public float Beta;

    /// <summary>Plan's shape bucket; selects the pinned algorithm (S2)</summary>
    public uint ShapeBucket;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Output[i] = table[ids[i]].</summary>
/// <remarks>Mirror of <c>RoostEmbeddingParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostEmbeddingParams")]
[StructLayout(LayoutKind.Sequential)]
public struct EmbeddingParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[tokens] int32</summary>
    public DeviceSpan Ids;

    /// <summary>[vocab, dim]</summary>
    public DeviceSpan Table;

    /// <summary>[tokens, dim]</summary>
    public DeviceSpan Output;

    /// <summary>C field <c>tokens</c>, documented in the header.</summary>
    public uint Tokens;

    /// <summary>C field <c>dim</c>, documented in the header.</summary>
    public uint Dim;

    /// <summary>C field <c>vocab</c>, documented in the header.</summary>
    public uint Vocab;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Logits = hidden[rows] @ weight^T for the selected rows (last token of each sequence, or all draft rows).</summary>
/// <remarks>Mirror of <c>RoostLmHeadParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostLmHeadParams")]
[StructLayout(LayoutKind.Sequential)]
public struct LmHeadParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[tokens, dim]</summary>
    public DeviceSpan Hidden;

    /// <summary>[selected] int32 indices into hidden</summary>
    public DeviceSpan Rows;

    /// <summary>[vocab, dim]</summary>
    public DeviceSpan Weight;

    /// <summary>[selected, vocab_padded]</summary>
    public DeviceSpan Logits;

    /// <summary>C field <c>selected</c>, documented in the header.</summary>
    public uint Selected;

    /// <summary>C field <c>dim</c>, documented in the header.</summary>
    public uint Dim;

    /// <summary>C field <c>vocab</c>, documented in the header.</summary>
    public uint Vocab;

    /// <summary>C field <c>vocab_padded</c>, documented in the header.</summary>
    public uint VocabPadded;

    /// <summary>0 disables final logit soft-capping</summary>
    public float SoftCap;

    /// <summary>C field <c>shape_bucket</c>, documented in the header.</summary>
    public uint ShapeBucket;
}

/// <summary>Output = a + b (a may alias output).</summary>
/// <remarks>Mirror of <c>RoostResidualParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostResidualParams")]
[StructLayout(LayoutKind.Sequential)]
public struct ResidualParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>C field <c>a</c>, documented in the header.</summary>
    public DeviceSpan A;

    /// <summary>C field <c>b</c>, documented in the header.</summary>
    public DeviceSpan B;

    /// <summary>C field <c>output</c>, documented in the header.</summary>
    public DeviceSpan Output;

    /// <summary>C field <c>elements</c>, documented in the header.</summary>
    public ulong Elements;
}

/// <summary>Write new K and V rows into the paged pools at slot_mapping (page page_size + offset).</summary>
/// <remarks>Mirror of <c>RoostCacheWriteParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostCacheWriteParams")]
[StructLayout(LayoutKind.Sequential)]
public struct CacheWriteParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[tokens, kv_heads, head_dim]</summary>
    public DeviceSpan K;

    /// <summary>C field <c>v</c>, documented in the header.</summary>
    public DeviceSpan V;

    /// <summary>C field <c>k_cache</c>, documented in the header.</summary>
    public DeviceSpan KCache;

    /// <summary>C field <c>v_cache</c>, documented in the header.</summary>
    public DeviceSpan VCache;

    /// <summary>[tokens] int64; negative skips the token</summary>
    public DeviceSpan SlotMapping;

    /// <summary>C field <c>tokens</c>, documented in the header.</summary>
    public uint Tokens;

    /// <summary>C field <c>kv_heads</c>, documented in the header.</summary>
    public uint KvHeads;

    /// <summary>C field <c>head_dim</c>, documented in the header.</summary>
    public uint HeadDim;

    /// <summary>C field <c>page_size</c>, documented in the header.</summary>
    public uint PageSize;

    /// <summary>ROOST_LAYOUT_PAGED_*</summary>
    public TensorLayout KvLayout;

    /// <summary>Quantised KV only; 1 otherwise</summary>
    public float KScale;

    /// <summary>C field <c>v_scale</c>, documented in the header.</summary>
    public float VScale;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Copy whole blocks within or between pools: copy-on-write forks, compaction and tier moves.</summary>
/// <remarks>Mirror of <c>RoostCacheCopyParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostCacheCopyParams")]
[StructLayout(LayoutKind.Sequential)]
public struct CacheCopyParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>C field <c>src_pool</c>, documented in the header.</summary>
    public DeviceSpan SrcPool;

    /// <summary>May equal src_pool</summary>
    public DeviceSpan DstPool;

    /// <summary>[count, 2] int32: (src block, dst block)</summary>
    public DeviceSpan Pairs;

    /// <summary>C field <c>block_bytes</c>, documented in the header.</summary>
    public ulong BlockBytes;

    /// <summary>C field <c>count</c>, documented in the header.</summary>
    public uint Count;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Ids[r] = argmax(logits[r, :vocab]); ties go to the lowest index.</summary>
/// <remarks>Mirror of <c>RoostArgmaxParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostArgmaxParams")]
[StructLayout(LayoutKind.Sequential)]
public struct ArgmaxParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[rows, stride]</summary>
    public DeviceSpan Logits;

    /// <summary>[rows] int32</summary>
    public DeviceSpan Ids;

    /// <summary>C field <c>rows</c>, documented in the header.</summary>
    public uint Rows;

    /// <summary>C field <c>vocab</c>, documented in the header.</summary>
    public uint Vocab;

    /// <summary>0 means vocab</summary>
    public uint Stride;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

/// <summary>Temperature, top-k, top-p and min-p sampling with per-row parameters and seeds; optional token masks.</summary>
/// <remarks>Mirror of <c>RoostSampleParams</c> in <c>kernels/include/roost_kernels.h</c>.</remarks>
[NativeStruct("RoostSampleParams")]
[StructLayout(LayoutKind.Sequential)]
public struct SampleParams
{
    /// <summary>Size of this struct as the caller sees it (C1 rule).</summary>
    public uint StructSize;

    /// <summary>C1 ABI minor the caller was built against.</summary>
    public uint AbiMinor;

    /// <summary>[rows, stride] f32</summary>
    public DeviceSpan Logits;

    /// <summary>[rows] f32; 0 means greedy</summary>
    public DeviceSpan Temperature;

    /// <summary>[rows] int32; 0 disables</summary>
    public DeviceSpan TopK;

    /// <summary>[rows] f32; 1 disables</summary>
    public DeviceSpan TopP;

    /// <summary>[rows] f32; 0 disables</summary>
    public DeviceSpan MinP;

    /// <summary>[rows] u64</summary>
    public DeviceSpan Seeds;

    /// <summary>[rows] u64, Philox offsets</summary>
    public DeviceSpan Offsets;

    /// <summary>Optional [rows, ceil(vocab / 32)] u32 allowed-token bitmask; pointer NULL when unused</summary>
    public DeviceSpan Mask;

    /// <summary>[rows] int32 output</summary>
    public DeviceSpan Ids;

    /// <summary>Optional [rows] f32 output of the chosen token</summary>
    public DeviceSpan Logprobs;

    /// <summary>C field <c>rows</c>, documented in the header.</summary>
    public uint Rows;

    /// <summary>C field <c>vocab</c>, documented in the header.</summary>
    public uint Vocab;

    /// <summary>C field <c>stride</c>, documented in the header.</summary>
    public uint Stride;

    /// <summary>Reserved; zero.</summary>
    public uint Reserved0;
}

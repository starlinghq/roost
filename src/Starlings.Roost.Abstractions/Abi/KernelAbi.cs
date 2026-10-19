namespace Starlings.Roost.Abi;

/// <summary>Versions of the kernel library C ABI (contract C1, <c>kernels/include/roost_kernels.h</c>).</summary>
public static class KernelAbi
{
    /// <summary><c>ROOST_ABI_MAJOR</c>. A loader accepts only a library with the same major.</summary>
    public const int Major = 1;

    /// <summary><c>ROOST_ABI_MINOR</c>. A loader accepts a library whose minor is at least this.</summary>
    public const int Minor = 1;

    /// <summary><c>ROOST_ABI_VERSION</c>, the value <c>roost_abi_version()</c> returns: major in the high 16 bits.</summary>
    public const uint Version = (Major << 16) | Minor;

    /// <summary>Whether a library reporting <paramref name="libraryVersion"/> may be loaded by this caller.</summary>
    /// <param name="libraryVersion">The value the library's <c>roost_abi_version()</c> returned.</param>
    /// <returns>True when the majors are equal and the library minor is at least <see cref="Minor"/>.</returns>
    public static bool IsCompatible(uint libraryVersion) =>
        (libraryVersion >> 16) == Major && (libraryVersion & 0xFFFF) >= Minor;
}

/// <summary>C1 status codes (<c>ROOST_STATUS_*</c>).</summary>
public enum KernelStatus
{
    /// <summary>Success.</summary>
    Ok = 0,

    /// <summary>A field is out of range, inconsistent or a reserved field is not zero.</summary>
    InvalidArgument = 1,

    /// <summary>The variant does not support this shape, type or device.</summary>
    Unsupported = 2,

    /// <summary>A CUDA call failed; <c>roost_last_error()</c> names it.</summary>
    CudaError = 3,

    /// <summary>An internal error in the library.</summary>
    Internal = 4,

    /// <summary>A struct size or ABI minor the library cannot read.</summary>
    AbiMismatch = 5,

    /// <summary>A host allocation in setup failed. Entry points never allocate device memory.</summary>
    OutOfMemory = 6,

    /// <summary>A caller-owned workspace is smaller than the workspace query reported.</summary>
    WorkspaceTooSmall = 7,

    /// <summary><c>run()</c> was called without a matching <c>plan()</c>.</summary>
    NotPlanned = 8,
}

/// <summary>Element types (<c>ROOST_DTYPE_*</c>), shared by C1, C2 and C3.</summary>
public enum DType : uint
{
    /// <summary>No type; a zero span.</summary>
    Undefined = 0,

    /// <summary>IEEE binary32.</summary>
    F32 = 1,

    /// <summary>IEEE binary16.</summary>
    F16 = 2,

    /// <summary>bfloat16.</summary>
    BF16 = 3,

    /// <summary>IEEE binary64 (CPU reference).</summary>
    F64 = 4,

    /// <summary>FP8 E4M3.</summary>
    F8E4M3 = 5,

    /// <summary>FP8 E5M2.</summary>
    F8E5M2 = 6,

    /// <summary>Signed 8-bit integer.</summary>
    I8 = 7,

    /// <summary>Unsigned 8-bit integer.</summary>
    U8 = 8,

    /// <summary>Signed 16-bit integer.</summary>
    I16 = 9,

    /// <summary>Signed 32-bit integer.</summary>
    I32 = 10,

    /// <summary>Unsigned 32-bit integer.</summary>
    U32 = 11,

    /// <summary>Signed 64-bit integer.</summary>
    I64 = 12,

    /// <summary>Unsigned 64-bit integer.</summary>
    U64 = 13,

    /// <summary>Two signed 4-bit values per byte, low nibble first.</summary>
    I4 = 14,

    /// <summary>Two unsigned 4-bit values per byte, low nibble first.</summary>
    U4 = 15,

    /// <summary>FP4 E2M1.</summary>
    F4E2M1 = 16,
}

/// <summary>Buffer layouts (<c>ROOST_LAYOUT_*</c>).</summary>
public enum TensorLayout : uint
{
    /// <summary>Row-major (C order).</summary>
    RowMajor = 0,

    /// <summary>Column-major.</summary>
    ColMajor = 1,

    /// <summary>Paged KV pool, <c>[pages, page_size, kv_heads, head_dim]</c>.</summary>
    PagedNhd = 2,

    /// <summary>Paged KV pool, <c>[pages, kv_heads, page_size, head_dim]</c>.</summary>
    PagedHnd = 3,
}

/// <summary>
/// Op ids (<c>ROOST_OP_*</c>) of the registration table. The numbering is permanent; contract C3 names the same ops.
/// </summary>
public enum OpId : uint
{
    /// <summary>Not an op.</summary>
    None = 0,

    /// <summary>M0.</summary>
    RmsNorm = 1,

    /// <summary>M0.</summary>
    QkNorm = 2,

    /// <summary>M0: default and llama3 scaling.</summary>
    Rope = 3,

    /// <summary>M0: decode and prefill.</summary>
    PagedAttention = 4,

    /// <summary>M0.</summary>
    SiluAndMul = 5,

    /// <summary>M0.</summary>
    GeluAndMul = 6,

    /// <summary>M0.</summary>
    Gemm = 7,

    /// <summary>M0.</summary>
    Embedding = 8,

    /// <summary>M0.</summary>
    LmHead = 9,

    /// <summary>M0.</summary>
    Residual = 10,

    /// <summary>M0 cache op: write new K and V into pages.</summary>
    CacheWrite = 11,

    /// <summary>M0 cache op: block copies and tier moves.</summary>
    CacheCopy = 12,

    /// <summary>M0.</summary>
    Argmax = 13,

    /// <summary>M0.</summary>
    Sample = 14,

    /// <summary>Optional fused: Residual then RmsNorm.</summary>
    AddRmsNorm = 64,

    /// <summary>Optional fused: AddRmsNorm then Quantize.</summary>
    AddRmsNormQuant = 65,

    /// <summary>Optional fused: SiluAndMul then Quantize.</summary>
    SiluMulQuant = 66,

    /// <summary>Optional fused: QKV projection split, QkNorm, Rope and CacheWrite.</summary>
    QkvQkNormRopeCacheWrite = 67,

    /// <summary>Optional fused: Gemm with a LoRA expand epilogue.</summary>
    GemmLoraEpilogue = 68,

    /// <summary>Reserved, M1.</summary>
    GatedDeltaRule = 128,

    /// <summary>Reserved, M1.</summary>
    CausalConv1d = 129,

    /// <summary>Reserved, M1: recurrent state slot copy.</summary>
    RecurrentStateCopy = 130,

    /// <summary>Reserved, M1: recurrent state snapshot write and restore.</summary>
    RecurrentStateSnapshot = 131,

    /// <summary>Reserved, M1.</summary>
    SlidingWindowAttention = 132,

    /// <summary>Reserved, M2: YaRN, longrope, dynamic NTK.</summary>
    RopeScaled = 160,

    /// <summary>Reserved, M2.</summary>
    Quantize = 161,

    /// <summary>Reserved, M2.</summary>
    Dequantize = 162,

    /// <summary>Reserved, M2.</summary>
    GemmW4A16 = 163,

    /// <summary>Reserved, M2.</summary>
    GemmFp8 = 164,

    /// <summary>Reserved, M2.</summary>
    TreeAttention = 165,

    /// <summary>Reserved, M2.</summary>
    RejectSample = 166,

    /// <summary>Reserved, M2.</summary>
    MoeRouteTopK = 167,

    /// <summary>Reserved, M2.</summary>
    MoePermute = 168,

    /// <summary>Reserved, M2.</summary>
    MoeUnpermute = 169,

    /// <summary>Reserved, M2.</summary>
    GroupedGemm = 170,

    /// <summary>Reserved, M2.</summary>
    FusedMoe = 171,

    /// <summary>Reserved, M3.</summary>
    LoraShrink = 192,

    /// <summary>Reserved, M3.</summary>
    LoraExpand = 193,

    /// <summary>Reserved, M4.</summary>
    AllReduce = 224,

    /// <summary>Reserved, M4.</summary>
    AllGather = 225,

    /// <summary>Reserved, M4.</summary>
    ReduceScatter = 226,

    /// <summary>Reserved, M4.</summary>
    AllToAll = 227,

    /// <summary>Reserved, M4.</summary>
    MlaAttention = 228,

    /// <summary>Reserved, M5.</summary>
    PatchEmbed = 256,

    /// <summary>Reserved, M5.</summary>
    BidirectionalAttention = 257,

    /// <summary>Reserved, M5.</summary>
    MelSpectrogram = 258,

    /// <summary>Reserved, M5.</summary>
    AudioConvSubsample = 259,

    /// <summary>Reserved, M5.</summary>
    DequantizeGguf = 260,
}

/// <summary>Registration flags of a variant (<c>ROOST_FLAG_*</c>).</summary>
[Flags]
public enum VariantTraits : uint
{
    /// <summary>No flags.</summary>
    None = 0,

    /// <summary><c>run()</c> may be captured in a CUDA graph.</summary>
    GraphCapturable = 0x1,

    /// <summary>Bitwise identical output for identical input and launch shape.</summary>
    Deterministic = 0x2,

    /// <summary>A row's output does not depend on the other rows in the batch.</summary>
    BatchInvariant = 0x4,

    /// <summary><c>plan()</c> must run on the host before <c>run()</c>, and is never captured.</summary>
    NeedsPlan = 0x8,

    /// <summary>A plan stays valid for every call with the same shape bucket.</summary>
    PlanPerShape = 0x10,
}

/// <summary>Library context creation flags (<c>ROOST_CONTEXT_*</c>).</summary>
[Flags]
public enum ContextOptions : uint
{
    /// <summary>No flags.</summary>
    None = 0,

    /// <summary>Choose only variants flagged deterministic and batch invariant.</summary>
    Deterministic = 0x1,
}

/// <summary>Rope scaling kinds of the M0 Rope op (<c>ROOST_ROPE_*</c>).</summary>
public enum RopeKind : uint
{
    /// <summary>Unscaled rotary embedding.</summary>
    Default = 0,

    /// <summary>Llama 3 frequency scaling.</summary>
    Llama3 = 1,
}

/// <summary>Paged attention modes (<c>ROOST_ATTENTION_*</c>).</summary>
public enum AttentionMode : uint
{
    /// <summary>One query token per sequence.</summary>
    Decode = 0,

    /// <summary>Several query tokens per sequence.</summary>
    Prefill = 1,
}

/// <summary>GELU forms (<c>ROOST_GELU_*</c>).</summary>
public enum GeluApproximation : uint
{
    /// <summary>The erf form.</summary>
    Exact = 0,

    /// <summary>The tanh approximation.</summary>
    Tanh = 1,
}

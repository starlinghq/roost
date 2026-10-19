/*
 * roost_kernels.h: the kernel library C ABI and registration table (contract C1, build plan section 5).
 *
 * ABI major 1, minor 2. Frozen as contract v0 by R-004 (1.0); 1.1 adds RoostGatedDeltaRuleParams with row strides for
 * q, k and v (2026-10-19); 1.2 adds the Qwen3.5 gating ops (GatedDeltaRuleGates, GatedRmsNorm, SigmoidMul) and the params
 * of CausalConv1d, RecurrentStateCopy and RecurrentStateSnapshot (contract version 3). The ADR is the "Contracts" section
 * of README.md.
 *
 * Rules (the ADR gives the reasons):
 *   - Fixed-width integer types, float, double and pointers only. No size_t, long, bool, enum-typed fields or POSIX
 *     types, so the layout is the same under gcc, clang and MSVC on x86-64 and aarch64. 32-bit targets are not
 *     supported.
 *   - Every params struct begins with uint32_t struct_size and uint32_t abi_minor. A library rejects a struct_size
 *     smaller than it needs for the abi_minor it was given with ROOST_STATUS_ABI_MISMATCH. Fields are only ever
 *     appended, in a minor bump; nothing is reordered or removed within a major.
 *   - The loader accepts a library whose major equals ROOST_ABI_MAJOR and whose minor is at least the caller's.
 *   - Every entry point takes a library context created by roost_context_create (cuBLASLt handles, pinned algorithm
 *     choices, loaded modules), the params, an optional caller-owned workspace and a CUDA stream (cudaStream_t).
 *   - Entry points never allocate device memory, never synchronise the host, and leave workspace contents undefined
 *     on entry: a kernel that needs zeroed scratch zeroes it itself on the stream.
 *   - Errors return a status; roost_last_error() gives the message for the calling thread.
 *   - Reserved fields are zero. A library may reject a non-zero reserved field with ROOST_STATUS_INVALID_ARGUMENT.
 *   - Libraries link the shared CUDA runtime (S1): the host process and every kernel library share one runtime, so
 *     streams and graphs pass between them.
 *   - Batch invariance and determinism are registration flags and context options, never environment variables (S2).
 *
 * Params structs exist for the M0 ops and for the hybrid ops Qwen3.5-0.8B needs (moved into M0 by decision 15). Op ids for
 * later phases are reserved below so their numbering never moves; their params structs are added, never changed, when
 * their items land.
 */
#ifndef ROOST_KERNELS_H_
#define ROOST_KERNELS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(ROOST_KERNELS_BUILD)
#define ROOST_API __declspec(dllexport)
#else
#define ROOST_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define ROOST_API __attribute__((visibility("default")))
#else
#define ROOST_API
#endif

/* ---- Versions ------------------------------------------------------------------------------------------------ */

#define ROOST_ABI_MAJOR 1
#define ROOST_ABI_MINOR 2
/* roost_abi_version() returns (major << 16) | minor. */
#define ROOST_ABI_VERSION ((ROOST_ABI_MAJOR << 16) | ROOST_ABI_MINOR)

/* ---- Status codes (int32_t) ---------------------------------------------------------------------------------- */

typedef int32_t RoostStatus;

#define ROOST_STATUS_OK 0
#define ROOST_STATUS_INVALID_ARGUMENT 1
#define ROOST_STATUS_UNSUPPORTED 2
#define ROOST_STATUS_CUDA_ERROR 3
#define ROOST_STATUS_INTERNAL 4
#define ROOST_STATUS_ABI_MISMATCH 5
#define ROOST_STATUS_OUT_OF_MEMORY 6
#define ROOST_STATUS_WORKSPACE_TOO_SMALL 7
#define ROOST_STATUS_NOT_PLANNED 8

/* ---- Element types (uint32_t) -------------------------------------------------------------------------------- */

#define ROOST_DTYPE_UNDEFINED 0
#define ROOST_DTYPE_F32 1
#define ROOST_DTYPE_F16 2
#define ROOST_DTYPE_BF16 3
#define ROOST_DTYPE_F64 4
#define ROOST_DTYPE_F8E4M3 5
#define ROOST_DTYPE_F8E5M2 6
#define ROOST_DTYPE_I8 7
#define ROOST_DTYPE_U8 8
#define ROOST_DTYPE_I16 9
#define ROOST_DTYPE_I32 10
#define ROOST_DTYPE_U32 11
#define ROOST_DTYPE_I64 12
#define ROOST_DTYPE_U64 13
#define ROOST_DTYPE_I4 14   /* two signed 4-bit values per byte, low nibble first */
#define ROOST_DTYPE_U4 15   /* two unsigned 4-bit values per byte, low nibble first */
#define ROOST_DTYPE_F4E2M1 16
#define ROOST_DTYPE_COUNT 17

/* ---- Layouts (uint32_t) -------------------------------------------------------------------------------------- */

#define ROOST_LAYOUT_ROW_MAJOR 0
#define ROOST_LAYOUT_COL_MAJOR 1
/* Paged KV pools: [pages, page_size, kv_heads, head_dim] (NHD) or [pages, kv_heads, page_size, head_dim] (HND). */
#define ROOST_LAYOUT_PAGED_NHD 2
#define ROOST_LAYOUT_PAGED_HND 3

/* ---- Op ids (uint32_t) --------------------------------------------------------------------------------------- */
/* Numbering is permanent. Mirrored by Starlings.Roost.Abi.OpId; contract C3 names the same ops. */

/* M0 */
#define ROOST_OP_RMS_NORM 1
#define ROOST_OP_QK_NORM 2
#define ROOST_OP_ROPE 3
#define ROOST_OP_PAGED_ATTENTION 4
#define ROOST_OP_SILU_AND_MUL 5
#define ROOST_OP_GELU_AND_MUL 6
#define ROOST_OP_GEMM 7
#define ROOST_OP_EMBEDDING 8
#define ROOST_OP_LM_HEAD 9
#define ROOST_OP_RESIDUAL 10
#define ROOST_OP_CACHE_WRITE 11
#define ROOST_OP_CACHE_COPY 12
#define ROOST_OP_ARGMAX 13
#define ROOST_OP_SAMPLE 14
/* Optional fused ops, each defined as its unfused parts */
#define ROOST_OP_ADD_RMS_NORM 64
#define ROOST_OP_ADD_RMS_NORM_QUANT 65
#define ROOST_OP_SILU_MUL_QUANT 66
#define ROOST_OP_QKV_QK_NORM_ROPE_CACHE_WRITE 67
#define ROOST_OP_GEMM_LORA_EPILOGUE 68
/* Reserved, M1 */
#define ROOST_OP_GATED_DELTA_RULE 128
#define ROOST_OP_CAUSAL_CONV1D 129
#define ROOST_OP_RECURRENT_STATE_COPY 130
#define ROOST_OP_RECURRENT_STATE_SNAPSHOT 131
#define ROOST_OP_SLIDING_WINDOW_ATTENTION 132
/* Qwen3.5 gating (contract version 3, ABI 1.2); numbered in the hybrid block next to GatedDeltaRule */
#define ROOST_OP_GATED_DELTA_RULE_GATES 133
#define ROOST_OP_GATED_RMS_NORM 134
#define ROOST_OP_SIGMOID_MUL 135
/* Reserved, M2 */
#define ROOST_OP_ROPE_SCALED 160
#define ROOST_OP_QUANTIZE 161
#define ROOST_OP_DEQUANTIZE 162
#define ROOST_OP_GEMM_W4A16 163
#define ROOST_OP_GEMM_FP8 164
#define ROOST_OP_TREE_ATTENTION 165
#define ROOST_OP_REJECT_SAMPLE 166
#define ROOST_OP_MOE_ROUTE_TOP_K 167
#define ROOST_OP_MOE_PERMUTE 168
#define ROOST_OP_MOE_UNPERMUTE 169
#define ROOST_OP_GROUPED_GEMM 170
#define ROOST_OP_FUSED_MOE 171
/* Reserved, M3 */
#define ROOST_OP_LORA_SHRINK 192
#define ROOST_OP_LORA_EXPAND 193
/* Reserved, M4 */
#define ROOST_OP_ALL_REDUCE 224
#define ROOST_OP_ALL_GATHER 225
#define ROOST_OP_REDUCE_SCATTER 226
#define ROOST_OP_ALL_TO_ALL 227
#define ROOST_OP_MLA_ATTENTION 228
/* Reserved, M5 */
#define ROOST_OP_PATCH_EMBED 256
#define ROOST_OP_BIDIRECTIONAL_ATTENTION 257
#define ROOST_OP_MEL_SPECTROGRAM 258
#define ROOST_OP_AUDIO_CONV_SUBSAMPLE 259
#define ROOST_OP_DEQUANTIZE_GGUF 260

/* ---- Registration flags (uint32_t) --------------------------------------------------------------------------- */

#define ROOST_FLAG_GRAPH_CAPTURABLE 0x1u  /* run() may be captured in a CUDA graph */
#define ROOST_FLAG_DETERMINISTIC 0x2u     /* bitwise identical output for identical input and launch shape */
#define ROOST_FLAG_BATCH_INVARIANT 0x4u   /* a row's output does not depend on the other rows in the batch */
#define ROOST_FLAG_NEEDS_PLAN 0x8u        /* plan() must run on the host before run(); plan() is never captured */
#define ROOST_FLAG_PLAN_PER_SHAPE 0x10u   /* a plan stays valid for every call with the same shape bucket */

/* ---- Common records ------------------------------------------------------------------------------------------ */

/* A buffer: device (or, in CPU tests, host) address, element count and element type. Mirrors C3 DeviceSpan.
 * Worker-side only: never placed in shared memory between processes. */
typedef struct RoostDeviceSpan {
  void* address;
  uint64_t length; /* elements, not bytes; for packed 4-bit types, logical elements */
  uint32_t dtype;  /* ROOST_DTYPE_* */
  uint32_t reserved0;
} RoostDeviceSpan;

/* Sizes a call needs, from an entry's workspace query. Zero means none. */
typedef struct RoostWorkspaceSizes {
  uint32_t struct_size;
  uint32_t abi_minor;
  uint64_t device_bytes;      /* device scratch */
  uint64_t pinned_host_bytes; /* page-locked host staging used by plan() */
  uint64_t plan_bytes;        /* host plan record written by plan() and read by run() */
  uint32_t device_alignment;  /* minimum alignment of the device workspace in bytes */
  uint32_t reserved0;
} RoostWorkspaceSizes;

/* Caller-owned workspaces, sized from RoostWorkspaceSizes. */
typedef struct RoostWorkspace {
  uint32_t struct_size;
  uint32_t abi_minor;
  void* device;
  uint64_t device_bytes;
  void* pinned_host;
  uint64_t pinned_host_bytes;
  void* plan;
  uint64_t plan_bytes;
} RoostWorkspace;

/* Constraints under which a variant is valid. A zero mask or zero bound means "no constraint" except where noted. */
typedef struct RoostConstraints {
  uint32_t sm_min;            /* compute capability as major * 10 + minor, e.g. 86; required */
  uint32_t sm_max;            /* inclusive; 0 means no upper bound */
  uint64_t act_dtype_mask;    /* bit ROOST_DTYPE_x: activation types accepted */
  uint64_t weight_dtype_mask; /* bit ROOST_DTYPE_x: weight types accepted */
  uint64_t kv_dtype_mask;     /* bit ROOST_DTYPE_x: KV cache types accepted */
  uint64_t head_dim_mask;     /* bit i: head dimension 16 * (i + 1) accepted */
  uint32_t layout_mask;       /* bit ROOST_LAYOUT_x */
  uint32_t page_size_mask;    /* bit i: page (block) size 2^i accepted */
  uint32_t min_alignment;     /* minimum pointer alignment in bytes of every buffer argument */
  uint32_t max_batch;         /* sequences per call */
  uint64_t max_tokens;        /* tokens per call */
  uint32_t group_size_mask;   /* bit i: query heads per KV head of i + 1 accepted (attention) */
  uint32_t reserved0;
} RoostConstraints;

/* ---- Library context ----------------------------------------------------------------------------------------- */

/* Per-library, per-device state that outlives a call: cuBLASLt handles, pinned Gemm algorithms, loaded modules.
 * A context is used by one thread at a time; the worker owns one per loaded library and device. */
typedef struct RoostContext RoostContext;

typedef struct RoostContextCreateParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  int32_t device;           /* CUDA device ordinal, current on the calling thread */
  uint32_t flags;           /* ROOST_CONTEXT_* */
  void* stream;             /* stream for setup work, may be NULL for the legacy default stream */
  uint64_t reserved0;
} RoostContextCreateParams;

#define ROOST_CONTEXT_DETERMINISTIC 0x1u /* choose only variants flagged deterministic and batch invariant */

/* ---- Entry points -------------------------------------------------------------------------------------------- */

/* The params pointer is the op's Roost<Op>Params; ws may be NULL when the workspace query reports zero sizes. */
typedef RoostStatus (*RoostRunFn)(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream);
/* Host-side planning: reads host copies of the batch layout, writes ws->plan and ws->pinned_host, copies to the
 * device on stream. Never captured in a graph. */
typedef RoostStatus (*RoostPlanFn)(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream);
typedef RoostStatus (*RoostWorkspaceFn)(RoostContext* ctx, const void* params, RoostWorkspaceSizes* sizes);

typedef struct RoostOpEntry {
  uint32_t op;                  /* ROOST_OP_* */
  uint32_t flags;               /* ROOST_FLAG_* */
  const char* variant_id;       /* stable ASCII id, unique per op across loaded libraries; conflicts resolve by it */
  uint32_t params_struct_size;  /* sizeof the op's params struct this library was built against */
  uint32_t params_abi_minor;    /* the minor of that struct */
  RoostConstraints constraints;
  RoostRunFn run;               /* required */
  RoostPlanFn plan;             /* required when ROOST_FLAG_NEEDS_PLAN, else NULL */
  RoostWorkspaceFn workspace;   /* NULL means no workspace */
  uint64_t reserved0;
} RoostOpEntry;

/* Filled by roost_register. Entries are static data owned by the library and live until it unloads. */
typedef struct RoostOpTable {
  uint32_t struct_size;     /* set by the caller */
  uint32_t abi_minor;       /* set by the caller */
  uint32_t entry_size;      /* set by the library: sizeof(RoostOpEntry) it was built with */
  uint32_t entry_count;     /* set by the library */
  const RoostOpEntry* entries;
  const char* library_name; /* ASCII, for diagnostics and evidence */
  uint32_t library_abi_major;
  uint32_t library_abi_minor;
} RoostOpTable;

/* ---- M0 params ----------------------------------------------------------------------------------------------- */
/* Rows are tokens unless a field says otherwise. Strides are in elements. */

/* out = x * rsqrt(mean(x^2) + eps) * weight, per row of width dim. */
typedef struct RoostRmsNormParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan input;  /* [rows, dim] */
  RoostDeviceSpan weight; /* [dim] */
  RoostDeviceSpan output; /* [rows, dim] */
  uint32_t rows;
  uint32_t dim;
  uint32_t input_stride;  /* elements between rows; 0 means dim */
  uint32_t output_stride; /* 0 means dim */
  float eps;
  uint32_t reserved0;
} RoostRmsNormParams;

/* residual += x; x = RmsNorm(residual). Fused form of Residual then RmsNorm. */
typedef struct RoostAddRmsNormParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan input;    /* [rows, dim], overwritten with the normalised output */
  RoostDeviceSpan residual; /* [rows, dim], overwritten with input + residual */
  RoostDeviceSpan weight;   /* [dim] */
  uint32_t rows;
  uint32_t dim;
  float eps;
  uint32_t reserved0;
} RoostAddRmsNormParams;

/* Per-head RmsNorm of query and key heads, in place. */
typedef struct RoostQkNormParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan q;        /* [tokens, q_heads, head_dim] at q_stride */
  RoostDeviceSpan k;        /* [tokens, kv_heads, head_dim] at k_stride */
  RoostDeviceSpan q_weight; /* [head_dim] */
  RoostDeviceSpan k_weight; /* [head_dim] */
  uint32_t tokens;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t q_stride; /* elements between tokens; 0 means q_heads * head_dim */
  uint32_t k_stride; /* 0 means kv_heads * head_dim */
  float eps;
  uint32_t reserved0;
} RoostQkNormParams;

#define ROOST_ROPE_DEFAULT 0
#define ROOST_ROPE_LLAMA3 1

/* Rotary position embedding of q and k, in place, NeoX (half-split) style unless interleaved is set. */
typedef struct RoostRopeParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan q;         /* [tokens, q_heads, head_dim] */
  RoostDeviceSpan k;         /* [tokens, kv_heads, head_dim] */
  RoostDeviceSpan positions; /* [tokens] int32 */
  uint32_t tokens;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t rotary_dim; /* 0 means head_dim */
  uint32_t kind;       /* ROOST_ROPE_* */
  uint32_t interleaved;
  float theta;
  float llama3_factor;
  float llama3_low_freq_factor;
  float llama3_high_freq_factor;
  uint32_t llama3_original_max_position;
} RoostRopeParams;

#define ROOST_ATTENTION_DECODE 0
#define ROOST_ATTENTION_PREFILL 1

/* Paged attention over a state group's KV pools. Batch layout tables are device copies; the *_host fields are host
 * copies of the same tables read by plan() only (S2). */
typedef struct RoostPagedAttentionParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan q;              /* [tokens, q_heads, head_dim] */
  RoostDeviceSpan output;         /* [tokens, q_heads, head_dim] */
  RoostDeviceSpan lse;            /* optional [tokens, q_heads] f32; address NULL when unused */
  RoostDeviceSpan k_cache;        /* pool, layout as kv_layout */
  RoostDeviceSpan v_cache;
  RoostDeviceSpan q_indptr;       /* [batch + 1] int32, token offsets per sequence */
  RoostDeviceSpan kv_indptr;      /* [batch + 1] int32, offsets into kv_indices */
  RoostDeviceSpan kv_indices;     /* page ids */
  RoostDeviceSpan kv_last_page_len; /* [batch] int32, 1..page_size */
  const int32_t* q_indptr_host;   /* [batch + 1], plan() only */
  const int32_t* kv_indptr_host;  /* [batch + 1], plan() only */
  const int32_t* kv_last_page_len_host; /* [batch], plan() only */
  uint32_t mode;                  /* ROOST_ATTENTION_* */
  uint32_t batch;
  uint32_t tokens;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t page_size;
  uint32_t kv_layout;             /* ROOST_LAYOUT_PAGED_* */
  float sm_scale;                 /* 0 means 1 / sqrt(head_dim) */
  float logits_soft_cap;          /* 0 disables */
  int32_t window_left;            /* -1 disables the sliding window */
  uint32_t causal;                /* prefill only */
} RoostPagedAttentionParams;

/* out = act(x[:, :dim]) * x[:, dim:] for SiluAndMul and GeluAndMul. */
#define ROOST_GELU_EXACT 0
#define ROOST_GELU_TANH 1

typedef struct RoostActAndMulParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan input;  /* [rows, 2 * dim] */
  RoostDeviceSpan output; /* [rows, dim] */
  uint32_t rows;
  uint32_t dim;
  uint32_t gelu_approximation; /* ROOST_GELU_*, GeluAndMul only */
  uint32_t reserved0;
} RoostActAndMulParams;

/* d = alpha * a @ b (+ beta * c), a [m, k], b [k, n] or its transpose, fp32 accumulation. */
typedef struct RoostGemmParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan a;
  RoostDeviceSpan b;
  RoostDeviceSpan c;    /* optional; address NULL when beta is 0 */
  RoostDeviceSpan d;
  RoostDeviceSpan bias; /* optional [n] */
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t lda;
  uint64_t ldb;
  uint64_t ldd;
  uint32_t transpose_a;
  uint32_t transpose_b;
  float alpha;
  float beta;
  /* Upper bound on m that plan() chose the pinned algorithm for (S2), in rows, not an index. 0 means the call's own m.
   * plan() picks the algorithm for M = shape_bucket; run() accepts any m up to the planned bucket and returns
   * ROOST_STATUS_INVALID_ARGUMENT above it. Every call with the same bucket, n, k, types and transposes uses one
   * algorithm, which is what makes a row's result independent of the other rows (contract version 3). */
  uint32_t shape_bucket;
  uint32_t reserved0;
} RoostGemmParams;

/* output[i] = table[ids[i]]. */
typedef struct RoostEmbeddingParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan ids;    /* [tokens] int32 */
  RoostDeviceSpan table;  /* [vocab, dim] */
  RoostDeviceSpan output; /* [tokens, dim] */
  uint32_t tokens;
  uint32_t dim;
  uint32_t vocab;
  uint32_t reserved0;
} RoostEmbeddingParams;

/* logits = hidden[rows] @ weight^T for the selected rows (last token of each sequence, or all draft rows). */
typedef struct RoostLmHeadParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan hidden;  /* [tokens, dim] */
  RoostDeviceSpan rows;    /* [selected] int32 indices into hidden */
  RoostDeviceSpan weight;  /* [vocab, dim] */
  RoostDeviceSpan logits;  /* [selected, vocab_padded] */
  uint32_t selected;
  uint32_t dim;
  uint32_t vocab;
  uint32_t vocab_padded;
  float soft_cap; /* 0 disables final logit soft-capping */
  uint32_t shape_bucket; /* upper bound on selected, as RoostGemmParams.shape_bucket bounds m; 0 means selected */
} RoostLmHeadParams;

/* output = a + b (a may alias output). */
typedef struct RoostResidualParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan a;
  RoostDeviceSpan b;
  RoostDeviceSpan output;
  uint64_t elements;
} RoostResidualParams;

/* Write new K and V rows into the paged pools at slot_mapping (page * page_size + offset). */
typedef struct RoostCacheWriteParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan k;            /* [tokens, kv_heads, head_dim] */
  RoostDeviceSpan v;
  RoostDeviceSpan k_cache;
  RoostDeviceSpan v_cache;
  RoostDeviceSpan slot_mapping; /* [tokens] int64; negative skips the token */
  uint32_t tokens;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t page_size;
  uint32_t kv_layout;           /* ROOST_LAYOUT_PAGED_* */
  float k_scale;                /* quantised KV only; 1 otherwise */
  float v_scale;
  uint32_t reserved0;
} RoostCacheWriteParams;

/* Copy whole blocks within or between pools: copy-on-write forks, compaction and tier moves. */
typedef struct RoostCacheCopyParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan src_pool;
  RoostDeviceSpan dst_pool;  /* may equal src_pool */
  RoostDeviceSpan pairs;     /* [count, 2] int32: (src block, dst block) */
  uint64_t block_bytes;
  uint32_t count;
  uint32_t reserved0;
} RoostCacheCopyParams;

/* ids[r] = argmax(logits[r, :vocab]); ties go to the lowest index. */
typedef struct RoostArgmaxParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan logits; /* [rows, stride] */
  RoostDeviceSpan ids;    /* [rows] int32 */
  uint32_t rows;
  uint32_t vocab;
  uint32_t stride;        /* 0 means vocab */
  uint32_t reserved0;
} RoostArgmaxParams;

/* Temperature, top-k, top-p and min-p sampling with per-row parameters and seeds; optional token masks. */
typedef struct RoostSampleParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan logits;      /* [rows, stride] f32 */
  RoostDeviceSpan temperature; /* [rows] f32; 0 means greedy */
  RoostDeviceSpan top_k;       /* [rows] int32; 0 disables */
  RoostDeviceSpan top_p;       /* [rows] f32; 1 disables */
  RoostDeviceSpan min_p;       /* [rows] f32; 0 disables */
  RoostDeviceSpan seeds;       /* [rows] u64 */
  RoostDeviceSpan offsets;     /* [rows] u64, Philox offsets */
  RoostDeviceSpan mask;        /* optional [rows, ceil(vocab / 32)] u32 allowed-token bitmask; address NULL when unused */
  RoostDeviceSpan ids;         /* [rows] int32 output */
  RoostDeviceSpan logprobs;    /* optional [rows] f32 output of the chosen token */
  uint32_t rows;
  uint32_t vocab;
  uint32_t stride;
  uint32_t reserved0;
} RoostSampleParams;

/* ---- Hybrid (recurrent) params, ABI 1.1 and 1.2 -------------------------------------------------------------- */

/* Gated delta rule over packed sequences (contract C3 GatedDeltaRule; CpuOps.GatedDeltaRule is the exact definition).
 * Per sequence b (tokens cu_seqlens[b] to cu_seqlens[b + 1]) and value head h, with key head h / (v_heads / k_heads)
 * and state S [k_dim, v_dim] from slot state_slots[b]: q and k are l2-normalised when l2_norm_qk is set, q is scaled,
 * S = S * exp(g_t), S = S + k (outer) (beta_t * (v_t - S^T k)), o_t = S^T q; S is written back to the slot.
 *
 * The l2 norm of q and k happens here and nowhere else (contract version 3): with l2_norm_qk set, each key head's row of
 * k_dim elements of q and of k is multiplied by 1 / sqrt(sum(x^2) + 1e-6), eps fixed at 1e-6, before the query scale.
 * This is transformers' use_qk_l2norm_in_kernel=True, which Qwen3.5 always passes; a Qwen3.5 caller sets l2_norm_qk and
 * scale 0, and no separate normalisation op runs on the views. g and beta come from GatedDeltaRuleGates below.
 *
 * q, k and v are row-strided views: row t of q starts at element t * q_stride of the q span (likewise k and v), and the
 * heads of a row are contiguous. A stride of 0 means a contiguous tensor (k_heads * k_dim, or v_heads * v_dim). For
 * Qwen3.5 the three views are column slices of the causal conv1d output [tokens, 2 * k_heads * k_dim + v_heads * v_dim]:
 * q at column 0, k at column k_heads * k_dim, v at column 2 * k_heads * k_dim, all with that row width as stride (6144
 * for Qwen3.5-0.8B). Output, g and beta are contiguous. */
typedef struct RoostGatedDeltaRuleParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan q;           /* [tokens, k_heads, k_dim] at q_stride */
  RoostDeviceSpan k;           /* [tokens, k_heads, k_dim] at k_stride */
  RoostDeviceSpan v;           /* [tokens, v_heads, v_dim] at v_stride */
  RoostDeviceSpan g;           /* [tokens, v_heads] log decay */
  RoostDeviceSpan beta;        /* [tokens, v_heads] */
  RoostDeviceSpan output;      /* [tokens, v_heads, v_dim], contiguous */
  RoostDeviceSpan state;       /* recurrent pool [slots, v_heads, k_dim, v_dim] f32 (C4) */
  RoostDeviceSpan state_slots; /* [batch] int32 slot per sequence */
  RoostDeviceSpan cu_seqlens;  /* [batch + 1] int32 token offsets */
  const int32_t* cu_seqlens_host; /* [batch + 1] host copy for plan() (chunk tables, S3); NULL when no plan is needed */
  uint32_t batch;
  uint32_t tokens;
  uint32_t k_heads;
  uint32_t v_heads;            /* a multiple of k_heads */
  uint32_t k_dim;
  uint32_t v_dim;
  uint32_t q_stride;           /* elements between token rows of q; 0 means k_heads * k_dim */
  uint32_t k_stride;           /* 0 means k_heads * k_dim */
  uint32_t v_stride;           /* 0 means v_heads * v_dim */
  uint32_t use_initial_state;  /* nonzero: start from the slot's state; zero: start from zero */
  uint32_t l2_norm_qk;         /* nonzero: l2-normalise q and k inside the op */
  float scale;                 /* query scale; 0 means 1 / sqrt(k_dim) */
} RoostGatedDeltaRuleParams;

/* Gated DeltaNet gates from the raw a and b projections (contract C3 GatedDeltaRuleGates; ABI 1.2). Per token t and value
 * head h: g = -exp(a_log[h]) * softplus(a[t, h] + dt_bias[h]) and beta = sigmoid(b[t, h]), where softplus(x) is
 * log(1 + exp(x)), and x itself above 20 as in torch.nn.functional.softplus. This is transformers 5.20.1
 * Qwen3_5GatedDeltaNet.forward: g is computed in float32 and beta in the activation type, so a Qwen3.5 caller passes g as
 * an f32 span and beta in the activation type, which is what GatedDeltaRule reads. a and b are row-strided like q, k and
 * v above, so they may be column views of a fused projection. Each span's own dtype gives its element type. */
typedef struct RoostGatedDeltaRuleGatesParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan a;       /* [tokens, v_heads] at a_stride */
  RoostDeviceSpan b;       /* [tokens, v_heads] at b_stride */
  RoostDeviceSpan a_log;   /* [v_heads] */
  RoostDeviceSpan dt_bias; /* [v_heads] */
  RoostDeviceSpan g;       /* [tokens, v_heads] output, contiguous */
  RoostDeviceSpan beta;    /* [tokens, v_heads] output, contiguous */
  uint32_t tokens;
  uint32_t v_heads;
  uint32_t a_stride;       /* elements between token rows of a; 0 means v_heads */
  uint32_t b_stride;       /* 0 means v_heads */
} RoostGatedDeltaRuleGatesParams;

/* Gated RMS norm, norm before gate (contract C3 GatedRmsNorm; ABI 1.2): per row of dim elements,
 * out = x * rsqrt(mean(x^2) + eps) * weight * silu(gate), with silu(z) = z * sigmoid(z). The weight multiplies as stored
 * (it is not zero-centred, unlike the decoder RMS norms). Qwen3.5 applies it to the gated delta rule output per value
 * head: rows = tokens * v_heads, dim = v_dim, gate = in_proj_z's output, weight = linear_attn.norm.weight (float32 in the
 * checkpoint) and eps = rms_norm_eps (transformers 5.20.1 Qwen3_5RMSNormGated). input may alias output. */
typedef struct RoostGatedRmsNormParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan input;  /* [rows, dim] */
  RoostDeviceSpan gate;   /* [rows, dim] */
  RoostDeviceSpan weight; /* [dim], its own dtype (f32 for Qwen3.5) */
  RoostDeviceSpan output; /* [rows, dim] */
  uint32_t rows;
  uint32_t dim;
  float eps;
  uint32_t reserved0;
} RoostGatedRmsNormParams;

/* out = input * sigmoid(gate), elementwise over [rows, dim] (contract C3 SigmoidMul; ABI 1.2). Qwen3.5's full-attention
 * output gate: attn_out * sigmoid(gate) before o_proj, rows = tokens, dim = heads * head_dim. q_proj packs each head's
 * query rows and then its gate rows ([heads, 2 * head_dim] per token), so the gate is not a column slice of q_proj's
 * output; the weight loader splits the rows into a query and a gate projection. input may alias output. */
typedef struct RoostSigmoidMulParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan input;  /* [rows, dim] */
  RoostDeviceSpan gate;   /* [rows, dim] */
  RoostDeviceSpan output; /* [rows, dim] */
  uint32_t rows;
  uint32_t dim;
} RoostSigmoidMulParams;

/* Depthwise causal convolution over packed sequences, one filter per channel (contract C3 CausalConv1d; ABI 1.2, lifted
 * unchanged from R-034's provisional kernels/src/ops/hybrid_params.h). y_t = bias + sum_{j < width} weight[c, j] *
 * x_{t - (width - 1) + j}, x before the sequence start coming from the slot's conv_state (oldest first);
 * weight[c, width - 1] multiplies the current token. SiLU is applied last when silu is set. Afterwards the slot holds the
 * last width - 1 inputs of [state, sequence]. The state is always read: a fresh sequence needs a zeroed slot. input and
 * output must not overlap. A sequence of zero tokens leaves its slot untouched. */
typedef struct RoostCausalConv1dParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan input;       /* [tokens, channels] */
  RoostDeviceSpan weight;      /* [channels, width] */
  RoostDeviceSpan bias;        /* optional [channels]; address NULL when unused */
  RoostDeviceSpan output;      /* [tokens, channels] */
  RoostDeviceSpan conv_state;  /* pool [slots, channels, width - 1] */
  RoostDeviceSpan state_slots; /* [batch] int32 */
  RoostDeviceSpan cu_seqlens;  /* [batch + 1] int32 */
  uint32_t batch;
  uint32_t tokens;
  uint32_t channels;
  uint32_t width;
  uint32_t silu;               /* nonzero: apply SiLU */
  uint32_t reserved0;
} RoostCausalConv1dParams;

/* Copies slot pairs[i, 0] to slot pairs[i, 1] of pool as slot_bytes bytes, in pair order: a later pair sees an earlier
 * pair's write. A pair with equal slots is a no-op. Slot indices are not range-checked (they are device data). ABI 1.2. */
typedef struct RoostRecurrentStateCopyParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan pool;
  RoostDeviceSpan pairs; /* [count, 2] int32 (source slot, destination slot) */
  uint64_t slot_bytes;
  uint32_t count;
  uint32_t reserved0;
} RoostRecurrentStateCopyParams;

/* pairs[i, 0] is a pool slot and pairs[i, 1] a snapshots entry. restore == 0 copies slot to snapshot, nonzero copies
 * snapshot to slot. Byte copies in pair order. ABI 1.2. */
typedef struct RoostRecurrentStateSnapshotParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan pool;
  RoostDeviceSpan snapshots;
  RoostDeviceSpan pairs; /* [count, 2] int32 (slot, snapshot) */
  uint64_t slot_bytes;
  uint32_t count;
  uint32_t restore;
} RoostRecurrentStateSnapshotParams;

/* ---- Exports ------------------------------------------------------------------------------------------------- */

ROOST_API uint32_t roost_abi_version(void);
/* Message for the last failed call on this thread; valid until the next call on this thread. Never NULL. */
ROOST_API const char* roost_last_error(void);
ROOST_API RoostStatus roost_register(RoostOpTable* table);
ROOST_API RoostStatus roost_context_create(const RoostContextCreateParams* params, RoostContext** out);
ROOST_API RoostStatus roost_context_destroy(RoostContext* ctx);

/* Function pointer types for loaders that resolve the exports by name. */
typedef uint32_t (*RoostAbiVersionFn)(void);
typedef const char* (*RoostLastErrorFn)(void);
typedef RoostStatus (*RoostRegisterFn)(RoostOpTable* table);
typedef RoostStatus (*RoostContextCreateFn)(const RoostContextCreateParams* params, RoostContext** out);
typedef RoostStatus (*RoostContextDestroyFn)(RoostContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* ROOST_KERNELS_H_ */

/* S2 spike (R-006): illustrative C ABI for one paged attention op.
 *
 * This is NOT contract C1. It shows the shape that a torch-free FlashInfer
 * wrapper takes (plan on the host, run on a stream, workspace query, status
 * code plus last-error string) so that R-004 can decide the real header.
 * Fixed-width types only; no POSIX types.
 */
#ifndef ROOST_S2_ATTN_H_
#define ROOST_S2_ATTN_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define ROOST_S2_API __declspec(dllexport)
#else
#define ROOST_S2_API __attribute__((visibility("default")))
#endif

typedef enum RoostS2Status {
  ROOST_S2_OK = 0,
  ROOST_S2_INVALID_ARGUMENT = 1,
  ROOST_S2_UNSUPPORTED = 2,
  ROOST_S2_CUDA_ERROR = 3,
  ROOST_S2_INTERNAL = 4
} RoostS2Status;

/* Paged GQA decode, one query token per sequence.
 * bf16 q/k/v/o, head_dim 128, NHD pages: [num_pages, page_size, num_kv_heads, 128].
 * q and o: [batch, num_qo_heads, 128], contiguous. K and V are separate pools with equal strides. */
typedef struct RoostS2PagedDecodeParams {
  uint32_t struct_size; /* sizeof(RoostS2PagedDecodeParams) as the caller sees it */
  uint32_t abi_minor;   /* 0 */

  const void* q;
  void* o;
  float* lse; /* optional, [batch, num_qo_heads], may be NULL */
  const void* k_cache;
  const void* v_cache;

  /* Device copies, int32. */
  const int32_t* kv_indptr;        /* [batch + 1] */
  const int32_t* kv_indices;       /* [kv_indptr[batch]] page ids */
  const int32_t* kv_last_page_len; /* [batch], 1..page_size */
  /* Host copy of kv_indptr, needed by the planner only. */
  const int32_t* kv_indptr_host;

  uint32_t batch_size;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t page_size;
  uint32_t head_dim; /* 128 only */
  float sm_scale;    /* 0 means 1/sqrt(head_dim) */
  int32_t window_left; /* -1 disables the window */
  uint32_t reserved0;

  /* Caller-owned workspaces, sized by roost_s2_paged_decode_workspace(). */
  void* float_workspace;
  size_t float_workspace_bytes;
  void* int_workspace; /* device */
  size_t int_workspace_bytes;
  void* pinned_int_workspace; /* page-locked host, same size as int_workspace */
} RoostS2PagedDecodeParams;

/* Opaque plan record produced by plan() and consumed by run(). */
typedef struct RoostS2DecodePlan {
  int64_t words[10];
} RoostS2DecodePlan;

ROOST_S2_API uint32_t roost_s2_abi_version(void); /* major << 16 | minor */
ROOST_S2_API const char* roost_s2_last_error(void);

/* Workspace sizes for the batch described in params (host indptr required). */
ROOST_S2_API int32_t roost_s2_paged_decode_workspace(const RoostS2PagedDecodeParams* p,
                                                     size_t* float_bytes, size_t* int_bytes);
/* Host-side planning: fills the int workspace and copies it to the device on `stream`. */
ROOST_S2_API int32_t roost_s2_paged_decode_plan(const RoostS2PagedDecodeParams* p,
                                                RoostS2DecodePlan* plan, void* stream);
/* Launch only: no allocation, no host synchronisation. */
ROOST_S2_API int32_t roost_s2_paged_decode_run(const RoostS2PagedDecodeParams* p,
                                               const RoostS2DecodePlan* plan, void* stream);

#ifdef __cplusplus
}
#endif
#endif

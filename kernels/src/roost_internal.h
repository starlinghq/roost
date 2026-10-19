// Declarations shared by the translation units of one kernel library. Not part of the C1 ABI.
#ifndef ROOST_INTERNAL_H_
#define ROOST_INTERNAL_H_

#include <cstdint>

#include "roost_kernels.h"

#define ROOST_STRINGIZE_(x) #x
#define ROOST_STRINGIZE(x) ROOST_STRINGIZE_(x)

// Per-library, per-device state (C1).
struct RoostContext {
  int32_t device;
  uint32_t flags;
  void* cublaslt;  // cublasLtHandle_t of the device, created with the context (R-017); never touched by other ops
};

// Records the message roost_last_error returns on this thread.
void roost_set_error(const char* message);

extern "C" RoostStatus roost_residual_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws,
                                          void* stream);

// The M0 ops of R-017: each op's entry points and the entries that register them (m0_registry.cu).
extern "C" {
RoostStatus roost_rms_norm_run(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_qk_norm_run(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_rope_run(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_silu_and_mul_run(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_embedding_run(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_cache_write_run(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_cache_copy_run(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_paged_attention_workspace(RoostContext*, const void*, RoostWorkspaceSizes*);
RoostStatus roost_paged_attention_plan(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_paged_attention_run(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_gemm_workspace(RoostContext*, const void*, RoostWorkspaceSizes*);
RoostStatus roost_gemm_plan(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_gemm_run(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_lm_head_workspace(RoostContext*, const void*, RoostWorkspaceSizes*);
RoostStatus roost_lm_head_plan(RoostContext*, const void*, const RoostWorkspace*, void*);
RoostStatus roost_lm_head_run(RoostContext*, const void*, const RoostWorkspace*, void*);
}

// Creates and destroys the cuBLASLt handle held by a context (gemm.cu).
RoostStatus roost_gemm_context_init(RoostContext* ctx);
void roost_gemm_context_release(RoostContext* ctx);

// The registration entries of the M0 ops (m0_registry.cu).
const RoostOpEntry* roost_m0_entries(uint32_t* count);

#endif  // ROOST_INTERNAL_H_

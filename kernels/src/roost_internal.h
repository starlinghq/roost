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
  void* hybrid = nullptr;  // loaded GatedDeltaRule cubins (R-034), owned by gated_delta_rule.cu
};

// The registration entries of the hybrid ops (R-034, src/ops/hybrid_entries.cu).
const RoostOpEntry* roost_hybrid_entries(uint32_t* count);

// Load and release the hybrid ops' per-context state (R-034): the AOT cubins when the library was built with them.
RoostStatus roost_hybrid_context_init(RoostContext* ctx);
void roost_hybrid_context_release(RoostContext* ctx);

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

extern "C" RoostStatus roost_causal_conv1d_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws,
                                               void* stream);
extern "C" RoostStatus roost_recurrent_state_copy_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws,
                                                      void* stream);
extern "C" RoostStatus roost_recurrent_state_snapshot_run(RoostContext* ctx, const void* params,
                                                          const RoostWorkspace* ws, void* stream);

extern "C" RoostStatus roost_gdn_chunk_workspace(RoostContext* ctx, const void* params, RoostWorkspaceSizes* sizes);
extern "C" RoostStatus roost_gdn_chunk_plan(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream);
extern "C" RoostStatus roost_gdn_chunk_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream);
extern "C" RoostStatus roost_gdn_recurrent_workspace(RoostContext* ctx, const void* params, RoostWorkspaceSizes* sizes);
extern "C" RoostStatus roost_gdn_recurrent_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream);

#endif  // ROOST_INTERNAL_H_

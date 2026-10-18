// Spike S6 C ABI sketch for a Qwen3-style MoE layer, shaped like contract C1 (struct_size and abi_minor first,
// entry points take (const Params*, void* stream)). Throwaway evidence: not the final roost_kernels.h.
#ifndef ROOST_MOE_SPIKE_H
#define ROOST_MOE_SPIKE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define ROOST_MOE_SPIKE_ABI_MINOR 0u

typedef struct RoostMoeLayerParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  int32_t num_tokens, hidden, inter, num_experts, topk, norm_topk_prob;
  const void* x;           // bf16 [T, H]
  const void* w_router;    // bf16 [E, H]
  const void* w_gate_up;   // bf16 [E, 2I, H]; rows 0..I-1 gate, I..2I-1 up
  const void* w_down;      // bf16 [E, H, I]
  void* out;               // bf16 [T, H]
  void* workspace;         // device memory, size from roost_moe_layer_workspace_bytes
  size_t workspace_bytes;
  // Optional stage outputs for conformance (device pointers, may be null).
  void* dbg_logits;        // bf16 [T, E]
  void* dbg_topk_weights;  // f32  [T, K]
  void* dbg_topk_ids;      // i32  [T, K]
  void* dbg_gemm1;         // bf16 [T*K, 2I], rows grouped by expert
  void* dbg_act;           // bf16 [T*K, I]
  void* dbg_gemm2;         // bf16 [T*K, H]
  void* dbg_inv;           // i32  [T*K]: row of (token, k) in the grouped order
} RoostMoeLayerParams;

int roost_moe_layer_workspace_bytes(const RoostMoeLayerParams* p, size_t* bytes);
int roost_moe_layer(const RoostMoeLayerParams* p, void* stream);
const char* roost_moe_last_error(void);

#ifdef __cplusplus
}
#endif
#endif

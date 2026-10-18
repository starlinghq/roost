// S2 spike (R-006): compile-only probe. Instantiates FlashInfer's paged prefill for Qwen3 shapes
// (bf16, head_dim 128, causal, no rope/soft-cap) to measure build time and object size.
// No wrapper or test: prefill planning (PrefillPlan) is the remaining work for R-017.
#include <cuda_bf16.h>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>

namespace flashinfer {
using DT = __nv_bfloat16;
using P = BatchPrefillPagedParams<DT, DT, DT, int32_t>;
using V = DefaultAttention<false, false, false, false>;
#ifndef ROOST_TILES
#define ROOST_TILES 128
#endif
#define INST(TILE)                                                                              \
  template cudaError_t BatchPrefillWithPagedKVCacheDispatched<true, TILE, 128, 128,             \
                                                              PosEncodingMode::kNone, false,    \
                                                              MaskMode::kCausal, V, P>(         \
      P, DT*, float*, bool, cudaStream_t);
#if ROOST_TILES == 128
INST(128)
#elif ROOST_TILES == 3
INST(16) INST(64) INST(128)
#endif
}  // namespace flashinfer

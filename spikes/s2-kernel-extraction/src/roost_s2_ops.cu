// S2 spike (R-006): non-attention M0 ops without torch. FlashInfer headers are used unmodified;
// the vLLM rms_norm kernel comes from the patched csrc/layernorm_kernels.cu.
#include "roost_s2_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <string>

#include <flashinfer/activation.cuh>
#include <flashinfer/norm.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

#include "layernorm_kernels.cu"  // vLLM, patched: host launchers compiled out (patches/)

using bf16 = __nv_bfloat16;

namespace {
int32_t Map(cudaError_t e) { return e == cudaSuccess ? ROOST_S2_OK : ROOST_S2_CUDA_ERROR; }
__device__ __forceinline__ float silu(const float& v) { return v / (1.0f + __expf(-v)); }
}  // namespace

extern "C" {

ROOST_S2_API int32_t roost_s2_rms_norm_flashinfer(const void* x, const void* w, void* out,
                                                  uint32_t rows, uint32_t d, uint32_t stride_x,
                                                  float eps, void* stream) {
  return Map(flashinfer::norm::RMSNorm(static_cast<bf16*>(const_cast<void*>(x)),
                                       static_cast<bf16*>(const_cast<void*>(w)),
                                       static_cast<bf16*>(out), rows, d, stride_x, d, eps, false,
                                       static_cast<cudaStream_t>(stream)));
}

ROOST_S2_API int32_t roost_s2_rms_norm_vllm(const void* x, const void* w, void* out, uint32_t rows,
                                            uint32_t d, uint32_t stride_x, float eps, void* stream) {
  if (d % 8 != 0 || d > 8 * 1024) return ROOST_S2_UNSUPPORTED;
  const uint32_t block = d / 8 < 1024 ? d / 8 : 1024;
  vllm::rms_norm_kernel<bf16, 8, 2, true><<<rows, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<bf16*>(out), static_cast<const bf16*>(x), stride_x, 0, 0, 0, 0,
      static_cast<const bf16*>(w), 0, eps, static_cast<int>(rows), static_cast<int>(d));
  return Map(cudaGetLastError());
}

ROOST_S2_API int32_t roost_s2_fused_add_rms_norm(void* x, void* residual, const void* w,
                                                 uint32_t rows, uint32_t d, float eps,
                                                 void* stream) {
  return Map(flashinfer::norm::FusedAddRMSNorm(
      static_cast<bf16*>(x), static_cast<bf16*>(residual), static_cast<bf16*>(const_cast<void*>(w)),
      rows, d, d, d, eps, false, static_cast<cudaStream_t>(stream)));
}

ROOST_S2_API int32_t roost_s2_qk_rms_norm(const void* x, const void* w, void* out, uint32_t tokens,
                                          uint32_t heads, uint32_t d, float eps, void* stream) {
  return Map(flashinfer::norm::QKRMSNorm(
      static_cast<bf16*>(const_cast<void*>(x)), static_cast<bf16*>(const_cast<void*>(w)),
      static_cast<bf16*>(out), tokens, heads, d, heads * d, d, heads * d, d, eps, false,
      static_cast<cudaStream_t>(stream)));
}

ROOST_S2_API int32_t roost_s2_silu_and_mul(const void* in, void* out, uint32_t tokens, uint32_t d,
                                           void* stream) {
  // The launcher is not in the headers (FlashInfer generates it from a Python template), so it is
  // written here: 8 elements per 16-byte vector.
  if (d % 8 != 0) return ROOST_S2_UNSUPPORTED;
  const uint32_t block = d / 8 < 256 ? d / 8 : 256;
  flashinfer::activation::act_and_mul_kernel<bf16, silu, 8>
      <<<tokens, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<bf16*>(out), static_cast<const bf16*>(in), static_cast<int>(d));
  return Map(cudaGetLastError());
}

ROOST_S2_API int32_t roost_s2_rope_neox(void* q, void* k, const int32_t* pos, uint32_t nnz,
                                        uint32_t qh, uint32_t kvh, uint32_t hd, float theta,
                                        void* stream) {
  return Map(flashinfer::BatchQKApplyRotaryPosIds(
      static_cast<bf16*>(q), static_cast<bf16*>(k), static_cast<bf16*>(q), static_cast<bf16*>(k),
      const_cast<int32_t*>(pos), nnz, qh, kvh, hd, hd, size_t(qh) * hd, hd, size_t(kvh) * hd, hd,
      size_t(qh) * hd, hd, size_t(kvh) * hd, hd, /*interleave=*/false, 1.0f, theta,
      static_cast<cudaStream_t>(stream)));
}

ROOST_S2_API int32_t roost_s2_append_kv(void* k_pool, void* v_pool, const int32_t* kv_indices,
                                        const int32_t* kv_indptr, const int32_t* last_page_len,
                                        const void* k, const void* v, const int32_t* batch_idx,
                                        const int32_t* pos, uint32_t batch, uint32_t nnz,
                                        uint32_t kvh, uint32_t hd, uint32_t page, void* stream) {
  flashinfer::paged_kv_t<bf16, int32_t> kv(
      kvh, page, hd, batch, flashinfer::QKVLayout::kNHD, static_cast<bf16*>(k_pool),
      static_cast<bf16*>(v_pool), const_cast<int32_t*>(kv_indices), const_cast<int32_t*>(kv_indptr),
      const_cast<int32_t*>(last_page_len));
  return Map(flashinfer::AppendPagedKVCache(
      kv, static_cast<bf16*>(const_cast<void*>(k)), static_cast<bf16*>(const_cast<void*>(v)),
      const_cast<int32_t*>(batch_idx), const_cast<int32_t*>(pos), nnz, size_t(kvh) * hd, hd,
      size_t(kvh) * hd, hd, static_cast<cudaStream_t>(stream)));
}

}  // extern "C"

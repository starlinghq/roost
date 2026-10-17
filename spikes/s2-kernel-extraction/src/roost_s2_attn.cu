// S2 spike (R-006): FlashInfer batch paged decode behind a C ABI, no torch and no TVM-FFI.
// Replaces the framework glue of FlashInfer csrc/batch_decode.cu (Plan, WorkspaceSize, Run)
// with the code below. The FlashInfer headers are used unmodified.
#include "roost_s2_attn.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <exception>
#include <string>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

using namespace flashinfer;

namespace roost_s2 {

using DT = nv_bfloat16;
using Params = BatchDecodeParams<DT, DT, DT, int32_t>;
using Variant = DefaultAttention</*custom_mask=*/false, /*sliding_window=*/false,
                                 /*soft_cap=*/false, /*alibi=*/false>;
constexpr uint32_t kHeadDim = 128;
constexpr PosEncodingMode kPos = PosEncodingMode::kNone;

}  // namespace roost_s2

namespace {

using namespace roost_s2;
thread_local std::string g_err;

int32_t Fail(int32_t code, const std::string& msg) {
  g_err = msg;
  return code;
}

int32_t Validate(const RoostS2PagedDecodeParams* p, bool need_dev) {
  if (p == nullptr) return Fail(ROOST_S2_INVALID_ARGUMENT, "params is null");
  if (p->struct_size < sizeof(RoostS2PagedDecodeParams))
    return Fail(ROOST_S2_INVALID_ARGUMENT, "struct_size too small");
  if (p->head_dim != kHeadDim) return Fail(ROOST_S2_UNSUPPORTED, "only head_dim 128 is built");
  if (p->num_kv_heads == 0 || p->num_qo_heads % p->num_kv_heads != 0)
    return Fail(ROOST_S2_INVALID_ARGUMENT, "num_qo_heads must be a multiple of num_kv_heads");
  if (p->batch_size == 0 || p->page_size == 0)
    return Fail(ROOST_S2_INVALID_ARGUMENT, "batch_size and page_size must be non-zero");
  if (p->kv_indptr_host == nullptr)
    return Fail(ROOST_S2_INVALID_ARGUMENT, "kv_indptr_host is required");
  if (need_dev && (p->q == nullptr || p->o == nullptr || p->k_cache == nullptr ||
                   p->v_cache == nullptr || p->kv_indptr == nullptr || p->kv_indices == nullptr ||
                   p->kv_last_page_len == nullptr))
    return Fail(ROOST_S2_INVALID_ARGUMENT, "null device pointer");
  return ROOST_S2_OK;
}

// Group size 1..8 is what FlashInfer dispatches; Qwen3 uses 2 (1.7B) and 4 (0.6B).
template <typename F>
cudaError_t DispatchGroup(uint32_t group, F&& f) {
  switch (group) {
    case 2:
      return f(std::integral_constant<uint32_t, 2>{});
    case 4:
      return f(std::integral_constant<uint32_t, 4>{});
    default:
      return cudaErrorNotSupported;
  }
}

}  // namespace

extern "C" {

ROOST_S2_API uint32_t roost_s2_abi_version(void) { return (1u << 16) | 0u; }
ROOST_S2_API const char* roost_s2_last_error(void) { return g_err.c_str(); }

ROOST_S2_API int32_t roost_s2_paged_decode_workspace(const RoostS2PagedDecodeParams* p,
                                                     size_t* float_bytes, size_t* int_bytes) {
  try {
    if (int32_t s = Validate(p, false)) return s;
    size_t fb = 0, ib = 0;
    auto* indptr = const_cast<int32_t*>(p->kv_indptr_host);
    cudaError_t st = DispatchGroup(p->num_qo_heads / p->num_kv_heads, [&](auto g) {
      constexpr uint32_t G = decltype(g)::value;
      return DecodePlanWorkspaceSize<kHeadDim, kPos, Variant, Params>(
          fb, ib, indptr, p->batch_size, p->num_qo_heads, p->page_size, false, nullptr,
          BatchDecodeWithPagedKVCacheWorkEstimationDispatched<G, kHeadDim, kPos, Variant, Params>);
    });
    if (st != cudaSuccess) return Fail(ROOST_S2_CUDA_ERROR, cudaGetErrorString(st));
    *float_bytes = fb;
    *int_bytes = ib;
    return ROOST_S2_OK;
  } catch (const std::exception& e) {
    return Fail(ROOST_S2_INTERNAL, e.what());
  }
}

ROOST_S2_API int32_t roost_s2_paged_decode_plan(const RoostS2PagedDecodeParams* p,
                                                RoostS2DecodePlan* plan, void* stream) {
  try {
    if (int32_t s = Validate(p, false)) return s;
    if (plan == nullptr) return Fail(ROOST_S2_INVALID_ARGUMENT, "plan is null");
    DecodePlanInfo info;
    auto* indptr = const_cast<int32_t*>(p->kv_indptr_host);
    cudaError_t st = DispatchGroup(p->num_qo_heads / p->num_kv_heads, [&](auto g) {
      constexpr uint32_t G = decltype(g)::value;
      return DecodePlan<kHeadDim, kPos, Variant, Params>(
          p->float_workspace, p->float_workspace_bytes, p->int_workspace,
          p->pinned_int_workspace, p->int_workspace_bytes, info, indptr, p->batch_size,
          p->num_qo_heads, p->page_size, false, static_cast<cudaStream_t>(stream),
          BatchDecodeWithPagedKVCacheWorkEstimationDispatched<G, kHeadDim, kPos, Variant, Params>);
    });
    if (st != cudaSuccess) return Fail(ROOST_S2_CUDA_ERROR, cudaGetErrorString(st));
    auto v = info.ToVector();
    for (size_t i = 0; i < 10; ++i) plan->words[i] = v[i];
    return ROOST_S2_OK;
  } catch (const std::exception& e) {
    return Fail(ROOST_S2_INTERNAL, e.what());
  }
}

ROOST_S2_API int32_t roost_s2_paged_decode_run(const RoostS2PagedDecodeParams* p,
                                               const RoostS2DecodePlan* plan, void* stream) {
  try {
    if (int32_t s = Validate(p, true)) return s;
    if (plan == nullptr) return Fail(ROOST_S2_INVALID_ARGUMENT, "plan is null");
    DecodePlanInfo info;
    info.FromVector(std::vector<int64_t>(plan->words, plan->words + 10));

    auto* kc = static_cast<DT*>(const_cast<void*>(p->k_cache));
    auto* vc = static_cast<DT*>(const_cast<void*>(p->v_cache));
    paged_kv_t<DT, int32_t> kv(p->num_kv_heads, p->page_size, p->head_dim, p->batch_size,
                               QKVLayout::kNHD, kc, vc,
                               const_cast<int32_t*>(p->kv_indices),
                               const_cast<int32_t*>(p->kv_indptr),
                               const_cast<int32_t*>(p->kv_last_page_len));
    Params params;
    params.q = static_cast<DT*>(const_cast<void*>(p->q));
    params.paged_kv = kv;
    params.o = static_cast<DT*>(p->o);
    params.lse = p->lse;
    params.num_qo_heads = p->num_qo_heads;
    params.q_stride_n = static_cast<int32_t>(p->num_qo_heads * p->head_dim);
    params.q_stride_h = static_cast<int32_t>(p->head_dim);
    params.window_left = p->window_left;
    params.sm_scale = p->sm_scale != 0.f ? p->sm_scale : 1.f / sqrtf(float(p->head_dim));
    params.padded_batch_size = static_cast<uint32_t>(info.padded_batch_size);
    params.request_indices =
        GetPtrFromBaseOffset<int32_t>(p->int_workspace, info.request_indices_offset);
    params.kv_tile_indices =
        GetPtrFromBaseOffset<int32_t>(p->int_workspace, info.kv_tile_indices_offset);
    params.o_indptr = GetPtrFromBaseOffset<int32_t>(p->int_workspace, info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        GetPtrFromBaseOffset<int32_t>(p->int_workspace, info.kv_chunk_size_ptr_offset);
    DT* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (info.split_kv) {
      tmp_v = GetPtrFromBaseOffset<DT>(p->float_workspace, info.v_offset);
      tmp_s = GetPtrFromBaseOffset<float>(p->float_workspace, info.s_offset);
    }
    cudaError_t st = BatchDecodeWithPagedKVCacheDispatched<kHeadDim, kPos, Variant, Params>(
        params, tmp_v, tmp_s, /*enable_pdl=*/false, static_cast<cudaStream_t>(stream));
    if (st != cudaSuccess) return Fail(ROOST_S2_CUDA_ERROR, cudaGetErrorString(st));
    return ROOST_S2_OK;
  } catch (const std::exception& e) {
    return Fail(ROOST_S2_INTERNAL, e.what());
  }
}

}  // extern "C"

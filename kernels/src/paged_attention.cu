// PagedAttention (ROOST_OP_PAGED_ATTENTION), R-017: FlashInfer's batch decode and batch prefill kernels over a paged KV
// cache, behind the C1 ABI. The FlashInfer headers are used unmodified (pinned in kernels/manifest.toml); this file
// replaces the framework glue of csrc/batch_decode.cu and csrc/batch_prefill_paged.cuh (plan, workspace sizes, run).
//
// Specialisation (Qwen3.5-0.8B full-attention layers, S2 and decision 15): head_dim 256, 8 query heads over 2 KV heads
// (group size 4), bf16 Q, K, V and output, int32 page tables, NHD or HND pools, any page size. Sliding window and logit
// soft capping are not instantiated (R-113 adds the window) and are refused with ROOST_STATUS_UNSUPPORTED.
//
// Planning is a host step (S2 finding 1): plan() reads the host copies of the batch tables, writes FlashInfer's int
// workspace through the pinned staging buffer and records the scheduler's decisions as raw bytes in ws->plan. run() reads
// that record with a memcpy, builds the kernel parameters on the stack and launches: no allocation of any kind, no
// synchronisation.
//
// Device workspace layout: [int workspace | float workspace], the float part starting at the next 256-byte boundary.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <type_traits>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

#include "op_common.h"

namespace roost_attn {

using namespace flashinfer;
using roost_ops::AlignUp;
using roost_ops::CheckHeader;
using roost_ops::CheckSpan;
using roost_ops::Fail;
using roost_ops::FromCuda;
using roost_ops::Guarded;

using DT = nv_bfloat16;
using IdT = int32_t;
using DecodeParams = BatchDecodeParams<DT, DT, DT, IdT>;
using PrefillParams = BatchPrefillPagedParams<DT, DT, DT, IdT>;
using Variant = DefaultAttention</*custom_mask=*/false, /*sliding_window=*/false, /*soft_cap=*/false,
                                 /*alibi=*/false>;

constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kGroup = 4;
constexpr PosEncodingMode kPos = PosEncodingMode::kNone;
constexpr uint32_t kWorkspaceAlignment = 256;
constexpr uint32_t kPlanMagic = 0x52415031;  // "RAP1"

// What plan() leaves for run(). Trivially copyable: run() reads it with memcpy. The shape fields let run() refuse a plan
// made for another batch.
struct PlanRecord {
  uint32_t magic;
  uint32_t mode;
  uint32_t batch;
  uint32_t tokens;
  uint32_t page_size;
  uint32_t kv_layout;
  uint64_t pages;  // kv_indptr_host[batch]: how many entries run() may read from kv_indices
  uint64_t int_bytes;
  uint64_t float_bytes;
  uint64_t float_offset;
  DecodePlanInfo decode;
  PrefillPlanInfo prefill;
};
static_assert(std::is_trivially_copyable_v<PlanRecord>, "the plan record is copied as bytes");

struct Sizes {
  uint64_t int_bytes;
  uint64_t float_bytes;
};

RoostStatus Validate(const RoostPagedAttentionParams& p, bool for_run) {
  constexpr char kOp[] = "PagedAttention";
  if (p.mode != ROOST_ATTENTION_DECODE && p.mode != ROOST_ATTENTION_PREFILL) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: mode must be DECODE or PREFILL");
  }
  if (p.head_dim != kHeadDim) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "PagedAttention: only head_dim 256 is built");
  }
  if (p.kv_heads == 0 || p.q_heads != p.kv_heads * kGroup) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "PagedAttention: only 4 query heads per KV head are built");
  }
  if (p.window_left != -1) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "PagedAttention: sliding window is not built (window_left must be -1)");
  }
  if (p.logits_soft_cap != 0.0f) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "PagedAttention: logit soft capping is not built");
  }
  if (p.kv_layout != ROOST_LAYOUT_PAGED_NHD && p.kv_layout != ROOST_LAYOUT_PAGED_HND) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: kv_layout must be PAGED_NHD or PAGED_HND");
  }
  if (p.batch == 0 || p.page_size == 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: batch and page_size must be non-zero");
  }
  if (p.mode == ROOST_ATTENTION_DECODE && p.tokens != p.batch) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: decode takes exactly one query token per sequence");
  }
  if (!for_run) {
    // The host copies of the batch tables are read by the workspace query and plan() only (C1); run() never touches them, so a
    // replayed or captured run may be given stale or NULL host pointers.
    if (p.kv_indptr_host == nullptr || p.q_indptr_host == nullptr) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: q_indptr_host and kv_indptr_host are required");
    }
    if (p.q_indptr_host[0] != 0 || p.q_indptr_host[p.batch] != static_cast<int32_t>(p.tokens)) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: q_indptr_host must start at 0 and end at tokens");
    }
    if (p.kv_indptr_host[0] != 0 || p.kv_indptr_host[p.batch] < 0) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: kv_indptr_host must start at 0 and not decrease");
    }
    return ROOST_STATUS_OK;
  }
  const uint64_t q_elems = static_cast<uint64_t>(p.tokens) * p.q_heads * p.head_dim;
  if (auto s = CheckSpan(p.q, ROOST_DTYPE_BF16, q_elems, kOp, "q")) return s;
  if (auto s = CheckSpan(p.output, ROOST_DTYPE_BF16, q_elems, kOp, "output")) return s;
  if (auto s = CheckSpan(p.lse, ROOST_DTYPE_F32, static_cast<uint64_t>(p.tokens) * p.q_heads, kOp, "lse", true)) {
    return s;
  }
  if (auto s = CheckSpan(p.k_cache, ROOST_DTYPE_BF16, 1, kOp, "k_cache")) return s;
  if (auto s = CheckSpan(p.v_cache, ROOST_DTYPE_BF16, 1, kOp, "v_cache")) return s;
  if (auto s = CheckSpan(p.q_indptr, ROOST_DTYPE_I32, p.batch + 1ull, kOp, "q_indptr")) return s;
  if (auto s = CheckSpan(p.kv_indptr, ROOST_DTYPE_I32, p.batch + 1ull, kOp, "kv_indptr")) return s;
  if (auto s = CheckSpan(p.kv_last_page_len, ROOST_DTYPE_I32, p.batch, kOp, "kv_last_page_len")) return s;
  return ROOST_STATUS_OK;
}

QKVLayout Layout(uint32_t kv_layout) {
  return kv_layout == ROOST_LAYOUT_PAGED_HND ? QKVLayout::kHND : QKVLayout::kNHD;
}

cudaStream_t ToStream(void* s) { return static_cast<cudaStream_t>(s); }

// Prefill tile heights FlashInfer's planner may choose for head_dim 256 (it never returns 128 here).
template <typename F>
cudaError_t DispatchCtaTileQ(int64_t cta_tile_q, F&& f) {
  switch (cta_tile_q) {
    case 16: return f(std::integral_constant<uint32_t, 16>{});
    case 32: return f(std::integral_constant<uint32_t, 32>{});
    case 64: return f(std::integral_constant<uint32_t, 64>{});
    default: return cudaErrorNotSupported;
  }
}

template <MaskMode kMask>
cudaError_t PrefillDispatch(int64_t cta_tile_q, PrefillParams params, DT* tmp_v, float* tmp_s, cudaStream_t stream) {
  return DispatchCtaTileQ(cta_tile_q, [&](auto tile) {
    constexpr uint32_t kTile = decltype(tile)::value;
    return BatchPrefillWithPagedKVCacheDispatched</*SAME_KV_STRIDES=*/true, kTile, kHeadDim, kHeadDim, kPos,
                                                  /*use_fp16_qk_reduction=*/false, kMask, Variant, PrefillParams>(
        params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
  });
}

RoostStatus QuerySizes(const RoostPagedAttentionParams& p, Sizes* out) {
  size_t fb = 0, ib = 0;
  cudaError_t st = cudaSuccess;
  if (p.mode == ROOST_ATTENTION_DECODE) {
    st = DecodePlanWorkspaceSize<kHeadDim, kPos, Variant, DecodeParams>(
        fb, ib, const_cast<IdT*>(p.kv_indptr_host), p.batch, p.q_heads, p.page_size, false, nullptr,
        BatchDecodeWithPagedKVCacheWorkEstimationDispatched<kGroup, kHeadDim, kPos, Variant, DecodeParams>);
  } else {
    st = PrefillPlanWorkspaceSize<IdT>(fb, ib, const_cast<IdT*>(p.q_indptr_host), const_cast<IdT*>(p.kv_indptr_host),
                                       p.tokens, p.batch, p.q_heads, p.kv_heads, kHeadDim, kHeadDim, p.page_size,
                                       false, sizeof(DT), p.window_left, /*fixed_split_size=*/-1,
                                       /*disable_split_kv=*/false, /*num_colocated_ctas=*/0, /*uniform_q_len=*/0,
                                       nullptr, sizeof(DT));
  }
  if (st != cudaSuccess) {
    return Fail(ROOST_STATUS_CUDA_ERROR, cudaGetErrorString(st));
  }
  out->int_bytes = ib;
  out->float_bytes = fb;
  return ROOST_STATUS_OK;
}

}  // namespace roost_attn

extern "C" RoostStatus roost_paged_attention_workspace(RoostContext*, const void* params, RoostWorkspaceSizes* sizes) {
  using namespace roost_attn;
  if (auto s = CheckHeader(params, sizeof(RoostPagedAttentionParams), "PagedAttention")) return s;
  if (sizes == nullptr) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: sizes is NULL");
  const auto& p = *static_cast<const RoostPagedAttentionParams*>(params);
  return Guarded("PagedAttention", [&]() -> RoostStatus {
    if (auto s = Validate(p, false)) return s;
    Sizes sz{};
    if (auto s = QuerySizes(p, &sz)) return s;
    sizes->device_bytes = AlignUp(sz.int_bytes, kWorkspaceAlignment) + sz.float_bytes;
    sizes->pinned_host_bytes = sz.int_bytes;
    sizes->plan_bytes = sizeof(PlanRecord);
    sizes->device_alignment = kWorkspaceAlignment;
    return ROOST_STATUS_OK;
  });
}

extern "C" RoostStatus roost_paged_attention_plan(RoostContext*, const void* params, const RoostWorkspace* ws,
                                                  void* stream) {
  using namespace roost_attn;
  if (auto s = CheckHeader(params, sizeof(RoostPagedAttentionParams), "PagedAttention")) return s;
  const auto& p = *static_cast<const RoostPagedAttentionParams*>(params);
  return Guarded("PagedAttention", [&]() -> RoostStatus {
    if (auto s = Validate(p, false)) return s;
    if (ws == nullptr || ws->plan == nullptr) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: plan needs a workspace with a plan record");
    }
    Sizes sz{};
    if (auto s = QuerySizes(p, &sz)) return s;
    const uint64_t float_offset = AlignUp(sz.int_bytes, kWorkspaceAlignment);
    if (ws->plan_bytes < sizeof(PlanRecord) || ws->device_bytes < float_offset + sz.float_bytes ||
        ws->pinned_host_bytes < sz.int_bytes || (sz.int_bytes > 0 && ws->pinned_host == nullptr) ||
        ws->device == nullptr) {
      return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "PagedAttention: workspace is smaller than the query reported");
    }
    PlanRecord rec{};
    rec.magic = kPlanMagic;
    rec.mode = p.mode;
    rec.batch = p.batch;
    rec.tokens = p.tokens;
    rec.page_size = p.page_size;
    rec.kv_layout = p.kv_layout;
    rec.pages = static_cast<uint64_t>(p.kv_indptr_host[p.batch]);
    rec.int_bytes = sz.int_bytes;
    rec.float_bytes = sz.float_bytes;
    rec.float_offset = float_offset;
    void* int_buffer = ws->device;
    void* float_buffer = static_cast<char*>(ws->device) + float_offset;
    cudaError_t st = cudaSuccess;
    if (p.mode == ROOST_ATTENTION_DECODE) {
      st = DecodePlan<kHeadDim, kPos, Variant, DecodeParams>(
          float_buffer, sz.float_bytes, int_buffer, ws->pinned_host, sz.int_bytes, rec.decode,
          const_cast<IdT*>(p.kv_indptr_host), p.batch, p.q_heads, p.page_size, false, ToStream(stream),
          BatchDecodeWithPagedKVCacheWorkEstimationDispatched<kGroup, kHeadDim, kPos, Variant, DecodeParams>);
    } else {
      st = PrefillPlan<IdT>(float_buffer, sz.float_bytes, int_buffer, ws->pinned_host, sz.int_bytes, rec.prefill,
                            const_cast<IdT*>(p.q_indptr_host), const_cast<IdT*>(p.kv_indptr_host), p.tokens, p.batch,
                            p.q_heads, p.kv_heads, kHeadDim, kHeadDim, p.page_size, false, sizeof(DT), p.window_left,
                            /*fixed_split_size=*/-1, /*disable_split_kv=*/false, /*num_colocated_ctas=*/0,
                            /*uniform_q_len=*/0, ToStream(stream), sizeof(DT));
    }
    if (st != cudaSuccess) {
      return Fail(ROOST_STATUS_CUDA_ERROR, cudaGetErrorString(st));
    }
    std::memcpy(ws->plan, &rec, sizeof(rec));
    return ROOST_STATUS_OK;
  });
}

extern "C" RoostStatus roost_paged_attention_run(RoostContext*, const void* params, const RoostWorkspace* ws,
                                                 void* stream) {
  using namespace roost_attn;
  if (auto s = CheckHeader(params, sizeof(RoostPagedAttentionParams), "PagedAttention")) return s;
  const auto& p = *static_cast<const RoostPagedAttentionParams*>(params);
  return Guarded("PagedAttention", [&]() -> RoostStatus {
    if (auto s = Validate(p, true)) return s;
    if (ws == nullptr || ws->plan == nullptr || ws->device == nullptr) {
      return ROOST_STATUS_NOT_PLANNED;
    }
    if (ws->plan_bytes < sizeof(PlanRecord)) {
      return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "PagedAttention: plan record is too small");
    }
    PlanRecord rec;
    std::memcpy(&rec, ws->plan, sizeof(rec));
    if (rec.magic != kPlanMagic) {
      roost_set_error("PagedAttention: no plan was made in this workspace");
      return ROOST_STATUS_NOT_PLANNED;
    }
    if (rec.mode != p.mode || rec.batch != p.batch || rec.tokens != p.tokens || rec.page_size != p.page_size ||
        rec.kv_layout != p.kv_layout) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "PagedAttention: the plan was made for another batch shape");
    }
    if (auto s = CheckSpan(p.kv_indices, ROOST_DTYPE_I32, rec.pages, "PagedAttention", "kv_indices")) return s;
    if (ws->device_bytes < rec.float_offset + rec.float_bytes) {
      return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "PagedAttention: device workspace is smaller than the plan");
    }
    auto* int_buffer = ws->device;
    auto* float_buffer = static_cast<char*>(ws->device) + rec.float_offset;
    const float sm_scale = p.sm_scale != 0.0f ? p.sm_scale : 1.0f / sqrtf(static_cast<float>(p.head_dim));
    auto* kc = static_cast<DT*>(p.k_cache.address);
    auto* vc = static_cast<DT*>(p.v_cache.address);
    paged_kv_t<DT, IdT> kv(p.kv_heads, p.page_size, p.head_dim, p.batch, Layout(p.kv_layout), kc, vc,
                           static_cast<IdT*>(p.kv_indices.address), static_cast<IdT*>(p.kv_indptr.address),
                           static_cast<IdT*>(p.kv_last_page_len.address));
    auto* lse = static_cast<float*>(p.lse.address);
    cudaError_t st = cudaSuccess;
    if (p.mode == ROOST_ATTENTION_DECODE) {
      DecodeParams dp;
      dp.q = static_cast<DT*>(p.q.address);
      dp.paged_kv = kv;
      dp.o = static_cast<DT*>(p.output.address);
      dp.lse = lse;
      dp.num_qo_heads = p.q_heads;
      dp.q_stride_n = static_cast<IdT>(p.q_heads * p.head_dim);
      dp.q_stride_h = static_cast<IdT>(p.head_dim);
      dp.window_left = -1;
      dp.sm_scale = sm_scale;
      const DecodePlanInfo& info = rec.decode;
      dp.padded_batch_size = static_cast<uint32_t>(info.padded_batch_size);
      dp.request_indices = GetPtrFromBaseOffset<IdT>(int_buffer, info.request_indices_offset);
      dp.kv_tile_indices = GetPtrFromBaseOffset<IdT>(int_buffer, info.kv_tile_indices_offset);
      dp.o_indptr = GetPtrFromBaseOffset<IdT>(int_buffer, info.o_indptr_offset);
      dp.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdT>(int_buffer, info.kv_chunk_size_ptr_offset);
      dp.partition_kv = info.split_kv;
      DT* tmp_v = nullptr;
      float* tmp_s = nullptr;
      if (info.split_kv) {
        tmp_v = GetPtrFromBaseOffset<DT>(float_buffer, info.v_offset);
        tmp_s = GetPtrFromBaseOffset<float>(float_buffer, info.s_offset);
      }
      st = BatchDecodeWithPagedKVCacheDispatched<kHeadDim, kPos, Variant, DecodeParams>(dp, tmp_v, tmp_s, false,
                                                                                         ToStream(stream));
    } else {
      PrefillParams pp;
      pp.q = static_cast<DT*>(p.q.address);
      pp.paged_kv = kv;
      pp.q_indptr = static_cast<IdT*>(p.q_indptr.address);
      pp.o = static_cast<DT*>(p.output.address);
      pp.lse = lse;
      pp.num_qo_heads = p.q_heads;
      pp.group_size = uint_fastdiv(p.q_heads / p.kv_heads);
      pp.q_stride_n = static_cast<IdT>(p.q_heads * p.head_dim);
      pp.q_stride_h = static_cast<IdT>(p.head_dim);
      pp.window_left = -1;
      pp.sm_scale = sm_scale;
      const PrefillPlanInfo& info = rec.prefill;
      pp.request_indices = GetPtrFromBaseOffset<IdT>(int_buffer, info.request_indices_offset);
      pp.qo_tile_indices = GetPtrFromBaseOffset<IdT>(int_buffer, info.qo_tile_indices_offset);
      pp.kv_tile_indices = GetPtrFromBaseOffset<IdT>(int_buffer, info.kv_tile_indices_offset);
      pp.o_indptr = GetPtrFromBaseOffset<IdT>(int_buffer, info.o_indptr_offset);
      pp.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdT>(int_buffer, info.kv_chunk_size_ptr_offset);
      pp.padded_batch_size = static_cast<uint32_t>(info.padded_batch_size);
      pp.max_total_num_rows = static_cast<uint32_t>(info.total_num_rows);
      DT* tmp_v = nullptr;
      float* tmp_s = nullptr;
      if (info.split_kv) {
        pp.merge_indptr = GetPtrFromBaseOffset<IdT>(int_buffer, info.merge_indptr_offset);
        tmp_v = GetPtrFromBaseOffset<DT>(float_buffer, info.v_offset);
        tmp_s = GetPtrFromBaseOffset<float>(float_buffer, info.s_offset);
      }
      pp.partition_kv = info.split_kv;
      st = p.causal ? PrefillDispatch<MaskMode::kCausal>(info.cta_tile_q, pp, tmp_v, tmp_s, ToStream(stream))
                    : PrefillDispatch<MaskMode::kNone>(info.cta_tile_q, pp, tmp_v, tmp_s, ToStream(stream));
    }
    return FromCuda(st);
  });
}

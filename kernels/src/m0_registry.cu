// Registration entries of the M0 attention-side ops (R-017): which variants this library offers, under which
// constraints and flags. The kernels are in m0_ops.cu, paged_attention.cu and gemm.cu.
//
// Flags follow what the tests measured (kernels/tests/ops-cuda): DETERMINISTIC and BATCH_INVARIANT are claimed only where the
// run-twice and batch-split checks passed on hardware (SM 86); the attention kernels are deterministic but not batch invariant
// (the split-KV schedule depends on the whole batch). The RoostConstraints bit layout is documented in roost_kernels.h.

#include <cstdint>

#include "roost_internal.h"

namespace {

constexpr uint64_t Bit(uint32_t dtype) { return uint64_t{1} << dtype; }
constexpr uint64_t kAct16 = Bit(ROOST_DTYPE_BF16) | Bit(ROOST_DTYPE_F16);
constexpr uint64_t kHeadDim256 = uint64_t{1} << (256 / 16 - 1);  // bit i is head_dim 16 * (i + 1)
constexpr uint64_t kHeadDims = (uint64_t{1} << (64 / 16 - 1)) | (uint64_t{1} << (128 / 16 - 1)) | kHeadDim256;
constexpr uint32_t kNhdHnd = (1u << ROOST_LAYOUT_PAGED_NHD) | (1u << ROOST_LAYOUT_PAGED_HND);
constexpr uint32_t kPages16To64 = (1u << 4) | (1u << 5) | (1u << 6);  // bit i is page size 2^i
constexpr uint32_t kGroup4 = 1u << 3;                                 // bit i is i + 1 query heads per KV head

// GRAPH_CAPTURABLE is not claimed yet: the graph replay check of the tests (the "graph" group) has not passed on hardware.
constexpr uint32_t kElementwise = ROOST_FLAG_DETERMINISTIC | ROOST_FLAG_BATCH_INVARIANT;

// Gemm and LmHead: a plan pins one cuBLASLt algorithm per shape bucket, so a row does not depend on the other rows of the call.
constexpr uint32_t kPlannedGemm = ROOST_FLAG_DETERMINISTIC | ROOST_FLAG_BATCH_INVARIANT | ROOST_FLAG_NEEDS_PLAN | ROOST_FLAG_PLAN_PER_SHAPE;

RoostConstraints Row(uint64_t act, uint64_t head_dim_mask, uint32_t alignment) {
  RoostConstraints c = {};
  c.sm_min = 80;
  c.act_dtype_mask = act;
  c.head_dim_mask = head_dim_mask;
  c.layout_mask = 1u << ROOST_LAYOUT_ROW_MAJOR;
  c.min_alignment = alignment;
  return c;
}

RoostConstraints Paged(uint64_t head_dim_mask, uint64_t kv, uint32_t layouts, uint32_t pages, uint32_t alignment) {
  RoostConstraints c = {};
  c.sm_min = 80;
  c.kv_dtype_mask = kv;
  c.head_dim_mask = head_dim_mask;
  c.layout_mask = layouts;
  c.page_size_mask = pages;
  c.min_alignment = alignment;
  return c;
}

RoostOpEntry Entry(uint32_t op, uint32_t flags, const char* id, uint32_t params_size, const RoostConstraints& c,
                   RoostRunFn run, RoostPlanFn plan = nullptr, RoostWorkspaceFn workspace = nullptr) {
  RoostOpEntry e = {};
  e.op = op;
  e.flags = flags;
  e.variant_id = id;
  e.params_struct_size = params_size;
  e.params_abi_minor = ROOST_ABI_MINOR;
  e.constraints = c;
  e.run = run;
  e.plan = plan;
  e.workspace = workspace;
  return e;
}

RoostConstraints WithWeights(RoostConstraints c, uint64_t weight) {
  c.weight_dtype_mask = weight;
  return c;
}

RoostConstraints WithKv(RoostConstraints c, uint64_t kv) {
  c.kv_dtype_mask = kv;
  return c;
}

const RoostOpEntry kEntries[] = {
    Entry(ROOST_OP_RMS_NORM, kElementwise, "rms-norm-flashinfer", sizeof(RoostRmsNormParams), Row(kAct16, 0, 16),
          roost_rms_norm_run),
    Entry(ROOST_OP_QK_NORM, kElementwise, "qk-norm-flashinfer", sizeof(RoostQkNormParams), Row(kAct16, kHeadDim256, 16),
          roost_qk_norm_run),
    Entry(ROOST_OP_ROPE, kElementwise, "rope-roost", sizeof(RoostRopeParams), Row(kAct16, kHeadDims, 2),
          roost_rope_run),
    Entry(ROOST_OP_SILU_AND_MUL, kElementwise, "silu-and-mul-flashinfer", sizeof(RoostActAndMulParams),
          Row(kAct16, 0, 16), roost_silu_and_mul_run),
    Entry(ROOST_OP_EMBEDDING, kElementwise, "embedding-roost", sizeof(RoostEmbeddingParams),
          WithWeights(Row(kAct16 | Bit(ROOST_DTYPE_F32), 0, 2), kAct16 | Bit(ROOST_DTYPE_F32)), roost_embedding_run),
    Entry(ROOST_OP_CACHE_WRITE, kElementwise, "cache-write-roost", sizeof(RoostCacheWriteParams),
          Paged(kHeadDim256, kAct16 | Bit(ROOST_DTYPE_F32), kNhdHnd, kPages16To64, 2), roost_cache_write_run),
    Entry(ROOST_OP_CACHE_COPY, kElementwise, "cache-copy-roost", sizeof(RoostCacheCopyParams),
          Paged(0, kAct16 | Bit(ROOST_DTYPE_F32) | Bit(ROOST_DTYPE_U8), kNhdHnd, 0, 1), roost_cache_copy_run),
    Entry(ROOST_OP_PAGED_ATTENTION, ROOST_FLAG_DETERMINISTIC | ROOST_FLAG_NEEDS_PLAN, "paged-attention-flashinfer-fa2",
          sizeof(RoostPagedAttentionParams),
          [] {
            RoostConstraints c = WithKv(Paged(kHeadDim256, Bit(ROOST_DTYPE_BF16), kNhdHnd, kPages16To64, 16),
                                        Bit(ROOST_DTYPE_BF16));
            c.act_dtype_mask = Bit(ROOST_DTYPE_BF16);
            c.group_size_mask = kGroup4;
            return c;
          }(),
          roost_paged_attention_run, roost_paged_attention_plan, roost_paged_attention_workspace),
    Entry(ROOST_OP_GEMM, kPlannedGemm, "gemm-cublaslt",
          sizeof(RoostGemmParams), WithWeights(Row(kAct16, 0, 16), kAct16), roost_gemm_run, roost_gemm_plan,
          roost_gemm_workspace),
    Entry(ROOST_OP_LM_HEAD, kPlannedGemm, "lm-head-cublaslt", sizeof(RoostLmHeadParams), WithWeights(Row(kAct16, 0, 16), kAct16), roost_lm_head_run,
          roost_lm_head_plan, roost_lm_head_workspace),
};

}  // namespace

const RoostOpEntry* roost_m0_entries(uint32_t* count) {
  *count = static_cast<uint32_t>(sizeof(kEntries) / sizeof(kEntries[0]));
  return kEntries;
}

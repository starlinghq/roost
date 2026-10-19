// Registration entries of the hybrid ops (R-034), merged into the library's table by roost_register. Kept apart from
// registry.cu so the attention-side ops (R-017) and these add their entries in their own files.

#include <cstdint>

#include "roost_internal.h"
#include "roost_kernels.h"

namespace roost_hybrid_entries_impl {

constexpr uint64_t Bit(uint32_t dtype) { return uint64_t{1} << dtype; }

constexpr uint32_t kStateFlags = ROOST_FLAG_GRAPH_CAPTURABLE | ROOST_FLAG_DETERMINISTIC | ROOST_FLAG_BATCH_INVARIANT;

constexpr RoostConstraints PlainConstraints(uint64_t act_mask, uint32_t alignment) {
  return {
      /* sm_min */ 80,
      /* sm_max */ 0,
      /* act_dtype_mask */ act_mask,
      /* weight_dtype_mask */ 0,
      /* kv_dtype_mask */ 0,
      /* head_dim_mask */ 0,
      /* layout_mask */ 1u << ROOST_LAYOUT_ROW_MAJOR,
      /* page_size_mask */ 0,
      /* min_alignment */ alignment,
      /* max_batch */ 0,
      /* max_tokens */ 0,
      /* group_size_mask */ 0,
      /* reserved0 */ 0,
  };
}

#ifdef ROOST_HYBRID_GDN
constexpr RoostConstraints GdnConstraints() {
  RoostConstraints c = PlainConstraints(Bit(ROOST_DTYPE_BF16), 16);
  c.head_dim_mask = uint64_t{1} << 7;  // head size 128 = 16 * (7 + 1)
  c.group_size_mask = 1u;               // 16 value heads per 16 key heads: one value head per key head
  return c;
}
#endif

const RoostOpEntry kHybridEntries[] = {
#ifdef ROOST_HYBRID_GDN
    {
        /* op */ ROOST_OP_GATED_DELTA_RULE,
        /* flags */ ROOST_FLAG_DETERMINISTIC | ROOST_FLAG_BATCH_INVARIANT | ROOST_FLAG_NEEDS_PLAN,
        /* variant_id */ "gated-delta-rule-chunk-aot",
        /* params_struct_size */ static_cast<uint32_t>(sizeof(RoostGatedDeltaRuleParams)),
        /* params_abi_minor */ ROOST_ABI_MINOR,
        /* constraints */ GdnConstraints(),
        /* run */ roost_gdn_chunk_run,
        /* plan */ roost_gdn_chunk_plan,
        /* workspace */ roost_gdn_chunk_workspace,
        /* reserved0 */ 0,
    },
    {
        /* op */ ROOST_OP_GATED_DELTA_RULE,
        /* flags */ kStateFlags,
        /* variant_id */ "gated-delta-rule-recurrent-aot",
        /* params_struct_size */ static_cast<uint32_t>(sizeof(RoostGatedDeltaRuleParams)),
        /* params_abi_minor */ ROOST_ABI_MINOR,
        /* constraints */ GdnConstraints(),
        /* run */ roost_gdn_recurrent_run,
        /* plan */ nullptr,
        /* workspace */ roost_gdn_recurrent_workspace,
        /* reserved0 */ 0,
    },
#endif
    {
        /* op */ ROOST_OP_CAUSAL_CONV1D,
        /* flags */ kStateFlags,
        /* variant_id */ "causal-conv1d-cuda",
        /* params_struct_size */ static_cast<uint32_t>(sizeof(RoostCausalConv1dParams)),
        /* params_abi_minor */ ROOST_ABI_MINOR,
        /* constraints */ PlainConstraints(Bit(ROOST_DTYPE_BF16) | Bit(ROOST_DTYPE_F32), 4),
        /* run */ roost_causal_conv1d_run,
        /* plan */ nullptr,
        /* workspace */ nullptr,
        /* reserved0 */ 0,
    },
    {
        /* op */ ROOST_OP_RECURRENT_STATE_COPY,
        /* flags */ kStateFlags,
        /* variant_id */ "recurrent-state-copy-cuda",
        /* params_struct_size */ static_cast<uint32_t>(sizeof(RoostRecurrentStateCopyParams)),
        /* params_abi_minor */ ROOST_ABI_MINOR,
        /* constraints */ PlainConstraints(0, 1),
        /* run */ roost_recurrent_state_copy_run,
        /* plan */ nullptr,
        /* workspace */ nullptr,
        /* reserved0 */ 0,
    },
    {
        /* op */ ROOST_OP_RECURRENT_STATE_SNAPSHOT,
        /* flags */ kStateFlags,
        /* variant_id */ "recurrent-state-snapshot-cuda",
        /* params_struct_size */ static_cast<uint32_t>(sizeof(RoostRecurrentStateSnapshotParams)),
        /* params_abi_minor */ ROOST_ABI_MINOR,
        /* constraints */ PlainConstraints(0, 1),
        /* run */ roost_recurrent_state_snapshot_run,
        /* plan */ nullptr,
        /* workspace */ nullptr,
        /* reserved0 */ 0,
    },
    {
        /* op */ ROOST_OP_GATED_DELTA_RULE_GATES,
        /* flags */ kStateFlags,
        /* variant_id */ "gated-delta-rule-gates-cuda",
        /* params_struct_size */ static_cast<uint32_t>(sizeof(RoostGatedDeltaRuleGatesParams)),
        /* params_abi_minor */ ROOST_ABI_MINOR,
        /* constraints */ PlainConstraints(Bit(ROOST_DTYPE_BF16) | Bit(ROOST_DTYPE_F32), 4),
        /* run */ roost_gated_delta_rule_gates_run,
        /* plan */ nullptr,
        /* workspace */ nullptr,
        /* reserved0 */ 0,
    },
    {
        /* op */ ROOST_OP_GATED_RMS_NORM,
        /* flags */ kStateFlags,
        /* variant_id */ "gated-rms-norm-cuda",
        /* params_struct_size */ static_cast<uint32_t>(sizeof(RoostGatedRmsNormParams)),
        /* params_abi_minor */ ROOST_ABI_MINOR,
        /* constraints */ PlainConstraints(Bit(ROOST_DTYPE_BF16) | Bit(ROOST_DTYPE_F32), 4),
        /* run */ roost_gated_rms_norm_run,
        /* plan */ nullptr,
        /* workspace */ nullptr,
        /* reserved0 */ 0,
    },
    {
        /* op */ ROOST_OP_SIGMOID_MUL,
        /* flags */ kStateFlags,
        /* variant_id */ "sigmoid-mul-cuda",
        /* params_struct_size */ static_cast<uint32_t>(sizeof(RoostSigmoidMulParams)),
        /* params_abi_minor */ ROOST_ABI_MINOR,
        /* constraints */ PlainConstraints(Bit(ROOST_DTYPE_BF16) | Bit(ROOST_DTYPE_F32), 4),
        /* run */ roost_sigmoid_mul_run,
        /* plan */ nullptr,
        /* workspace */ nullptr,
        /* reserved0 */ 0,
    },
};

}  // namespace roost_hybrid_entries_impl

using roost_hybrid_entries_impl::kHybridEntries;

const RoostOpEntry* roost_hybrid_entries(uint32_t* count) {
  *count = static_cast<uint32_t>(sizeof(kHybridEntries) / sizeof(kHybridEntries[0]));
  return kHybridEntries;
}

#ifndef ROOST_HYBRID_GDN
// Built without the Triton AOT kernels: GatedDeltaRule is not registered and there is no per-context state.
RoostStatus roost_hybrid_context_init(RoostContext* /*ctx*/) { return ROOST_STATUS_OK; }
void roost_hybrid_context_release(RoostContext* /*ctx*/) {}
#endif

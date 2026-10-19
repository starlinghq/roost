// Registration entries of the hybrid ops (R-034), merged into the library's table by roost_register. Kept apart from
// registry.cu so the attention-side ops (R-017) and these add their entries in their own files.

#include <cstdint>

#include "hybrid_params.h"
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

const RoostOpEntry kHybridEntries[] = {
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
};

}  // namespace roost_hybrid_entries_impl

using roost_hybrid_entries_impl::kHybridEntries;

const RoostOpEntry* roost_hybrid_entries(uint32_t* count) {
  *count = static_cast<uint32_t>(sizeof(kHybridEntries) / sizeof(kHybridEntries[0]));
  return kHybridEntries;
}

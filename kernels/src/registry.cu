// C1 exports and the registration table of the Roost kernel library (R-016). One library is built per SM, so the
// digest of each release asset is per SM and the loader's refusal of a missing SM is a file lookup.
//
// The table lists the variants the library was built with: the Residual stub that proves the build, registration and run
// path (R-016) and the M0 attention-side ops (R-017). R-034 adds the hybrid ops behind the same table.

#include <cstdint>
#include <cstring>
#include <iterator>
#include <new>
#include <string>
#include <vector>

#include "roost_kernels.h"
#include "roost_internal.h"

#ifndef ROOST_SM
#error "ROOST_SM (the compute capability as major * 10 + minor) must be defined by the build"
#endif

namespace {

thread_local std::string g_last_error;

constexpr uint64_t Bit(uint32_t dtype) { return uint64_t{1} << dtype; }

// Entries are static data owned by the library until it unloads (C1).
const RoostOpEntry kEntries[] = {
    {
        /* op */ ROOST_OP_RESIDUAL,
        /* flags */ ROOST_FLAG_GRAPH_CAPTURABLE | ROOST_FLAG_DETERMINISTIC | ROOST_FLAG_BATCH_INVARIANT,
        /* variant_id */ "residual-stub",
        /* params_struct_size */ static_cast<uint32_t>(sizeof(RoostResidualParams)),
        /* params_abi_minor */ ROOST_ABI_MINOR,
        /* constraints */
        {
            /* sm_min */ 80,
            /* sm_max */ 0,
            /* act_dtype_mask */ Bit(ROOST_DTYPE_F32) | Bit(ROOST_DTYPE_F16) | Bit(ROOST_DTYPE_BF16),
            /* weight_dtype_mask */ 0,
            /* kv_dtype_mask */ 0,
            /* head_dim_mask */ 0,
            /* layout_mask */ 1u << ROOST_LAYOUT_ROW_MAJOR,
            /* page_size_mask */ 0,
            /* min_alignment */ 4,
            /* max_batch */ 0,
            /* max_tokens */ 0,
            /* group_size_mask */ 0,
            /* reserved0 */ 0,
        },
        /* run */ roost_residual_run,
        /* plan */ nullptr,
        /* workspace */ nullptr,
        /* reserved0 */ 0,
    },
};

// The table the library registers: the R-016 stub followed by each item's group of entries (R-017: m0_registry.cu).
const std::vector<RoostOpEntry>& AllEntries() {
  static const std::vector<RoostOpEntry> all = [] {
    std::vector<RoostOpEntry> v(std::begin(kEntries), std::end(kEntries));
    uint32_t count = 0;
    const RoostOpEntry* m0 = roost_m0_entries(&count);
    v.insert(v.end(), m0, m0 + count);
    return v;
  }();
  return all;
}

}  // namespace

void roost_set_error(const char* message) { g_last_error = message; }

extern "C" {

ROOST_API uint32_t roost_abi_version(void) { return ROOST_ABI_VERSION; }

ROOST_API const char* roost_last_error(void) { return g_last_error.c_str(); }

ROOST_API RoostStatus roost_register(RoostOpTable* table) {
  if (table == nullptr) {
    roost_set_error("roost_register: table is NULL");
    return ROOST_STATUS_INVALID_ARGUMENT;
  }
  // The caller sets struct_size and abi_minor; a table smaller than this library writes is an ABI mismatch.
  if (table->struct_size < sizeof(RoostOpTable) || table->abi_minor > ROOST_ABI_MINOR) {
    roost_set_error("roost_register: table struct_size or abi_minor is not supported by this library");
    return ROOST_STATUS_ABI_MISMATCH;
  }
  table->entry_size = static_cast<uint32_t>(sizeof(RoostOpEntry));
  const std::vector<RoostOpEntry>& all = AllEntries();
  table->entry_count = static_cast<uint32_t>(all.size());
  table->entries = all.data();
  table->library_name = "roost-kernels-sm" ROOST_STRINGIZE(ROOST_SM);
  table->library_abi_major = ROOST_ABI_MAJOR;
  table->library_abi_minor = ROOST_ABI_MINOR;
  return ROOST_STATUS_OK;
}

ROOST_API RoostStatus roost_context_create(const RoostContextCreateParams* params, RoostContext** out) {
  if (params == nullptr || out == nullptr) {
    roost_set_error("roost_context_create: params or out is NULL");
    return ROOST_STATUS_INVALID_ARGUMENT;
  }
  if (params->struct_size < sizeof(RoostContextCreateParams) || params->abi_minor > ROOST_ABI_MINOR) {
    roost_set_error("roost_context_create: params struct_size or abi_minor is not supported by this library");
    return ROOST_STATUS_ABI_MISMATCH;
  }
  if (params->reserved0 != 0) {
    roost_set_error("roost_context_create: reserved0 must be zero");
    return ROOST_STATUS_INVALID_ARGUMENT;
  }
  *out = new (std::nothrow) RoostContext{params->device, params->flags, nullptr};
  if (*out == nullptr) {
    roost_set_error("roost_context_create: out of host memory");
    return ROOST_STATUS_OUT_OF_MEMORY;
  }
  const RoostStatus status = roost_gemm_context_init(*out);
  if (status != ROOST_STATUS_OK) {
    delete *out;
    *out = nullptr;
    return status;
  }
  return ROOST_STATUS_OK;
}

ROOST_API RoostStatus roost_context_destroy(RoostContext* ctx) {
  roost_gemm_context_release(ctx);
  delete ctx;
  return ROOST_STATUS_OK;
}

}  // extern "C"

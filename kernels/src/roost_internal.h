// Declarations shared by the translation units of one kernel library. Not part of the C1 ABI.
#ifndef ROOST_INTERNAL_H_
#define ROOST_INTERNAL_H_

#include <cstdint>

#include "roost_kernels.h"

#define ROOST_STRINGIZE_(x) #x
#define ROOST_STRINGIZE(x) ROOST_STRINGIZE_(x)

// Per-library, per-device state (C1). Later ops add cuBLASLt handles and pinned algorithms here.
struct RoostContext {
  int32_t device;
  uint32_t flags;
};

// The registration entries of the hybrid ops (R-034, src/ops/hybrid_entries.cu).
const RoostOpEntry* roost_hybrid_entries(uint32_t* count);

// Records the message roost_last_error returns on this thread.
void roost_set_error(const char* message);

extern "C" RoostStatus roost_residual_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws,
                                          void* stream);

extern "C" RoostStatus roost_causal_conv1d_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws,
                                               void* stream);
extern "C" RoostStatus roost_recurrent_state_copy_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws,
                                                      void* stream);
extern "C" RoostStatus roost_recurrent_state_snapshot_run(RoostContext* ctx, const void* params,
                                                          const RoostWorkspace* ws, void* stream);

#endif  // ROOST_INTERNAL_H_

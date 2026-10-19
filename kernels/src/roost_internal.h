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

// Records the message roost_last_error returns on this thread.
void roost_set_error(const char* message);

extern "C" RoostStatus roost_residual_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws,
                                          void* stream);

#endif  // ROOST_INTERNAL_H_

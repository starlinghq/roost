// Helpers shared by the hybrid op translation units (R-034). Not part of the C1 ABI.
#ifndef ROOST_HYBRID_COMMON_H_
#define ROOST_HYBRID_COMMON_H_

#include <cuda_runtime.h>

#include <cstdint>

#include "roost_internal.h"
#include "roost_kernels.h"

namespace roost_hybrid {

inline RoostStatus Fail(RoostStatus status, const char* message) {
  roost_set_error(message);
  return status;
}

inline RoostStatus CudaFail(cudaError_t err) {
  roost_set_error(cudaGetErrorString(err));
  return ROOST_STATUS_CUDA_ERROR;
}

// A span the op reads or writes: dtype as expected, reserved zero, at least `elements` long, non-NULL when non-empty.
inline bool SpanOk(const RoostDeviceSpan& s, uint32_t dtype, uint64_t elements) {
  if (s.reserved0 != 0 || s.dtype != dtype || s.length < elements) {
    return false;
  }
  return elements == 0 || s.address != nullptr;
}

inline bool Aligned(const void* p, uint64_t bytes) { return (reinterpret_cast<uintptr_t>(p) % bytes) == 0; }

inline uint32_t CeilDiv(uint64_t a, uint64_t b) { return static_cast<uint32_t>((a + b - 1) / b); }

}  // namespace roost_hybrid

#endif  // ROOST_HYBRID_COMMON_H_

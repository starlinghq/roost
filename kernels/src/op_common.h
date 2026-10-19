// Helpers shared by the M0 op translation units (R-017). Not part of the C1 ABI.
//
// Every entry point validates its params struct the same way (size, minor, reserved fields, spans), turns exceptions
// into a status so that nothing crosses the C ABI, and reports the message through roost_set_error.
#ifndef ROOST_OP_COMMON_H_
#define ROOST_OP_COMMON_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <string>

#include "roost_internal.h"
#include "roost_kernels.h"

namespace roost_ops {

constexpr uint64_t DtypeBit(uint32_t dtype) { return uint64_t{1} << dtype; }

inline uint32_t DtypeBytes(uint32_t dtype) {
  switch (dtype) {
    case ROOST_DTYPE_F32:
    case ROOST_DTYPE_I32:
    case ROOST_DTYPE_U32: return 4;
    case ROOST_DTYPE_F16:
    case ROOST_DTYPE_BF16:
    case ROOST_DTYPE_I16: return 2;
    case ROOST_DTYPE_F64:
    case ROOST_DTYPE_I64:
    case ROOST_DTYPE_U64: return 8;
    case ROOST_DTYPE_I8:
    case ROOST_DTYPE_U8:
    case ROOST_DTYPE_F8E4M3:
    case ROOST_DTYPE_F8E5M2: return 1;
    default: return 0;
  }
}

inline RoostStatus Fail(RoostStatus status, const std::string& message) {
  roost_set_error(message.c_str());
  return status;
}

inline RoostStatus FromCuda(cudaError_t err) {
  if (err == cudaSuccess) {
    return ROOST_STATUS_OK;
  }
  roost_set_error(cudaGetErrorString(err));
  return ROOST_STATUS_CUDA_ERROR;
}

// Checks the leading struct_size / abi_minor of any params struct (C1): a struct smaller than this library needs for
// the minor it was given is an ABI mismatch.
inline RoostStatus CheckHeader(const void* params, uint32_t needed_size, const char* op) {
  if (params == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, std::string(op) + ": params is NULL");
  }
  const auto* header = static_cast<const uint32_t*>(params);
  if (header[0] < needed_size || header[1] > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH,
                std::string(op) + ": params struct_size or abi_minor is not supported by this library");
  }
  return ROOST_STATUS_OK;
}

// A span must have the expected dtype, a non-NULL address when elements > 0, at least `elements` elements and a zero
// reserved field. `optional` spans may have a NULL address and zero length.
inline RoostStatus CheckSpan(const RoostDeviceSpan& s, uint32_t dtype, uint64_t elements, const char* op,
                             const char* name, bool optional = false) {
  if (s.reserved0 != 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, std::string(op) + ": " + name + ".reserved0 must be zero");
  }
  if (optional && s.address == nullptr) {
    return ROOST_STATUS_OK;
  }
  if (s.dtype != dtype) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, std::string(op) + ": " + name + " has an unexpected dtype");
  }
  if (elements > 0 && s.address == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, std::string(op) + ": " + name + " has a NULL address");
  }
  if (s.length < elements) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, std::string(op) + ": " + name + " is shorter than the shape needs");
  }
  return ROOST_STATUS_OK;
}

inline bool Aligned(const void* p, uintptr_t bytes) { return (reinterpret_cast<uintptr_t>(p) % bytes) == 0; }

constexpr uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// Runs `body` and maps an escaping exception (FlashInfer's FLASHINFER_CHECK throws) to a status.
template <typename F>
RoostStatus Guarded(const char* op, F&& body) {
  try {
    return body();
  } catch (const std::exception& e) {
    return Fail(ROOST_STATUS_INTERNAL, std::string(op) + ": " + e.what());
  } catch (...) {
    return Fail(ROOST_STATUS_INTERNAL, std::string(op) + ": unknown exception");
  }
}

}  // namespace roost_ops

#endif  // ROOST_OP_COMMON_H_

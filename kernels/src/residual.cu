// Residual (ROOST_OP_RESIDUAL): output = a + b, a may alias output. This is the registered stub of R-016: a small,
// correct op that exercises the full path (build, registration, params validation, launch on the caller's stream)
// that R-017's real ops reuse. The arithmetic is done in float and rounded once to the element type.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "roost_internal.h"

namespace roost_residual {

__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T FromFloat(float v);
template <>
__device__ __forceinline__ float FromFloat<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half FromFloat<__half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 FromFloat<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template <typename T>
__global__ void ResidualKernel(const T* a, const T* b, T* out, uint64_t n) {
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = FromFloat<T>(ToFloat(a[i]) + ToFloat(b[i]));
  }
}

template <typename T>
RoostStatus Launch(const RoostResidualParams& p, cudaStream_t stream) {
  constexpr uint32_t kThreads = 256;
  constexpr uint64_t kMaxBlocks = 65535;
  const uint64_t wanted = (p.elements + kThreads - 1) / kThreads;
  const uint32_t blocks = static_cast<uint32_t>(wanted < kMaxBlocks ? wanted : kMaxBlocks);
  ResidualKernel<T><<<blocks, kThreads, 0, stream>>>(static_cast<const T*>(p.a.address),
                                                     static_cast<const T*>(p.b.address),
                                                     static_cast<T*>(p.output.address), p.elements);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    roost_set_error(cudaGetErrorString(err));
    return ROOST_STATUS_CUDA_ERROR;
  }
  return ROOST_STATUS_OK;
}

}  // namespace roost_residual

using roost_residual::Launch;

extern "C" RoostStatus roost_residual_run(RoostContext* ctx, const void* params, const RoostWorkspace* /*ws*/,
                                          void* stream) {
  if (ctx == nullptr || params == nullptr) {
    roost_set_error("Residual: ctx or params is NULL");
    return ROOST_STATUS_INVALID_ARGUMENT;
  }
  const auto& p = *static_cast<const RoostResidualParams*>(params);
  if (p.struct_size < sizeof(RoostResidualParams) || p.abi_minor > ROOST_ABI_MINOR) {
    roost_set_error("Residual: params struct_size or abi_minor is not supported by this library");
    return ROOST_STATUS_ABI_MISMATCH;
  }
  if (p.a.dtype != p.b.dtype || p.a.dtype != p.output.dtype) {
    roost_set_error("Residual: a, b and output must have the same dtype");
    return ROOST_STATUS_INVALID_ARGUMENT;
  }
  if (p.a.reserved0 != 0 || p.b.reserved0 != 0 || p.output.reserved0 != 0) {
    roost_set_error("Residual: reserved fields must be zero");
    return ROOST_STATUS_INVALID_ARGUMENT;
  }
  if (p.a.length < p.elements || p.b.length < p.elements || p.output.length < p.elements) {
    roost_set_error("Residual: a span is shorter than elements");
    return ROOST_STATUS_INVALID_ARGUMENT;
  }
  if (p.elements == 0) {
    return ROOST_STATUS_OK;
  }
  if (p.a.address == nullptr || p.b.address == nullptr || p.output.address == nullptr) {
    roost_set_error("Residual: a span has a NULL address");
    return ROOST_STATUS_INVALID_ARGUMENT;
  }
  const auto s = static_cast<cudaStream_t>(stream);
  switch (p.a.dtype) {
    case ROOST_DTYPE_F32: return Launch<float>(p, s);
    case ROOST_DTYPE_F16: return Launch<__half>(p, s);
    case ROOST_DTYPE_BF16: return Launch<__nv_bfloat16>(p, s);
    default:
      roost_set_error("Residual: dtype is not supported (f32, f16 and bf16 only)");
      return ROOST_STATUS_UNSUPPORTED;
  }
}

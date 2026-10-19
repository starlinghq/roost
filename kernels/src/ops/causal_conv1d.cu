// CausalConv1d (ROOST_OP_CAUSAL_CONV1D) for the Qwen3.5-0.8B linear-attention layers (R-034): depthwise causal
// convolution of width <= 16 (4 for Qwen3.5) over packed sequences, [tokens, channels] layout, per-sequence conv state
// in a pool slot. CpuOps.CausalConv1d is the definition. S3 did not cover this op, so it is plain CUDA C++ rather than
// Triton AOT. Two launches: outputs (one thread per token and channel; reads the old state), then the state update
// (one thread per sequence and channel), so no thread reads a state another thread has already overwritten.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "hybrid_common.h"
#include "hybrid_params.h"

namespace roost_hybrid_conv {

constexpr int kMaxWidth = 16;

__device__ __forceinline__ float ToF(float v) { return v; }
__device__ __forceinline__ float ToF(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T>
__device__ __forceinline__ T FromF(float v);
template <>
__device__ __forceinline__ float FromF<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 FromF<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Index of the sequence holding token t: the last b with cu[b] <= t < cu[b + 1] (empty sequences are skipped).
__device__ __forceinline__ int SequenceOf(const int32_t* cu, int batch, int t) {
  int lo = 0;
  int hi = batch - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (cu[mid] <= t) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// Element i of [state (width - 1 entries), sequence inputs] for one channel of one sequence.
template <typename T, typename S>
__device__ __forceinline__ float History(const T* input, const S* state, int t0, int channels, int c, int i, int sl) {
  return i < sl ? ToF(state[i]) : ToF(input[static_cast<int64_t>(t0 + i - sl) * channels + c]);
}

template <typename T, typename S>
__global__ void OutputKernel(const T* input, const T* weight, const T* bias, T* output, const S* conv_state,
                             const int32_t* slots, const int32_t* cu, int batch, int tokens, int channels, int width,
                             int silu) {
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  const int t = blockIdx.x;
  if (c >= channels || t >= tokens) {
    return;
  }
  const int b = SequenceOf(cu, batch, t);
  const int t0 = cu[b];
  const int sl = width - 1;
  const S* state = conv_state + (static_cast<int64_t>(slots[b]) * channels + c) * sl;
  float y = bias != nullptr ? ToF(bias[c]) : 0.0f;
  for (int j = 0; j < width; ++j) {
    y += ToF(weight[static_cast<int64_t>(c) * width + j]) * History(input, state, t0, channels, c, (t - t0) + j, sl);
  }
  if (silu != 0) {
    y = y / (1.0f + expf(-y));
  }
  output[static_cast<int64_t>(t) * channels + c] = FromF<T>(y);
}

template <typename T, typename S>
__global__ void StateKernel(const T* input, S* conv_state, const int32_t* slots, const int32_t* cu, int batch,
                            int channels, int width) {
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  const int b = blockIdx.x;
  if (c >= channels || b >= batch) {
    return;
  }
  const int t0 = cu[b];
  const int n = cu[b + 1] - t0;
  if (n <= 0) {
    return;
  }
  const int sl = width - 1;
  S* state = conv_state + (static_cast<int64_t>(slots[b]) * channels + c) * sl;
  float next[kMaxWidth];
  for (int j = 0; j < sl; ++j) {
    next[j] = History(input, state, t0, channels, c, n + j, sl);
  }
  for (int j = 0; j < sl; ++j) {
    state[j] = FromF<S>(next[j]);
  }
}

template <typename T, typename S>
RoostStatus Launch(const RoostCausalConv1dParams& p, cudaStream_t stream) {
  const T* bias = static_cast<const T*>(p.bias.address);
  dim3 block(128);
  // Tokens and sequences go on grid x (up to 2^31 - 1); channel blocks go on y, which Validate bounds.
  dim3 grid(p.tokens, roost_hybrid::CeilDiv(p.channels, block.x));
  OutputKernel<T, S><<<grid, block, 0, stream>>>(
      static_cast<const T*>(p.input.address), static_cast<const T*>(p.weight.address), bias,
      static_cast<T*>(p.output.address), static_cast<const S*>(p.conv_state.address),
      static_cast<const int32_t*>(p.state_slots.address), static_cast<const int32_t*>(p.cu_seqlens.address),
      static_cast<int>(p.batch), static_cast<int>(p.tokens), static_cast<int>(p.channels), static_cast<int>(p.width),
      static_cast<int>(p.silu));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return roost_hybrid::CudaFail(err);
  }
  dim3 sgrid(p.batch, roost_hybrid::CeilDiv(p.channels, block.x));
  StateKernel<T, S><<<sgrid, block, 0, stream>>>(
      static_cast<const T*>(p.input.address), static_cast<S*>(p.conv_state.address),
      static_cast<const int32_t*>(p.state_slots.address), static_cast<const int32_t*>(p.cu_seqlens.address),
      static_cast<int>(p.batch), static_cast<int>(p.channels), static_cast<int>(p.width));
  err = cudaGetLastError();
  return err == cudaSuccess ? ROOST_STATUS_OK : roost_hybrid::CudaFail(err);
}

}  // namespace roost_hybrid_conv

extern "C" RoostStatus roost_causal_conv1d_run(RoostContext* ctx, const void* params, const RoostWorkspace* /*ws*/,
                                               void* stream) {
  using roost_hybrid::Fail;
  using roost_hybrid::SpanOk;
  if (ctx == nullptr || params == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CausalConv1d: ctx or params is NULL");
  }
  const auto& p = *static_cast<const RoostCausalConv1dParams*>(params);
  if (p.struct_size < sizeof(RoostCausalConv1dParams) || p.abi_minor > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH, "CausalConv1d: params struct_size or abi_minor is not supported by this library");
  }
  if (p.reserved0 != 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CausalConv1d: reserved0 must be zero");
  }
  if (p.width < 1 || p.width > roost_hybrid_conv::kMaxWidth || p.channels == 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CausalConv1d: width must be in [1, 16] and channels non-zero");
  }
  if (p.channels > 65535u * 128u || p.tokens > 0x7fffffffu || p.batch > 0x7fffffffu) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CausalConv1d: too many channels, tokens or sequences for one call");
  }
  if (p.batch == 0 || p.tokens == 0) {
    return ROOST_STATUS_OK;
  }
  const uint32_t act = p.input.dtype;
  if (act != ROOST_DTYPE_BF16 && act != ROOST_DTYPE_F32) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "CausalConv1d: activation dtype must be bf16 or f32");
  }
  const uint32_t st = p.conv_state.dtype == ROOST_DTYPE_UNDEFINED ? act : p.conv_state.dtype;
  if (st != ROOST_DTYPE_BF16 && st != ROOST_DTYPE_F32) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "CausalConv1d: conv state dtype must be bf16 or f32");
  }
  const uint64_t sl = p.width - 1;
  const bool has_bias = p.bias.address != nullptr;
  if (!SpanOk(p.input, act, uint64_t{p.tokens} * p.channels) || !SpanOk(p.output, act, uint64_t{p.tokens} * p.channels) ||
      !SpanOk(p.weight, act, uint64_t{p.channels} * p.width) ||
      (has_bias && !SpanOk(p.bias, act, p.channels)) ||
      !SpanOk(p.state_slots, ROOST_DTYPE_I32, p.batch) || !SpanOk(p.cu_seqlens, ROOST_DTYPE_I32, uint64_t{p.batch} + 1)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CausalConv1d: a span has the wrong dtype, is too short or has a NULL address");
  }
  if (sl > 0 && (p.conv_state.reserved0 != 0 || p.conv_state.address == nullptr)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CausalConv1d: conv_state is NULL");
  }
  if (p.input.address == p.output.address) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CausalConv1d: input and output must not alias");
  }
  const auto s = static_cast<cudaStream_t>(stream);
  if (act == ROOST_DTYPE_BF16) {
    return st == ROOST_DTYPE_BF16 ? roost_hybrid_conv::Launch<__nv_bfloat16, __nv_bfloat16>(p, s)
                                  : roost_hybrid_conv::Launch<__nv_bfloat16, float>(p, s);
  }
  return st == ROOST_DTYPE_BF16 ? roost_hybrid_conv::Launch<float, __nv_bfloat16>(p, s)
                                : roost_hybrid_conv::Launch<float, float>(p, s);
}

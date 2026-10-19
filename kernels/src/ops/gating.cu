// The Qwen3.5 gating ops of contract version 3 (ABI 1.2), R-034: GatedDeltaRuleGates, GatedRmsNorm and SigmoidMul.
// CpuOps.GatedDeltaRuleGates, GatedRmsNorm and SigmoidMul are the definitions; the formulas and dtypes are the header's.
// Plain CUDA C++ (S3 did not cover them). Every element is computed in float and rounded once to its span's type, which
// differs from the double reference by float rounding of the intermediates only. No thread's result depends on another
// row or token (GatedRmsNorm reduces a row in a fixed order), so the ops are deterministic and batch invariant, and they
// allocate nothing and use only the given stream, so they are graph capturable.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "hybrid_common.h"

namespace roost_hybrid_gating {

// An element of a span whose type is chosen at run time: bf16 or float32 (the types Qwen3.5 uses).
__device__ __forceinline__ float Load(const void* p, uint32_t dtype, int64_t i) {
  return dtype == ROOST_DTYPE_BF16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i])
                                   : static_cast<const float*>(p)[i];
}

__device__ __forceinline__ void Store(void* p, uint32_t dtype, int64_t i, float v) {
  if (dtype == ROOST_DTYPE_BF16) {
    static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
  } else {
    static_cast<float*>(p)[i] = v;
  }
}

__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

constexpr float kSoftplusThreshold = 20.0f;  // torch.nn.functional.softplus: x itself above it

// One thread per (token, head).
__global__ void GatesKernel(const void* a, const void* b, const void* a_log, const void* dt_bias, void* g, void* beta,
                            uint32_t a_t, uint32_t b_t, uint32_t alog_t, uint32_t dt_t, uint32_t g_t, uint32_t beta_t,
                            int64_t total, int32_t v_heads, int64_t a_stride, int64_t b_stride) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) {
    return;
  }
  const int64_t t = i / v_heads;
  const int32_t h = static_cast<int32_t>(i - t * v_heads);
  const float x = Load(a, a_t, t * a_stride + h) + Load(dt_bias, dt_t, h);
  const float softplus = x > kSoftplusThreshold ? x : log1pf(expf(x));
  Store(g, g_t, i, -expf(Load(a_log, alog_t, h)) * softplus);
  Store(beta, beta_t, i, Sigmoid(Load(b, b_t, t * b_stride + h)));
}

// One block per row; the row's sum of squares is reduced in a fixed order (strided partial sums, then a tree).
__global__ void GatedRmsNormKernel(const void* input, const void* gate, const void* weight, void* output, uint32_t act,
                                   uint32_t weight_t, int32_t dim, float eps) {
  extern __shared__ float partial[];
  const int64_t row = static_cast<int64_t>(blockIdx.x) * dim;
  float ss = 0.0f;
  for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = Load(input, act, row + i);
    ss += v * v;
  }
  partial[threadIdx.x] = ss;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      partial[threadIdx.x] += partial[threadIdx.x + s];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(partial[0] / static_cast<float>(dim) + eps);
  for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float z = Load(gate, act, row + i);
    const float y = Load(input, act, row + i) * inv * Load(weight, weight_t, i) * (z * Sigmoid(z));
    Store(output, act, row + i, y);
  }
}

__global__ void SigmoidMulKernel(const void* input, const void* gate, void* output, uint32_t act, int64_t total) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
    Store(output, act, i, Load(input, act, i) * Sigmoid(Load(gate, act, i)));
  }
}

inline bool FloatType(uint32_t dtype) { return dtype == ROOST_DTYPE_BF16 || dtype == ROOST_DTYPE_F32; }

inline uint64_t ElementBytes(uint32_t dtype) { return dtype == ROOST_DTYPE_BF16 ? 2 : 4; }

// A span used for `elements` elements must start at a multiple of its element size.
inline bool Natural(const RoostDeviceSpan& s) { return roost_hybrid::Aligned(s.address, ElementBytes(s.dtype)); }

// Two device ranges that overlap without starting at the same address. Starting together is elementwise-safe for these ops.
inline bool PartialOverlap(const RoostDeviceSpan& x, uint64_t x_elems, const RoostDeviceSpan& y, uint64_t y_elems) {
  const auto xb = reinterpret_cast<uintptr_t>(x.address);
  const auto yb = reinterpret_cast<uintptr_t>(y.address);
  if (xb == yb) {
    return false;
  }
  return xb < yb + y_elems * ElementBytes(y.dtype) && yb < xb + x_elems * ElementBytes(x.dtype);
}

}  // namespace roost_hybrid_gating

extern "C" RoostStatus roost_gated_delta_rule_gates_run(RoostContext* ctx, const void* params,
                                                         const RoostWorkspace* /*ws*/, void* stream) {
  using roost_hybrid::Fail;
  using roost_hybrid::SpanOk;
  using namespace roost_hybrid_gating;
  if (ctx == nullptr || params == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRuleGates: ctx or params is NULL");
  }
  const auto& p = *static_cast<const RoostGatedDeltaRuleGatesParams*>(params);
  if (p.struct_size < sizeof(RoostGatedDeltaRuleGatesParams) || p.abi_minor > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH,
                "GatedDeltaRuleGates: params struct_size or abi_minor is not supported by this library");
  }
  if (p.v_heads == 0 || p.v_heads > 0x7fffffffu) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRuleGates: v_heads must be in [1, 2^31 - 1]");
  }
  if (p.tokens == 0) {
    return ROOST_STATUS_OK;
  }
  const uint64_t vh = p.v_heads;
  const uint64_t a_stride = p.a_stride == 0 ? vh : p.a_stride;
  const uint64_t b_stride = p.b_stride == 0 ? vh : p.b_stride;
  if (a_stride < vh || b_stride < vh) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRuleGates: a_stride or b_stride is below v_heads");
  }
  const RoostDeviceSpan* all[] = {&p.a, &p.b, &p.a_log, &p.dt_bias, &p.g, &p.beta};
  for (const RoostDeviceSpan* s : all) {
    if (!FloatType(s->dtype)) {
      return Fail(ROOST_STATUS_UNSUPPORTED, "GatedDeltaRuleGates: every span must be bf16 or f32");
    }
  }
  const uint64_t total = uint64_t{p.tokens} * vh;
  const uint64_t a_reach = (uint64_t{p.tokens} - 1) * a_stride + vh;
  const uint64_t b_reach = (uint64_t{p.tokens} - 1) * b_stride + vh;
  if (!SpanOk(p.a, p.a.dtype, a_reach) || !SpanOk(p.b, p.b.dtype, b_reach) || !SpanOk(p.a_log, p.a_log.dtype, vh) ||
      !SpanOk(p.dt_bias, p.dt_bias.dtype, vh) || !SpanOk(p.g, p.g.dtype, total) || !SpanOk(p.beta, p.beta.dtype, total)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRuleGates: a span is too short, has a NULL address or reserved0 set");
  }
  for (const RoostDeviceSpan* s : all) {
    if (!Natural(*s)) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRuleGates: a span is not aligned to its element size");
    }
  }
  if (PartialOverlap(p.g, total, p.a, a_reach) || PartialOverlap(p.g, total, p.b, b_reach) ||
      PartialOverlap(p.beta, total, p.a, a_reach) || PartialOverlap(p.beta, total, p.b, b_reach) ||
      PartialOverlap(p.g, total, p.beta, total) || p.g.address == p.beta.address) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRuleGates: g and beta must not overlap each other or the inputs");
  }
  const uint64_t blocks = (total + 255) / 256;
  if (blocks > 0x7fffffffu) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRuleGates: too many elements for one call");
  }
  GatesKernel<<<static_cast<uint32_t>(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      p.a.address, p.b.address, p.a_log.address, p.dt_bias.address, p.g.address, p.beta.address, p.a.dtype, p.b.dtype,
      p.a_log.dtype, p.dt_bias.dtype, p.g.dtype, p.beta.dtype, static_cast<int64_t>(total), static_cast<int32_t>(vh),
      static_cast<int64_t>(a_stride), static_cast<int64_t>(b_stride));
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? ROOST_STATUS_OK : roost_hybrid::CudaFail(err);
}

extern "C" RoostStatus roost_gated_rms_norm_run(RoostContext* ctx, const void* params, const RoostWorkspace* /*ws*/,
                                                 void* stream) {
  using roost_hybrid::Fail;
  using roost_hybrid::SpanOk;
  using namespace roost_hybrid_gating;
  if (ctx == nullptr || params == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedRmsNorm: ctx or params is NULL");
  }
  const auto& p = *static_cast<const RoostGatedRmsNormParams*>(params);
  if (p.struct_size < sizeof(RoostGatedRmsNormParams) || p.abi_minor > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH, "GatedRmsNorm: params struct_size or abi_minor is not supported by this library");
  }
  if (p.reserved0 != 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedRmsNorm: reserved0 must be zero");
  }
  if (!(p.eps >= 0.0f) || !std::isfinite(p.eps)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedRmsNorm: eps must be finite and non-negative");
  }
  if (p.dim == 0 || p.dim > 0x7fffffffu || p.rows > 0x7fffffffu) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedRmsNorm: dim must be in [1, 2^31 - 1] and rows at most 2^31 - 1");
  }
  if (p.rows == 0) {
    return ROOST_STATUS_OK;
  }
  const uint32_t act = p.input.dtype;
  if (!FloatType(act) || !FloatType(p.weight.dtype)) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "GatedRmsNorm: activations and weight must be bf16 or f32");
  }
  const uint64_t n = uint64_t{p.rows} * p.dim;
  if (!SpanOk(p.input, act, n) || !SpanOk(p.gate, act, n) || !SpanOk(p.output, act, n) ||
      !SpanOk(p.weight, p.weight.dtype, p.dim)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT,
                "GatedRmsNorm: a span has the wrong dtype, is too short, has a NULL address or reserved0 set");
  }
  if (!Natural(p.input) || !Natural(p.gate) || !Natural(p.output) || !Natural(p.weight)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedRmsNorm: a span is not aligned to its element size");
  }
  if (PartialOverlap(p.output, n, p.input, n) || PartialOverlap(p.output, n, p.gate, n) ||
      PartialOverlap(p.output, n, p.weight, p.dim) || (p.output.address == p.weight.address)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedRmsNorm: output may alias input or gate only from the same address");
  }
  uint32_t threads = 32;
  while (threads < p.dim && threads < 1024) {
    threads *= 2;
  }
  GatedRmsNormKernel<<<p.rows, threads, threads * sizeof(float), static_cast<cudaStream_t>(stream)>>>(
      p.input.address, p.gate.address, p.weight.address, p.output.address, act, p.weight.dtype,
      static_cast<int32_t>(p.dim), p.eps);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? ROOST_STATUS_OK : roost_hybrid::CudaFail(err);
}

extern "C" RoostStatus roost_sigmoid_mul_run(RoostContext* ctx, const void* params, const RoostWorkspace* /*ws*/,
                                              void* stream) {
  using roost_hybrid::Fail;
  using roost_hybrid::SpanOk;
  using namespace roost_hybrid_gating;
  if (ctx == nullptr || params == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "SigmoidMul: ctx or params is NULL");
  }
  const auto& p = *static_cast<const RoostSigmoidMulParams*>(params);
  if (p.struct_size < sizeof(RoostSigmoidMulParams) || p.abi_minor > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH, "SigmoidMul: params struct_size or abi_minor is not supported by this library");
  }
  const uint64_t n = uint64_t{p.rows} * p.dim;
  if (n == 0) {
    return ROOST_STATUS_OK;
  }
  const uint32_t act = p.input.dtype;
  if (!FloatType(act)) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "SigmoidMul: activations must be bf16 or f32");
  }
  if (!SpanOk(p.input, act, n) || !SpanOk(p.gate, act, n) || !SpanOk(p.output, act, n)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT,
                "SigmoidMul: a span has the wrong dtype, is too short, has a NULL address or reserved0 set");
  }
  if (!Natural(p.input) || !Natural(p.gate) || !Natural(p.output)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "SigmoidMul: a span is not aligned to its element size");
  }
  if (PartialOverlap(p.output, n, p.input, n) || PartialOverlap(p.output, n, p.gate, n)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "SigmoidMul: output may alias input or gate only from the same address");
  }
  uint64_t blocks = (n + 255) / 256;
  if (blocks > (uint64_t{1} << 20)) {
    blocks = uint64_t{1} << 20;  // grid-stride loop covers the rest
  }
  SigmoidMulKernel<<<static_cast<uint32_t>(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      p.input.address, p.gate.address, p.output.address, act, static_cast<int64_t>(n));
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? ROOST_STATUS_OK : roost_hybrid::CudaFail(err);
}

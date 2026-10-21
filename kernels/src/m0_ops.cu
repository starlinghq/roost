// The non-attention, non-Gemm M0 ops (R-017): RmsNorm, QkNorm, Rope, SiluAndMul, Embedding, CacheWrite and CacheCopy.
//
// RmsNorm and QkNorm are FlashInfer's norm::RMSNorm and norm::QKRMSNorm, unmodified. SiluAndMul is FlashInfer's
// act_and_mul_kernel with a launcher written here (FlashInfer generates its launcher from a Python template, S2). Rope,
// Embedding, CacheWrite and CacheCopy are Roost's own kernels: FlashInfer's rotary kernel uses fast-math trigonometry whose
// error grows with the position (see Rope below), and C1's slot_mapping CacheWrite and block-pair CacheCopy do not match
// FlashInfer's append API (S2).
//
// Every call is a single kernel launch on the caller's stream: no allocation, no synchronisation, no workspace.
// A device-side bound that cannot be reported (an out-of-range token id, slot or block pair) makes the kernel skip or
// zero that item instead of reading or writing outside its buffer: a bad id never becomes a fault.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <numeric>

#include <flashinfer/activation.cuh>
#include <flashinfer/norm.cuh>

#include "op_common.h"

namespace roost_m0 {

using namespace roost_ops;

// ---- dtype dispatch ------------------------------------------------------------------------------------------------

template <typename F>
RoostStatus Dispatch16(uint32_t dtype, const char* op, F&& f) {
  switch (dtype) {
    case ROOST_DTYPE_BF16: return f(__nv_bfloat16{});
    case ROOST_DTYPE_F16: return f(__half{});
    default: return Fail(ROOST_STATUS_UNSUPPORTED, std::string(op) + ": dtype is not supported (bf16 and f16 only)");
  }
}

cudaStream_t ToStream(void* s) { return static_cast<cudaStream_t>(s); }

// FlashInfer loads 16-byte vectors of gcd(8, width) elements: every row start must be aligned to a whole vector.
RoostStatus CheckVectorAccess(const char* op, const char* name, const void* base, uint64_t stride_elems, uint32_t width,
                              uint32_t elem_bytes) {
  const uint32_t vec = std::gcd(16u / elem_bytes, width);
  const uintptr_t need = static_cast<uintptr_t>(vec) * elem_bytes;
  if (!Aligned(base, need) || (stride_elems % vec) != 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT,
                std::string(op) + ": " + name + " must be aligned to " + std::to_string(need) +
                    " bytes with a row stride that is a multiple of " + std::to_string(vec) + " elements");
  }
  return ROOST_STATUS_OK;
}

// ---- RmsNorm -------------------------------------------------------------------------------------------------------

RoostStatus RmsNormRun(RoostContext*, const void* params, const RoostWorkspace*, void* stream) {
  constexpr char kOp[] = "RmsNorm";
  if (auto s = CheckHeader(params, sizeof(RoostRmsNormParams), kOp)) return s;
  const auto& p = *static_cast<const RoostRmsNormParams*>(params);
  return Guarded(kOp, [&]() -> RoostStatus {
    if (p.reserved0 != 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "RmsNorm: reserved0 must be zero");
    if (p.dim == 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "RmsNorm: dim must be non-zero");
    const uint32_t in_stride = p.input_stride == 0 ? p.dim : p.input_stride;
    const uint32_t out_stride = p.output_stride == 0 ? p.dim : p.output_stride;
    if (in_stride < p.dim || out_stride < p.dim) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "RmsNorm: a row stride is smaller than dim");
    }
    if (p.rows == 0) return ROOST_STATUS_OK;
    const uint64_t in_len = (p.rows - 1ull) * in_stride + p.dim;
    const uint64_t out_len = (p.rows - 1ull) * out_stride + p.dim;
    return Dispatch16(p.input.dtype, kOp, [&](auto tag) -> RoostStatus {
      using T = decltype(tag);
      const uint32_t dt = p.input.dtype;
      if (auto s = CheckSpan(p.input, dt, in_len, kOp, "input")) return s;
      if (auto s = CheckSpan(p.weight, dt, p.dim, kOp, "weight")) return s;
      if (auto s = CheckSpan(p.output, dt, out_len, kOp, "output")) return s;
      if (auto s = CheckVectorAccess(kOp, "input", p.input.address, in_stride, p.dim, sizeof(T))) return s;
      if (auto s = CheckVectorAccess(kOp, "output", p.output.address, out_stride, p.dim, sizeof(T))) return s;
      if (auto s = CheckVectorAccess(kOp, "weight", p.weight.address, 0, p.dim, sizeof(T))) return s;
      return FromCuda(flashinfer::norm::RMSNorm(static_cast<T*>(p.input.address), static_cast<T*>(p.weight.address),
                                                static_cast<T*>(p.output.address), p.rows, p.dim, in_stride,
                                                out_stride, p.eps, false, ToStream(stream)));
    });
  });
}

// ---- QkNorm --------------------------------------------------------------------------------------------------------

RoostStatus QkNormRun(RoostContext*, const void* params, const RoostWorkspace*, void* stream) {
  constexpr char kOp[] = "QkNorm";
  if (auto s = CheckHeader(params, sizeof(RoostQkNormParams), kOp)) return s;
  const auto& p = *static_cast<const RoostQkNormParams*>(params);
  return Guarded(kOp, [&]() -> RoostStatus {
    if (p.reserved0 != 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "QkNorm: reserved0 must be zero");
    if (p.head_dim == 0 || p.q_heads == 0 || p.kv_heads == 0) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "QkNorm: head_dim, q_heads and kv_heads must be non-zero");
    }
    const uint32_t q_stride = p.q_stride == 0 ? p.q_heads * p.head_dim : p.q_stride;
    const uint32_t k_stride = p.k_stride == 0 ? p.kv_heads * p.head_dim : p.k_stride;
    if (q_stride < p.q_heads * p.head_dim || k_stride < p.kv_heads * p.head_dim) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "QkNorm: a token stride is smaller than heads * head_dim");
    }
    if (p.tokens == 0) return ROOST_STATUS_OK;
    const uint64_t q_len = (p.tokens - 1ull) * q_stride + static_cast<uint64_t>(p.q_heads) * p.head_dim;
    const uint64_t k_len = (p.tokens - 1ull) * k_stride + static_cast<uint64_t>(p.kv_heads) * p.head_dim;
    return Dispatch16(p.q.dtype, kOp, [&](auto tag) -> RoostStatus {
      using T = decltype(tag);
      const uint32_t dt = p.q.dtype;
      if (auto s = CheckSpan(p.q, dt, q_len, kOp, "q")) return s;
      if (auto s = CheckSpan(p.k, dt, k_len, kOp, "k")) return s;
      if (auto s = CheckSpan(p.q_weight, dt, p.head_dim, kOp, "q_weight")) return s;
      if (auto s = CheckSpan(p.k_weight, dt, p.head_dim, kOp, "k_weight")) return s;
      if (auto s = CheckVectorAccess(kOp, "q", p.q.address, q_stride, p.head_dim, sizeof(T))) return s;
      if (auto s = CheckVectorAccess(kOp, "k", p.k.address, k_stride, p.head_dim, sizeof(T))) return s;
      if (auto s = CheckVectorAccess(kOp, "q_weight", p.q_weight.address, 0, p.head_dim, sizeof(T))) return s;
      if (auto s = CheckVectorAccess(kOp, "k_weight", p.k_weight.address, 0, p.head_dim, sizeof(T))) return s;
      auto* q = static_cast<T*>(p.q.address);
      auto* k = static_cast<T*>(p.k.address);
      // In place: each warp reads a whole head into registers before it writes it back.
      if (auto e = flashinfer::norm::QKRMSNorm(q, static_cast<T*>(p.q_weight.address), q, p.tokens, p.q_heads,
                                               p.head_dim, q_stride, p.head_dim, q_stride, p.head_dim, p.eps, false,
                                               ToStream(stream))) {
        return FromCuda(e);
      }
      return FromCuda(flashinfer::norm::QKRMSNorm(k, static_cast<T*>(p.k_weight.address), k, p.tokens, p.kv_heads,
                                                  p.head_dim, k_stride, p.head_dim, k_stride, p.head_dim, p.eps,
                                                  false, ToStream(stream)));
    });
  });
}

// ---- Rope ----------------------------------------------------------------------------------------------------------

// Roost's own kernel, not FlashInfer's BatchQKApplyRotaryPosIds: that one forms the angle as float(pos) * powf-based
// frequency and calls the __sincosf intrinsic, which does no range reduction, so its error grows with the position. The
// error figures (7.8e-2 absolute on outputs of RMS 1 at positions up to 2^18, 1.6e-2 at positions up to 4096) are FAIL lines
// excerpted unedited in kernels/results/r017-rejected-configs-sm86.log, an excerpt of an earlier run whose full log was not
// committed; kernels/results/r017-ops-sm86.log has only the passing rope-roost lines. Here the
// frequency and angle are formed in double (as CpuOps does), the angle is reduced modulo 2 pi in double, and sincosf of the
// reduced angle gives cos and sin to about one fp32 ulp. The cost is a few double operations per (token, pair); the kernel is
// memory-bound.
//
// Pairs follow CpuOps: NeoX (i, i + R/2) or interleaved (2i, 2i + 1), R = rotary_dim; elements at and beyond R are untouched.
constexpr uint32_t kRopeMaxPairs = 256;
constexpr uint32_t kRopeTokensPerBlock = 4;

template <typename T>
__device__ __forceinline__ float RopeLoad(const T* p);
template <>
__device__ __forceinline__ float RopeLoad<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <>
__device__ __forceinline__ float RopeLoad<__half>(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void RopeStore(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void RopeStore(__half* p, float v) { *p = __float2half_rn(v); }

template <typename T>
__global__ void RopeKernel(T* q, T* k, const int32_t* positions, uint32_t tokens, uint32_t q_heads, uint32_t kv_heads,
                           uint32_t head_dim, uint32_t rot, bool interleaved, double theta, bool llama3, double factor,
                           double low_freq_factor, double high_freq_factor, double original_max) {
  __shared__ double freq[kRopeMaxPairs];
  const uint32_t half_rot = rot / 2;
  for (uint32_t i = threadIdx.x; i < half_rot; i += blockDim.x) {
    double f = pow(theta, -2.0 * i / rot);
    if (llama3) {
      const double wavelen = 2.0 * 3.14159265358979323846 / f;
      const double low = original_max / low_freq_factor;
      const double high = original_max / high_freq_factor;
      if (wavelen < high) {
        // unchanged
      } else if (wavelen > low) {
        f = f / factor;
      } else {
        const double smooth = (original_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        f = (1.0 - smooth) * f / factor + smooth * f;
      }
    }
    freq[i] = f;
  }
  __syncthreads();
  const uint32_t first = blockIdx.x * kRopeTokensPerBlock;
  const uint32_t last = min(first + kRopeTokensPerBlock, tokens);
  const uint32_t jobs = (last - first) * half_rot;
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  for (uint32_t job = threadIdx.x; job < jobs; job += blockDim.x) {
    const uint32_t tok = first + job / half_rot;
    const uint32_t i = job % half_rot;
    double angle = static_cast<double>(positions[tok]) * freq[i];
    angle -= kTwoPi * rint(angle / kTwoPi);
    float s, c;
    sincosf(static_cast<float>(angle), &s, &c);
    const uint32_t ia = interleaved ? 2 * i : i;
    const uint32_t ib = interleaved ? 2 * i + 1 : i + half_rot;
    T* qt = q + static_cast<uint64_t>(tok) * q_heads * head_dim;
    for (uint32_t h = 0; h < q_heads; ++h) {
      T* x = qt + static_cast<uint64_t>(h) * head_dim;
      const float x1 = RopeLoad(x + ia), x2 = RopeLoad(x + ib);
      RopeStore(x + ia, x1 * c - x2 * s);
      RopeStore(x + ib, x2 * c + x1 * s);
    }
    T* kt = k + static_cast<uint64_t>(tok) * kv_heads * head_dim;
    for (uint32_t h = 0; h < kv_heads; ++h) {
      T* x = kt + static_cast<uint64_t>(h) * head_dim;
      const float x1 = RopeLoad(x + ia), x2 = RopeLoad(x + ib);
      RopeStore(x + ia, x1 * c - x2 * s);
      RopeStore(x + ib, x2 * c + x1 * s);
    }
  }
}

RoostStatus RopeRun(RoostContext*, const void* params, const RoostWorkspace*, void* stream) {
  constexpr char kOp[] = "Rope";
  if (auto s = CheckHeader(params, sizeof(RoostRopeParams), kOp)) return s;
  const auto& p = *static_cast<const RoostRopeParams*>(params);
  return Guarded(kOp, [&]() -> RoostStatus {
    if (p.kind != ROOST_ROPE_DEFAULT && p.kind != ROOST_ROPE_LLAMA3) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Rope: kind must be DEFAULT or LLAMA3");
    }
    if (p.head_dim == 0 || p.q_heads == 0 || p.kv_heads == 0) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Rope: head_dim, q_heads and kv_heads must be non-zero");
    }
    const uint32_t rot = p.rotary_dim == 0 ? p.head_dim : p.rotary_dim;
    if (rot > p.head_dim || (rot & 1u) != 0 || rot == 0) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Rope: rotary_dim must be even and at most head_dim");
    }
    if (rot / 2 > kRopeMaxPairs) {
      return Fail(ROOST_STATUS_UNSUPPORTED, "Rope: rotary_dim above 512 is not built");
    }
    if (!(p.theta > 0.0f)) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Rope: theta must be positive");
    if (p.kind == ROOST_ROPE_LLAMA3 &&
        (!(p.llama3_factor > 0.0f) || !(p.llama3_low_freq_factor > 0.0f) ||
         !(p.llama3_high_freq_factor > p.llama3_low_freq_factor) || p.llama3_original_max_position == 0)) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Rope: llama3 needs factor > 0, 0 < low < high and a positive max");
    }
    if (p.tokens == 0) return ROOST_STATUS_OK;
    const uint64_t q_len = static_cast<uint64_t>(p.tokens) * p.q_heads * p.head_dim;
    const uint64_t k_len = static_cast<uint64_t>(p.tokens) * p.kv_heads * p.head_dim;
    return Dispatch16(p.q.dtype, kOp, [&](auto tag) -> RoostStatus {
      using T = decltype(tag);
      const uint32_t dt = p.q.dtype;
      if (auto s = CheckSpan(p.q, dt, q_len, kOp, "q")) return s;
      if (auto s = CheckSpan(p.k, dt, k_len, kOp, "k")) return s;
      if (auto s = CheckSpan(p.positions, ROOST_DTYPE_I32, p.tokens, kOp, "positions")) return s;
      if (!Aligned(p.q.address, 2) || !Aligned(p.k.address, 2)) {
        return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Rope: q and k must be 2-byte aligned");
      }
      const uint32_t blocks = (p.tokens + kRopeTokensPerBlock - 1) / kRopeTokensPerBlock;
      RopeKernel<T><<<blocks, 128, 0, ToStream(stream)>>>(
          static_cast<T*>(p.q.address), static_cast<T*>(p.k.address), static_cast<const int32_t*>(p.positions.address),
          p.tokens, p.q_heads, p.kv_heads, p.head_dim, rot, p.interleaved != 0, static_cast<double>(p.theta),
          p.kind == ROOST_ROPE_LLAMA3, p.llama3_factor, p.llama3_low_freq_factor, p.llama3_high_freq_factor,
          static_cast<double>(p.llama3_original_max_position));
      return FromCuda(cudaGetLastError());
    });
  });
}

// ---- SiluAndMul ----------------------------------------------------------------------------------------------------

__device__ float Silu(const float& v) { return v / (1.0f + expf(-v)); }

RoostStatus SiluAndMulRun(RoostContext*, const void* params, const RoostWorkspace*, void* stream) {
  constexpr char kOp[] = "SiluAndMul";
  if (auto s = CheckHeader(params, sizeof(RoostActAndMulParams), kOp)) return s;
  const auto& p = *static_cast<const RoostActAndMulParams*>(params);
  return Guarded(kOp, [&]() -> RoostStatus {
    if (p.reserved0 != 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "SiluAndMul: reserved0 must be zero");
    if (p.dim == 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "SiluAndMul: dim must be non-zero");
    if (p.rows == 0) return ROOST_STATUS_OK;
    return Dispatch16(p.input.dtype, kOp, [&](auto tag) -> RoostStatus {
      using T = decltype(tag);
      const uint32_t dt = p.input.dtype;
      if (auto s = CheckSpan(p.input, dt, 2ull * p.rows * p.dim, kOp, "input")) return s;
      if (auto s = CheckSpan(p.output, dt, static_cast<uint64_t>(p.rows) * p.dim, kOp, "output")) return s;
      // The launcher picks the largest vector that divides dim, so every access stays aligned (FlashInfer notes).
      const uint32_t vec = std::gcd(16u / static_cast<uint32_t>(sizeof(T)), p.dim);
      const uintptr_t need = static_cast<uintptr_t>(vec) * sizeof(T);
      if (!Aligned(p.input.address, need) || !Aligned(p.output.address, need)) {
        return Fail(ROOST_STATUS_INVALID_ARGUMENT, "SiluAndMul: buffers must be aligned to the vector width");
      }
      const uint32_t block = std::min<uint32_t>(256, std::max<uint32_t>(1, p.dim / vec));
      auto* out = static_cast<T*>(p.output.address);
      const auto* in = static_cast<const T*>(p.input.address);
      const auto s = ToStream(stream);
      const int d = static_cast<int>(p.dim);
      switch (vec) {
        case 8:
          flashinfer::activation::act_and_mul_kernel<T, Silu, 8><<<p.rows, block, 0, s>>>(out, in, d);
          break;
        case 4:
          flashinfer::activation::act_and_mul_kernel<T, Silu, 4><<<p.rows, block, 0, s>>>(out, in, d);
          break;
        case 2:
          flashinfer::activation::act_and_mul_kernel<T, Silu, 2><<<p.rows, block, 0, s>>>(out, in, d);
          break;
        default:
          flashinfer::activation::act_and_mul_kernel<T, Silu, 1><<<p.rows, block, 0, s>>>(out, in, d);
          break;
      }
      return FromCuda(cudaGetLastError());
    });
  });
}

// ---- Embedding -----------------------------------------------------------------------------------------------------

// One block per token. An id outside [0, vocab) writes a zero row.
__global__ void EmbeddingKernel(const int32_t* ids, const uint8_t* table, uint8_t* out, uint32_t vocab,
                                uint64_t row_bytes, uint32_t unit) {
  const uint32_t token = blockIdx.x;
  const int32_t id = ids[token];
  const bool ok = id >= 0 && static_cast<uint32_t>(id) < vocab;
  const uint8_t* src = table + static_cast<uint64_t>(ok ? id : 0) * row_bytes;
  uint8_t* dst = out + static_cast<uint64_t>(token) * row_bytes;
  if (unit == 16) {
    for (uint64_t i = threadIdx.x; i < row_bytes / 16; i += blockDim.x) {
      const uint4 v = ok ? reinterpret_cast<const uint4*>(src)[i] : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4*>(dst)[i] = v;
    }
  } else if (unit == 4) {
    for (uint64_t i = threadIdx.x; i < row_bytes / 4; i += blockDim.x) {
      reinterpret_cast<uint32_t*>(dst)[i] = ok ? reinterpret_cast<const uint32_t*>(src)[i] : 0u;
    }
  } else {
    for (uint64_t i = threadIdx.x; i < row_bytes / 2; i += blockDim.x) {
      reinterpret_cast<uint16_t*>(dst)[i] = ok ? reinterpret_cast<const uint16_t*>(src)[i] : uint16_t{0};
    }
  }
}

RoostStatus EmbeddingRun(RoostContext*, const void* params, const RoostWorkspace*, void* stream) {
  constexpr char kOp[] = "Embedding";
  if (auto s = CheckHeader(params, sizeof(RoostEmbeddingParams), kOp)) return s;
  const auto& p = *static_cast<const RoostEmbeddingParams*>(params);
  return Guarded(kOp, [&]() -> RoostStatus {
    if (p.reserved0 != 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Embedding: reserved0 must be zero");
    const uint32_t es = DtypeBytes(p.table.dtype);
    if (p.table.dtype != ROOST_DTYPE_BF16 && p.table.dtype != ROOST_DTYPE_F16 && p.table.dtype != ROOST_DTYPE_F32) {
      return Fail(ROOST_STATUS_UNSUPPORTED, "Embedding: table dtype must be bf16, f16 or f32");
    }
    if (p.dim == 0 || p.vocab == 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Embedding: dim and vocab must be non-zero");
    if (p.tokens == 0) return ROOST_STATUS_OK;
    if (auto s = CheckSpan(p.ids, ROOST_DTYPE_I32, p.tokens, kOp, "ids")) return s;
    if (auto s = CheckSpan(p.table, p.table.dtype, static_cast<uint64_t>(p.vocab) * p.dim, kOp, "table")) return s;
    if (auto s = CheckSpan(p.output, p.table.dtype, static_cast<uint64_t>(p.tokens) * p.dim, kOp, "output")) return s;
    const uint64_t row_bytes = static_cast<uint64_t>(p.dim) * es;
    uint32_t unit = es == 4 ? 4 : 2;
    if (row_bytes % 16 == 0 && Aligned(p.table.address, 16) && Aligned(p.output.address, 16)) {
      unit = 16;
    } else if (row_bytes % 4 == 0 && Aligned(p.table.address, 4) && Aligned(p.output.address, 4)) {
      unit = 4;
    } else if (!Aligned(p.table.address, 2) || !Aligned(p.output.address, 2)) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Embedding: table and output must be aligned to the element size");
    }
    EmbeddingKernel<<<p.tokens, 256, 0, ToStream(stream)>>>(static_cast<const int32_t*>(p.ids.address),
                                                            static_cast<const uint8_t*>(p.table.address),
                                                            static_cast<uint8_t*>(p.output.address), p.vocab,
                                                            row_bytes, unit);
    return FromCuda(cudaGetLastError());
  });
}

// ---- CacheWrite ----------------------------------------------------------------------------------------------------

// One block per token. slot = page * page_size + offset; a negative slot skips the token, and so does a slot beyond the
// pool. Bit-exact copy: the activation and cache dtypes are the same (quantised KV is R-043's).
__global__ void CacheWriteKernel(const uint8_t* k, const uint8_t* v, uint8_t* k_cache, uint8_t* v_cache,
                                 const int64_t* slots, uint64_t pool_rows, uint32_t kv_heads, uint32_t head_dim,
                                 uint32_t page_size, bool hnd, uint32_t elem_bytes) {
  const uint32_t token = blockIdx.x;
  const int64_t slot = slots[token];
  if (slot < 0) return;
  const uint64_t block = static_cast<uint64_t>(slot) / page_size;
  const uint64_t off = static_cast<uint64_t>(slot) % page_size;
  // pool_rows = number of (page, offset) rows the pool holds.
  if (block * page_size + off >= pool_rows) return;
  const uint64_t head_bytes = static_cast<uint64_t>(head_dim) * elem_bytes;
  for (uint64_t i = threadIdx.x; i < static_cast<uint64_t>(kv_heads) * (head_bytes / 2); i += blockDim.x) {
    const uint64_t h = i / (head_bytes / 2);
    const uint64_t w = i % (head_bytes / 2);  // 16-bit word inside the head
    const uint64_t row = hnd ? ((block * kv_heads + h) * page_size + off) : ((block * page_size + off) * kv_heads + h);
    const uint64_t src = (static_cast<uint64_t>(token) * kv_heads + h) * (head_bytes / 2) + w;
    const uint64_t dst = row * (head_bytes / 2) + w;
    reinterpret_cast<uint16_t*>(k_cache)[dst] = reinterpret_cast<const uint16_t*>(k)[src];
    reinterpret_cast<uint16_t*>(v_cache)[dst] = reinterpret_cast<const uint16_t*>(v)[src];
  }
}

RoostStatus CacheWriteRun(RoostContext*, const void* params, const RoostWorkspace*, void* stream) {
  constexpr char kOp[] = "CacheWrite";
  if (auto s = CheckHeader(params, sizeof(RoostCacheWriteParams), kOp)) return s;
  const auto& p = *static_cast<const RoostCacheWriteParams*>(params);
  return Guarded(kOp, [&]() -> RoostStatus {
    if (p.reserved0 != 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CacheWrite: reserved0 must be zero");
    if (p.kv_layout != ROOST_LAYOUT_PAGED_NHD && p.kv_layout != ROOST_LAYOUT_PAGED_HND) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CacheWrite: kv_layout must be PAGED_NHD or PAGED_HND");
    }
    if (p.kv_heads == 0 || p.head_dim == 0 || p.page_size == 0) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CacheWrite: kv_heads, head_dim and page_size must be non-zero");
    }
    const uint32_t dt = p.k.dtype;
    if (dt != ROOST_DTYPE_BF16 && dt != ROOST_DTYPE_F16 && dt != ROOST_DTYPE_F32) {
      return Fail(ROOST_STATUS_UNSUPPORTED, "CacheWrite: k dtype must be bf16, f16 or f32");
    }
    if (p.k_cache.dtype != dt || p.v_cache.dtype != dt || p.v.dtype != dt) {
      return Fail(ROOST_STATUS_UNSUPPORTED, "CacheWrite: quantised KV (cache dtype != activation dtype) is not built");
    }
    if (p.k_scale != 1.0f || p.v_scale != 1.0f) {
      return Fail(ROOST_STATUS_UNSUPPORTED, "CacheWrite: k_scale and v_scale must be 1 without quantised KV");
    }
    if (p.tokens == 0) return ROOST_STATUS_OK;
    const uint64_t per_token = static_cast<uint64_t>(p.kv_heads) * p.head_dim;  // elements per (page, offset) row
    if (auto s = CheckSpan(p.k, dt, p.tokens * per_token, kOp, "k")) return s;
    if (auto s = CheckSpan(p.v, dt, p.tokens * per_token, kOp, "v")) return s;
    if (auto s = CheckSpan(p.k_cache, dt, per_token, kOp, "k_cache")) return s;
    if (auto s = CheckSpan(p.v_cache, dt, per_token, kOp, "v_cache")) return s;
    if (auto s = CheckSpan(p.slot_mapping, ROOST_DTYPE_I64, p.tokens, kOp, "slot_mapping")) return s;
    const uint64_t pool_rows = std::min(p.k_cache.length, p.v_cache.length) / per_token;
    if ((per_token * DtypeBytes(dt)) % 2 != 0 || !Aligned(p.k.address, 2) || !Aligned(p.v.address, 2) ||
        !Aligned(p.k_cache.address, 2) || !Aligned(p.v_cache.address, 2)) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CacheWrite: buffers must be 2-byte aligned");
    }
    CacheWriteKernel<<<p.tokens, 256, 0, ToStream(stream)>>>(
        static_cast<const uint8_t*>(p.k.address), static_cast<const uint8_t*>(p.v.address),
        static_cast<uint8_t*>(p.k_cache.address), static_cast<uint8_t*>(p.v_cache.address),
        static_cast<const int64_t*>(p.slot_mapping.address), pool_rows, p.kv_heads, p.head_dim, p.page_size,
        p.kv_layout == ROOST_LAYOUT_PAGED_HND, DtypeBytes(dt));
    return FromCuda(cudaGetLastError());
  });
}

// ---- CacheCopy -----------------------------------------------------------------------------------------------------

// One block per (src, dst) pair. Pairs of one call must be independent: no pair's destination is another pair's source or
// destination (the CPU reference copies in pair order). A pair outside either pool is skipped.
__global__ void CacheCopyKernel(const uint8_t* src_pool, uint8_t* dst_pool, const int32_t* pairs, uint64_t block_bytes,
                                uint64_t src_bytes, uint64_t dst_bytes, uint32_t unit) {
  const int64_t s = pairs[2ull * blockIdx.x];
  const int64_t d = pairs[2ull * blockIdx.x + 1];
  if (s < 0 || d < 0 || (static_cast<uint64_t>(s) + 1) * block_bytes > src_bytes ||
      (static_cast<uint64_t>(d) + 1) * block_bytes > dst_bytes) {
    return;
  }
  if (src_pool == dst_pool && s == d) return;
  const uint8_t* src = src_pool + static_cast<uint64_t>(s) * block_bytes;
  uint8_t* dst = dst_pool + static_cast<uint64_t>(d) * block_bytes;
  if (unit == 16) {
    for (uint64_t i = threadIdx.x; i < block_bytes / 16; i += blockDim.x) {
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
    }
  } else {
    for (uint64_t i = threadIdx.x; i < block_bytes; i += blockDim.x) {
      dst[i] = src[i];
    }
  }
}

RoostStatus CacheCopyRun(RoostContext*, const void* params, const RoostWorkspace*, void* stream) {
  constexpr char kOp[] = "CacheCopy";
  if (auto s = CheckHeader(params, sizeof(RoostCacheCopyParams), kOp)) return s;
  const auto& p = *static_cast<const RoostCacheCopyParams*>(params);
  return Guarded(kOp, [&]() -> RoostStatus {
    if (p.reserved0 != 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CacheCopy: reserved0 must be zero");
    if (p.count == 0) return ROOST_STATUS_OK;
    if (p.block_bytes == 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "CacheCopy: block_bytes must be non-zero");
    const uint32_t se = DtypeBytes(p.src_pool.dtype), de = DtypeBytes(p.dst_pool.dtype);
    if (se == 0 || de == 0) return Fail(ROOST_STATUS_UNSUPPORTED, "CacheCopy: pool dtype has no byte size");
    if (auto s = CheckSpan(p.src_pool, p.src_pool.dtype, 1, kOp, "src_pool")) return s;
    if (auto s = CheckSpan(p.dst_pool, p.dst_pool.dtype, 1, kOp, "dst_pool")) return s;
    if (auto s = CheckSpan(p.pairs, ROOST_DTYPE_I32, 2ull * p.count, kOp, "pairs")) return s;
    const uint32_t unit = (p.block_bytes % 16 == 0 && Aligned(p.src_pool.address, 16) && Aligned(p.dst_pool.address, 16))
                              ? 16
                              : 1;
    CacheCopyKernel<<<p.count, 256, 0, ToStream(stream)>>>(
        static_cast<const uint8_t*>(p.src_pool.address), static_cast<uint8_t*>(p.dst_pool.address),
        static_cast<const int32_t*>(p.pairs.address), p.block_bytes, p.src_pool.length * se, p.dst_pool.length * de,
        unit);
    return FromCuda(cudaGetLastError());
  });
}

}  // namespace roost_m0

extern "C" {
RoostStatus roost_rms_norm_run(RoostContext* c, const void* p, const RoostWorkspace* w, void* s) {
  return roost_m0::RmsNormRun(c, p, w, s);
}
RoostStatus roost_qk_norm_run(RoostContext* c, const void* p, const RoostWorkspace* w, void* s) {
  return roost_m0::QkNormRun(c, p, w, s);
}
RoostStatus roost_rope_run(RoostContext* c, const void* p, const RoostWorkspace* w, void* s) {
  return roost_m0::RopeRun(c, p, w, s);
}
RoostStatus roost_silu_and_mul_run(RoostContext* c, const void* p, const RoostWorkspace* w, void* s) {
  return roost_m0::SiluAndMulRun(c, p, w, s);
}
RoostStatus roost_embedding_run(RoostContext* c, const void* p, const RoostWorkspace* w, void* s) {
  return roost_m0::EmbeddingRun(c, p, w, s);
}
RoostStatus roost_cache_write_run(RoostContext* c, const void* p, const RoostWorkspace* w, void* s) {
  return roost_m0::CacheWriteRun(c, p, w, s);
}
RoostStatus roost_cache_copy_run(RoostContext* c, const void* p, const RoostWorkspace* w, void* s) {
  return roost_m0::CacheCopyRun(c, p, w, s);
}
}

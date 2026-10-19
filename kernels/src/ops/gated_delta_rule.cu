// GatedDeltaRule (ROOST_OP_GATED_DELTA_RULE) for Qwen3.5-0.8B (R-034): the S3 Triton AOT kernels of
// flash-linear-attention behind the C ABI, reading q, k and v through row-strided views and the recurrent state from
// pool slots. CpuOps.GatedDeltaRule is the exact definition (docs/m0/roost-s3-triton-aot-gdn.md for the kernels).
//
// Two variants. "gated-delta-rule-chunk-aot" is FLA's chunked prefill (cumsum, kkt+solve, w/u, chunk state, chunk
// output); it needs plan() for the chunk tables. "gated-delta-rule-recurrent-aot" is FLA's fused recurrent kernel
// (decode and short continuations); it needs no plan and can be captured in a CUDA graph.
//
// The stock cubins read q, k and v densely and the state from a dense [sequence, head, k, v] tensor. Around them this
// file gathers the strided views into workspace (l2-normalising q and k for the chunk path, as FLA's l2norm kernel
// does), gathers each sequence's state from its pool slot and scatters the final state back, all on the caller's stream.
// A kernel patch that reads the views and the slots directly is the follow-up this costs (about 2 MiB of state traffic
// per sequence and layer). The cubins are loaded when the context is created, never inside a call.
//
// Specialised: 16 key and value heads, head size 128, bf16 q/k/v/beta/output, float32 g and state, l2 norm of q and k
// inside the op. Anything else returns ROOST_STATUS_UNSUPPORTED.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "gdn_aot.h"
#include "hybrid_common.h"

namespace roost_gdn_op {

using roost_gdn::GdnKernel;
using roost_gdn::GdnRole;
using roost_hybrid::Fail;

constexpr uint32_t kHeads = 16;
constexpr uint32_t kDim = 128;
constexpr uint32_t kRow = kHeads * kDim;      // elements per q, k, v and output row
constexpr uint32_t kChunk = 64;               // FLA chunk size (BT)
constexpr uint64_t kStateElems = uint64_t{kHeads} * kDim * kDim;  // floats per sequence
constexpr uint64_t kMaxTokens = uint64_t{1} << 24;
constexpr uint32_t kPlanMagic = 0x504e4447;   // "GDNP"

// What the table of a build must say about each kernel's run-time arguments; the launch code below passes them in
// exactly this order. A mismatch means the specs and this file drifted apart, so the context refuses to load.
const char* const kExpectedSignature[roost_gdn::kGdnKernelCount] = {
    "s:*fp32,o:*fp32,scale:fp32,cu_seqlens:*i32,chunk_indices:*i32,T:i32",
    "k:*bf16,g:*fp32,beta:*bf16,A:*bf16,cu_seqlens:*i32,chunk_indices:*i32,T:i32",
    "k:*bf16,v:*bf16,beta:*bf16,w:*bf16,u:*bf16,A:*bf16,g:*fp32,cu_seqlens:*i32,chunk_indices:*i32,T:i32",
    "k:*bf16,v:*bf16,w:*bf16,v_new:*bf16,g:*fp32,h:*bf16,h0:*fp32,ht:*fp32,cu_seqlens:*i32,chunk_offsets:*i64,T:i32",
    "k:*bf16,v:*bf16,w:*bf16,v_new:*bf16,g:*fp32,h:*bf16,ht:*fp32,cu_seqlens:*i32,chunk_offsets:*i64,T:i32",
    "q:*bf16,k:*bf16,v:*bf16,h:*bf16,g:*fp32,o:*bf16,cu_seqlens:*i32,chunk_indices:*i32,scale:fp32,T:i32",
    "q:*bf16,k:*bf16,v:*bf16,g:*fp32,beta:*bf16,o:*bf16,h0:*fp32,ht:*fp32,cu_seqlens:*i32,scale:fp32,T:i32",
};

struct Loaded {
  cudaLibrary_t library[roost_gdn::kGdnKernelCount];
  cudaKernel_t kernel[roost_gdn::kGdnKernelCount];
  int count;
};

// ---- Workspace layouts ----------------------------------------------------------------------------------------------

uint64_t Align(uint64_t x, uint64_t a = 256) { return (x + a - 1) / a * a; }

// Chunk variant. The first three regions are the tables plan() writes; the rest is scratch run() owns.
struct ChunkLayout {
  uint64_t cu, chunk_indices, chunk_offsets, tables_bytes;
  uint64_t qn, kn, vc, gc, a, w, u, vnew, h, h0, ht, total;
  uint32_t nt_max;
};

ChunkLayout MakeChunkLayout(uint64_t tokens, uint64_t batch) {
  ChunkLayout l{};
  l.nt_max = static_cast<uint32_t>(tokens / kChunk + batch);
  uint64_t p = 0;
  auto take = [&p](uint64_t bytes) {
    const uint64_t at = p;
    p += Align(bytes);
    return at;
  };
  l.cu = take(4 * (batch + 1));
  l.chunk_indices = take(8 * uint64_t{l.nt_max});
  l.chunk_offsets = take(8 * (batch + 1));
  l.tables_bytes = p;
  l.qn = take(tokens * kRow * 2);
  l.kn = take(tokens * kRow * 2);
  l.vc = take(tokens * kRow * 2);
  l.gc = take(tokens * kHeads * 4);
  l.a = take(tokens * kHeads * kChunk * 2);
  l.w = take(tokens * kRow * 2);
  l.u = take(tokens * kRow * 2);
  l.vnew = take(tokens * kRow * 2);
  l.h = take(uint64_t{l.nt_max} * kStateElems * 2);
  l.h0 = take(batch * kStateElems * 4);
  l.ht = take(batch * kStateElems * 4);
  l.total = p;
  return l;
}

struct RecurrentLayout {
  uint64_t cu, q, k, v, h0, ht, total;
};

RecurrentLayout MakeRecurrentLayout(uint64_t tokens, uint64_t batch) {
  RecurrentLayout l{};
  uint64_t p = 0;
  auto take = [&p](uint64_t bytes) {
    const uint64_t at = p;
    p += Align(bytes);
    return at;
  };
  l.cu = take(4 * (batch + 1));
  l.q = take(tokens * kRow * 2);
  l.k = take(tokens * kRow * 2);
  l.v = take(tokens * kRow * 2);
  l.h0 = take(batch * kStateElems * 4);
  l.ht = take(batch * kStateElems * 4);
  l.total = p;
  return l;
}

struct PlanRecord {
  uint32_t magic;
  uint32_t batch;
  uint32_t tokens;
  uint32_t nt;
};

// ---- Gather and scatter kernels -------------------------------------------------------------------------------------

// dst[t, h, :] = (norm ? src_row / sqrt(sum(src_row^2) + 1e-6) : src_row) for a [tokens, 16, 128] view whose rows are
// `stride` elements apart. One warp per (token, head); lanes take elements lane, lane + 32, ... so no alignment is needed.
// blockIdx.y selects q, k or v.
struct Views {
  const __nv_bfloat16* src[3];
  uint64_t stride[3];
  __nv_bfloat16* dst[3];
  int norm[3];
};

__global__ void GatherViewsKernel(Views views, uint32_t tokens) {
  const int which = blockIdx.y;
  const uint32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= tokens * kHeads) {
    return;
  }
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t t = row / kHeads;
  const uint32_t h = row % kHeads;
  const __nv_bfloat16* s = views.src[which] + static_cast<uint64_t>(t) * views.stride[which] + h * kDim;
  float x[kDim / 32];
  float ss = 0.0f;
  for (int i = 0; i < kDim / 32; ++i) {
    x[i] = __bfloat162float(s[lane + 32 * i]);
    ss += x[i] * x[i];
  }
  float r = 1.0f;
  if (views.norm[which] != 0) {
    for (int off = 16; off > 0; off >>= 1) {
      ss += __shfl_xor_sync(0xffffffffu, ss, off);
    }
    r = 1.0f / sqrtf(ss + 1e-6f);
  }
  __nv_bfloat16* d = views.dst[which] + static_cast<uint64_t>(row) * kDim;
  for (int i = 0; i < kDim / 32; ++i) {
    d[lane + 32 * i] = __float2bfloat16_rn(x[i] * r);
  }
}

// dense[b] = pool[slots[b]] for sequences with tokens (empty or negative-slot sequences are skipped: their slot may be
// unset). blockIdx.x is the sequence (x allows 2^31 - 1 sequences, y only 65535); blockIdx.y strides the state.
__global__ void GatherStateKernel(const float4* pool, const int32_t* slots, const int32_t* cu, float4* dense) {
  const int b = blockIdx.x;
  const int32_t slot = slots[b];
  if (cu[b + 1] <= cu[b] || slot < 0) {
    return;
  }
  const uint64_t n = kStateElems / 4;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<uint64_t>(gridDim.y) * blockDim.x) {
    dense[static_cast<uint64_t>(b) * n + i] = pool[static_cast<uint64_t>(slot) * n + i];
  }
}

// pool[slots[b]] = dense[b], same skip rule.
__global__ void ScatterStateKernel(const float4* dense, const int32_t* slots, const int32_t* cu, float4* pool) {
  const int b = blockIdx.x;
  const int32_t slot = slots[b];
  if (cu[b + 1] <= cu[b] || slot < 0) {
    return;
  }
  const uint64_t n = kStateElems / 4;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<uint64_t>(gridDim.y) * blockDim.x) {
    pool[static_cast<uint64_t>(slot) * n + i] = dense[static_cast<uint64_t>(b) * n + i];
  }
}

// ---- Launch helpers -------------------------------------------------------------------------------------------------

RoostStatus LaunchTriton(const Loaded& lib, GdnRole role, dim3 grid, void** args, cudaStream_t stream) {
  const GdnKernel& k = roost_gdn::kGdnKernels[role];
  const cudaError_t err = cudaLaunchKernel(reinterpret_cast<const void*>(lib.kernel[role]), grid, dim3(k.threads), args,
                                           k.shared_bytes, stream);
  if (err != cudaSuccess) {
    return roost_hybrid::CudaFail(err);
  }
  return ROOST_STATUS_OK;
}

#define ROOST_CHECK_CUDA(expr)                          \
  do {                                                  \
    const cudaError_t e_ = (expr);                      \
    if (e_ != cudaSuccess) {                            \
      return roost_hybrid::CudaFail(e_);                \
    }                                                   \
  } while (0)

#define ROOST_CHECK(expr)                               \
  do {                                                  \
    const RoostStatus s_ = (expr);                      \
    if (s_ != ROOST_STATUS_OK) {                        \
      return s_;                                        \
    }                                                   \
  } while (0)

uint32_t Stride(uint32_t stride, uint32_t row) { return stride == 0 ? row : stride; }

// The checks both variants share. Returns OK, or the status and message to give back.
RoostStatus Validate(const RoostGatedDeltaRuleParams& p, bool need_host_cu) {
  using roost_hybrid::SpanOk;
  if (p.struct_size < sizeof(RoostGatedDeltaRuleParams) || p.abi_minor > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH, "GatedDeltaRule: params struct_size or abi_minor is not supported by this library");
  }
  if (p.k_heads != kHeads || p.v_heads != kHeads || p.k_dim != kDim || p.v_dim != kDim) {
    return Fail(ROOST_STATUS_UNSUPPORTED,
                "GatedDeltaRule: this library is specialised for 16 key heads, 16 value heads and head size 128 (Qwen3.5-0.8B)");
  }
  if (p.l2_norm_qk == 0) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "GatedDeltaRule: only l2_norm_qk != 0 is built (Qwen3.5 normalises q and k)");
  }
  if (p.tokens > kMaxTokens || p.batch > (1u << 20)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: too many tokens or sequences for one call");
  }
  const uint32_t qs = Stride(p.q_stride, kRow);
  const uint32_t ks = Stride(p.k_stride, kRow);
  const uint32_t vs = Stride(p.v_stride, kRow);
  if (qs < kRow || ks < kRow || vs < kRow) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: a q, k or v stride is smaller than the row width");
  }
  const uint64_t t = p.tokens;
  const uint64_t view_tail = t == 0 ? 0 : (t - 1);
  if (!SpanOk(p.q, ROOST_DTYPE_BF16, view_tail * qs + kRow) || !SpanOk(p.k, ROOST_DTYPE_BF16, view_tail * ks + kRow) ||
      !SpanOk(p.v, ROOST_DTYPE_BF16, view_tail * vs + kRow)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT,
                "GatedDeltaRule: q, k and v must be bf16 spans reaching (tokens - 1) * stride + 2048 elements");
  }
  if (!SpanOk(p.g, ROOST_DTYPE_F32, t * kHeads) || !SpanOk(p.beta, ROOST_DTYPE_BF16, t * kHeads) ||
      !SpanOk(p.output, ROOST_DTYPE_BF16, t * kRow)) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "GatedDeltaRule: g must be float32, beta and output bf16, each long enough");
  }
  if (!SpanOk(p.state, ROOST_DTYPE_F32, kStateElems) || p.state.length % kStateElems != 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: state must be a float32 pool of whole [16, 128, 128] slots");
  }
  if (!SpanOk(p.state_slots, ROOST_DTYPE_I32, p.batch) || !SpanOk(p.cu_seqlens, ROOST_DTYPE_I32, uint64_t{p.batch} + 1)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: state_slots and cu_seqlens must be int32 spans of batch and batch + 1");
  }
  // The Triton cubins were specialised for 16-byte aligned pointers; the state kernels move float4.
  if (!roost_hybrid::Aligned(p.g.address, 16) || !roost_hybrid::Aligned(p.beta.address, 16) ||
      !roost_hybrid::Aligned(p.output.address, 16) || !roost_hybrid::Aligned(p.state.address, 16)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: g, beta, output and state must be 16-byte aligned");
  }
  if (need_host_cu && p.cu_seqlens_host == nullptr && p.batch > 0 && p.tokens > 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: plan() needs cu_seqlens_host");
  }
  return ROOST_STATUS_OK;
}

float Scale(const RoostGatedDeltaRuleParams& p) { return p.scale == 0.0f ? 0.08838834764831845f : p.scale; }

RoostStatus GatherViews(const RoostGatedDeltaRuleParams& p, __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v, bool norm,
                        cudaStream_t stream) {
  Views views{};
  views.src[0] = static_cast<const __nv_bfloat16*>(p.q.address);
  views.src[1] = static_cast<const __nv_bfloat16*>(p.k.address);
  views.src[2] = static_cast<const __nv_bfloat16*>(p.v.address);
  views.stride[0] = Stride(p.q_stride, kRow);
  views.stride[1] = Stride(p.k_stride, kRow);
  views.stride[2] = Stride(p.v_stride, kRow);
  views.dst[0] = q;
  views.dst[1] = k;
  views.dst[2] = v;
  views.norm[0] = norm ? 1 : 0;
  views.norm[1] = norm ? 1 : 0;
  views.norm[2] = 0;
  constexpr uint32_t kWarps = 4;
  const dim3 grid(roost_hybrid::CeilDiv(uint64_t{p.tokens} * kHeads, kWarps), 3);
  GatherViewsKernel<<<grid, kWarps * 32, 0, stream>>>(views, p.tokens);
  ROOST_CHECK_CUDA(cudaGetLastError());
  return ROOST_STATUS_OK;
}

RoostStatus StateMove(bool scatter, const void* a, const RoostGatedDeltaRuleParams& p, const int32_t* cu, void* b,
                      cudaStream_t stream) {
  const dim3 grid(p.batch, 256);
  const auto* slots = static_cast<const int32_t*>(p.state_slots.address);
  if (scatter) {
    ScatterStateKernel<<<grid, 256, 0, stream>>>(static_cast<const float4*>(a), slots, cu, static_cast<float4*>(b));
  } else {
    GatherStateKernel<<<grid, 256, 0, stream>>>(static_cast<const float4*>(a), slots, cu, static_cast<float4*>(b));
  }
  ROOST_CHECK_CUDA(cudaGetLastError());
  return ROOST_STATUS_OK;
}

bool WorkspaceOk(const RoostWorkspace* ws, uint64_t bytes) {
  return ws != nullptr && ws->device != nullptr && ws->device_bytes >= bytes && roost_hybrid::Aligned(ws->device, 256);
}

}  // namespace roost_gdn_op

// ---- Context: cubins are loaded when the context is created ---------------------------------------------------------

RoostStatus roost_hybrid_context_init(RoostContext* ctx) {
  using namespace roost_gdn_op;
  auto* loaded = new (std::nothrow) Loaded{};
  if (loaded == nullptr) {
    return Fail(ROOST_STATUS_OUT_OF_MEMORY, "GatedDeltaRule: out of host memory");
  }
  for (int i = 0; i < roost_gdn::kGdnKernelCount; ++i) {
    const GdnKernel& k = roost_gdn::kGdnKernels[i];
    if (std::strcmp(k.signature, kExpectedSignature[i]) != 0) {
      roost_set_error((std::string("GatedDeltaRule: the AOT table's argument list for ") + k.role +
                       " differs from the launcher's: " + k.signature).c_str());
      for (int j = 0; j < i; ++j) {
        cudaLibraryUnload(loaded->library[j]);
      }
      roost_hybrid_context_release(ctx);
      delete loaded;
      return ROOST_STATUS_INTERNAL;
    }
    cudaError_t err = cudaLibraryLoadData(&loaded->library[i], k.cubin, nullptr, nullptr, 0, nullptr, nullptr, 0);
    if (err == cudaSuccess) {
      err = cudaLibraryGetKernel(&loaded->kernel[i], loaded->library[i], k.kernel_name);
    }
    if (err != cudaSuccess) {
      for (int j = 0; j < i; ++j) {
        cudaLibraryUnload(loaded->library[j]);
      }
      delete loaded;
      return roost_hybrid::CudaFail(err);
    }
    loaded->count = i + 1;
    if (k.shared_bytes > 48 * 1024) {
      for (int j = 0; j <= i; ++j) {
        cudaLibraryUnload(loaded->library[j]);
      }
      delete loaded;
      return Fail(ROOST_STATUS_INTERNAL, "GatedDeltaRule: a kernel needs more than 48 KiB of shared memory, which the launcher does not opt in to");
    }
  }
  ctx->hybrid = loaded;
  return ROOST_STATUS_OK;
}

void roost_hybrid_context_release(RoostContext* ctx) {
  auto* loaded = static_cast<roost_gdn_op::Loaded*>(ctx->hybrid);
  if (loaded != nullptr) {
    for (int i = 0; i < loaded->count; ++i) {
      cudaLibraryUnload(loaded->library[i]);
    }
    delete loaded;
    ctx->hybrid = nullptr;
  }
}

// ---- Chunked prefill ------------------------------------------------------------------------------------------------

extern "C" RoostStatus roost_gdn_chunk_workspace(RoostContext* /*ctx*/, const void* params, RoostWorkspaceSizes* sizes) {
  using namespace roost_gdn_op;
  if (params == nullptr || sizes == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: params or sizes is NULL");
  }
  const auto& p = *static_cast<const RoostGatedDeltaRuleParams*>(params);
  if (p.struct_size < sizeof(RoostGatedDeltaRuleParams) || p.abi_minor > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH, "GatedDeltaRule: params struct_size or abi_minor is not supported by this library");
  }
  if (p.tokens > kMaxTokens || p.batch > (1u << 20)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: too many tokens or sequences for one call");
  }
  const ChunkLayout l = MakeChunkLayout(p.tokens, p.batch);
  sizes->device_bytes = l.total;
  sizes->pinned_host_bytes = l.tables_bytes;
  sizes->plan_bytes = sizeof(PlanRecord);
  sizes->device_alignment = 256;
  return ROOST_STATUS_OK;
}

extern "C" RoostStatus roost_gdn_chunk_plan(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream) {
  using namespace roost_gdn_op;
  if (ctx == nullptr || params == nullptr || ws == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule plan: ctx, params or workspace is NULL");
  }
  const auto& p = *static_cast<const RoostGatedDeltaRuleParams*>(params);
  ROOST_CHECK(Validate(p, true));
  const ChunkLayout l = MakeChunkLayout(p.tokens, p.batch);
  if (!WorkspaceOk(ws, l.total) || ws->pinned_host == nullptr || ws->pinned_host_bytes < l.tables_bytes ||
      ws->plan == nullptr || ws->plan_bytes < sizeof(PlanRecord)) {
    return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "GatedDeltaRule plan: workspace is missing or smaller than the workspace query said");
  }
  PlanRecord record{kPlanMagic, p.batch, p.tokens, 0};
  if (p.batch == 0 || p.tokens == 0) {
    std::memcpy(ws->plan, &record, sizeof(record));
    return ROOST_STATUS_OK;
  }
  const int32_t* host_cu = p.cu_seqlens_host;
  if (host_cu[0] != 0 || static_cast<uint32_t>(host_cu[p.batch]) != p.tokens) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule plan: cu_seqlens_host must start at 0 and end at tokens");
  }
  auto* base = static_cast<unsigned char*>(ws->pinned_host);
  std::memset(base, 0, l.tables_bytes);
  auto* cu = reinterpret_cast<int32_t*>(base + l.cu);
  auto* indices = reinterpret_cast<int32_t*>(base + l.chunk_indices);
  auto* offsets = reinterpret_cast<int64_t*>(base + l.chunk_offsets);
  uint32_t nt = 0;
  offsets[0] = 0;
  cu[0] = 0;
  for (uint32_t s = 0; s < p.batch; ++s) {
    if (host_cu[s + 1] < host_cu[s]) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule plan: cu_seqlens_host is not non-decreasing");
    }
    cu[s + 1] = host_cu[s + 1];
    const uint32_t len = static_cast<uint32_t>(host_cu[s + 1] - host_cu[s]);
    const uint32_t chunks = (len + kChunk - 1) / kChunk;
    for (uint32_t j = 0; j < chunks; ++j) {
      indices[2 * nt] = static_cast<int32_t>(s);
      indices[2 * nt + 1] = static_cast<int32_t>(j);
      ++nt;
    }
    offsets[s + 1] = nt;
  }
  record.nt = nt;
  std::memcpy(ws->plan, &record, sizeof(record));
  ROOST_CHECK_CUDA(cudaMemcpyAsync(ws->device, base, l.tables_bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
  return ROOST_STATUS_OK;
}

extern "C" RoostStatus roost_gdn_chunk_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream) {
  using namespace roost_gdn_op;
  if (ctx == nullptr || params == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: ctx or params is NULL");
  }
  const auto& p = *static_cast<const RoostGatedDeltaRuleParams*>(params);
  ROOST_CHECK(Validate(p, false));
  if (p.batch == 0 || p.tokens == 0) {
    return ROOST_STATUS_OK;
  }
  const auto* loaded = static_cast<const Loaded*>(ctx->hybrid);
  if (loaded == nullptr) {
    return Fail(ROOST_STATUS_INTERNAL, "GatedDeltaRule: the context holds no loaded kernels");
  }
  const ChunkLayout l = MakeChunkLayout(p.tokens, p.batch);
  if (!WorkspaceOk(ws, l.total) || ws->plan == nullptr || ws->plan_bytes < sizeof(PlanRecord)) {
    return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "GatedDeltaRule: workspace or plan is missing or too small");
  }
  PlanRecord record;
  std::memcpy(&record, ws->plan, sizeof(record));
  if (record.magic != kPlanMagic || record.batch != p.batch || record.tokens != p.tokens || record.nt > l.nt_max) {
    return Fail(ROOST_STATUS_NOT_PLANNED, "GatedDeltaRule: plan() has not run for these tokens and sequences");
  }
  const uint32_t nt = record.nt;
  const auto stream_ = static_cast<cudaStream_t>(stream);
  auto* dev = static_cast<unsigned char*>(ws->device);
  auto* cu = reinterpret_cast<int32_t*>(dev + l.cu);
  auto* indices = reinterpret_cast<int32_t*>(dev + l.chunk_indices);
  auto* offsets = reinterpret_cast<int64_t*>(dev + l.chunk_offsets);
  auto* qn = reinterpret_cast<__nv_bfloat16*>(dev + l.qn);
  auto* kn = reinterpret_cast<__nv_bfloat16*>(dev + l.kn);
  auto* vc = reinterpret_cast<__nv_bfloat16*>(dev + l.vc);
  auto* gc = reinterpret_cast<float*>(dev + l.gc);
  auto* a = reinterpret_cast<__nv_bfloat16*>(dev + l.a);
  auto* w = reinterpret_cast<__nv_bfloat16*>(dev + l.w);
  auto* u = reinterpret_cast<__nv_bfloat16*>(dev + l.u);
  auto* vnew = reinterpret_cast<__nv_bfloat16*>(dev + l.vnew);
  auto* h = reinterpret_cast<__nv_bfloat16*>(dev + l.h);
  auto* h0 = reinterpret_cast<float*>(dev + l.h0);
  auto* ht = reinterpret_cast<float*>(dev + l.ht);

  // The fused kkt+solve kernel writes only the lower-triangular blocks of A and relies on zeroed memory (S3).
  ROOST_CHECK_CUDA(cudaMemsetAsync(a, 0, uint64_t{p.tokens} * kHeads * kChunk * 2, stream_));
  ROOST_CHECK(GatherViews(p, qn, kn, vc, true, stream_));
  // h0 is zeroed first so a sequence the gather skips (negative slot) starts from a zero state, deterministically.
  ROOST_CHECK_CUDA(cudaMemsetAsync(h0, 0, uint64_t{p.batch} * kStateElems * 4, stream_));
  if (p.use_initial_state != 0) {
    ROOST_CHECK(StateMove(false, p.state.address, p, cu, h0, stream_));
  }

  float scale_arg = Scale(p);
  const float rcp_ln2 = 1.4426950216f;
  const int32_t t = static_cast<int32_t>(p.tokens);
  void* g = p.g.address;
  void* beta = p.beta.address;
  void* out = p.output.address;
  {
    float scale_gate = rcp_ln2;  // the cumsum kernel's `scale` argument: log2(e), so later kernels can use exp2
    void* gp = g;
    void* gcp = gc;
    void* cup = cu;
    void* idxp = indices;
    void* null_ptr = nullptr;
    void* args[] = {&gp, &gcp, &scale_gate, &cup, &idxp, const_cast<int32_t*>(&t), &null_ptr, &null_ptr};
    ROOST_CHECK(LaunchTriton(*loaded, roost_gdn::kCumsum, dim3(nt * kHeads), args, stream_));
  }
  {
    void* kp = kn;
    void* gcp = gc;
    void* bp = beta;
    void* ap = a;
    void* cup = cu;
    void* idxp = indices;
    void* null_ptr = nullptr;
    void* args[] = {&kp, &gcp, &bp, &ap, &cup, &idxp, const_cast<int32_t*>(&t), &null_ptr, &null_ptr};
    ROOST_CHECK(LaunchTriton(*loaded, roost_gdn::kKktSolve, dim3(nt, kHeads), args, stream_));
  }
  {
    void* kp = kn;
    void* vp = vc;
    void* bp = beta;
    void* wp = w;
    void* up = u;
    void* ap = a;
    void* gcp = gc;
    void* cup = cu;
    void* idxp = indices;
    void* null_ptr = nullptr;
    void* args[] = {&kp, &vp, &bp, &wp, &up, &ap, &gcp, &cup, &idxp, const_cast<int32_t*>(&t), &null_ptr, &null_ptr};
    ROOST_CHECK(LaunchTriton(*loaded, roost_gdn::kRecomputeWU, dim3(nt, kHeads), args, stream_));
  }
  {
    void* kp = kn;
    void* up = u;
    void* wp = w;
    void* vnp = vnew;
    void* gcp = gc;
    void* hp = h;
    void* h0p = h0;
    void* htp = ht;
    void* cup = cu;
    void* offp = offsets;
    void* null_ptr = nullptr;
    const dim3 grid(4 * p.batch * kHeads);
    if (p.use_initial_state != 0) {
      void* args[] = {&kp, &up, &wp, &vnp, &gcp, &hp, &h0p, &htp, &cup, &offp, const_cast<int32_t*>(&t), &null_ptr, &null_ptr};
      ROOST_CHECK(LaunchTriton(*loaded, roost_gdn::kChunkHInit, grid, args, stream_));
    } else {
      void* args[] = {&kp, &up, &wp, &vnp, &gcp, &hp, &htp, &cup, &offp, const_cast<int32_t*>(&t), &null_ptr, &null_ptr};
      ROOST_CHECK(LaunchTriton(*loaded, roost_gdn::kChunkH, grid, args, stream_));
    }
  }
  {
    void* qp = qn;
    void* kp = kn;
    void* vnp = vnew;
    void* hp = h;
    void* gcp = gc;
    void* op = out;
    void* cup = cu;
    void* idxp = indices;
    void* null_ptr = nullptr;
    void* args[] = {&qp, &kp, &vnp, &hp, &gcp, &op, &cup, &idxp, &scale_arg, const_cast<int32_t*>(&t), &null_ptr, &null_ptr};
    ROOST_CHECK(LaunchTriton(*loaded, roost_gdn::kChunkO, dim3(4 * nt, kHeads), args, stream_));
  }
  ROOST_CHECK(StateMove(true, ht, p, cu, p.state.address, stream_));
  return ROOST_STATUS_OK;
}

// ---- Recurrent (decode and short continuations) ---------------------------------------------------------------------

extern "C" RoostStatus roost_gdn_recurrent_workspace(RoostContext* /*ctx*/, const void* params, RoostWorkspaceSizes* sizes) {
  using namespace roost_gdn_op;
  if (params == nullptr || sizes == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: params or sizes is NULL");
  }
  const auto& p = *static_cast<const RoostGatedDeltaRuleParams*>(params);
  if (p.struct_size < sizeof(RoostGatedDeltaRuleParams) || p.abi_minor > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH, "GatedDeltaRule: params struct_size or abi_minor is not supported by this library");
  }
  if (p.tokens > kMaxTokens || p.batch > (1u << 20)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: too many tokens or sequences for one call");
  }
  sizes->device_bytes = MakeRecurrentLayout(p.tokens, p.batch).total;
  sizes->device_alignment = 256;
  return ROOST_STATUS_OK;
}

extern "C" RoostStatus roost_gdn_recurrent_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream) {
  using namespace roost_gdn_op;
  if (ctx == nullptr || params == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "GatedDeltaRule: ctx or params is NULL");
  }
  const auto& p = *static_cast<const RoostGatedDeltaRuleParams*>(params);
  ROOST_CHECK(Validate(p, false));
  if (p.batch == 0 || p.tokens == 0) {
    return ROOST_STATUS_OK;
  }
  const auto* loaded = static_cast<const Loaded*>(ctx->hybrid);
  if (loaded == nullptr) {
    return Fail(ROOST_STATUS_INTERNAL, "GatedDeltaRule: the context holds no loaded kernels");
  }
  const RecurrentLayout l = MakeRecurrentLayout(p.tokens, p.batch);
  if (!WorkspaceOk(ws, l.total)) {
    return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "GatedDeltaRule: workspace is missing or smaller than the workspace query said");
  }
  const auto stream_ = static_cast<cudaStream_t>(stream);
  auto* dev = static_cast<unsigned char*>(ws->device);
  auto* cu = reinterpret_cast<int32_t*>(dev + l.cu);
  auto* q = reinterpret_cast<__nv_bfloat16*>(dev + l.q);
  auto* k = reinterpret_cast<__nv_bfloat16*>(dev + l.k);
  auto* v = reinterpret_cast<__nv_bfloat16*>(dev + l.v);
  auto* h0 = reinterpret_cast<float*>(dev + l.h0);
  auto* ht = reinterpret_cast<float*>(dev + l.ht);

  // The kernel needs 16-byte aligned cu_seqlens; the caller's copy may not be, so it is copied into the workspace.
  ROOST_CHECK_CUDA(cudaMemcpyAsync(cu, p.cu_seqlens.address, 4 * (uint64_t{p.batch} + 1), cudaMemcpyDeviceToDevice, stream_));
  // The recurrent kernel normalises q and k itself: the views are copied raw.
  ROOST_CHECK(GatherViews(p, q, k, v, false, stream_));
  ROOST_CHECK_CUDA(cudaMemsetAsync(h0, 0, uint64_t{p.batch} * kStateElems * 4, stream_));
  if (p.use_initial_state != 0) {
    ROOST_CHECK(StateMove(false, p.state.address, p, cu, h0, stream_));
  }
  float scale_arg = Scale(p);
  const int32_t t = static_cast<int32_t>(p.tokens);
  void* qp = q;
  void* kp = k;
  void* vp = v;
  void* gp = p.g.address;
  void* bp = p.beta.address;
  void* op = p.output.address;
  void* h0p = h0;
  void* htp = ht;
  void* cup = cu;
  void* null_ptr = nullptr;
  void* args[] = {&qp, &kp, &vp, &gp, &bp, &op, &h0p, &htp, &cup, &scale_arg, const_cast<int32_t*>(&t), &null_ptr, &null_ptr};
  // 16 value blocks of 8 columns per (sequence, head).
  ROOST_CHECK(LaunchTriton(*loaded, roost_gdn::kRecurrent, dim3(16 * p.batch * kHeads), args, stream_));
  ROOST_CHECK(StateMove(true, ht, p, cu, p.state.address, stream_));
  return ROOST_STATUS_OK;
}

// Spike S6: one Qwen3 MoE layer without torch. Router GEMM, vLLM top-k softmax (patched, separate TU),
// permute, CUTLASS grouped GEMM (bf16, SM80 tensor ops), SiLU-and-mul, grouped GEMM, weighted unpermute.
#include "roost_moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"

extern "C" int roost_vllm_topk_softmax_bf16(const void*, float*, int*, int*, float*, int, int, int, int, void*);

namespace {

using Elem = cutlass::bfloat16_t;
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Elem, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    Elem, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    Elem, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Elem, 8, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4>::GemmKernel;
using GroupedGemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

thread_local char g_err[256] = "";
int fail(const char* what, cudaError_t e = cudaSuccess) {
  std::snprintf(g_err, sizeof g_err, "%s%s%s", what, e ? ": " : "", e ? cudaGetErrorString(e) : "");
  return 1;
}

// Everything carved out of the caller's workspace.
struct Layout {
  size_t logits, tw, tid, tei, smws, counts, offsets, cursor, inv, rowtok, xp, g1, act, y, probs1, probs2, args;
  size_t total;
};
size_t al(size_t& o, size_t n) { size_t r = o; o += (n + 255) & ~size_t(255); return r; }
Layout layout(const RoostMoeLayerParams& p) {
  size_t T = p.num_tokens, H = p.hidden, I = p.inter, E = p.num_experts, K = p.topk, TK = T * K, o = 0;
  Layout l{};
  l.logits = al(o, T * E * 2);
  l.tw = al(o, TK * 4); l.tid = al(o, TK * 4); l.tei = al(o, TK * 4);
  l.smws = al(o, T * E * 4);
  l.counts = al(o, E * 4); l.offsets = al(o, (E + 1) * 4); l.cursor = al(o, E * 4);
  l.inv = al(o, TK * 4); l.rowtok = al(o, TK * 4);
  l.xp = al(o, TK * H * 2); l.g1 = al(o, TK * 2 * I * 2); l.act = al(o, TK * I * 2); l.y = al(o, TK * H * 2);
  l.probs1 = al(o, E * sizeof(cutlass::gemm::GemmCoord)); l.probs2 = al(o, E * sizeof(cutlass::gemm::GemmCoord));
  // pointer and ld arrays for both GEMMs: 3 pointer arrays + 3 ld arrays each.
  l.args = al(o, 2 * (3 * E * 8 + 3 * E * 8));
  l.total = o;
  return l;
}

struct GemmArgs {  // device-side argument arrays for one grouped GEMM
  Elem **a, **b, **d;
  int64_t *lda, *ldb, *ldd;
};

__global__ void router_gemm(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* logits, int T, int H, int E) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (warp >= T * E) return;
  int t = warp / E, e = warp % E;
  const uint4* xv = reinterpret_cast<const uint4*>(x + size_t(t) * H);
  const uint4* wv = reinterpret_cast<const uint4*>(w + size_t(e) * H);
  float acc = 0.f;
  for (int i = lane; i < H / 8; i += 32) {
    uint4 a = xv[i], b = wv[i];
    const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&a);
    const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&b);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += __bfloat162float(pa[j]) * __bfloat162float(pb[j]);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) logits[size_t(t) * E + e] = __float2bfloat16(acc);
}

__global__ void count_experts(const int* ids, int n, int* counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&counts[ids[i]], 1);
}

// One block: prefix sum over experts, then fill both grouped-GEMM argument tables.
__global__ void build_args(const int* counts, int* offsets, int E, int H, int I, Elem* xp, Elem* g1, Elem* act, Elem* y,
                           const Elem* wgu, const Elem* wdn, cutlass::gemm::GemmCoord* pr1, cutlass::gemm::GemmCoord* pr2,
                           GemmArgs a1, GemmArgs a2) {
  if (threadIdx.x != 0) return;
  int acc = 0;
  for (int e = 0; e < E; ++e) {
    int m = counts[e];
    offsets[e] = acc;
    pr1[e] = cutlass::gemm::GemmCoord(m, 2 * I, H);
    a1.a[e] = xp + size_t(acc) * H; a1.b[e] = const_cast<Elem*>(wgu) + size_t(e) * 2 * I * H; a1.d[e] = g1 + size_t(acc) * 2 * I;
    a1.lda[e] = H; a1.ldb[e] = H; a1.ldd[e] = 2 * I;
    pr2[e] = cutlass::gemm::GemmCoord(m, H, I);
    a2.a[e] = act + size_t(acc) * I; a2.b[e] = const_cast<Elem*>(wdn) + size_t(e) * H * I; a2.d[e] = y + size_t(acc) * H;
    a2.lda[e] = I; a2.ldb[e] = I; a2.ldd[e] = H;
    acc += m;
  }
  offsets[E] = acc;
}

__global__ void scatter_rows(const int* ids, int n, int K, const int* offsets, int* cursor, int* inv, int* rowtok) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int e = ids[i];
  int pos = offsets[e] + atomicAdd(&cursor[e], 1);
  inv[i] = pos;
  rowtok[pos] = i / K;
}

__global__ void gather_rows(const __nv_bfloat16* x, const int* rowtok, __nv_bfloat16* xp, int H) {
  int r = blockIdx.x;
  const uint4* s = reinterpret_cast<const uint4*>(x + size_t(rowtok[r]) * H);
  uint4* d = reinterpret_cast<uint4*>(xp + size_t(r) * H);
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x) d[i] = s[i];
}

__global__ void silu_mul(const __nv_bfloat16* g1, __nv_bfloat16* act, int rows, int I) {
  size_t n = size_t(rows) * I;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    size_t r = i / I, c = i % I;
    float g = __bfloat162float(g1[r * 2 * I + c]), u = __bfloat162float(g1[r * 2 * I + I + c]);
    act[i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
  }
}

__global__ void combine(const __nv_bfloat16* y, const int* inv, const float* tw, __nv_bfloat16* out, int K, int H) {
  int t = blockIdx.x;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += tw[t * K + k] * __bfloat162float(y[size_t(inv[t * K + k]) * H + h]);
    out[size_t(t) * H + h] = __float2bfloat16(acc);
  }
}

template <class T> T* at(void* base, size_t off) { return reinterpret_cast<T*>(static_cast<char*>(base) + off); }

int run_grouped(cutlass::gemm::GemmCoord* probs, int E, const GemmArgs& a, cudaStream_t s) {
  typename GroupedGemm::Arguments args(probs, E, GroupedGemm::sufficient(nullptr, E),
                                       typename GemmKernel::Epilogue::OutputOp::Params(1.f, 0.f),
                                       a.a, a.b, a.d, a.d, a.lda, a.ldb, a.ldd, a.ldd, nullptr);
  GroupedGemm op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return fail("grouped gemm can_implement");
  size_t ws = op.get_workspace_size(args);
  if (ws != 0) return fail("grouped gemm wants a workspace");
  if (op.initialize(args, nullptr, s) != cutlass::Status::kSuccess) return fail("grouped gemm initialize");
  if (op.run(s) != cutlass::Status::kSuccess) return fail("grouped gemm run");
  return 0;
}

}  // namespace

extern "C" {

const char* roost_moe_last_error(void) { return g_err; }

int roost_moe_layer_workspace_bytes(const RoostMoeLayerParams* p, size_t* bytes) {
  if (!p || !bytes || p->struct_size < sizeof(RoostMoeLayerParams) || p->abi_minor > ROOST_MOE_SPIKE_ABI_MINOR)
    return fail("bad params header");
  *bytes = layout(*p).total;
  return 0;
}

int roost_moe_layer(const RoostMoeLayerParams* p, void* stream_) {
  if (!p || p->struct_size < sizeof(RoostMoeLayerParams) || p->abi_minor > ROOST_MOE_SPIKE_ABI_MINOR)
    return fail("bad params header");
  const int T = p->num_tokens, H = p->hidden, I = p->inter, E = p->num_experts, K = p->topk, TK = T * K;
  if (H % 8 || I % 8 || (E & (E - 1)) || E > 512 || K > E) return fail("unsupported shape");
  Layout l = layout(*p);
  if (p->workspace_bytes < l.total) return fail("workspace too small");
  auto stream = static_cast<cudaStream_t>(stream_);
  void* ws = p->workspace;
  auto* logits = at<__nv_bfloat16>(ws, l.logits);
  auto* tw = at<float>(ws, l.tw); auto* tid = at<int>(ws, l.tid); auto* tei = at<int>(ws, l.tei);
  auto* counts = at<int>(ws, l.counts); auto* offsets = at<int>(ws, l.offsets); auto* cursor = at<int>(ws, l.cursor);
  auto* inv = at<int>(ws, l.inv); auto* rowtok = at<int>(ws, l.rowtok);
  auto* xp = at<Elem>(ws, l.xp); auto* g1 = at<Elem>(ws, l.g1); auto* act = at<Elem>(ws, l.act); auto* y = at<Elem>(ws, l.y);
  auto* pr1 = at<cutlass::gemm::GemmCoord>(ws, l.probs1); auto* pr2 = at<cutlass::gemm::GemmCoord>(ws, l.probs2);
  char* ap = static_cast<char*>(ws) + l.args;
  auto carve = [&](GemmArgs& a) {
    a.a = reinterpret_cast<Elem**>(ap); ap += E * 8; a.b = reinterpret_cast<Elem**>(ap); ap += E * 8;
    a.d = reinterpret_cast<Elem**>(ap); ap += E * 8; a.lda = reinterpret_cast<int64_t*>(ap); ap += E * 8;
    a.ldb = reinterpret_cast<int64_t*>(ap); ap += E * 8; a.ldd = reinterpret_cast<int64_t*>(ap); ap += E * 8;
  };
  GemmArgs a1, a2; carve(a1); carve(a2);

  cudaMemsetAsync(counts, 0, E * 4, stream); cudaMemsetAsync(cursor, 0, E * 4, stream);
  router_gemm<<<(T * E * 32 + 255) / 256, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(p->x),
      static_cast<const __nv_bfloat16*>(p->w_router), logits, T, H, E);
  if (roost_vllm_topk_softmax_bf16(logits, tw, tid, tei, at<float>(ws, l.smws), T, E, K, p->norm_topk_prob, stream))
    return fail("topk launch", cudaGetLastError());
  count_experts<<<(TK + 255) / 256, 256, 0, stream>>>(tid, TK, counts);
  build_args<<<1, 32, 0, stream>>>(counts, offsets, E, H, I, xp, g1, act, y, static_cast<const Elem*>(p->w_gate_up),
                                   static_cast<const Elem*>(p->w_down), pr1, pr2, a1, a2);
  scatter_rows<<<(TK + 255) / 256, 256, 0, stream>>>(tid, TK, K, offsets, cursor, inv, rowtok);
  gather_rows<<<TK, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(p->x), rowtok, reinterpret_cast<__nv_bfloat16*>(xp), H);
  if (run_grouped(pr1, E, a1, stream)) return 1;
  silu_mul<<<1024, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(g1), reinterpret_cast<__nv_bfloat16*>(act), TK, I);
  if (run_grouped(pr2, E, a2, stream)) return 1;
  combine<<<T, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(y), inv, tw, static_cast<__nv_bfloat16*>(p->out), K, H);

  auto cp = [&](void* dst, const void* src, size_t n) { if (dst) cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, stream); };
  cp(p->dbg_logits, logits, size_t(T) * E * 2); cp(p->dbg_topk_weights, tw, size_t(TK) * 4);
  cp(p->dbg_topk_ids, tid, size_t(TK) * 4); cp(p->dbg_gemm1, g1, size_t(TK) * 2 * I * 2);
  cp(p->dbg_act, act, size_t(TK) * I * 2); cp(p->dbg_gemm2, y, size_t(TK) * H * 2); cp(p->dbg_inv, inv, size_t(TK) * 4);
  cudaError_t e = cudaGetLastError();
  return e ? fail("launch", e) : 0;
}

}  // extern "C"

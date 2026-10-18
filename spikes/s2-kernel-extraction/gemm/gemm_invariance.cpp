// S2 spike (R-006): does pinning one cuBLASLt algo make row results independent of the batch size M?
// Plans once at M=64, then runs the same algo with M=64, 17, 5 and 1 (leading rows identical) and
// compares output rows bitwise. Also reports what the heuristic picks per M when not pinned.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return uint16_t(u >> 16); }

struct Ctx {
  cublasLtHandle_t lt; void* ws; size_t wsb = 32u << 20; int N, K;
  void* dw; cublasLtMatmulDesc_t op;
};

static cublasLtMatmulHeuristicResult_t Plan(Ctx& c, int M) {
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, c.K, c.N, c.K);
  cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, c.K, M, c.K);
  cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, c.N, M, c.N);
  cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c.wsb, sizeof c.wsb);
  cublasLtMatmulHeuristicResult_t h; int n = 0;
  cublasLtMatmulAlgoGetHeuristic(c.lt, c.op, la, lb, lc, lc, pref, 1, &h, &n);
  return h;
}
static void Run(Ctx& c, const cublasLtMatmulAlgo_t& algo, const void* a, void* out, int M) {
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, c.K, c.N, c.K);
  cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, c.K, M, c.K);
  cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, c.N, M, c.N);
  float one = 1, zero = 0;
  cublasLtMatmul(c.lt, c.op, &one, c.dw, la, a, lb, &zero, out, lc, out, lc, &algo, c.ws, c.wsb, nullptr);
  cudaDeviceSynchronize();
}
static int AlgoId(const cublasLtMatmulAlgo_t& a) { int id = -1; size_t w; cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w); return id; }
static int Splitk(const cublasLtMatmulAlgo_t& a) { int v = -1; size_t w; cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &v, sizeof v, &w); return v; }

int main() {
  Ctx c; c.N = 2048; c.K = 6144;  // Qwen3-1.7B down_proj
  cublasLtCreate(&c.lt); cudaMalloc(&c.ws, c.wsb);
  cublasLtMatmulDescCreate(&c.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(c.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t);
  cublasLtMatmulDescSetAttribute(c.op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof n);
  std::mt19937 rng(1); std::normal_distribution<float> nd(0, 1);
  std::vector<uint16_t> hw(size_t(c.N) * c.K), ha(size_t(64) * c.K);
  for (auto& x : hw) x = f2bf(nd(rng) * 0.05f);
  for (auto& x : ha) x = f2bf(nd(rng));
  void *da; cudaMalloc(&c.dw, hw.size() * 2); cudaMalloc(&da, ha.size() * 2);
  cudaMemcpy(c.dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(da, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice);
  void* dout[4]; for (auto& p : dout) cudaMalloc(&p, size_t(64) * c.N * 2);
  const int Ms[4] = {64, 17, 5, 1};
  for (int m : Ms) { auto h = Plan(c, m); std::printf("heuristic M=%-3d algo id=%d splitk=%d\n", m, AlgoId(h.algo), Splitk(h.algo)); }
  auto pinned = Plan(c, 64);
  std::printf("pinned: algo id=%d splitk=%d\n", AlgoId(pinned.algo), Splitk(pinned.algo));
  for (int i = 0; i < 4; ++i) Run(c, pinned.algo, da, dout[i], Ms[i]);
  std::vector<uint16_t> ref(size_t(Ms[0]) * c.N), cmp(ref.size());
  cudaMemcpy(ref.data(), dout[0], ref.size() * 2, cudaMemcpyDeviceToHost);
  int bad_total = 0;
  for (int i = 1; i < 4; ++i) {
    cudaMemcpy(cmp.data(), dout[i], size_t(Ms[i]) * c.N * 2, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (size_t j = 0; j < size_t(Ms[i]) * c.N; ++j) bad += cmp[j] != ref[j];
    std::printf("pinned algo, M=%-3d vs M=64: %d of %zu elements differ bitwise\n", Ms[i], bad, size_t(Ms[i]) * c.N);
    bad_total += bad;
  }
  // Same comparison with the heuristic's own pick per M (unpinned).
  for (int i = 1; i < 4; ++i) {
    auto h = Plan(c, Ms[i]);
    Run(c, h.algo, da, dout[i], Ms[i]);
    cudaMemcpy(cmp.data(), dout[i], size_t(Ms[i]) * c.N * 2, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (size_t j = 0; j < size_t(Ms[i]) * c.N; ++j) bad += cmp[j] != ref[j];
    std::printf("heuristic algo, M=%-3d vs M=64: %d of %zu elements differ bitwise\n", Ms[i], bad, size_t(Ms[i]) * c.N);
  }
  return 0;
}

// S2 spike (R-006): cuBLASLt vs one-tile CUTLASS bf16 GEMM on Qwen3-1.7B projection shapes.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gemm_cutlass.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { std::printf("cublas status %d line %d\n", int(s_), __LINE__); return 2; } } while (0)

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return uint16_t(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

template <class F>
static float TimeUs(F&& f, cudaStream_t s) {
  for (int i = 0; i < 10; ++i) f();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, s);
  const int it = 100;
  for (int i = 0; i < it; ++i) f();
  cudaEventRecord(b, s); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  return ms * 1000.f / it;
}

int main() {
  cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  void* ws; size_t wsb = 32u << 20; CK(cudaMalloc(&ws, wsb));
  struct Shape { const char* name; int m, n, k; };
  const Shape shapes[] = {
      {"decode b=1   gate_up", 1, 12288, 2048}, {"decode b=16  gate_up", 16, 12288, 2048},
      {"decode b=16  down", 16, 2048, 6144},    {"decode b=16  qkv(q)", 16, 2048, 2048},
      {"prefill 512  gate_up", 512, 12288, 2048}, {"prefill 512  down", 512, 2048, 6144},
      {"prefill 2048 N=8192", 2048, 8192, 2048}};
  std::mt19937 rng(3); std::normal_distribution<float> nd(0, 0.5f);
  for (auto sh : shapes) {
    const int M = sh.m, N = sh.n, K = sh.k;
    std::vector<uint16_t> ha(size_t(M) * K), hw(size_t(N) * K);
    for (auto& x : ha) x = f2bf(nd(rng));
    for (auto& x : hw) x = f2bf(nd(rng) * 0.05f);
    void *da, *dw, *dc1, *dc2, *dc3;
    CK(cudaMalloc(&da, ha.size() * 2)); CK(cudaMalloc(&dw, hw.size() * 2));
    CK(cudaMalloc(&dc1, size_t(M) * N * 2)); CK(cudaMalloc(&dc2, size_t(M) * N * 2)); CK(cudaMalloc(&dc3, size_t(M) * N * 2));
    CK(cudaMemcpy(da, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    // cuBLASLt, column-major view: C^T[N,M] = W[N,K] (OP_T of col-major [K,N]) * A^T.
    cublasLtMatmulDesc_t op; CB(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
    cublasLtMatrixLayout_t la, lb, lc;
    CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));
    CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K));
    CB(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
    cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb));
    cublasLtMatmulHeuristicResult_t h; int nret = 0;
    CB(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &h, &nret));
    if (!nret) { std::printf("no cublasLt algo\n"); return 2; }
    float one = 1.f, zero = 0.f;
    auto lt_run = [&](void* out) { cublasLtMatmul(lt, op, &one, dw, la, da, lb, &zero, out, lc, out, lc, &h.algo, ws, wsb, s); };
    lt_run(dc1); CK(cudaStreamSynchronize(s));
    float us_lt = TimeUs([&] { lt_run(dc1); }, s);
    int r2 = roost_s2_gemm_cutlass_128x128x32(da, dw, dc2, M, N, K, ws, s);
    int r3 = roost_s2_gemm_cutlass_32x64x64(da, dw, dc3, M, N, K, ws, s);
    CK(cudaStreamSynchronize(s));
    float us_l = r2 == 0 ? TimeUs([&] { roost_s2_gemm_cutlass_128x128x32(da, dw, dc2, M, N, K, ws, s); }, s) : -1;
    float us_s = r3 == 0 ? TimeUs([&] { roost_s2_gemm_cutlass_32x64x64(da, dw, dc3, M, N, K, ws, s); }, s) : -1;
    std::vector<uint16_t> o1(size_t(M) * N), o2(o1.size()), o3(o1.size());
    CK(cudaMemcpy(o1.data(), dc1, o1.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o2.data(), dc2, o1.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o3.data(), dc3, o1.size() * 2, cudaMemcpyDeviceToHost));
    double d2 = 0, d3 = 0, mx = 0;
    for (size_t i = 0; i < o1.size(); ++i) {
      d2 = std::max(d2, (double)std::fabs(bf2f(o1[i]) - bf2f(o2[i])));
      d3 = std::max(d3, (double)std::fabs(bf2f(o1[i]) - bf2f(o3[i])));
      mx = std::max(mx, (double)std::fabs(bf2f(o1[i])));
    }
    double fl = 2.0 * M * N * K;
    std::printf("%-24s M=%-5d N=%-6d K=%-5d | cublasLt %8.1f us %6.1f TF | cutlass128x128x32 %8.1f us (st %d, diff %.2e) | cutlass32x64x64 %8.1f us (st %d, diff %.2e) | max|c|=%.2f\n",
                sh.name, M, N, K, us_lt, fl / us_lt * 1e-6, us_l, r2, d2, us_s, r3, d3, mx);
    cudaFree(da); cudaFree(dw); cudaFree(dc1); cudaFree(dc2); cudaFree(dc3);
  }
  return 0;
}

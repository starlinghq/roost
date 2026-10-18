// S2 spike (R-006): C ABI test for the FlashInfer paged decode wrapper against a CPU reference.
// Uses only the C header and the CUDA runtime. Exit code 0 means every case is within tolerance.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "roost_s2_attn.h"

#define CK(x)                                                                     \
  do {                                                                            \
    cudaError_t e_ = (x);                                                         \
    if (e_ != cudaSuccess) {                                                      \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      return 2;                                                                   \
    }                                                                             \
  } while (0)

static uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);  // round to nearest even
  return static_cast<uint16_t>(u >> 16);
}
static float bf2f(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static int RunCase(const char* name, const std::vector<int>& kv_len, uint32_t qo, uint32_t kvh,
                   uint32_t page, uint32_t seed, double* out_max_abs) {
  const uint32_t D = 128, B = kv_len.size();
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);

  std::vector<int32_t> indptr(B + 1, 0), last(B);
  for (uint32_t b = 0; b < B; ++b) {
    int pages = (kv_len[b] + page - 1) / page;
    indptr[b + 1] = indptr[b] + pages;
    last[b] = kv_len[b] - (pages - 1) * page;
  }
  const uint32_t total_pages = indptr[B];
  const uint32_t pool_pages = total_pages + 7;  // spare pages, ids shuffled
  std::vector<int32_t> perm(pool_pages);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<int32_t> indices(perm.begin(), perm.begin() + total_pages);

  const size_t kv_elems = size_t(pool_pages) * page * kvh * D;
  std::vector<uint16_t> hk(kv_elems), hv(kv_elems), hq(size_t(B) * qo * D);
  for (auto& x : hk) x = f2bf(nd(rng));
  for (auto& x : hv) x = f2bf(nd(rng));
  for (auto& x : hq) x = f2bf(nd(rng));

  void *dk, *dv, *dq, *dop, *dfw, *diw, *piw;
  int32_t *dind, *didx, *dlast;
  CK(cudaMalloc(&dk, kv_elems * 2));
  CK(cudaMalloc(&dv, kv_elems * 2));
  CK(cudaMalloc(&dq, hq.size() * 2));
  CK(cudaMalloc(&dop, hq.size() * 2));
  CK(cudaMemcpy(dk, hk.data(), kv_elems * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv, hv.data(), kv_elems * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemset(dop, 0xff, hq.size() * 2));  // NaN pattern so unwritten output is caught
  CK(cudaMalloc(&dind, indptr.size() * 4));
  CK(cudaMalloc(&didx, indices.size() * 4));
  CK(cudaMalloc(&dlast, last.size() * 4));
  CK(cudaMemcpy(dind, indptr.data(), indptr.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(didx, indices.data(), indices.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dlast, last.data(), last.size() * 4, cudaMemcpyHostToDevice));

  RoostS2PagedDecodeParams p;
  std::memset(&p, 0, sizeof p);
  p.struct_size = sizeof p;
  p.q = dq;
  p.o = dop;
  p.k_cache = dk;
  p.v_cache = dv;
  p.kv_indptr = dind;
  p.kv_indices = didx;
  p.kv_last_page_len = dlast;
  p.kv_indptr_host = indptr.data();
  p.batch_size = B;
  p.num_qo_heads = qo;
  p.num_kv_heads = kvh;
  p.page_size = page;
  p.head_dim = D;
  p.window_left = -1;

  size_t fb = 0, ib = 0;
  if (roost_s2_paged_decode_workspace(&p, &fb, &ib) != 0) {
    std::printf("%s: workspace failed: %s\n", name, roost_s2_last_error());
    return 1;
  }
  CK(cudaMalloc(&dfw, fb ? fb : 16));
  CK(cudaMalloc(&diw, ib ? ib : 16));
  CK(cudaMallocHost(&piw, ib ? ib : 16));
  p.float_workspace = dfw;
  p.float_workspace_bytes = fb;
  p.int_workspace = diw;
  p.int_workspace_bytes = ib;
  p.pinned_int_workspace = piw;

  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  RoostS2DecodePlan plan;
  if (roost_s2_paged_decode_plan(&p, &plan, s) != 0) {
    std::printf("%s: plan failed: %s\n", name, roost_s2_last_error());
    return 1;
  }
  if (roost_s2_paged_decode_run(&p, &plan, s) != 0) {
    std::printf("%s: run failed: %s\n", name, roost_s2_last_error());
    return 1;
  }
  CK(cudaStreamSynchronize(s));
  CK(cudaGetLastError());
  std::vector<uint16_t> ho(hq.size());
  CK(cudaMemcpy(ho.data(), dop, ho.size() * 2, cudaMemcpyDeviceToHost));

  // CPU reference in double, from the same bf16 inputs.
  const uint32_t group = qo / kvh;
  const double scale = 1.0 / std::sqrt(double(D));
  double max_abs = 0, max_rel = 0, mean_abs = 0;
  size_t n = 0;
  bool nonfinite = false;
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t h = 0; h < qo; ++h) {
      uint32_t g = h / group;
      std::vector<double> sc(kv_len[b]);
      double mx = -1e300;
      for (int t = 0; t < kv_len[b]; ++t) {
        int pg = indices[indptr[b] + t / page];
        size_t base = ((size_t(pg) * page + t % page) * kvh + g) * D;
        double a = 0;
        for (uint32_t d = 0; d < D; ++d) a += double(bf2f(hq[(size_t(b) * qo + h) * D + d])) * bf2f(hk[base + d]);
        sc[t] = a * scale;
        mx = std::max(mx, sc[t]);
      }
      double den = 0;
      for (auto& x : sc) { x = std::exp(x - mx); den += x; }
      for (uint32_t d = 0; d < D; ++d) {
        double acc = 0;
        for (int t = 0; t < kv_len[b]; ++t) {
          int pg = indices[indptr[b] + t / page];
          size_t base = ((size_t(pg) * page + t % page) * kvh + g) * D;
          acc += sc[t] * bf2f(hv[base + d]);
        }
        acc /= den;
        float got = bf2f(ho[(size_t(b) * qo + h) * D + d]);
        if (!std::isfinite(got)) nonfinite = true;
        double ae = std::fabs(got - acc);
        max_abs = std::max(max_abs, ae);
        max_rel = std::max(max_rel, ae / (std::fabs(acc) + 1e-2));
        mean_abs += ae;
        ++n;
      }
    }
  }
  mean_abs /= n;
  // bf16 has 8 bits of mantissa; outputs here are O(0.1..1), so one ulp is about 2e-3..4e-3.
  const bool ok = !nonfinite && max_abs < 1e-2;
  std::printf("%-28s B=%u qo=%u kv=%u page=%u split_kv=%d padded_batch=%lld float_ws=%zu int_ws=%zu "
              "max_abs=%.3e mean_abs=%.3e max_rel=%.3e %s\n",
              name, B, qo, kvh, page, int(plan.words[9]), (long long)plan.words[0], fb, ib,
              max_abs, mean_abs, max_rel, ok ? "PASS" : "FAIL");
  *out_max_abs = max_abs;

  // Timing of run() only (plan reused), 200 launches.
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0, s));
  for (int i = 0; i < 200; ++i) roost_s2_paged_decode_run(&p, &plan, s);
  CK(cudaEventRecord(e1, s));
  CK(cudaEventSynchronize(e1));
  float ms;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  std::printf("%-28s run() mean %.1f us\n", name, ms * 1000.f / 200.f);

  cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dop); cudaFree(dfw); cudaFree(diw);
  cudaFreeHost(piw); cudaFree(dind); cudaFree(didx); cudaFree(dlast); cudaStreamDestroy(s);
  return ok ? 0 : 1;
}

int main() {
  std::printf("roost_s2 abi=0x%x\n", roost_s2_abi_version());
  int rc = 0;
  double m;
  // Qwen3-1.7B shape: 16 query heads, 8 KV heads (group 2), head_dim 128, page 16.
  rc |= RunCase("mixed lens, partial pages", {5, 37, 130, 16}, 16, 8, 16, 1, &m);
  rc |= RunCase("single long seq (split-kv)", {6000}, 16, 8, 16, 2, &m);
  rc |= RunCase("batch 8 mid lens", {900, 17, 1, 2048, 333, 64, 1024, 777}, 16, 8, 16, 3, &m);
  { std::vector<int> many(256); for (size_t i = 0; i < many.size(); ++i) many[i] = 1 + int((i * 37) % 300);
    rc |= RunCase("batch 256 (no split-kv)", many, 16, 8, 16, 5, &m); }
  // Group-4 variant (32 query heads, 8 KV heads), for the larger cards.
  rc |= RunCase("group 4 (32q/8kv)", {100, 515}, 32, 8, 16, 4, &m);
  // A rejected shape must return a status and a message, not throw across the ABI.
  RoostS2PagedDecodeParams bad;
  std::memset(&bad, 0, sizeof bad);
  bad.struct_size = sizeof bad;
  RoostS2DecodePlan pl;
  int32_t st = roost_s2_paged_decode_plan(&bad, &pl, nullptr);
  std::printf("invalid params -> status %d (%s)\n", st, roost_s2_last_error());
  if (st == 0) rc |= 1;
  std::printf(rc ? "S2 TEST FAILED\n" : "S2 TEST PASSED\n");
  return rc;
}

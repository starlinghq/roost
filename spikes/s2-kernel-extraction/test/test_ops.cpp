// S2 spike (R-006): C ABI test for the non-attention ops against a CPU reference (double).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "roost_s2_ops.h"

static uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return uint16_t(u >> 16);
}
static float bf2f(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static std::mt19937 rng(7);
static std::vector<uint16_t> Rand(size_t n, float sd = 1.f, float mean = 0.f) {
  std::normal_distribution<float> nd(mean, sd);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = f2bf(nd(rng));
  return v;
}
template <class T>
static T* Dev(const std::vector<T>& h) {
  void* d;
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return (T*)d;
}
static std::vector<uint16_t> Get(const void* d, size_t n) {
  std::vector<uint16_t> h(n);
  cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost);
  return h;
}
static int g_fail = 0;
static void Report(const char* name, int32_t st, const std::vector<uint16_t>& got,
                   const std::vector<double>& ref, double tol) {
  double mx = 0;
  bool bad = st != 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double g = bf2f(got[i]);
    if (!std::isfinite(g)) bad = true;
    mx = std::max(mx, std::fabs(g - ref[i]) / (1.0 + std::fabs(ref[i])));
  }
  if (mx > tol) bad = true;
  std::printf("%-34s status=%d max_err(abs/(1+|ref|))=%.3e tol=%.0e %s%s\n", name, st, mx, tol,
              bad ? "FAIL" : "PASS", st ? roost_s2_last_error() : "");
  g_fail |= bad;
}

int main() {
  const uint32_t R = 37, D = 2048;  // Qwen3-1.7B hidden size
  const float eps = 1e-6f;
  auto x = Rand(R * D), w = Rand(D, 0.2f, 1.f);
  uint16_t *dx = Dev(x), *dw = Dev(w), *dout;
  cudaMalloc((void**)&dout, R * D * 2);
  std::vector<double> ref(R * D);
  for (uint32_t r = 0; r < R; ++r) {
    double ss = 0;
    for (uint32_t i = 0; i < D; ++i) ss += double(bf2f(x[r * D + i])) * bf2f(x[r * D + i]);
    double inv = 1.0 / std::sqrt(ss / D + eps);
    for (uint32_t i = 0; i < D; ++i) ref[r * D + i] = bf2f(x[r * D + i]) * inv * bf2f(w[i]);
  }
  int32_t st = roost_s2_rms_norm_flashinfer(dx, dw, dout, R, D, D, eps, nullptr);
  cudaDeviceSynchronize();
  Report("rms_norm (flashinfer)", st, Get(dout, R * D), ref, 2e-2);
  cudaMemset(dout, 0, R * D * 2);
  st = roost_s2_rms_norm_vllm(dx, dw, dout, R, D, D, eps, nullptr);
  cudaDeviceSynchronize();
  Report("rms_norm (vllm, patched)", st, Get(dout, R * D), ref, 2e-2);

  // fused add + rms_norm: residual += x, then x = rms(residual) * w
  {
    auto res = Rand(R * D);
    uint16_t *dxa = Dev(x), *dres = Dev(res);
    std::vector<double> r2(R * D), rr(R * D);
    for (size_t i = 0; i < R * D; ++i) rr[i] = double(bf2f(x[i])) + bf2f(res[i]);
    for (uint32_t r = 0; r < R; ++r) {
      double ss = 0;
      for (uint32_t i = 0; i < D; ++i) ss += rr[r * D + i] * rr[r * D + i];
      double inv = 1.0 / std::sqrt(ss / D + eps);
      for (uint32_t i = 0; i < D; ++i) r2[r * D + i] = rr[r * D + i] * inv * bf2f(w[i]);
    }
    st = roost_s2_fused_add_rms_norm(dxa, dres, dw, R, D, eps, nullptr);
    cudaDeviceSynchronize();
    Report("fused_add_rms_norm (flashinfer)", st, Get(dxa, R * D), r2, 2e-2);
    Report("  residual output", st, Get(dres, R * D), rr, 2e-2);
  }
  // QkNorm: [tokens, heads, 128], weight length 128
  {
    const uint32_t T = 9, H = 16, HD = 128;
    auto q = Rand(T * H * HD), qw = Rand(HD, 0.2f, 1.f);
    uint16_t *dq = Dev(q), *dqw = Dev(qw), *dqo;
    cudaMalloc((void**)&dqo, T * H * HD * 2);
    std::vector<double> rq(T * H * HD);
    for (uint32_t g = 0; g < T * H; ++g) {
      double ss = 0;
      for (uint32_t i = 0; i < HD; ++i) ss += double(bf2f(q[g * HD + i])) * bf2f(q[g * HD + i]);
      double inv = 1.0 / std::sqrt(ss / HD + eps);
      for (uint32_t i = 0; i < HD; ++i) rq[g * HD + i] = bf2f(q[g * HD + i]) * inv * bf2f(qw[i]);
    }
    st = roost_s2_qk_rms_norm(dq, dqw, dqo, T, H, HD, eps, nullptr);
    cudaDeviceSynchronize();
    Report("qk_rms_norm (flashinfer)", st, Get(dqo, T * H * HD), rq, 2e-2);
  }
  // SiluAndMul: Qwen3-1.7B intermediate 6144
  {
    const uint32_t T = 5, I = 6144;
    auto in = Rand(T * 2 * I);
    uint16_t *din = Dev(in), *dso;
    cudaMalloc((void**)&dso, T * I * 2);
    std::vector<double> rs(T * I);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t i = 0; i < I; ++i) {
        double a = bf2f(in[t * 2 * I + i]), b = bf2f(in[t * 2 * I + I + i]);
        rs[t * I + i] = a / (1.0 + std::exp(-a)) * b;
      }
    st = roost_s2_silu_and_mul(din, dso, T, I, nullptr);
    cudaDeviceSynchronize();
    Report("silu_and_mul (flashinfer)", st, Get(dso, T * I), rs, 2e-2);
  }
  // RoPE NeoX: q [nnz,16,128], k [nnz,8,128], theta 1e6 (Qwen3)
  {
    const uint32_t N = 11, QH = 16, KH = 8, HD = 128;
    const float theta = 1e6f;
    auto q = Rand(N * QH * HD), k = Rand(N * KH * HD);
    std::vector<int32_t> pos(N);
    for (uint32_t i = 0; i < N; ++i) pos[i] = 3 + 97 * i;
    uint16_t *dq = Dev(q), *dk = Dev(k);
    int32_t* dpos = Dev(pos);
    auto rope = [&](const std::vector<uint16_t>& a, uint32_t heads) {
      std::vector<double> o(a.size());
      for (uint32_t n = 0; n < N; ++n)
        for (uint32_t h = 0; h < heads; ++h)
          for (uint32_t i = 0; i < HD / 2; ++i) {
            double f = pos[n] * std::pow((double)theta, -2.0 * i / HD);
            double c = std::cos(f), s = std::sin(f);
            size_t b = (size_t(n) * heads + h) * HD;
            double x1 = bf2f(a[b + i]), x2 = bf2f(a[b + i + HD / 2]);
            o[b + i] = x1 * c - x2 * s;
            o[b + i + HD / 2] = x2 * c + x1 * s;
          }
      return o;
    };
    st = roost_s2_rope_neox(dq, dk, dpos, N, QH, KH, HD, theta, nullptr);
    cudaDeviceSynchronize();
    Report("rope_neox q (flashinfer)", st, Get(dq, q.size()), rope(q, QH), 2e-2);
    Report("rope_neox k (flashinfer)", st, Get(dk, k.size()), rope(k, KH), 2e-2);
  }
  // Append to paged cache: 2 sequences, 20 and 5 existing tokens, append 3 and 1; page 16.
  {
    const uint32_t KH = 8, HD = 128, P = 16, B = 2, POOL = 6;
    std::vector<int32_t> indptr = {0, 2, 3}, indices = {4, 1, 3}, last = {7, 6};  // after append
    std::vector<int32_t> bidx = {0, 0, 0, 1}, pos = {17, 18, 19, 5};
    const uint32_t NNZ = 4;
    auto kn = Rand(NNZ * KH * HD), vn = Rand(NNZ * KH * HD);
    std::vector<uint16_t> zero(POOL * P * KH * HD, 0);
    uint16_t *kp = Dev(zero), *vp = Dev(zero), *dk = Dev(kn), *dv = Dev(vn);
    int32_t *di = Dev(indices), *dp = Dev(indptr), *dl = Dev(last), *db = Dev(bidx), *dpo = Dev(pos);
    st = roost_s2_append_kv(kp, vp, di, dp, dl, dk, dv, db, dpo, B, NNZ, KH, HD, P, nullptr);
    cudaDeviceSynchronize();
    std::vector<double> rk(zero.size(), 0.0), rv(zero.size(), 0.0);
    for (uint32_t t = 0; t < NNZ; ++t) {
      uint32_t page = indices[indptr[bidx[t]] + pos[t] / P], off = pos[t] % P;
      for (uint32_t i = 0; i < KH * HD; ++i) {
        rk[(size_t(page) * P + off) * KH * HD + i] = bf2f(kn[t * KH * HD + i]);
        rv[(size_t(page) * P + off) * KH * HD + i] = bf2f(vn[t * KH * HD + i]);
      }
    }
    Report("append_paged_kv K (flashinfer)", st, Get(kp, zero.size()), rk, 0.0);
    Report("append_paged_kv V (flashinfer)", st, Get(vp, zero.size()), rv, 0.0);
  }
  std::printf(g_fail ? "S2 OPS TEST FAILED\n" : "S2 OPS TEST PASSED\n");
  return g_fail;
}

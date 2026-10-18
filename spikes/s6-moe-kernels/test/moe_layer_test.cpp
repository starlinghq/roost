// Spike S6 conformance driver: random Qwen3-30B-A3B-shaped weights (never written to disk), the C ABI layer on the GPU,
// and a double-precision CPU reference. Reports per-stage errors against (a) a stage-isolated reference: each stage's
// reference takes the GPU's previous stage output as exact input and is compared within one bf16 ulp plus an absolute
// term, and (b) a pure double reference with no intermediate rounding (end-to-end).
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <set>
#include <vector>

#include "roost_moe.h"

using bf16 = uint16_t;
static float bf2f(bf16 v) { uint32_t u = uint32_t(v) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {  // round to nearest even
  uint32_t u; std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return bf16(u >> 16);
}
static const double kRtol = 0.0079, kAtol = 1e-4;  // one bf16 ulp, plus f32-accumulation noise near zero
static double rbf(double d) { return bf2f(f2bf(float(d))); }

static uint64_t mix(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static void fill_normal(bf16* p, size_t n, uint64_t seed, float scale) {
  for (size_t i = 0; i < n; i += 2) {
    uint64_t a = mix(seed * 0x100000001b3ULL + i), b = mix(a ^ 0xabcdefULL);
    double u1 = ((a >> 11) + 1) * (1.0 / 9007199254740993.0), u2 = (b >> 11) * (1.0 / 9007199254740992.0);
    double r = std::sqrt(-2.0 * std::log(u1));
    p[i] = f2bf(float(scale * r * std::cos(6.283185307179586 * u2)));
    if (i + 1 < n) p[i + 1] = f2bf(float(scale * r * std::sin(6.283185307179586 * u2)));
  }
}

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(2);} } while (0)

struct Stat {
  double maxabs = 0, maxref = 0, sumsq_d = 0, sumsq_r = 0; size_t n = 0, bad = 0;
  void add(double gpu, double ref, double rtol, double atol) {
    double d = std::fabs(gpu - ref);
    maxabs = std::max(maxabs, d); maxref = std::max(maxref, std::fabs(ref)); sumsq_d += d * d; sumsq_r += ref * ref; ++n;
    if (d > atol + rtol * std::fabs(ref)) ++bad;
  }
  void print(const char* name) const {
    std::printf("  %-26s n=%-9zu max_abs=%.3e  max_abs/max_ref=%.3e  rms_rel=%.3e  outside_tol=%zu\n", name, n, maxabs,
                maxref > 0 ? maxabs / maxref : 0.0, sumsq_r > 0 ? std::sqrt(sumsq_d / sumsq_r) : 0.0, bad);
  }
};

int main(int argc, char** argv) {
  const int H = 2048, I = 768, E = 128, K = 8;  // Qwen/Qwen3-30B-A3B config.json
  std::vector<int> Ts; bool dup = false;
  for (int i = 1; i < argc; ++i) { if (!std::strcmp(argv[i], "dup")) dup = true; else Ts.push_back(std::atoi(argv[i])); }
  if (Ts.empty()) Ts = {1, 16, 128};
  const int Tmax = *std::max_element(Ts.begin(), Ts.end());
  const size_t nx = size_t(Tmax) * H, nr = size_t(E) * H, ngu = size_t(E) * 2 * I * H, ndn = size_t(E) * H * I;
  std::vector<bf16> x(nx), wr(nr), wgu(ngu), wdn(ndn);
  fill_normal(x.data(), nx, 1, 1.0f);
  fill_normal(wr.data(), nr, 2, 3.0f / std::sqrt(float(H)));
#pragma omp parallel for num_threads(4)
  for (int e = 0; e < E; ++e) {
    fill_normal(wgu.data() + size_t(e) * 2 * I * H, size_t(2 * I) * H, 1000 + e, 1.0f / std::sqrt(float(H)));
    fill_normal(wdn.data() + size_t(e) * H * I, size_t(H) * I, 5000 + e, 1.0f / std::sqrt(float(I)));
  }
  if (dup) for (int t = 1; t < Tmax; ++t) std::memcpy(&x[size_t(t) * H], x.data(), H * 2);

  void *dx, *dwr, *dwgu, *dwdn;
  CK(cudaMalloc(&dx, nx * 2)); CK(cudaMalloc(&dwr, nr * 2)); CK(cudaMalloc(&dwgu, ngu * 2)); CK(cudaMalloc(&dwdn, ndn * 2));
  CK(cudaMemcpy(dx, x.data(), nx * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dwr, wr.data(), nr * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwgu, wgu.data(), ngu * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dwdn, wdn.data(), ndn * 2, cudaMemcpyHostToDevice));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s sm_%d%d; shapes H=%d I=%d E=%d K=%d; case=%s\n", prop.name, prop.major, prop.minor, H, I, E, K, dup ? "all-tokens-identical" : "random");

  int exit_code = 0;
  for (int T : Ts) {
    RoostMoeLayerParams p{};
    p.struct_size = sizeof p; p.abi_minor = ROOST_MOE_SPIKE_ABI_MINOR;
    p.num_tokens = T; p.hidden = H; p.inter = I; p.num_experts = E; p.topk = K; p.norm_topk_prob = 1;
    p.x = dx; p.w_router = dwr; p.w_gate_up = dwgu; p.w_down = dwdn;
    size_t wsb = 0;
    if (roost_moe_layer_workspace_bytes(&p, &wsb)) { std::fprintf(stderr, "ws: %s\n", roost_moe_last_error()); return 2; }
    void *dws, *dout, *dlog, *dtw, *dti, *dg1, *dact, *dg2, *dinv;
    const size_t TK = size_t(T) * K;
    CK(cudaMalloc(&dws, wsb)); CK(cudaMalloc(&dout, size_t(T) * H * 2)); CK(cudaMalloc(&dlog, size_t(T) * E * 2));
    CK(cudaMalloc(&dtw, TK * 4)); CK(cudaMalloc(&dti, TK * 4)); CK(cudaMalloc(&dg1, TK * 2 * I * 2));
    CK(cudaMalloc(&dact, TK * I * 2)); CK(cudaMalloc(&dg2, TK * H * 2)); CK(cudaMalloc(&dinv, TK * 4));
    p.workspace = dws; p.workspace_bytes = wsb; p.out = dout; p.dbg_logits = dlog; p.dbg_topk_weights = dtw;
    p.dbg_topk_ids = dti; p.dbg_gemm1 = dg1; p.dbg_act = dact; p.dbg_gemm2 = dg2; p.dbg_inv = dinv;
    if (roost_moe_layer(&p, nullptr)) { std::fprintf(stderr, "layer: %s\n", roost_moe_last_error()); return 2; }
    CK(cudaDeviceSynchronize());

    std::vector<bf16> out(size_t(T) * H), glog(size_t(T) * E), gg1(TK * 2 * I), gact(TK * I), gg2(TK * H);
    std::vector<float> gtw(TK); std::vector<int> gti(TK), ginv(TK);
    CK(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(glog.data(), dlog, glog.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gtw.data(), dtw, TK * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gti.data(), dti, TK * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gg1.data(), dg1, gg1.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gact.data(), dact, gact.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gg2.data(), dg2, gg2.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ginv.data(), dinv, TK * 4, cudaMemcpyDeviceToHost));

    Stat sLog, sTw, sG1, sAct, sG2m, sOutM, sOutP, sG2p;
    size_t id_sets_equal = 0, id_orders_total = 0; size_t tokens_compared = 0;
    std::vector<int> tok_ok(T, 0);
    // Reference: per token, independent of the GPU, except that stage comparison is limited to tokens whose top-k set matches.
    std::vector<Stat> sl(T), stw(T), sg1(T), sact(T), sg2m(T), sg2p(T), som(T), sop(T);
    std::vector<int> set_eq(T, 0);
#pragma omp parallel for num_threads(4) schedule(dynamic)
    for (int t = 0; t < T; ++t) {
      std::vector<double> lg(E), pr(E);
      for (int e = 0; e < E; ++e) {
        double a = 0; for (int h = 0; h < H; ++h) a += double(bf2f(x[size_t(t) * H + h])) * bf2f(wr[size_t(e) * H + h]);
        lg[e] = rbf(a); sl[t].add(bf2f(glog[size_t(t) * E + e]), lg[e], kRtol, kAtol);
      }
      double mx = *std::max_element(lg.begin(), lg.end()), s = 0;
      for (int e = 0; e < E; ++e) { pr[e] = std::exp(lg[e] - mx); s += pr[e]; }
      std::vector<int> ord(E); std::iota(ord.begin(), ord.end(), 0);
      std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return pr[a] > pr[b]; });
      double ssel = 0; for (int k = 0; k < K; ++k) ssel += pr[ord[k]];
      std::set<int> rs(ord.begin(), ord.begin() + K), gs(gti.begin() + size_t(t) * K, gti.begin() + size_t(t) * K + K);
      if (rs != gs) continue;
      set_eq[t] = 1;
      std::vector<double> yp(size_t(K) * H), yg(size_t(K) * H), w(K);
      for (int k = 0; k < K; ++k) {
        int e = gti[size_t(t) * K + k]; w[k] = pr[e] / ssel;
        stw[t].add(gtw[size_t(t) * K + k], w[k], 1e-5, 0);
        const bf16* Wgu = &wgu[size_t(e) * 2 * I * H]; const bf16* Wdn = &wdn[size_t(e) * H * I];
        std::vector<double> g(2 * I), actp(I);
        for (int j = 0; j < 2 * I; ++j) { double a = 0; for (int h = 0; h < H; ++h) a += double(bf2f(x[size_t(t) * H + h])) * bf2f(Wgu[size_t(j) * H + h]); g[j] = a; }
        size_t row = size_t(ginv[size_t(t) * K + k]);
        // Stage-isolated checks: each stage's reference takes the GPU's previous stage as exact input.
        for (int j = 0; j < 2 * I; ++j) sg1[t].add(bf2f(gg1[row * 2 * I + j]), rbf(g[j]), kRtol, kAtol);
        std::vector<double> actg(I);
        for (int j = 0; j < I; ++j) {
          double gg = bf2f(gg1[row * 2 * I + j]), uu = bf2f(gg1[row * 2 * I + I + j]);
          actg[j] = rbf(gg / (1 + std::exp(-gg)) * uu);
          actp[j] = g[j] / (1 + std::exp(-g[j])) * g[I + j];
          sact[t].add(bf2f(gact[row * I + j]), actg[j], kRtol, kAtol);
        }
        for (int h = 0; h < H; ++h) {
          double am = 0, ap = 0;
          for (int j = 0; j < I; ++j) { double wv = bf2f(Wdn[size_t(h) * I + j]); am += double(bf2f(gact[row * I + j])) * wv; ap += actp[j] * wv; }
          yg[size_t(k) * H + h] = bf2f(gg2[row * H + h]); yp[size_t(k) * H + h] = ap;
          sg2m[t].add(bf2f(gg2[row * H + h]), rbf(am), kRtol, kAtol);
          sg2p[t].add(bf2f(gg2[row * H + h]), ap, 0.0, 0.0);
        }
      }
      for (int h = 0; h < H; ++h) {
        double og = 0, op = 0;
        for (int k = 0; k < K; ++k) { og += double(gtw[size_t(t) * K + k]) * yg[size_t(k) * H + h]; op += w[k] * yp[size_t(k) * H + h]; }
        double g = bf2f(out[size_t(t) * H + h]);
        som[t].add(g, rbf(og), kRtol, kAtol); sop[t].add(g, op, 0.0, 0.0);
      }
    }
    auto merge = [](Stat& d, const Stat& s) { d.maxabs = std::max(d.maxabs, s.maxabs); d.maxref = std::max(d.maxref, s.maxref); d.sumsq_d += s.sumsq_d; d.sumsq_r += s.sumsq_r; d.n += s.n; d.bad += s.bad; };
    for (int t = 0; t < T; ++t) {
      merge(sLog, sl[t]); id_sets_equal += set_eq[t];
      if (!set_eq[t]) continue;
      ++tokens_compared; merge(sTw, stw[t]); merge(sG1, sg1[t]); merge(sAct, sact[t]); merge(sG2m, sg2m[t]); merge(sG2p, sg2p[t]); merge(sOutM, som[t]); merge(sOutP, sop[t]);
    }
    (void)id_orders_total; (void)tok_ok;
    std::printf("T=%d workspace=%.1f MiB; top-k expert set equals reference for %zu/%d tokens\n", T, wsb / 1048576.0, id_sets_equal, T);
    sLog.print("router logits (bf16 ulp)"); sTw.print("top-k weights (f32, 1e-5)");
    sG1.print("grouped gemm1 (stage)"); sAct.print("silu*up (stage)"); sG2m.print("grouped gemm2 (stage)");
    sOutM.print("combine (stage)");  sOutP.print("layer out vs pure f64");
    if (sOutP.sumsq_r > 0) std::printf("  pure-f64 end-to-end rms_rel=%.3e (bf16 pipeline floor is ~2^-9 per rounding)\n", std::sqrt(sOutP.sumsq_d / sOutP.sumsq_r));
    bool pass = id_sets_equal == size_t(T) && sTw.bad == 0 && sG1.bad == 0 && sAct.bad == 0 && sG2m.bad == 0 && sOutM.bad == 0 && sLog.bad == 0 && std::sqrt(sOutP.sumsq_d / sOutP.sumsq_r) < 0.01 && sOutP.maxabs / sOutP.maxref < 0.03;
    std::printf("  RESULT T=%d: %s\n", T, pass ? "PASS" : "FAIL");
    if (!pass) exit_code = 1;

    // Timing (informational; fidelity above comes first).
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    p.dbg_logits = p.dbg_topk_weights = p.dbg_topk_ids = p.dbg_gemm1 = p.dbg_act = p.dbg_gemm2 = p.dbg_inv = nullptr;
    for (int i = 0; i < 3; ++i) roost_moe_layer(&p, nullptr);
    cudaEventRecord(a); for (int i = 0; i < 20; ++i) roost_moe_layer(&p, nullptr); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    std::printf("  layer time %.3f ms (mean of 20, no debug copies)\n", ms / 20);
    cudaFree(dws); cudaFree(dout); cudaFree(dlog); cudaFree(dtw); cudaFree(dti); cudaFree(dg1); cudaFree(dact); cudaFree(dg2); cudaFree(dinv);
  }
  return exit_code;
}

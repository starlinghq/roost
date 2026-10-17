// Spike S8 (R-065): worker process and shared-memory rings (C8).
// Throwaway evidence code; not part of Roost.slnx. One binary, several subcommands:
//   bench    per-step latency: direct (no rings), inproc (rings, worker thread), proc (rings, worker process)
//   fault    sticky / hang / recoverable / crash injection, detection and recovery
//   respawn  kill, respawn, re-handshake and weight reload timing
//   calib    GPU time to read N bytes of device memory (sets the synthetic step time)
//   worker   internal: the worker process entry point
#include <cuda_runtime.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <x86intrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

static inline u64 now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return u64(ts.tv_sec) * 1000000000ull + u64(ts.tv_nsec);
}

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      fprintf(stderr, "FATAL %s:%d %s: %s\n", __FILE__, __LINE__, #x,                 \
              cudaGetErrorString(e_));                                                \
      exit(2);                                                                        \
    }                                                                                 \
  } while (0)

// ---- shared-memory layout (a stand-in for C2/C8; blittable, atomics are lock-free) ----
constexpr u32 kSlots = 8;
constexpr size_t kPlanBytes = 4096;  // a batch-1 plan with sequence table, ids, sampling row, cache ops
constexpr size_t kResBytes = 1024;   // ids, finish flags, timings
constexpr u32 kAbi = 0x00010000;     // major 1, minor 0
constexpr u32 kMagic = 0x38534f52;   // "ROS8"

enum Kind : u32 { kNormal = 0, kIllegal = 1, kHang = 2, kRecoverable = 3, kCrash = 4 };
enum Status : u32 { kOk = 0, kRecoverableFault = 1, kStickyFault = 2 };
enum State : u32 { kInit = 0, kReady = 1, kFaulted = 2, kRejected = 3 };

struct alignas(64) Cursor {
  std::atomic<u64> v;
  char pad[56];
};

struct PlanSlot {
  u32 kind;
  u32 pad;
  u64 seq;
  u64 t_pub;   // host CLOCK_MONOTONIC when published
  u64 gpu_ns;  // synthetic GPU time of the step
  u64 target;  // address the illegal-access kernel writes to
  unsigned char body[kPlanBytes - 40];
};
struct ResSlot {
  u64 seq;
  u32 status;
  u32 cuda_err;
  u64 t_seen;  // worker saw the plan
  u64 t_done;  // worker published the result
  u64 gpu_event_ns;
  unsigned char body[kResBytes - 40];
};
static_assert(sizeof(PlanSlot) == kPlanBytes && sizeof(ResSlot) == kResBytes);

struct Shm {
  u32 magic;
  u32 abi_expected;  // host writes, worker compares with its own
  unsigned char manifest_expected[32];
  u64 weights_expected;
  u64 weights_loaded;
  u32 worker_abi;
  u32 reject_reason;
  std::atomic<u32> state;
  std::atomic<u32> stop;
  std::atomic<u64> worker_pid;
  std::atomic<u64> heartbeat;
  u64 t_start, t_cuda, t_registered, t_weights, t_ready;  // worker timeline
  Cursor plan_head, plan_tail, res_head, res_tail;
  PlanSlot plan[kSlots];
  ResSlot res[kSlots];
};
constexpr size_t shm_bytes() { return (sizeof(Shm) + 4095) & ~size_t(4095); }
static const unsigned char kManifest[32] = {'r', 'o', 'o', 's', 't', '-', 's', '8', '-', 'm', 'a', 'n', 'i', 'f', 'e', 's',
                                            't', '-', 'd', 'i', 'g', 'e', 's', 't', '-', '0', '0', '0', '0', '0', '0', '1'};

// ---- synthetic step kernel ----
__device__ __forceinline__ u64 gtimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__global__ void step_kernel(const PlanSlot* plan, ResSlot* res, volatile int* never_set) {
  if (threadIdx.x != 0) return;
  u32 kind = plan->kind;
  if (kind == kIllegal) {
    *(volatile int*)plan->target = 1;  // unmapped address: sticky illegal-address fault
    return;
  }
  if (kind == kHang) {
    while (*never_set == 0) {
    }
    return;
  }
  u64 t0 = gtimer();
  while (gtimer() - t0 < plan->gpu_ns) {
  }
  res->seq = plan->seq;
}

// ---- worker core: used by the worker process and by the in-process worker thread ----
struct WorkerArgs {
  std::string weights;      // "" or "-" for none
  bool exit_on_sticky = true;
  bool inproc = false;
};
struct Gpu {
  cudaStream_t s;
  PlanSlot* d_plan;
  ResSlot* d_res;
  int* d_never;
  void* d_weights = nullptr;
  cudaEvent_t e0, e1;
};

static cudaError_t gpu_step(Gpu& g, PlanSlot* p, ResSlot* r, u64* event_ns) {
  cudaError_t e;
  if ((e = cudaEventRecord(g.e0, g.s)) != cudaSuccess) return e;
  if ((e = cudaMemcpyAsync(g.d_plan, p, sizeof(PlanSlot), cudaMemcpyHostToDevice, g.s)) != cudaSuccess) return e;
  u32 threads = p->kind == kRecoverable ? 2048 : 32;  // 2048 threads/block: invalid configuration, not sticky
  step_kernel<<<1, threads, 0, g.s>>>(g.d_plan, g.d_res, g.d_never);
  if ((e = cudaGetLastError()) != cudaSuccess) return e;
  if ((e = cudaMemcpyAsync(r->body, g.d_res->body, sizeof(r->body), cudaMemcpyDeviceToHost, g.s)) != cudaSuccess) return e;
  if ((e = cudaEventRecord(g.e1, g.s)) != cudaSuccess) return e;
  if ((e = cudaStreamSynchronize(g.s)) != cudaSuccess) return e;
  float ms = 0;
  cudaEventElapsedTime(&ms, g.e0, g.e1);
  *event_ns = u64(ms * 1e6f);
  return cudaSuccess;
}

static void gpu_init(Shm* shm, Gpu& g, bool register_ring) {
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost));
  CK(cudaFree(0));
  shm->t_cuda = now_ns();
  if (register_ring) CK(cudaHostRegister(shm, shm_bytes(), cudaHostRegisterPortable));
  shm->t_registered = now_ns();
  CK(cudaStreamCreateWithFlags(&g.s, cudaStreamNonBlocking));
  CK(cudaMalloc(&g.d_plan, sizeof(PlanSlot)));
  CK(cudaMalloc(&g.d_res, sizeof(ResSlot)));
  CK(cudaMalloc(&g.d_never, 4));
  CK(cudaMemset(g.d_never, 0, 4));
  CK(cudaMemset(g.d_res, 0, sizeof(ResSlot)));
  CK(cudaEventCreate(&g.e0));
  CK(cudaEventCreate(&g.e1));
}

static bool load_weights(Shm* shm, Gpu& g, const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  struct stat st;
  fstat(fd, &st);
  size_t n = size_t(st.st_size);
  void* m = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
  if (m == MAP_FAILED) return false;
  CK(cudaMalloc(&g.d_weights, n));
  const size_t chunk = size_t(64) << 20;
  for (size_t off = 0; off < n; off += chunk) {
    size_t c = std::min(chunk, n - off);
    CK(cudaMemcpy((char*)g.d_weights + off, (char*)m + off, c, cudaMemcpyDefault));
  }
  munmap(m, n);
  close(fd);
  shm->weights_loaded = n;
  return true;
}

static int worker_run(Shm* shm, const WorkerArgs& a, Gpu* external = nullptr) {
  Gpu local;
  Gpu& g = external ? *external : local;
  shm->t_start = now_ns();
  if (!a.inproc) {
    rlimit rl{0, 0};  // a CUDA process core dump is large and slow; the host restarts the worker instead
    setrlimit(RLIMIT_CORE, &rl);
  }
  shm->worker_pid.store(u64(getpid()));
  shm->worker_abi = kAbi;
  if (shm->magic != kMagic || shm->abi_expected != kAbi || memcmp(shm->manifest_expected, kManifest, 32) != 0) {
    shm->reject_reason = shm->abi_expected != kAbi ? 1 : 2;
    shm->state.store(kRejected, std::memory_order_release);
    return 3;  // handshake rejected before any CUDA work
  }
  gpu_init(shm, g, true);
  if (!a.weights.empty() && a.weights != "-") {
    if (!load_weights(shm, g, a.weights)) return 5;
    if (shm->weights_expected && shm->weights_loaded != shm->weights_expected) {
      shm->reject_reason = 3;
      shm->state.store(kRejected, std::memory_order_release);
      return 3;
    }
  }
  shm->t_weights = now_ns();
  std::atomic<bool> hb_run{true};
  std::thread hb([&] {  // heartbeat: independent of the stepping thread, so a hung step still beats
    while (hb_run.load(std::memory_order_relaxed)) {
      shm->heartbeat.fetch_add(1, std::memory_order_relaxed);
      usleep(1000);
    }
  });
  shm->t_ready = now_ns();
  shm->state.store(kReady, std::memory_order_release);
  int rc = 0;
  while (!shm->stop.load(std::memory_order_relaxed)) {
    u64 tail = shm->plan_tail.v.load(std::memory_order_relaxed);
    if (shm->plan_head.v.load(std::memory_order_acquire) == tail) {
      _mm_pause();
      continue;
    }
    PlanSlot* p = &shm->plan[tail % kSlots];
    ResSlot* r = &shm->res[tail % kSlots];
    r->t_seen = now_ns();
    r->seq = p->seq;
    r->status = kOk;
    r->cuda_err = 0;
    if (p->kind == kCrash) {
      if (a.inproc) return 6;
      abort();  // SIGABRT; core dumps are disabled at worker start
    }
    u64 ev = 0;
    cudaError_t e = gpu_step(g, p, r, &ev);
    r->gpu_event_ns = ev;
    bool sticky = false;
    if (e != cudaSuccess) {
      r->cuda_err = u32(e);
      cudaGetLastError();  // clears non-sticky errors
      // classification probe: a recoverable error leaves the context usable, a sticky one fails again
      cudaError_t probe = cudaMemsetAsync(g.d_never, 0, 4, g.s);
      if (probe == cudaSuccess) probe = cudaStreamSynchronize(g.s);
      sticky = probe != cudaSuccess;
      r->status = sticky ? kStickyFault : kRecoverableFault;
    }
    r->t_done = now_ns();
    shm->plan_tail.v.store(tail + 1, std::memory_order_relaxed);
    shm->res_head.v.store(tail + 1, std::memory_order_release);
    if (sticky && a.exit_on_sticky) {
      shm->state.store(kFaulted, std::memory_order_release);
      rc = 4;
      break;
    }
  }
  hb_run = false;
  hb.join();
  return rc;
}

// ---- host side ----
struct Handle {
  Shm* shm = nullptr;
  std::string path;
  pid_t pid = 0;
  std::thread th;
  bool inproc = false;
  u64 t_spawn = 0;
  Gpu gpu{};
};
static int g_gen = 0;
static std::string g_self;

static Shm* map_shm(const std::string& path, bool create) {
  int fd = open(path.c_str(), create ? (O_CREAT | O_RDWR | O_TRUNC) : O_RDWR, 0600);
  if (fd < 0) {
    perror("open shm");
    exit(2);
  }
  if (create && ftruncate(fd, shm_bytes()) != 0) exit(2);
  void* m = mmap(nullptr, shm_bytes(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) {
    perror("mmap");
    exit(2);
  }
  close(fd);
  return (Shm*)m;
}

static void start(Handle& h, bool inproc, const WorkerArgs& a, bool bad_manifest = false, u64 weights_expected = 0) {
  h.inproc = inproc;
  h.path = "/dev/shm/roost-s8-" + std::to_string(getpid()) + "-" + std::to_string(g_gen++);
  h.shm = map_shm(h.path, true);
  memset(h.shm, 0, shm_bytes());
  h.shm->magic = kMagic;
  h.shm->abi_expected = kAbi;
  memcpy(h.shm->manifest_expected, kManifest, 32);
  if (bad_manifest) h.shm->manifest_expected[0] ^= 1;
  h.shm->weights_expected = weights_expected;
  h.t_spawn = now_ns();
  if (inproc) {
    h.th = std::thread([&h, a] { worker_run(h.shm, a, &h.gpu); });
  } else {
    pid_t pid = fork();
    if (pid == 0) {
      std::string w = a.weights.empty() ? "-" : a.weights;
      execl(g_self.c_str(), g_self.c_str(), "worker", h.path.c_str(), w.c_str(), a.exit_on_sticky ? "1" : "0", (char*)nullptr);
      _exit(127);
    }
    h.pid = pid;
  }
}

static bool wait_ready(Handle& h, double timeout_s) {
  u64 dl = now_ns() + u64(timeout_s * 1e9);
  while (now_ns() < dl) {
    u32 s = h.shm->state.load(std::memory_order_acquire);
    if (s == kReady) return true;
    if (s == kRejected) return false;
    if (!h.inproc && h.pid && waitpid(h.pid, nullptr, WNOHANG) == h.pid) {
      h.pid = 0;
      return false;
    }
    usleep(100);
  }
  return false;
}

static void submit(Handle& h, u32 kind, u64 gpu_ns, u64 seq) {
  u64 head = h.shm->plan_head.v.load(std::memory_order_relaxed);
  PlanSlot* p = &h.shm->plan[head % kSlots];
  p->kind = kind;
  p->seq = seq;
  p->gpu_ns = gpu_ns;
  p->target = 0x10;
  p->t_pub = now_ns();
  h.shm->plan_head.v.store(head + 1, std::memory_order_release);
}

enum Wait { kGot, kDeadline, kDied };
static Wait wait_result(Handle& h, ResSlot* out, u64* t_got, u64 deadline_ns) {
  u64 tail = h.shm->res_tail.v.load(std::memory_order_relaxed);
  u64 dl = now_ns() + deadline_ns;
  for (u32 it = 0;; ++it) {
    if (h.shm->res_head.v.load(std::memory_order_acquire) > tail) {
      *t_got = now_ns();
      *out = h.shm->res[tail % kSlots];
      h.shm->res_tail.v.store(tail + 1, std::memory_order_release);
      return kGot;
    }
    if ((it & 255) == 0) {
      if (now_ns() > dl) return kDeadline;
      if (!h.inproc && h.pid && waitpid(h.pid, nullptr, WNOHANG) == h.pid) {
        h.pid = 0;
        return kDied;
      }
    }
    _mm_pause();
  }
}

static void stop_worker(Handle& h) {
  if (h.shm) h.shm->stop.store(1);
  if (h.inproc) {
    if (h.th.joinable()) h.th.join();
  } else if (h.pid) {
    waitpid(h.pid, nullptr, 0);
    h.pid = 0;
  }
}

static double pct(std::vector<u64>& v, double p) {  // v sorted
  if (v.empty()) return 0;
  size_t i = size_t(p / 100.0 * double(v.size() - 1) + 0.5);
  return double(v[i]) / 1000.0;  // us
}
static void report(const char* name, std::vector<u64> v) {
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (u64 x : v) sum += double(x);
  printf("  %-22s n=%zu min=%.1f p50=%.1f p90=%.1f p99=%.1f p99.9=%.1f max=%.1f mean=%.1f (us)\n", name, v.size(), pct(v, 0), pct(v, 50),
         pct(v, 90), pct(v, 99), pct(v, 99.9), pct(v, 100), sum / double(v.size()) / 1000.0);
}

static std::string arg(int argc, char** argv, const char* key, const char* def) {
  for (int i = 2; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return argv[i + 1];
  return def;
}

// ---- bench ----
static int cmd_bench(int argc, char** argv) {
  std::string mode = arg(argc, argv, "--mode", "proc");
  long steps = atol(arg(argc, argv, "--steps", "5000").c_str());
  long warm = atol(arg(argc, argv, "--warmup", "200").c_str());
  u64 gpu_ns = u64(atof(arg(argc, argv, "--gpu-us", "5000").c_str()) * 1000);
  int depth = atoi(arg(argc, argv, "--depth", "1").c_str());
  std::string csv = arg(argc, argv, "--csv", "");
  std::vector<u64> lat, to_worker, from_worker, worker_busy, ev;
  u64 t_begin = 0, t_end = 0;
  printf("bench mode=%s steps=%ld gpu_us=%.0f depth=%d loadavg=", mode.c_str(), steps, double(gpu_ns) / 1000, depth);
  {
    double l[1];
    getloadavg(l, 1);
    printf("%.2f\n", l[0]);
  }
  if (mode == "direct") {
    Shm* shm = map_shm("/dev/shm/roost-s8-direct-" + std::to_string(getpid()), true);
    memset(shm, 0, shm_bytes());
    Gpu g;
    gpu_init(shm, g, true);
    for (long i = 0; i < steps + warm; ++i) {
      PlanSlot* p = &shm->plan[0];
      ResSlot* r = &shm->res[0];
      p->kind = kNormal;
      p->seq = u64(i);
      p->gpu_ns = gpu_ns;
      u64 t0 = now_ns();
      p->t_pub = t0;
      u64 e = 0;
      if (gpu_step(g, p, r, &e) != cudaSuccess) return 1;
      u64 t1 = now_ns();
      if (i == warm) t_begin = t0;
      if (i >= warm) {
        lat.push_back(t1 - t0);
        ev.push_back(e);
      }
      t_end = t1;
    }
    unlink(("/dev/shm/roost-s8-direct-" + std::to_string(getpid())).c_str());
  } else {
    Handle h;
    WorkerArgs a;
    a.inproc = mode == "inproc";
    start(h, mode == "inproc", a);
    if (!wait_ready(h, 30)) {
      fprintf(stderr, "worker not ready\n");
      return 1;
    }
    long sent = 0, got = 0;
    ResSlot r;
    u64 tg;
    std::vector<u64> pub_t(size_t(steps + warm));
    while (got < steps + warm) {
      while (sent < steps + warm && sent - got < depth) {
        if (sent == warm) t_begin = now_ns();
        submit(h, kNormal, gpu_ns, u64(sent));
        pub_t[size_t(sent)] = h.shm->plan[(h.shm->plan_head.v.load() - 1) % kSlots].t_pub;
        ++sent;
      }
      if (wait_result(h, &r, &tg, 10ull * 1000000000ull) != kGot) {
        fprintf(stderr, "result timeout\n");
        return 1;
      }
      if (r.status != kOk) {
        fprintf(stderr, "step fault\n");
        return 1;
      }
      if (got >= warm) {
        lat.push_back(tg - pub_t[size_t(r.seq)]);
        to_worker.push_back(r.t_seen - pub_t[size_t(r.seq)]);
        from_worker.push_back(tg - r.t_done);
        worker_busy.push_back(r.t_done - r.t_seen);
        ev.push_back(r.gpu_event_ns);
      }
      t_end = tg;
      ++got;
    }
    stop_worker(h);
    unlink(h.path.c_str());
  }
  double wall_per_step_us = double(t_end - t_begin) / double(steps) / 1000.0;
  report("step latency", lat);
  if (!to_worker.empty()) {
    report("publish->worker seen", to_worker);
    report("worker done->host seen", from_worker);
    report("worker busy", worker_busy);
  }
  report("gpu event (H2D..D2H)", ev);
  printf("  wall per step (throughput) = %.1f us\n", wall_per_step_us);
  if (!csv.empty()) {
    FILE* f = fopen(csv.c_str(), "w");
    for (u64 x : lat) fprintf(f, "%llu\n", (unsigned long long)x);
    fclose(f);
  }
  return 0;
}

// ---- fault injection ----
static const char* status_name(u32 s) { return s == kOk ? "ok" : s == kRecoverableFault ? "recoverable" : "sticky"; }

static int cmd_fault(int argc, char** argv) {
  std::string mode = arg(argc, argv, "--mode", "proc");
  std::string kind_s = arg(argc, argv, "--kind", "sticky");
  double deadline_ms = atof(arg(argc, argv, "--deadline-ms", "300").c_str());
  u32 kind = kind_s == "sticky" ? kIllegal : kind_s == "hang" ? kHang : kind_s == "recoverable" ? kRecoverable : kCrash;
  bool inproc = mode == "inproc";
  u64 dl = u64(deadline_ms * 1e6);
  u64 step_ns = 2000000;
  Handle h;
  WorkerArgs a;
  a.exit_on_sticky = !inproc;
  a.inproc = inproc;
  start(h, inproc, a);
  if (!wait_ready(h, 30)) return 1;
  ResSlot r;
  u64 tg;
  for (int i = 0; i < 5; ++i) {
    submit(h, kNormal, step_ns, u64(i));
    wait_result(h, &r, &tg, dl);
  }
  printf("fault mode=%s kind=%s deadline=%.0fms\n  warm steps ok (last status=%s)\n", mode.c_str(), kind_s.c_str(), deadline_ms, status_name(r.status));
  u64 hb0 = h.shm->heartbeat.load();
  u64 t_inj = now_ns();
  submit(h, kind, step_ns, 100);
  Wait w = wait_result(h, &r, &tg, dl);
  u64 t_det = now_ns();
  bool respawn = false;
  if (w == kGot) {
    printf("  detected via result ring after %.2f ms: status=%s cuda_err=%u (%s)\n", double(t_det - t_inj) / 1e6, status_name(r.status), r.cuda_err,
           r.cuda_err ? cudaGetErrorString(cudaError_t(r.cuda_err)) : "none");
    printf("  worker state after fault = %u (1 ready, 2 faulted)\n", h.shm->state.load());
    respawn = r.status == kStickyFault && !inproc;
    if (inproc && r.status == kStickyFault) {
      for (int i = 0; i < 3; ++i) {
        submit(h, kNormal, step_ns, u64(200 + i));
        Wait w2 = wait_result(h, &r, &tg, dl);
        printf("  follow-up normal step %d in the same process: %s status=%s cuda_err=%u (%s)\n", i, w2 == kGot ? "result" : "no result",
               w2 == kGot ? status_name(r.status) : "-", r.cuda_err, r.cuda_err ? cudaGetErrorString(cudaError_t(r.cuda_err)) : "none");
      }
      printf("  => context is poisoned for the lifetime of the process; only a process restart recovers\n");
    }
    if (r.status == kRecoverableFault) {
      submit(h, kNormal, step_ns, 300);
      Wait w2 = wait_result(h, &r, &tg, dl);
      printf("  follow-up normal step in the same worker: %s status=%s\n", w2 == kGot ? "result" : "no result", w2 == kGot ? status_name(r.status) : "-");
    }
  } else if (w == kDied) {
    printf("  detected via waitpid (worker exited without a result) after %.2f ms\n", double(t_det - t_inj) / 1e6);
    respawn = true;
  } else {
    u64 hb1 = h.shm->heartbeat.load();
    printf("  detected via step deadline after %.2f ms; heartbeat advanced by %llu in that time => %s\n", double(t_det - t_inj) / 1e6,
           (unsigned long long)(hb1 - hb0), hb1 > hb0 + 2 ? "worker alive but step not completing: HANG" : "worker silent: stalled/dead");
    if (inproc) {
      printf("  => in-process: the stepping thread is stuck in the driver and cannot be cancelled or killed; the host application must exit\n");
      fflush(stdout);
      _exit(0);
    }
    u64 tk = now_ns();
    kill(h.pid, SIGKILL);
    waitpid(h.pid, nullptr, 0);
    h.pid = 0;
    printf("  SIGKILL to reaped: %.2f ms\n", double(now_ns() - tk) / 1e6);
    respawn = true;
  }
  if (respawn) {
    if (h.pid) {
      waitpid(h.pid, nullptr, 0);
      h.pid = 0;
    }
    unlink(h.path.c_str());
    u64 t0 = now_ns();
    Handle h2;
    start(h2, false, a);
    bool ok = wait_ready(h2, 30);
    u64 t1 = now_ns();
    submit(h2, kNormal, step_ns, 400);
    Wait w2 = wait_result(h2, &r, &tg, dl);
    printf("  respawn (no weights): ready=%d in %.1f ms, first step %s status=%s at +%.1f ms\n", ok, double(t1 - t0) / 1e6, w2 == kGot ? "result" : "none",
           w2 == kGot ? status_name(r.status) : "-", double(tg - t0) / 1e6);
    stop_worker(h2);
    unlink(h2.path.c_str());
    return 0;
  }
  stop_worker(h);
  unlink(h.path.c_str());
  return 0;
}

// ---- respawn with weights ----
static int cmd_respawn(int argc, char** argv) {
  std::string weights = arg(argc, argv, "--weights", "");
  int rounds = atoi(arg(argc, argv, "--rounds", "5").c_str());
  struct stat st {};
  u64 wbytes = 0;
  if (!weights.empty() && stat(weights.c_str(), &st) == 0) wbytes = u64(st.st_size);
  WorkerArgs a;
  a.weights = weights;
  printf("respawn weights=%s (%llu bytes) rounds=%d\n", weights.empty() ? "none" : weights.c_str(), (unsigned long long)wbytes, rounds);
  printf("  round  kill->reaped  fork->state  exec+cuda_init  register  weights   handshake->ready  ready->first_step  spawn->first_step (ms)\n");
  for (int i = 0; i < rounds; ++i) {
    Handle h;
    start(h, false, a, false, wbytes);
    bool ok = wait_ready(h, 120);
    if (!ok) {
      printf("  worker failed to start\n");
      return 1;
    }
    u64 t_ready_seen = now_ns();
    ResSlot r;
    u64 tg;
    submit(h, kNormal, 1000000, 1);
    wait_result(h, &r, &tg, 5000000000ull);
    u64 tk = now_ns();
    kill(h.pid, SIGKILL);
    waitpid(h.pid, nullptr, 0);
    h.pid = 0;
    u64 tr = now_ns();
    double ms = 1e6;
    printf("  %3d    %8.2f      %8.2f     %8.2f      %7.2f   %8.2f   %8.2f          %8.2f           %8.2f\n", i, double(tr - tk) / ms,
           double(h.shm->t_start - h.t_spawn) / ms, double(h.shm->t_cuda - h.shm->t_start) / ms, double(h.shm->t_registered - h.shm->t_cuda) / ms,
           double(h.shm->t_weights - h.shm->t_registered) / ms, double(h.shm->t_ready - h.shm->t_weights) / ms, double(tg - h.shm->t_ready) / ms,
           double(tg - h.t_spawn) / ms);
    (void)t_ready_seen;
    unlink(h.path.c_str());
  }
  printf("  columns: fork->state = fork+exec until worker main starts; exec+cuda_init = cudaFree(0) context creation;\n"
         "  register = cudaHostRegister of the ring segment; weights = mmap + chunked H2D; handshake->ready = heartbeat thread + state publish.\n");
  // handshake rejection
  {
    Handle h;
    u64 t0 = now_ns();
    start(h, false, a, true, wbytes);
    bool ok = wait_ready(h, 30);
    waitpid(h.pid, nullptr, 0);
    printf("  bad manifest digest: ready=%d state=%u reject_reason=%u, worker gone after %.2f ms (before any CUDA init)\n", ok, h.shm->state.load(),
           h.shm->reject_reason, double(now_ns() - t0) / 1e6);
    unlink(h.path.c_str());
  }
  return 0;
}

// ---- calibration: device read bandwidth, to size the synthetic step ----
__global__ void read_sum(const uint4* p, size_t n, unsigned long long* out) {
  unsigned long long s = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint4 v = __ldcs(p + i);
    s += v.x ^ v.y ^ v.z ^ v.w;
  }
  if (s == 0x123456789ull) *out = s;
}
static int cmd_calib(int argc, char** argv) {
  double gb = atof(arg(argc, argv, "--gb", "3.44").c_str());
  size_t bytes = size_t(gb * 1e9) & ~size_t(15);
  void* d;
  unsigned long long* o;
  CK(cudaMalloc(&d, bytes));
  CK(cudaMalloc(&o, 8));
  CK(cudaMemset(d, 1, bytes));
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  std::vector<double> t;
  for (int i = 0; i < 40; ++i) {
    cudaEventRecord(a);
    read_sum<<<1024, 512>>>((const uint4*)d, bytes / 16, o);
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    if (i >= 5) t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  printf("calib read %.2f GB: min %.3f ms  p50 %.3f ms  max %.3f ms  => p50 %.0f GB/s (3090 peak 936 GB/s)\n", double(bytes) / 1e9, t.front(), t[t.size() / 2],
         t.back(), double(bytes) / 1e9 / (t[t.size() / 2] / 1e3));
  return 0;
}

int main(int argc, char** argv) {
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  g_self = n > 0 ? std::string(buf, size_t(n)) : argv[0];
  if (argc < 2) return 2;
  std::string c = argv[1];
  if (c == "worker") {
    Shm* shm = map_shm(argv[2], false);
    WorkerArgs a;
    a.weights = argv[3];
    a.exit_on_sticky = argv[4][0] == '1';
    return worker_run(shm, a);
  }
  if (c == "bench") return cmd_bench(argc, argv);
  if (c == "fault") return cmd_fault(argc, argv);
  if (c == "respawn") return cmd_respawn(argc, argv);
  if (c == "calib") return cmd_calib(argc, argv);
  return 2;
}

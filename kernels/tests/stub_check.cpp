// Loads a kernel library the way the worker's loader does (resolve the C1 exports by name), reads the registration
// table, checks the C1 rules on it, then runs the registered Residual stub on the current CUDA device and compares
// with a host reference. Usage: roost_stub_check <library> [--no-gpu]. R-016's GPU evidence; run it through
// tools/gpu-lease. --no-gpu stops after the registration table checks (CI containers have no device).

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "roost_kernels.h"

#if defined(_WIN32)
#include <windows.h>
static void* OpenLibrary(const char* path) { return reinterpret_cast<void*>(LoadLibraryA(path)); }
static void* Symbol(void* lib, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
}
#else
#include <dlfcn.h>
static void* OpenLibrary(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void* Symbol(void* lib, const char* name) { return dlsym(lib, name); }
#endif

#define CHECK(cond, ...)                      \
  do {                                        \
    if (!(cond)) {                            \
      std::fprintf(stderr, "FAIL: ");         \
      std::fprintf(stderr, __VA_ARGS__);      \
      std::fprintf(stderr, "\n");             \
      return 1;                               \
    }                                         \
  } while (0)

template <typename Fn>
static Fn Resolve(void* lib, const char* name) {
  return reinterpret_cast<Fn>(Symbol(lib, name));
}

static float BitsToFloat(uint16_t bf) {
  uint32_t u = static_cast<uint32_t>(bf) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

static uint16_t FloatToBf16(float f) {  // round to nearest even; inputs here are finite
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <library> [--no-gpu]\n", argv[0]);
    return 2;
  }
  const bool gpu = !(argc > 2 && std::strcmp(argv[2], "--no-gpu") == 0);
  void* lib = OpenLibrary(argv[1]);
  CHECK(lib != nullptr, "cannot load %s", argv[1]);

  auto abi_version = Resolve<RoostAbiVersionFn>(lib, "roost_abi_version");
  auto last_error = Resolve<RoostLastErrorFn>(lib, "roost_last_error");
  auto reg = Resolve<RoostRegisterFn>(lib, "roost_register");
  auto ctx_create = Resolve<RoostContextCreateFn>(lib, "roost_context_create");
  auto ctx_destroy = Resolve<RoostContextDestroyFn>(lib, "roost_context_destroy");
  CHECK(abi_version && last_error && reg && ctx_create && ctx_destroy, "a C1 export is missing");

  const uint32_t version = abi_version();
  CHECK((version >> 16) == ROOST_ABI_MAJOR && (version & 0xFFFFu) >= ROOST_ABI_MINOR, "ABI version 0x%x", version);

  // A table smaller than the library writes is refused, not overrun.
  RoostOpTable small = {};
  small.struct_size = 8;
  CHECK(reg(&small) == ROOST_STATUS_ABI_MISMATCH && last_error()[0] != '\0', "short table was not refused");

  RoostOpTable table = {};
  table.struct_size = sizeof table;
  table.abi_minor = ROOST_ABI_MINOR;
  CHECK(reg(&table) == ROOST_STATUS_OK, "roost_register: %s", last_error());
  CHECK(table.entry_size == sizeof(RoostOpEntry), "entry_size %u != %zu", table.entry_size, sizeof(RoostOpEntry));
  CHECK(table.entry_count >= 1 && table.entries != nullptr, "empty table");
  CHECK(table.library_abi_major == ROOST_ABI_MAJOR, "library ABI major");
  std::printf("library %s, abi %u.%u, %u entries\n", table.library_name, table.library_abi_major,
              table.library_abi_minor, table.entry_count);

  const RoostOpEntry* residual = nullptr;
  for (uint32_t i = 0; i < table.entry_count; ++i) {
    const RoostOpEntry& e = table.entries[i];
    CHECK(e.variant_id != nullptr && e.variant_id[0] != '\0', "entry %u has no variant id", i);
    CHECK(e.run != nullptr, "entry %s has no run", e.variant_id);
    CHECK(((e.flags & ROOST_FLAG_NEEDS_PLAN) != 0) == (e.plan != nullptr), "entry %s: plan/flag mismatch",
          e.variant_id);
    CHECK(e.constraints.sm_min != 0, "entry %s: sm_min is required", e.variant_id);
    std::printf("  op %u variant %s flags 0x%x sm_min %u\n", e.op, e.variant_id, e.flags, e.constraints.sm_min);
    if (e.op == ROOST_OP_RESIDUAL && std::strcmp(e.variant_id, "residual-stub") == 0) residual = &e;
  }
  CHECK(residual != nullptr, "residual-stub is not registered");
  CHECK(residual->params_struct_size == sizeof(RoostResidualParams), "params_struct_size");

  if (!gpu) {
    std::printf("registration table OK (no GPU)\n");
    return 0;
  }

  int device_count = 0;
  CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0, "no CUDA device");
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0) == cudaSuccess, "cudaGetDeviceProperties");
  std::printf("device %s, sm_%d%d\n", prop.name, prop.major, prop.minor);
  CHECK(static_cast<uint32_t>(prop.major * 10 + prop.minor) >= residual->constraints.sm_min, "SM below sm_min");

  RoostContextCreateParams cp = {};
  cp.struct_size = sizeof cp;
  cp.abi_minor = ROOST_ABI_MINOR;
  cp.device = 0;
  RoostContext* ctx = nullptr;
  CHECK(ctx_create(&cp, &ctx) == ROOST_STATUS_OK && ctx != nullptr, "context create: %s", last_error());

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream) == cudaSuccess, "stream");

  // bf16 case: 100003 elements (not a multiple of the block size), output aliasing a.
  const uint64_t n = 100003;
  std::vector<uint16_t> ha(n), hb(n), hout(n);
  std::vector<float> expect(n);
  for (uint64_t i = 0; i < n; ++i) {
    ha[i] = FloatToBf16(static_cast<float>(static_cast<int64_t>(i % 977) - 488) * 0.01f);
    hb[i] = FloatToBf16(static_cast<float>(static_cast<int64_t>(i % 613) - 306) * 0.02f);
    expect[i] = BitsToFloat(FloatToBf16(BitsToFloat(ha[i]) + BitsToFloat(hb[i])));
  }
  void *da = nullptr, *db = nullptr;
  CHECK(cudaMalloc(&da, n * 2) == cudaSuccess && cudaMalloc(&db, n * 2) == cudaSuccess, "cudaMalloc");
  CHECK(cudaMemcpy(da, ha.data(), n * 2, cudaMemcpyHostToDevice) == cudaSuccess, "copy a");
  CHECK(cudaMemcpy(db, hb.data(), n * 2, cudaMemcpyHostToDevice) == cudaSuccess, "copy b");

  RoostResidualParams p = {};
  p.struct_size = sizeof p;
  p.abi_minor = ROOST_ABI_MINOR;
  p.a = {da, n, ROOST_DTYPE_BF16, 0};
  p.b = {db, n, ROOST_DTYPE_BF16, 0};
  p.output = {da, n, ROOST_DTYPE_BF16, 0};
  p.elements = n;

  // Rejections come back as statuses with a message, never a crash or a launch.
  RoostResidualParams bad = p;
  bad.b.dtype = ROOST_DTYPE_F32;
  CHECK(residual->run(ctx, &bad, nullptr, stream) == ROOST_STATUS_INVALID_ARGUMENT, "dtype mismatch accepted");
  bad = p;
  bad.struct_size = 16;
  CHECK(residual->run(ctx, &bad, nullptr, stream) == ROOST_STATUS_ABI_MISMATCH, "short params accepted");
  bad = p;
  bad.a.dtype = bad.b.dtype = bad.output.dtype = ROOST_DTYPE_I8;
  CHECK(residual->run(ctx, &bad, nullptr, stream) == ROOST_STATUS_UNSUPPORTED, "i8 accepted");

  CHECK(residual->run(ctx, &p, nullptr, stream) == ROOST_STATUS_OK, "run: %s", last_error());
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "sync: %s", cudaGetErrorString(cudaGetLastError()));
  CHECK(cudaMemcpy(hout.data(), da, n * 2, cudaMemcpyDeviceToHost) == cudaSuccess, "copy out");
  uint64_t mismatches = 0;
  for (uint64_t i = 0; i < n; ++i) {
    if (BitsToFloat(hout[i]) != expect[i]) ++mismatches;
  }
  std::printf("Residual bf16, %llu elements, output aliasing a: %llu mismatches\n",
              static_cast<unsigned long long>(n), static_cast<unsigned long long>(mismatches));
  CHECK(mismatches == 0, "%llu elements differ from the host reference", static_cast<unsigned long long>(mismatches));

  cudaFree(da);
  cudaFree(db);
  cudaStreamDestroy(stream);
  ctx_destroy(ctx);
  std::printf("OK\n");
  return 0;
}

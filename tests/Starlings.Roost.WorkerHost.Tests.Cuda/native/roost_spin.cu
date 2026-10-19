// Synthetic step for the R-069 overhead measurement: the same step S8 used (spikes/s8-worker-process/s8.cu), as a C
// library the .NET worker executor calls. 4 KiB plan H2D from ring memory, one kernel that spins on %globaltimer for
// the requested GPU time, 1 KiB result D2H straight into ring memory, stream sync (spin schedule). The segment is
// cudaHostRegister'ed (portable) so both copies read and write ring memory directly. Built by native/build.sh into
// envs/; the binary is never committed.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
constexpr size_t kPlanBytes = 4096;
constexpr size_t kBodyBytes = 1024;

uint8_t* d_plan = nullptr;
uint8_t* d_res = nullptr;
cudaStream_t stream = nullptr;
cudaEvent_t e0 = nullptr, e1 = nullptr;

__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void step_kernel(const uint8_t* plan, uint8_t* res, size_t ns_offset) {
  if (threadIdx.x != 0) return;
  uint64_t gpu_ns = *reinterpret_cast<const uint64_t*>(plan + ns_offset);
  uint64_t t0 = gtimer();
  while (gtimer() - t0 < gpu_ns) {
  }
  *reinterpret_cast<uint64_t*>(res) = gpu_ns;
}
}  // namespace

extern "C" {

// Creates the context on `ordinal` and the step buffers. Writes the 16-byte device UUID. Returns a cudaError_t.
int roost_spin_open(int ordinal, uint8_t* uuid16) {
  cudaError_t e;
  if ((e = cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost)) != cudaSuccess) return e;
  if ((e = cudaSetDevice(ordinal)) != cudaSuccess) return e;
  if ((e = cudaFree(0)) != cudaSuccess) return e;
  cudaDeviceProp prop;
  if ((e = cudaGetDeviceProperties(&prop, ordinal)) != cudaSuccess) return e;
  std::memcpy(uuid16, prop.uuid.bytes, 16);
  if ((e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)) != cudaSuccess) return e;
  if ((e = cudaMalloc(&d_plan, kPlanBytes)) != cudaSuccess) return e;
  if ((e = cudaMalloc(&d_res, kBodyBytes)) != cudaSuccess) return e;
  if ((e = cudaMemset(d_res, 0, kBodyBytes)) != cudaSuccess) return e;
  if ((e = cudaEventCreate(&e0)) != cudaSuccess) return e;
  return cudaEventCreate(&e1);
}

int roost_spin_register(void* p, size_t bytes) { return cudaHostRegister(p, bytes, cudaHostRegisterPortable); }

// One step. `plan` holds at least 4096 bytes; the 8 bytes at `ns_offset` are the spin time. `body` receives 1024 bytes.
// `event_ns` receives the GPU event time from before the H2D to after the D2H.
int roost_spin_step(const void* plan, size_t ns_offset, void* body, uint64_t* event_ns) {
  cudaError_t e;
  if ((e = cudaEventRecord(e0, stream)) != cudaSuccess) return e;
  if ((e = cudaMemcpyAsync(d_plan, plan, kPlanBytes, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return e;
  step_kernel<<<1, 32, 0, stream>>>(d_plan, d_res, ns_offset);
  if ((e = cudaGetLastError()) != cudaSuccess) return e;
  if ((e = cudaMemcpyAsync(body, d_res, kBodyBytes, cudaMemcpyDeviceToHost, stream)) != cudaSuccess) return e;
  if ((e = cudaEventRecord(e1, stream)) != cudaSuccess) return e;
  if ((e = cudaStreamSynchronize(stream)) != cudaSuccess) return e;
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  *event_ns = static_cast<uint64_t>(ms * 1e6f);
  return cudaSuccess;
}

// S8 condition 2: after an error, cudaGetLastError, then a memset and a sync; a second failure is sticky.
int roost_spin_probe() {
  cudaGetLastError();
  if (cudaMemsetAsync(d_res, 0, kBodyBytes, stream) != cudaSuccess) return 1;
  return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : 1;
}

}  // extern "C"

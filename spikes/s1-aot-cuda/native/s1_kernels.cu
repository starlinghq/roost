// Spike S1: tiny C ABI library. Throwaway evidence, not production code.
// Built with: nvcc -arch=sm_86 -cudart shared -Xcompiler -fPIC -shared
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

// No-op export used to measure the bare P/Invoke transition.
int32_t s1_noop(void) { return 0; }
int32_t s1_add(int32_t a, int32_t b) { return a + b; }

__global__ void s1_inc_kernel(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] += 1.0f;
}

// Launches the increment kernel on `stream`. Returns cudaError_t as int.
int32_t s1_launch_inc(float* p, int32_t n, void* stream) {
    s1_inc_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(p, n);
    return (int32_t)cudaGetLastError();
}

}

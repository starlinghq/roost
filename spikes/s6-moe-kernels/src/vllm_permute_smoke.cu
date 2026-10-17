// Compile-only check that vLLM's permute/unpermute kernels build with the torch-free patch (not run; see report).
#include <cuda_bf16.h>
#include "moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h"

template void expandInputRowsKernelLauncher<__nv_bfloat16>(__nv_bfloat16 const*, __nv_bfloat16*, int const*, int*, int*,
    int64_t const*, int64_t const, int64_t const*, int64_t const, int const, int, cudaStream_t);
template void finalizeMoeRoutingKernelLauncher<__nv_bfloat16, __nv_bfloat16>(__nv_bfloat16 const*, __nv_bfloat16*, float const*,
    int const*, int64_t const, int64_t const, int64_t const, int64_t const*, cudaStream_t);

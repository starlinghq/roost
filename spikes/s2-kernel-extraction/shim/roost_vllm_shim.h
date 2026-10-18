// S2 spike (R-006): replaces vLLM's torch_utils.h / torch headeronly type tags.
// vLLM kernels use torch::headeronly::{Half,BFloat16} and c10::{Half,BFloat16} only as scalar type
// names, so the CUDA types stand in for them. Nothing here includes torch.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace torch::headeronly {
using Half = __half;
using BFloat16 = __nv_bfloat16;
}  // namespace torch::headeronly
namespace c10 {
using Half = __half;
using BFloat16 = __nv_bfloat16;
}  // namespace c10

// FP8 tag types: vectorization.cuh names them in a static_assert only.
namespace c10 {
struct Float8_e4m3fn { unsigned char x; };
struct Float8_e4m3fnuz { unsigned char x; };
}  // namespace c10

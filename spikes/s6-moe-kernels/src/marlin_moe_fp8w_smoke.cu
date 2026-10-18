// Compile-only check: vLLM Marlin MoE with FP8 (e4m3) weight-only quantisation and bf16 activations on SM 86.
// This is W8A16 (weights dequantised to bf16 in registers). It is not FP8 tensor-core math, which needs SM 89 or newer.
#include "kernel.h"
#include "marlin_template.h"

namespace marlin_moe_wna16 {
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(),
                                256, 1, 8, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
}

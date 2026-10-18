// Compile-only check: one vLLM Marlin MoE kernel (bf16 activations, GPTQ-style INT4 weights, group size 128, SM80 path)
// instantiated without torch. Mirrors what vLLM's generate_kernels.py emits; not run.
#include "kernel.h"
#include "marlin_template.h"

namespace marlin_moe_wna16 {
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(),
                                256, 1, 8, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
}

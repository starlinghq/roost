# S6 (R-063): MoE kernels without torch

Status: go for bf16 MoE on SM 86 with CUTLASS grouped GEMM plus vLLM's top-k kernel, no libtorch. INT4 and FP8 are partly answered (compile evidence only). The .NET-versus-native host question is answered for AVX2 only; the Vector512 half needs an AVX-512 host.

All code is under `spikes/s6-moe-kernels/`. It is throwaway evidence outside `Roost.slnx`. Upstream sources are fetched by full SHA into `envs/` by `fetch-upstream.sh` and never copied into the repository. Only patches are committed.

## Facts that shape the answer

1. **The development CPU has no AVX-512.** It is an Intel Core i7-5820K (Haswell-E, 6 cores / 12 threads, 15 MiB L3) with AVX2 and FMA only. .NET reports `Vector256.IsHardwareAccelerated=True`, `Vector512.IsHardwareAccelerated=False`, `Avx512F.IsSupported=False`. No Vector512 number below is a hardware number.
2. **FP8 tensor-core math does not exist on SM 86** (it starts at SM 89). FP8 is `untested` for the 3090. Nothing FP8 was run.
3. **No `CpuOps` exists yet** (`src/` holds only Abstractions). The "CpuOps" reference for the layer is a double-precision C++ reference written for the spike (`test/moe_layer_test.cpp`). Stream D's `CpuOps` should replace it as the conformance oracle when it lands.
4. The no-go condition in plan section 6 ("no upstream builds standalone for grouped GEMM after two attempts") is **not triggered**: CUTLASS grouped GEMM built and ran on the first attempt with a zero-line patch.

## Environment

| Item | Value |
| --- | --- |
| GPU | RTX 3090 (SM 86), via `tools/gpu-lease run --device 3090 --stream C` |
| Toolkit / host compiler | CUDA 13.3 (nvcc 13.3.73), gcc 16.2.1; nvcc accepted this pairing with no flags (checked with a trivial kernel at `-arch=sm_86`) |
| Build | nvcc directly, `-arch=sm_86`, `-O2`, one translation unit at a time (at most two jobs); no CMake needed for the spike |
| Host load during GEMV runs | load average 6.7 to 7.3 on 12 CPUs (shared box). Treat GEMV numbers as noisy |

## Per op: upstream, SHA, patch size, status

Patch size is lines added and removed against the pinned upstream file, counted from `patches/*.patch`. "Kernel bodies untouched" means every change is in includes, the torch launcher or a check macro.

| Op | Upstream and SHA | Patch (+/-) | Status on SM 86 |
| --- | --- | --- | --- |
| Top-k routing (softmax, top-k, renormalise) | vLLM `csrc/moe/topk_softmax_kernels.cu`, `7879f3687b5b6a4be4754242e70b9c21cf7b4183` (derived from TensorRT-LLM) | +13 / -141 of 901 lines (`vllm-topk-softmax.patch`). Drops torch includes and the torch launcher, adds a 12-line `extern "C"` entry. Kernel bodies untouched | Built, ran, matched reference in the layer test |
| Permute / unpermute | Roost's own kernels in `src/roost_moe.cu` (count, scan, scatter, gather, weighted combine) | n/a (about 70 lines of device code) | Built, ran, matched reference |
| Permute / unpermute (alternative) | vLLM `csrc/moe/permute_unpermute_kernels/` at the same SHA | +3 / -64 (`vllm-permute-unpermute.patch`; 59 of the removed lines are `dispatch.h`, which only holds torch dtype dispatch). Kernel bodies untouched | Compile-only (`src/vllm_permute_smoke.cu`, bf16). Not run. It also needs CUTLASS headers (`cutlass/array.h`) and CUB |
| Grouped GEMM, bf16 | NVIDIA CUTLASS v4.8.0, `098de2a652cf8f00fd70b2df54051c7eccbb855a`, header-only; `cutlass::gemm::device::GemmGrouped` with `DefaultGemmGrouped` (SM80 tensor-op kernel, runs on SM 86), tile 128x128x32, warp 64x64x32, 4 stages, `kDeviceOnly` scheduling | 0 (no patch) | Built, ran, matched reference. One translation unit compiles in 15 s with 0.77 GiB peak RSS |
| Grouped GEMM, FP8 | CUTLASS v4.8.0 has `examples/64_ada_fp8_gemm_grouped` (SM 89 target) and Hopper/Blackwell grouped examples (57, 68, 69, 75, 92) | not attempted | `untested` on SM 86: no FP8 tensor-core path. Weight-only FP8 with bf16 math (W8A16) via vLLM Marlin compiles (below) |
| Grouped GEMM, INT4 (weight-only, bf16 activations) | vLLM Marlin MoE, `csrc/moe/marlin_moe_wna16/marlin_template.h` and `csrc/quantization/marlin/*`, same vLLM SHA | +3 / -2 (`vllm-scalar-type.patch`, one include replaced by a check-macro shim in `core/scalar_type.hpp`). Kernel body untouched | One instantiation (bf16 activations, U4B8, group 128, 256 threads) compiles to an SM 86 cubin with tensor-core instructions (`src/marlin_moe_int4_smoke.cu`). Not run. A second instantiation with FP8 e4m3 weights also compiles (`src/marlin_moe_fp8w_smoke.cu`) |
| Marlin launcher and repack | vLLM `marlin_moe_wna16/ops.cu` (706 lines, about 108 lines touching torch types or checks) and `gptq_marlin_repack.cu` / `awq_marlin_repack.cu` | not done | Needs a rewrite as a C entry point plus an offline repack. Estimate: a few hundred lines of mostly mechanical code, with the repack the larger unknown. This is the work for R-077 / R-080 |
| Not examined | `grouped_topk_kernels.cu` (1512 lines, DeepSeek-style grouped top-k), `moe_align_block_size` in `moe_align_sum_kernels.cu` (1161 lines), `moe_wna16.cu` | not attempted | Qwen3 needs neither: CUTLASS grouped GEMM takes problem sizes from device memory, so block alignment is unnecessary |

Why not the other candidates named in the plan: Marlin standalone (IST-DASLab) was not needed because vLLM's Marlin MoE kernel template compiled with a 5-line patch. vLLM's Python/Triton fused-MoE path was not examined because Python is excluded from the runtime.

## Qwen3 MoE layer through a C ABI on SM 86

Shapes come from `Qwen/Qwen3-30B-A3B` `config.json` (read as data): hidden 2048, `moe_intermediate_size` 768, 128 experts, 8 per token, `norm_topk_prob` true, `silu`. Weights are random bf16 generated in memory (about 1.2 GB), never written to disk.

C ABI (`src/roost_moe.h`): shaped like C1. `RoostMoeLayerParams` starts with `struct_size` and `abi_minor`; entry points are `roost_moe_layer(const Params*, void* stream)` and `roost_moe_layer_workspace_bytes`. It is a sketch for the spike, not the final `roost_kernels.h`.

Pipeline, all on one stream: router GEMM (bf16 logits) -> vLLM top-k softmax -> expert counts, prefix scan and per-expert pointer tables built on the device -> row scatter and gather -> CUTLASS grouped GEMM for gate+up -> SiLU-and-mul -> CUTLASS grouped GEMM for down -> weighted combine.

Reference: double precision on the CPU, bf16 inputs. Two comparisons:

- **Stage-isolated:** each stage's reference takes the GPU's previous stage output as exact input. Tolerance is one bf16 ulp (relative 0.0079) plus 1e-4 absolute for float-accumulation noise near zero. Top-k weights use 1e-5.
- **End to end against pure f64** (no intermediate rounding): gate is rms relative error below 0.01 and max error below 0.03 of the max magnitude.

The top-k expert set must match the independent reference for every token. Stage comparisons run on every token, because every token matched.

Commands (each GPU run is a separate lease):

```sh
spikes/s6-moe-kernels/fetch-upstream.sh && spikes/s6-moe-kernels/build.sh
tools/gpu-lease run --device 3090 --need-mib 6000 --wait 3600 --stream C -- <build>/moe_layer_test 1 16 128 512
tools/gpu-lease run --device 3090 --need-mib 6000 --wait 3600 --stream C -- <build>/moe_layer_test 128 dup
tools/gpu-lease run --device 3090 --need-mib 6000 --wait 3600 --stream C -- compute-sanitizer --tool memcheck <build>/moe_layer_test 16
```

Results (all PASS):

| Tokens | Top-k sets matching | Router logits | gemm1 / SiLU-mul / gemm2 / combine (outside tolerance) | End-to-end vs pure f64: rms rel / max / max-ref |
| --- | --- | --- | --- | --- |
| 1 | 1/1 | exact | 0 / 0 / 0 / 0 | 3.89e-3 / 5.7e-3 / 4.9e-3 |
| 16 | 16/16 | exact | 0 / 0 / 0 / 0 | 3.86e-3 / 1.08e-2 / 5.4e-3 |
| 128 | 128/128 | max diff 1.95e-3 (one bf16 ulp), 0 outside tolerance | 0 / 0 / 0 / 0 | 3.86e-3 / 1.08e-2 / 4.5e-3 |
| 512 | 512/512 | max diff 1.95e-3, 0 outside tolerance | 0 / 0 / 0 / 0 | 3.88e-3 / 1.42e-2 / 5.7e-3 |
| 128, all tokens identical (8 experts get 128 rows, 120 get none) | 128/128 | n/a | pass | pass |

Top-k weights agree with the reference to 1.2e-7 at most. The end-to-end rms error of about 3.9e-3 is the floor of a bf16 pipeline with three roundings (about 2^-8 each); it does not grow with token count. `compute-sanitizer --tool memcheck` on the 16-token case reports 0 errors. The empty-expert groups (m = 0) work with the CUTLASS grouped kernel.

Not covered: `racecheck` and `synccheck`, CUDA-graph capture of the layer (the argument tables are built on the device, so it should be capturable; untested), batch invariance, and a deterministic permute (the scatter uses atomics, so row order within an expert varies between runs; per-row results do not depend on it, but this was not tested).

Smoke timing, 3090 shared with other agents, mean of 20 launches, no debug copies: 0.25 ms (1 token), 1.38 ms (16), 2.23 ms (128), 2.63 ms (512). These are not throughput claims. No starling fidelity gate has been run on a model, and the GPU was not exclusive.

## Host expert GEMV for bet X6

Kernel: one Qwen3 expert, batch 1, bf16 weights times f32 activation, f32 output. Shapes: gate+up 1536x2048 (6.3 MB) and down 2048x768 (3.1 MB). "rot=128" rotates over 128 experts (1.2 GB) so every call streams from DRAM; "rot=1" re-reads one expert and mostly hits L2/L3. Four rows share each activation load; bf16 widens by a 16-bit shift. Best of 7 repetitions of 60 to 400 calls. Accuracy against a scalar f64 loop: max abs error 9e-8 to 3e-7.

Implementations: .NET 10.0.12 `Vector256` (AVX2+FMA), .NET `Vector512` (portable API; software path here), and Rust 1.98.1 AVX2+FMA intrinsics built with `-C target-cpu=native`, LTO, one codegen unit. Native option chosen: Rust, because decision 22 names it for host kernels.

GB/s counts weight bytes only; GFLOP/s equals GB/s here (2 flops per 2-byte weight).

| Shape | rot | Threads | .NET Vector256 | .NET Vector512 (sw) | Rust AVX2 |
| --- | --- | --- | --- | --- | --- |
| gate+up | 128 (DRAM) | 1 | 16.6 GB/s, 379 us | 17.3, 363 us | 16.9, 373 us |
| gate+up | 128 (DRAM) | 4 | 31.3, 201 us | 33.7, 187 us | 27.1, 232 us |
| down | 128 (DRAM) | 1 | 9.7, 323 us | 10.2, 309 us | 10.6, 297 us |
| down | 128 (DRAM) | 4 | 22.9, 138 us | 21.5, 146 us | 17.3, 182 us |
| gate+up | 1 (cached) | 1 | 26.3, 239 us | 24.6, 255 us | 28.9, 218 us |
| gate+up | 1 (cached) | 4 | 58.3, 108 us | 51.1, 123 us | 48.5, 130 us |
| down | 1 (cached) | 1 | 25.8, 122 us | 21.9, 144 us | 27.2, 116 us |
| down | 1 (cached) | 4 | 51.7, 61 us | 51.3, 61 us | 31.6, 100 us |

Memory reference (Rust, xor-fold over 1 GiB): 14.7 GB/s on 1 thread, 42.1 GB/s on 4 threads. The simple fold underestimates single-thread bandwidth (the GEMV exceeds it), so use the 4-thread figure as the rough DRAM ceiling: the 4-thread DRAM-bound GEMVs reach about 40% to 80% of it.

Reading the numbers:

- Single-threaded, .NET `Vector256` and Rust AVX2 are within about 5% of each other on every DRAM-bound row. Multi-threaded and cached rows differ by up to 1.6x in either direction. The box ran at load 6.7 to 7.3 while these ran, and the two benchmarks ran back to back, not interleaved, so these differences are within run-to-run variance. This data does not show a performance gap between .NET and Rust for this kernel on AVX2.
- The batch-1 expert GEMV is bandwidth-bound. Moving a cold Qwen3-30B-A3B expert (9.4 MB bf16) costs about 340 us at 4 threads from DRAM (201 + 138). Eight cold experts in all 48 layers is 3.6 GB per token, which is about 0.1 s per token on this machine. Bet X6 therefore rests on the hot-expert hit rate, and quantised host kernels would cut the bytes.
- **Vector512 is not measured as hardware.** `Vector512.IsHardwareAccelerated` is false; the portable `Vector512` code ran, at about the same speed as `Vector256`. I did not inspect the generated code to say why. This says nothing about an AVX-512 CPU. Whether `Vector512` code matches a native Rust or C++ build there is open.
- Not measured: quantised (INT4/INT8) host GEMV, bf16-dot instructions (AVX512_BF16, AMX), larger batches, NUMA, and a C++ native option.

## Recommendation

- **Device side (R-080):** adopt CUTLASS v4.8.0 `GemmGrouped` (SM80 bf16 kernel, zero patches) for bf16 grouped GEMM, vLLM's top-k softmax kernel with the 154-line launcher patch, and Roost's own small permute/unpermute. The vLLM permute kernels are an option with a 67-line patch. Record CUTLASS and vLLM in `PROVENANCE.md` and `THIRD-PARTY-NOTICES.md` when they are adopted into `kernels/`; this spike adopts nothing.
- **INT4 (R-077):** vLLM Marlin MoE kernels compile torch-free with a 5-line patch. The remaining work is the launcher rewrite and the weight repack. Expect that, not the kernel body, to be the cost.
- **FP8:** mark `untested` for SM 86. Needs an SM 89 or newer device to evaluate CUTLASS FP8 grouped GEMM.
- **Host kernels (R-082):** .NET `Vector256` is competitive with Rust AVX2 on this CPU, so no evidence yet requires leaving .NET for the bf16 GEMV. Decision 22 should keep Rust available, because the quantised-weight and AVX-512/AMX questions are open. Re-run `host-gemv/` on an AVX-512 machine before deciding on Vector512.

## What is not done

- A running INT4 or FP8 layer (compile-only).
- Vector512 on AVX-512 hardware; a C++ native comparison; quantised host GEMV.
- vLLM permute kernels run; vLLM `moe_align_block_size` and grouped top-k examined.
- A starling fidelity gate on any model; real Qwen3 weights.
- `racecheck`/`synccheck`, graph capture, determinism and batch-invariance tests.
- The RTX 4060 Ti was not used.

## Files

`spikes/s6-moe-kernels/`: `fetch-upstream.sh`, `build.sh`, `patches/`, `src/roost_moe.{h,cu}`, `src/*_smoke.cu`, `test/moe_layer_test.cpp`, `host-gemv/dotnet/`, `host-gemv/rust/` (source and `Cargo.lock` only). Build outputs go to `/home/mark/Projects/roost-wave1/envs/R-063/`, outside the repository.

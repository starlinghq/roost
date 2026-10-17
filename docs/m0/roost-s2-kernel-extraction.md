# S2 (R-006): torch-free kernel extraction and the Gemm choice

Status: done for SM 86 (RTX 3090); SM 89 built but not run (section 6). Code: `spikes/s2-kernel-extraction/` (throwaway evidence, outside `Roost.slnx`).
Host: CUDA 13.3 (nvcc 13.3.73), driver 610.57, gcc 15, built with 2 jobs, `sm_86` / `sm_89` only. Every GPU run went through `tools/gpu-lease run --stream C`.

## 1. Answer

| Question (plan section 6) | Result |
| --- | --- |
| Which upstream provides each M0 op with no libtorch? | **FlashInfer provides every M0 op except Gemm, Embedding and the LmHead argmax, with zero patch lines.** vLLM `csrc/` is not needed; its torch coupling is a demonstrated 6 patch lines per file (section 4) and stays the fallback. This inverts the plan's expectation in decision 6. |
| One attention op from a C ABI on SM 86 and 89? | SM 86: yes, paged GQA decode, five cases against a CPU reference, all pass. SM 89: library built, **not run** (4060 Ti had 278 to 690 MiB free). |
| Gemm | **cuBLASLt for M0**, with the algorithm pinned per shape bucket (section 5). |
| Kill: patch set for one op over about 300 lines | **Not triggered.** 0 lines for FlashInfer; 6 lines (layernorm) and 6 lines (pos_encoding) for vLLM. |
| Kill: FlashInfer cannot be built standalone | **Not triggered.** The headers compile with plain `nvcc -I include`: no submodules fetched, no Python, no TVM-FFI, no torch. FlashInfer's own pinned CUTLASS (`b46b16d003484063bca4ed365e44095c4c6ed633`) is not needed for the M0 paths. |

## 2. Pinned upstreams (full SHAs)

| Upstream | Commit | Note | Licence |
| --- | --- | --- | --- |
| flashinfer-ai/flashinfer | `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94` | tag v0.7.2 (the tag object is `ca0f620c9eca61759dc488e343e9e1e6ba9e43c3`; pin the commit) | Apache-2.0 |
| vllm-project/vllm | `7879f3687b5b6a4be4754242e70b9c21cf7b4183` | main on 2026-10-17 | Apache-2.0 |
| NVIDIA/cutlass | `098de2a652cf8f00fd70b2df54051c7eccbb855a` | tag v4.8.0, Gemm evaluation only | BSD-3-Clause |

cuBLASLt ships with the CUDA toolkit under the CUDA EULA. Its redistribution terms belong to the decision 5 licence review; this report does not assert them. `fetch.sh` fetches the three by SHA into `envs/`; nothing fetched is committed.

## 3. Per-op table (M0 op list, R-017)

"Patch" is lines added plus removed against the upstream files. "Wrapper" is lines we write. "Run" means checked on the 3090 against a CPU reference computed in double from the same bf16 inputs (tolerance: abs error / (1 + |ref|) below 2e-2, which is a few bf16 ulps; measured maxima 2.6e-3 to 3.1e-3, i.e. one ulp).

| Op | Upstream and entry | Patch | Wrapper | Status |
| --- | --- | --- | --- | --- |
| PagedAttention decode (GQA, bf16, head_dim 128, page 16, NHD) | FlashInfer `BatchDecodeWithPagedKVCacheDispatched`, `DecodePlan` | 0 | about 190 (`roost_s2_attn.cu`) | Run, 5 cases pass (section 4) |
| PagedAttention prefill | FlashInfer `BatchPrefillWithPagedKVCacheDispatched`, `PrefillPlan` | 0 | not written | Compile probe only: 1 tile 10 s / 273 KB, 3 tiles (16, 64, 128) 16 s / 672 KB. Planner wrapper is the remaining work (about the size of the decode wrapper again). |
| RmsNorm | FlashInfer `norm::RMSNorm` | 0 | 6 | Run, pass |
| RmsNorm (alternative) | vLLM `layernorm_kernels.cu` `rms_norm_kernel` | 6 | 8 | Run, pass; same max error as FlashInfer on this case (3.0e-3) |
| Residual + RmsNorm | FlashInfer `norm::FusedAddRMSNorm` | 0 | 6 | Run, pass (both outputs) |
| QkNorm | FlashInfer `norm::QKRMSNorm` | 0 | 7 | Run, pass |
| SiluAndMul | FlashInfer `activation::act_and_mul_kernel` | 0 | 9 (the launcher is not in the headers; FlashInfer generates it from a Python template) | Run, pass |
| Rope default (NeoX, theta 1e6) | FlashInfer `BatchQKApplyRotaryPosIds` | 0 | 8 | Run, pass (q and k) |
| Rope llama3 | FlashInfer `BatchQKApplyLlama31RotaryPosIds` | 0 | not written | Present upstream, not exercised |
| Cache append (write new K/V into pages) | FlashInfer `AppendPagedKVCache` | 0 | 12 | Run, pass (bit-exact) |
| vLLM rotary kernel (fallback) | vLLM `pos_encoding_kernels.cu` | 6 | 0 | Compiles torch-free, **not run** |
| Cache copies and tier moves | `cudaMemcpyAsync` / small copy kernel | n/a | to write | Not attempted; vLLM `cache_kernels.cu` interleaves torch host code through 2078 lines (234 torch lines), so a guard patch is not cheap there; not needed for M0 |
| Embedding | gather kernel we write | n/a | about 20 | Not attempted |
| LmHead | Gemm plus argmax. FlashInfer has no argmax (`sampling.cuh` is top-k/top-p) | n/a | about 20 for argmax | Not attempted |
| Gemm | cuBLASLt (section 5) | n/a | about 60 | Benchmarked, section 5 |

Per translation unit build cost on SM 86 (nvcc, one job, host under load from other agents; wall seconds vary with contention, for example the decode TU took 30 to 38 s and once 13 s):

| TU | Wall | Object |
| --- | --- | --- |
| `roost_s2_attn.cu` (decode, group 2 and 4, all stage variants, merge kernels) | 13 to 38 s | 2.0 MB (3.1 MB `.so`) |
| `roost_s2_ops.cu` (norm x2, qk norm, add-norm, silu-mul, rope, append) | 18 to 23 s | 0.98 MB |
| `prefill_compile_probe.cu` | 10 s (1 tile), 16 s (3 tiles) | 273 KB, 672 KB |
| `gemm_cutlass.cu` (2 CUTLASS tile configs) | 10 s | 141 KB |

Everything above summed is about 1.5 minutes per SM at one job. R-016 builds six SMs (80, 86, 89, 90, 100, 120): expect roughly 10 minutes serial for the M0 ops, more once prefill variants, head sizes and FP8 arrive. SM 90, 100 and 120 were not compiled here. FlashInfer's FA2 path targets SM 80 and later; the Hopper and Blackwell specialised paths were not evaluated and are out of M0.

## 4. Attention: what was built and measured

`src/roost_s2_attn.{h,cu}` replace the TVM-FFI glue of `csrc/batch_decode.cu` (plan, workspace size, run) with `extern "C"` functions; the FlashInfer headers are used unmodified. Shape: bf16 Q/K/V/O, head_dim 128, NHD pages, int32 ids, group size 2 and 4, `enable_pdl=false`, `enable_cuda_graph=false`. Exceptions (`FLASHINFER_CHECK` throws) are caught at the boundary and returned as a status plus `roost_s2_last_error()`. The signature shape (params struct that begins with `struct_size` and `abi_minor`, `(params, stream)`, a workspace query) is illustrative, not C1.

Test `test/test_paged_decode.cpp`, RTX 3090, `tools/gpu-lease run --device 3090 --need-mib 2000 --stream C`. Page ids are shuffled and the pool has spare pages; output is pre-filled with a NaN pattern.

| Case | Batch | Heads q/kv | split-KV | Max abs error | Mean abs error | `run()` time |
| --- | --- | --- | --- | --- | --- | --- |
| mixed lengths 5, 37, 130, 16 (partial last pages) | 4 | 16/8 | yes | 3.9e-3 | 3.8e-4 | 10.0 us |
| one sequence of 6000 | 1 | 16/8 | yes | 2.7e-4 | 4.7e-5 | 37.8 us |
| lengths 900, 17, 1, 2048, 333, 64, 1024, 777 | 8 | 16/8 | yes | 2.4e-3 | 1.4e-4 | 33.3 us |
| 256 sequences, lengths 1 to 300 | 256 | 16/8 | no | 7.7e-3 | 1.8e-4 | 191 us |
| lengths 100, 515 | 2 | 32/8 (group 4) | yes | 1.9e-3 | 1.6e-4 | 22.5 us |

Pass threshold is 1e-2 absolute on outputs of order 0.1 to 1 (bf16 has 8 mantissa bits). `run()` time is the mean of 200 launches with the plan reused, measured with CUDA events; it is a launch-to-launch figure, not a fidelity-gated throughput claim. An unsupported head_dim returns status 2 with a message.

Qwen3-0.6B and 1.7B are both 16 query heads and 8 KV heads (group 2), head_dim 128; group 4 covers larger cards.

### Findings for contracts C1 and C2 (R-004)

1. **Planning is a host step.** FlashInfer's planner reads `kv_indptr` on the host, writes a page-locked staging buffer and copies it to a device buffer on the stream. The step plan therefore needs a host copy of the block-table offsets, and the worker needs a pinned workspace it owns. `plan()` is a host call that is not graph-capturable; `run()` allocates nothing and does not synchronise. This matches R-017's "no allocation inside calls" if the allocation moves into the worker (workspace sizes come from the query function).
2. **Split-KV is chosen by the planner** and changes how many CTAs write partial results; it needs a float workspace (41 KB to 388 KB in these cases).
3. **CUDA graph mode is untested.** With `enable_cuda_graph=true` FlashInfer pads the batch to a fixed size (`padded_batch_size = max_grid / gdy`). That path was not run.
4. vLLM reads `VLLM_BATCH_INVARIANT` from the environment inside `vllm_is_batch_invariant()` (in `core/utils.hpp`). If any vLLM kernel is adopted, the batch-invariant flag must come from C1, not the environment.

## 5. Gemm: cuBLASLt or CUTLASS

### Distribution cost (decision 12)

| Item | Size |
| --- | --- |
| `libcublasLt.so.13.6.0.2` | 516,598,272 bytes (493 MiB); `zstd -3`: 343.6 MB |
| `libcublas.so.13.6.0.2` | 55,354,008 bytes (53 MiB) (not needed if only cuBLASLt is called) |
| `libcudart.so.13.3.29` | 782,080 bytes |
| Our CUTLASS object, 2 tile configs, one SM | 141,512 bytes |

`ldd libcublasLt.so.13` shows only libc, libm, libdl, libpthread, librt and libgcc_s, so cuBLASLt stands alone at the size above. The library carries kernels for every SM, so a per-SM subset cannot be cut from it.

### Performance on the 3090 (`gemm/gemm_bench.cpp`, bf16 in and out, fp32 accumulate)

Qwen3-1.7B projection shapes (hidden 2048, intermediate 6144, so gate_up is N=12288). CUTLASS is `device::Gemm` with Sm80 tensor-op tiles, one fixed tile per column, no split-K, no shape heuristic. Times are means of 100 launches. The two backends agree to within one bf16 ulp (largest difference 3.1e-2 at |c| near 4 to 5); this is a cross-check between backends, not a conformance result (R-019 owns Gemm conformance).

| Shape | M | N | K | cuBLASLt | CUTLASS 128x128x32 | CUTLASS 32x64x64 |
| --- | --- | --- | --- | --- | --- | --- |
| decode gate_up | 1 | 12288 | 2048 | 62.1 us | 161.9 us | 75.3 us |
| decode gate_up | 16 | 12288 | 2048 | 79.4 us | 162.0 us | 75.3 us |
| decode down | 16 | 2048 | 6144 | 49.4 us | 237.5 us | 66.2 us |
| decode q | 16 | 2048 | 2048 | 19.0 us | 82.5 us | 25.0 us |
| prefill gate_up | 512 | 12288 | 2048 | 398.5 us (64.7 TFLOP/s) | 394.9 us | 505.0 us |
| prefill down | 512 | 2048 | 6144 | 218.4 us | 214.6 us | 282.8 us |
| prefill wide | 2048 | 8192 | 2048 | 992.7 us (69.2 TFLOP/s) | 923.2 us | 1435.1 us |

Reading: at M=1 cuBLASLt moves 50.3 MB of weights in 62.1 us, about 0.81 TB/s, which is near the 3090's memory bandwidth, with no tuning. CUTLASS matches or beats it at prefill with a large tile, and loses by 1.2 to 4.8 times at decode unless the tile suits the shape. The CUTLASS figures are a floor on effort (one tile, no split-K), not a ceiling on its performance; closing the decode gap is variant-selection work that belongs to R-072.

### Batch invariance (C1 flag) with cuBLASLt (`gemm/gemm_invariance.cpp`)

Qwen3 down_proj (N=2048, K=6144), same rows in each batch:

- Unpinned, the heuristic keeps algo id 6 but changes the split-K count with M (5 at M=64, 3 at M=17 and 5, 2 at M=1). Rows then differ bitwise from the M=64 result: 16588 of 34816 elements at M=17, 4944 of 10240 at M=5, 886 of 2048 at M=1.
- Pinned to the M=64 algo (split-K 5), M=17, 5 and 1 match the M=64 rows exactly (0 elements differ in 34816, 10240 and 2048).

So the default cuBLASLt path is not batch-invariant, and pinning one algo per shape (N, K) bucket makes it so for this shape on this device and library version. One shape, one device, one library build was tested; whether the pinned algo stays valid for every M in a bucket, and across library updates, must be a conformance case in R-019, and the pinned algo id must be recorded in the manifest next to the library digest.

### Recommendation

**Use cuBLASLt for Gemm in M0**, calling it from a small C ABI wrapper in the kernel library, with the algo chosen per (N, K, dtype) bucket at startup and pinned, and the choice recorded in the profile evidence.

- Decode is where Roost competes, and cuBLASLt is within a few percent of the memory-bandwidth limit there with no per-shape tuning. Matching it with CUTLASS means a tile and split-K variant per shape, then maintaining the selector.
- Build cost favours cuBLASLt (no per-shape template instantiation). CUTLASS builds cheaply per tile (about 5 s and 70 KB per tile per SM here), so build time is not the reason to avoid it; tuning effort is.
- The plan's TF32 argument does not apply to M0: the inputs are bf16, and `CUBLAS_COMPUTE_32F` accumulates in fp32 with no TF32 path involved. A TF32 policy matters only for fp32 matmuls, which M0 does not run.
- Decision 12: under option (b) (first-run fetch, pinned by digest) the 343.6 MB compressed / 493 MiB library is a download and a verification cost, not a package-size limit.

**Flip conditions.** (1) If decision 12 lands on (a), RID-specific native packages on nuget.org, then 516 MB cannot ride a package (250 MB limit) and the answer becomes CUTLASS (kernels stay in the kernel asset) or the system CUDA install. (2) If R-019 shows the pinned-algo invariance does not hold across M within a bucket or across cuBLASLt updates, use CUTLASS for the ops that carry the batch-invariant flag.

## 6. SM 89

- `build.sh 89` and `build-ops.sh 89` completed (decode TU 13 s / 2.08 MB object; ops TU 18 s). A first `build.sh 89` attempt at 15:58 was killed by the OS (`cc1plus` Killed) while the host had about 3 GB free memory; it was not retried in a loop. It succeeded once at 16:08 with about 32 GB available.
- `tools/gpu-lease run --device 4060 --need-mib 500 --wait 0` at 16:08:20 was refused: 278 MiB free of 16380 MiB. An earlier attempt with `--need-mib 300` was refused at 262 MiB free. The test was not run on SM 89. **SM 89 stays `untested`** (AGENTS.md). Run it with `tools/gpu-lease run --device 4060 --need-mib 500 --stream C -- <build>/sm89/test_paged_decode` when the device has room.

## 7. Not done or not covered

- SM 89 execution (above). SM 80, 90, 100, 120 not compiled.
- Prefill wrapper and test; Rope llama3 run; vLLM rotary run; Embedding; LmHead argmax; cache copy ops.
- CUDA graph mode for the planner.
- Any throughput claim: none is made. Numbers above are kernel timings for scale, not fidelity-gated results.
- Head dims other than 128, dtypes other than bf16, KV dtypes other than bf16, sliding window, soft cap.
- MSVC build of the wrapper (R-016's build-only CI job). The wrapper uses no POSIX types in the header; the FlashInfer headers were not built with MSVC here.

## 8. Reproducing

```sh
cd spikes/s2-kernel-extraction
./fetch.sh                # three upstreams by full SHA into ../../../envs/R-006 (override ROOST_S2_ENVS)
./build.sh 86             # attention wrapper, libroost_s2_attn.so, test_paged_decode
./build-ops.sh 86         # other ops (applies patches/vllm-*.patch to a copy), test_ops
./build-gemm.sh 86        # CUTLASS tiles and the cuBLASLt comparison
nvcc -std=c++17 -O2 -DNDEBUG --expt-relaxed-constexpr -gencode arch=compute_86,code=sm_86 \
  -DROOST_TILES=3 -I <flashinfer>/include -c src/prefill_compile_probe.cu -o <out>/prefill_probe.o
nvcc -std=c++17 -O1 gemm/gemm_invariance.cpp -o <out>/gemm_invariance -lcublasLt -lcudart
tools/gpu-lease run --device 3090 --need-mib 2000 --stream C -- <out>/test_paged_decode   # likewise test_ops, gemm_bench, gemm_invariance
```

The patch sets are `patches/vllm-layernorm-torch-free.patch` and `patches/vllm-pos-encoding-torch-free.patch`. Each swaps `#include "torch_utils.h"` for a shim header, drops the torch-only `dispatch_utils.h` include and wraps the torch host launchers in `#ifdef ROOST_VLLM_WITH_TORCH_HOST`. The vLLM device code names `torch::headeronly::Half`, `BFloat16` and the FP8 tag types only as scalar type names, so `shim/` supplies them from the CUDA types; nothing from torch is included. The shim directory mirrors torch's header paths so the vLLM `#include <torch/headeronly/...>` lines resolve without patching.

## 9. Consequences for the plan

- Decision 6: adopt FlashInfer for paged attention, norms, QkNorm, SiluAndMul, Rope and cache append; keep vLLM `csrc/` only as the six-line fallback; cuBLASLt for Gemm.
- R-016: pin only FlashInfer (and CUTLASS if a later op needs it) in `manifest.toml`; M0 needs no vLLM patch set, so `kernels/upstream/` holds none at M0.
- R-004: add a host copy of block-table offsets to the plan, a worker-owned pinned workspace, a separate plan step, and a workspace-size query to C1 (section 4). Record the cuBLASLt algo id per bucket in the manifest.
- R-019: add cuBLASLt pinned-algo invariance across M as a conformance case.

No third-party source is adopted by this spike: upstream trees are fetched into `envs/` and only our patches, wrappers and scripts are committed, so `PROVENANCE.md` and `THIRD-PARTY-NOTICES.md` are unchanged. R-016 adopts the pins in section 2 and updates both files in the same change.

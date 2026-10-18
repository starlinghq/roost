# S3 (R-007): Triton AOT for the Gated DeltaNet kernels

Branch `spike/s3-triton-gdn`, from `ff072f8`. Code and result files: `spikes/s3-triton-gdn/` (throwaway evidence, outside `Roost.slnx`; nothing is vendored, so `PROVENANCE.md` and `THIRD-PARTY-NOTICES.md` are unchanged). Stream C, measured on the RTX 3090 (SM 86) through `tools/gpu-lease`.

## Answer

Yes. All seven forward kernels that a Qwen3.5-0.8B linear-attention layer needs (flash-linear-attention's chunked and recurrent gated delta rule, plus the l2norm and gate cumsum that surround them) compile ahead of time to cubins for SM 80, 86, 89, 90, 100 and 120, and they run from a plain C ABI over the CUDA driver API (`cuModuleLoadData`, `cuLaunchKernel`). None needs Triton's global scratch, so the S3 stop condition ("AOT is not possible for a needed kernel") did not trigger. On SM 86 the AOT cubins are byte-identical (15 of 15 SHA-256 matches) to the cubins Python FLA runs with the same pinned configs, and the C-ABI output is bit-identical to Python FLA with the same pinned configs (0 mismatched elements in 7 cases, see below). Stock autotuned FLA is not the same thing: on the 3090 it produced different cubins for 3 of the 15 kernels (see "Autotune winners are not reproducible"). The C++ port for decision 23 is estimated at 13 to 15 engineer-weeks (rows sum to 12.7 to 14.7) for forward-only, SM 80 to 89 (section "Port estimate").

## Inputs and versions

| Item | Value |
| --- | --- |
| Model shape | Qwen3.5-0.8B `text_config`: 18 of 24 layers are linear attention; `linear_num_key_heads` 16, `linear_num_value_heads` 16, key and value head dim 128, conv kernel 4; activations bf16, recurrent state fp32 (`mamba_ssm_dtype`) |
| flash-linear-attention | `329b534245d3d7bb27cc03a99e87bdc51fb28d06` (fla 0.6.0, MIT), `fla-org/flash-linear-attention`, installed from a checkout under `envs/R-007/`, not vendored |
| Build-time Python closure | Python 3.12.14, torch 2.14.1+cu130, Triton 3.8.0, transformers 5.20.1 (FLA requires `>=5.6.2,<6`), einops 0.8.2, numpy 2.5.4 (these are the values for the C5 manifest's "Triton and Python versions for AOT") |
| Toolkit and driver | CUDA 13.3.73 (`nvcc`), driver 610.57.04; Triton uses its bundled ptxas. Recorded in `results/env_versions.txt` together with the Python package versions and the FLA SHA |
| Assumed call shape | `g` (log decay) and `beta` are computed upstream; q and k are l2-normalised by the l2norm kernel (prefill) or inside the recurrent kernel (decode); `scale = 128^-0.5`; chunk size 64; `state_v_first=False`. vLLM's gating kernels and the causal conv1d are not part of this spike |

FLA's `gated_delta_rule` forward path launches these Triton kernels (names as in the source):

| # | Kernel | Used for | Code lines (body) |
| --- | --- | --- | --- |
| 1 | `l2norm_fwd_kernel` | q and k normalisation (prefill) | 13 |
| 2 | `chunk_local_cumsum_scalar_kernel` | chunk-local cumulative gate | 23 |
| 3 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | beta K K^T and (I+A)^-1, fused | 202 |
| 4 | `recompute_w_u_fwd_kernel` | WY representation w, u | 39 |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | inter-chunk state recurrence | 250 |
| 6 | `chunk_fwd_kernel_o` | chunk output | 60 |
| 7 | `fused_recurrent_gated_delta_rule_fwd_kernel` | decode and short continuations | 104 |

## Matrix, size and build time

The specialisation matrix is the set of distinct Triton compilations the forward path produces for this model: `IS_VARLEN` (packed batch with `cu_seqlens` against a single sequence) times `USE_INITIAL_STATE` (chunk state kernel only), one config per kernel. `capture.py` records them from live Python FLA launches (signature, constexprs, divisibility attrs, `num_warps`, `num_stages`).

- **15 specialisations measured** (`results/capture_pinned_sm86.json`): l2norm 1, cumsum 2, kkt_solve 2, recompute_w_u 2, chunk state 4, chunk output 2, recurrent 2.
- **Engine minimum is 8**: a serving engine packs sequences (`cu_seqlens`), so only the varlen set is needed: l2norm, cumsum, kkt_solve, recompute_w_u, chunk state with and without initial state, chunk output, recurrent (stubs 00, 07 to 12, 14).
- Largest shared memory request in the pinned set is 47,364 B (chunk state), under the 48 KiB static limit, so no opt-in attribute is needed; the generated stub still sets it above 49,152 B.

Cubin bytes per SM (sources `results/manifest_sm*.json`):

| SM | Cubins | Total bytes (15) | Engine minimum (8) bytes | AOT compile, cold cache, wall | Sum of `triton.compile` | Hardware run |
| --- | --- | --- | --- | --- | --- | --- |
| 80 | 15 | 1,357,872 | 694,872 | 19.3 s | 12.2 s | none (untested) |
| 86 | 15 | 1,332,016 | 681,944 | 19.4 s | 12.5 s | RTX 3090, bit-exact vs FLA with the same pins |
| 89 | 15 | 1,330,992 | 681,432 | 19.7 s | 12.7 s | not run (see below) |
| 90 | 15 | 1,319,040 | 676,144 | 19.5 s | 12.5 s | none (untested) |
| 100 | 15 | 1,621,776 | 831,672 | 20.1 s | 13.1 s | none (untested) |
| 120 | 15 | 1,613,480 | 827,176 | 19.9 s | 13.0 s | none (untested) |

Per kernel on SM 86 (bytes, compile seconds, warps, stages, shared bytes): l2norm 22,144, 1.39, 2, 3, 64; cumsum 9,896 / 11,048, 0.06 / 0.07, 1, 3, 0; kkt_solve 193,520 / 194,672, 1.68 / 1.84, 1, 3, 9,216; recompute_w_u 93,424 / 94,064, 0.64 / 0.70, 2, 2, 20,480; chunk state 108,784 to 121,712, 0.96 to 1.05, 2, 2, 47,364; chunk output 85,616 / 85,232, 0.71 / 0.68, 2, 3, 20,480; recurrent 39,680 / 40,448, 0.35 / 0.34, 1, 3, 512 (fixed / varlen). Registers per thread from `cuobjdump --dump-resource-usage` on the SM 86 cubins (`results/cuobjdump_resources_sm86.txt`), all with zero stack and zero local memory: l2norm 64, cumsum 18 / 22, kkt_solve 222 / 222, recompute_w_u 208 / 214, chunk state 252 / 253 / 255 / 255, chunk output 188 / 188, recurrent 99 / 105. The chunk state kernel is at the 255-register hardware cap in the two varlen variants the engine needs (stubs 10 and 12) and next to it in the others, so any tuning there has no register headroom. Where this report gives one number for a kernel, it is the larger of its variants.

Build time is two separate costs:

1. **Autotune, once per SM and shape family**: about 101 to 104 s wall (cold Triton cache, the seven captured calls, 101 benchmark rounds over the autotuned kernels) on the 3090, including JIT compiles of every candidate. Two fresh runs on 2026-10-17 took 100.7 s and 104.2 s, each with exactly 101 rounds (`results/capture_autotuned_run{1,2}_sm86.log`; wall is the whole `capture.py` process including Python start). The first run, which produced the committed pins, was timed at 106.9 s but its output was not saved, so that figure is not backed by a log. This picks the configs; it needs the target GPU. AOT does not repeat it: `pins.json` freezes one config per kernel.
2. **AOT compile of the pinned set**: 19.3 to 20.1 s per SM for 15 cubins on one core (`total_compile_seconds` is measured inside `main()` after `import triton`, so process start is not in it; the roughly 7 s over the `triton.compile` sum is FLA module import, `create_binder` and formatting the cubins as C arrays). The sm_80 to sm_120 sets were built offline on the 3090 host.

Findings on the build step:

- **Needs a visible CUDA device.** With `CUDA_VISIBLE_DEVICES=""` (`results/build_aot_no_device.log`), `build_aot.py` fails in `JITFunction.create_binder` at `driver.active.get_current_target()` with `RuntimeError: 0 active drivers ([]). There should only be one.` The cross-SM compile (`GPUTarget("cuda", 89, 32)`) works once any NVIDIA driver is present, but a GPU-less CI box cannot run this build as written. I did not try to work around it.
- **Autotune winners are not reproducible.** Three cold autotuned captures on the 3090 are committed (`results/capture_autotuned_sm86.json` and `capture_autotuned_run{1,2}_sm86.json`). The same script chose different configs in them. Run 1 picked 4 warps and 40,960 B for the non-varlen `recompute_w_u_fwd_kernel` where the other two picked 2 warps and 20,480 B, and `BV=64`, 4 warps, 65,796 B for the varlen `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` without initial state where the other two picked `BV=32`, 4 warps, 49,412 B; the first capture picked 2 warps and 47,364 B for the varlen chunk state kernel with initial state where runs 1 and 2 picked 4 warps and 49,412 B. Within each run the chunk state kernel also got different configs for different call patterns (2 or 4 warps). `chunk_fwd_kernel_o` chose `BK=32`, `BV=32`, 2 warps in all three; an earlier note that one run chose `BK=128`, `BV=128`, 8 warps for it is not backed by any saved output and is withdrawn. Against stock autotuned FLA the pinned build differs in cubins: 3 of the 15 kernels (the chunk state kernel: both non-varlen variants and the varlen variant without initial state) were 4 warps and 49,412 B in the autotuned capture against 2 warps and 47,364 B pinned. The AOT build therefore must pin configs, not autotune, and bit-exactness is shown against Python FLA with the same pinned configs only.
- **Pins are from a 3090 at these shapes.** They are applied unchanged to the other SMs, so the non-86 cubins are compiled with configs nobody tuned for them. The autotune keys of the chunk kernels do not include T, so the pin is shape-stable for this head configuration; the l2norm `NB` constexpr changes above 65,536 tokens per call and would need another specialisation.
- **Triton's stock `triton.tools.compile` was not used.** Reading `compile.py` (Triton 3.8.0): it parses signature entries only as numbers or types, so `None` constexprs (`cu_seqlens=None`, `gk=None`) cannot be expressed; it expects a bare `JITFunction` while FLA wraps them in `Heuristics` and `Autotuner`; and it bakes the launch grid into the stub, whereas FLA's grids depend on T and `cu_seqlens` at run time. `build_aot.py` makes the same `triton.compile(ASTSource, target, options)` call and writes its own stub with a runtime grid. The cubins it produces are identical to what the JIT produced.

## Numeric diff

Basis: Python FLA with the same pinned configs (`capture.py`/`test_aot.py` reduce each FLA autotuner to the single pinned config), same inputs, same process, RTX 3090. The AOT side runs through the C ABI library (`gdn_api.c`, built with `gcc -O2 -shared -fPIC ... -lcuda`), called by ctypes with torch device pointers. The workspace is filled with 0xFF and the outputs `o` and `ht` with NaN before every call, so an unwritten element would show as NaN. The mismatch count is `a != b`, so a NaN counts as a mismatch.

| Case | Tokens | Elements compared (o, state) | Max abs diff | Mismatched |
| --- | --- | --- | --- | --- |
| prefill, packed [300,77,1500,64,1], initial state | 1,941 | 3,977,216 and 1,310,720 | 0 | 0 |
| prefill, packed [300,77,1500], no initial state | 1,877 | 3,844,096 and 786,432 | 0 | 0 |
| prefill, one sequence 1,024, initial state | 1,024 | 2,097,152 and 262,144 | 0 | 0 |
| prefill, one sequence 1,000, no initial state | 1,000 | 2,048,000 and 262,144 | 0 | 0 |
| decode, 4 sequences of 1 token | 4 | 8,192 and 1,048,576 | 0 | 0 |
| decode, packed [5,1,1] | 7 | 14,336 and 786,432 | 0 | 0 |
| decode, one sequence of 3 tokens | 3 | 6,144 and 262,144 | 0 | 0 |

Context, not a gate: against an fp32 recurrent reference (`naive_recurrent_gated_delta_rule` on l2-normalised fp32 inputs, one 256-token sequence with initial state), both the AOT build and Python FLA differ by max 5.0e-4 on `o` (reference max 8.9e-2) and 3.3e-3 on the final state (reference max 0.94), identically. That is bf16 intermediate rounding in the chunk path, inherent to FLA, not an AOT effect. The inputs are random (`common.make_inputs`), not Qwen3.5 activations, so these are not protected-fixture numbers and say nothing about model fidelity.

SM 89: cubins built (table above) and the C library compiled; the numeric check was **not run**. The RTX 4060 Ti had too little free memory; `tools/gpu-lease run --device 4060 --need-mib 1500 --wait 5` refused (exit 75): "has 278 MiB free, the job needs 1500 MiB" when first tried and "has 374 MiB free, the job needs 1500 MiB" when retried on 2026-10-17 (`results/gpu_lease_4060_refusal.log`). Run `run_all.sh` step 4 with `--device 4060` when it has room. SM 80, 90, 100 and 120 have cubins only; their status is `untested`.

## C ABI contract findings

These are what a Roost wrapper must reproduce (all in `gdn_api.c`):

- **Zero `A` before the kkt kernel.** The fused kkt+solve kernel writes only the lower-triangular blocks of the `[T, HV, 64]` buffer and relies on FLA's `torch.zeros`. With an uninitialised buffer the output was NaN; `cuMemsetD8Async` on `A` fixed it. `results/test_aot_no_memset_sm86.log` reproduces this: with the memset skipped (`GDN_SKIP_A_MEMSET=1`, a spike-only switch in `gdn_api.c`) the same four prefill cases return NaN for every element of `o` and the state (`mismatched` equals the element count), while `results/test_aot_sm86.log` has 0 mismatches. Every other buffer (including `h`, `v_new`, `w`, `u`, the normalised q and k, `o` and `ht`) was fully written by its kernel, shown by the NaN poisoning above.
- **Host-side tables.** The varlen kernels take `chunk_indices` (int32 `[NT,2]`: sequence, chunk in sequence) and `chunk_offsets` (int64 `[N+1]`) in addition to `cu_seqlens` (int32). FLA builds them with torch ops (a device sync on CUDA counts); `gdn_make_chunk_tables` does it in 12 lines of C on the host. `NT` is the sum over sequences of ceil(len/64) and sets grid sizes.
- **Grids** (replicating FLA's wrappers): l2norm `ceil(rows / 16)` with rows = tokens * 16; cumsum `NT*16`; kkt_solve and recompute_w_u `(NT, 16)`; chunk state `4*N*16`; chunk output `(4*NT, 16)`; recurrent `16*N*16`.
- **Argument lists** per stub are in `results/manifest_sm*.json` (`runtime_args`): pointers need 16-byte alignment (the specialisation carries `tt.divisibility 16`); `T` is a non-specialised `i32` (total tokens when varlen); `scale` is `fp32`.
- **Workspace** per prefill call, from `gdn_chunk_workspace_bytes`: normalised q and k, their rstd, gate cumsum, A, w, u, v_new, and `h` for `NT` chunks (`NT*16*128*128*2` B), about 30 MiB for 1,024 tokens.
- **Stub design.** One cubin per stub embedded as a C array, lazy `cuModuleLoadData`, runtime grid; single-threaded first use (no lock). Linking needs only `-lcuda`.

## Port estimate (decision 23)

Measured inputs: 691 lines of Triton across the seven kernels (counts in the first table), `tl.dot` call sites in the dot-heavy files (wy_fast 11, chunk_o 19, chunk_delta_h 33, chunk_fwd 26), and registers per thread above (no spills; chunk state at the cap). Estimate, engineer-weeks for one experienced CUDA engineer, forward only, `mma.sync` bf16 and tf32 (CuTe or hand-written), no wgmma or tcgen05:

| Kernel | Estimate | Why |
| --- | --- | --- |
| l2norm | 0.3 | row reduction |
| chunk-local cumsum | 0.3 | warp scan over 64 |
| recompute_w_u | 0.8 | two small tensor-core GEMMs per chunk |
| chunk output | 1.0 | q K^T, masked by decay, plus q h, two GEMMs |
| recurrent (decode) | 0.8 | CUDA cores only; also add the indexed-state-slot variant Roost's C4 pool needs |
| kkt_solve (fused) | 2.5 | beta K K^T over a K loop, then a blocked 16x16 triangular inverse in tf32 or fp32; 222 registers |
| chunk state | 3.0 | loop-carried 128x128 fp32 state across chunks, K split in 64-wide blocks, async copy pipelining; 252 to 255 registers, at the hardware cap |
| Subtotal, kernels | 8.7 | |
| C ABI wrapper, host tables, workspace sizing | 1.0 | `gdn_api.c` is the starting point |
| Conformance harness against this AOT build and starling references | 1.0 | not bit-exact; needs a tolerance agreed with the maintainer |
| Tuning to Triton parity on SM 86 and 89, then SM 80, 90 | 2 to 4 | no performance number was measured in this spike |
| **Total** | **12.7 to 14.7** (about 13 to 15) | the rows sum to 8.7 + 1.0 + 1.0 + (2 to 4); the wider range is 10 to 16 depending on how close to Triton the first tuned version must be |

Assumptions: Triton cubin performance is the bar (not measured here: the AOT cubins are the same bytes as FLA's, so the AOT path is at FLA's speed by construction); no Hopper or Blackwell-specific paths in the first version; backward is out of scope. Under 6 weeks would need an existing CUDA implementation of these kernels to adopt, and I did not look for one; over 20 would need wgmma or tcgen05 versions per architecture family. The plan's "port by M3" is plausible at this size. Until the port passes conformance the AOT path stays, with the costs above: a Python 3.12 + torch + Triton + transformers + FLA closure and a GPU-bearing build host.

## Caveats

- **Which vLLM.** The S3 question names the kernels vLLM uses, but this spike measured upstream `fla-org/flash-linear-attention` at `329b534`, not vLLM's vendored copy. The plan's baseline is vLLM v0.31.0; its vendored FLA revision was not read, so how far it differs from `329b534` (beyond the indexed-state-slot recurrent kernel below) is unmeasured. R-034 must diff the two before relying on these cubin sizes, registers or the port estimate.
- vLLM carries its own copy of the recurrent kernel with indexed state slots (`ssm_state_indices`) for paged recurrent state. This spike measured upstream FLA at the pinned SHA, which writes the final state to a dense `[N,HV,K,V]` tensor. Roost's recurrent-state pool (C4) needs the indexed variant, which is a source change to kernel 7 and a new specialisation.
- The causal conv1d and the gating kernels (`g`, `beta`) are outside this spike.
- Inputs are random; pins and bit-exactness are for SM 86 at the Qwen3.5-0.8B head configuration. Other Qwen3.5 sizes change `H`, `HV`, `K` or `V` and so recompile every kernel with new constexprs.
- Nothing here is a throughput claim.

## Reproduce

`spikes/s3-triton-gdn/run_all.sh` runs every step (step 1 writes its autotune winners to `out/pins_autotuned.json` and leaves the committed `pins.json` alone, since autotune is not reproducible), with each GPU run as its own `tools/gpu-lease run --device 3090 --need-mib N --stream C -- ...`. Result files committed: `results/capture_autotuned_sm86.json`, `results/capture_pinned_sm86.json`, `results/manifest_sm{80,86,89,90,100,120}.json`, `results/test_aot_sm86.log`, `results/test_aot_no_memset_sm86.log`, `results/capture_autotuned_run{1,2}_sm86.{json,log}`, `results/build_aot_no_device.log`, `results/gpu_lease_4060_refusal.log`, `results/cuobjdump_resources_sm86.txt`, `results/env_versions.txt`. Cubins, generated C and the `.so` live under the git-ignored `out/`.

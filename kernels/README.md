# Kernel library (stream C)

CUDA C++ behind the C1 ABI (`kernels/include/roost_kernels.h`, frozen). One shared library is built per SM,
`libroost_kernels_sm<N>.so`, so the digest of every release asset is per SM and a missing SM is a missing file.
Each library exports `roost_abi_version`, `roost_last_error`, `roost_register` (the registration table),
`roost_context_create` and `roost_context_destroy`. The table holds the `residual-stub` variant of `Residual` (R-016: a
correct, unoptimised op that proves the build, registration and run path) and the ten M0 attention-side variants of R-017
(next section). R-034 adds the hybrid ops.

## Files

| Path | Role |
| --- | --- |
| `manifest.toml` | Contract C5: SM list with a status per SM, upstream pins (full SHA, licence per path, patches), ops and variants, binary digests, Triton AOT closure. The CMake build reads its SM list and pins. |
| `CMakeLists.txt`, `cmake/RoostUpstreams.cmake` | The build. Upstreams are fetched at configure time with `git fetch --depth 1 origin <sha>`, refused unless HEAD is that SHA, then patched from `upstream/<name>.patches/` (each patch checked against the manifest's SHA-256). |
| `src/` | `registry.cu` (C1 exports and the table), `residual.cu` (the stub), `m0_registry.cu` (the R-017 entries and their constraints), `m0_ops.cu` (RmsNorm, QkNorm, Rope, SiluAndMul, Embedding, CacheWrite, CacheCopy), `paged_attention.cu`, `gemm.cu` (Gemm and LmHead), `op_common.h` (validation shared by the ops). |
| `tests/stub_check.cpp` | The host program that loads a library by its exports and runs the Residual stub. |
| `tests/ops-cuda/` | R-017's GPU conformance (`Roost.KernelOps.Tests.Cuda`, local only, not in `Roost.slnx`): loads a built library through its C1 exports and compares every M0 op with `CpuOps` on random inputs and on real Qwen3.5-0.8B weights. |
| `tools/manifest_cmake.py` | Manifest to CMake variables. |
| `tools/record_digests.py` | SHA-256 of the built libraries; `--write` records them in the manifest, `--check` compares. |
| `tools/check_provenance.py` | Fails a pin or patch-set change that does not change `PROVENANCE.md` and `THIRD-PARTY-NOTICES.md` (and any tree where the manifest and those files disagree). |
| `ci/build-all-sm.sh` | What the CI `kernels` job runs: every SM, built twice, byte-compared, registration table checked without a GPU. |
| `results/` | Raw text logs of GPU runs (`r016-stub-sm86.log`, `r017-ops-sm86.log`, `r017-graph-sm86.log`, `r017-stub-check-sm86.log`) and one excerpt (`r017-rejected-configs-sm86.log`). |

## Build

```sh
# cmake >= 3.24, ninja, CUDA 13.x, Python >= 3.11, git. Throwaway output goes outside the repository or under kernels/build/.
cmake -S kernels -B kernels/build -G Ninja -DROOST_SM_LIST=86 -DROOST_UPSTREAM_DIR=<cache dir>
cmake --build kernels/build -j2
python3 -I kernels/tools/record_digests.py kernels/build --check
tools/gpu-lease run --device 3090 --need-mib 500 --stream C -- kernels/build/roost_stub_check kernels/build/libroost_kernels_sm86.so
# R-017 conformance against CpuOps (needs the SM 86 library above and the .NET SDK; the locked checkpoint is optional):
dotnet build kernels/tests/ops-cuda -c Release -o <out dir>
tools/gpu-lease run --device 3090 --need-mib 6000 --stream C --wait 3600 -- \
  dotnet <out dir>/Roost.KernelOps.Tests.Cuda.dll kernels/build/libroost_kernels_sm86.so --log kernels/results/r017-ops-sm86.log
```

Leave `ROOST_SM_LIST` out to build every SM in the manifest. The M0 ops compile against the pinned FlashInfer headers and link
cuBLASLt (`libcublasLt.so.13` of the CUDA toolkit), so `-DROOST_FETCH_UPSTREAMS=OFF` is refused at configure time.

## M0 ops (R-017)

Specialised for the Qwen3.5-0.8B full-attention layers (config.json of `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`:
hidden 1024, 8 query heads over 2 KV heads, head_dim 256, `partial_rotary_factor` 0.25 so 64 rotary dimensions, `rope_theta` 1e7, MLP 3584,
vocabulary 248320, tied embeddings, `rms_norm_eps` 1e-6). The output gate of the attention layer, the `(1 + weight)` norm convention and the
mrope sections are model definition (R-035), not ops.

| Op | Variant id | Source | Shapes and types built | Flags | Workspace |
| --- | --- | --- | --- | --- | --- |
| RmsNorm | `rms-norm-flashinfer` | FlashInfer `norm::RMSNorm` | bf16, f16; any `dim`; row strides | deterministic, batch invariant | none |
| QkNorm | `qk-norm-flashinfer` | FlashInfer `norm::QKRMSNorm`, called for q then k, in place | bf16, f16; head_dim 256 registered | deterministic, batch invariant | none |
| Rope | `rope-roost` | Roost (see below) | bf16, f16; default and llama3; NeoX and interleaved; `rotary_dim` <= `head_dim`; head_dim 64, 128, 256 registered | deterministic, batch invariant | none |
| PagedAttention | `paged-attention-flashinfer-fa2` | FlashInfer FA2 batch decode and batch prefill | bf16 Q, KV and output; head_dim 256; 4 query heads per KV head; NHD and HND; page sizes 16, 32, 64 registered; causal or not; no sliding window, no soft cap | deterministic, needs plan | query: int and float scheduler buffers (0 to 8.2 MiB in the tested batches), pinned staging, plan record |
| SiluAndMul | `silu-and-mul-flashinfer` | FlashInfer `act_and_mul_kernel`, launcher written here | bf16, f16; any `dim` | deterministic, batch invariant | none |
| Gemm | `gemm-cublaslt` | cuBLASLt, algorithm pinned per shape bucket | bf16, f16 with fp32 accumulation; B as `[k,n]` or `[n,k]`; alpha, beta, C, bias | deterministic, batch invariant, needs plan, plan per shape | 8 MiB cuBLASLt scratch, plan record |
| Embedding | `embedding-roost` | Roost gather | bf16, f16, f32 tables | deterministic, batch invariant | none |
| LmHead | `lm-head-cublaslt` | gather of the selected rows, then cuBLASLt with fp32 logits and `ldd = vocab_padded`, then optional soft cap | bf16, f16 hidden and weight | deterministic, batch invariant, needs plan, plan per shape | gathered rows plus 8 MiB scratch, plan record |
| CacheWrite | `cache-write-roost` | Roost (C1's `slot_mapping` form does not match FlashInfer's append API) | bf16, f16, f32, same type as the activations; NHD and HND | deterministic, batch invariant | none |
| CacheCopy | `cache-copy-roost` | Roost | any element type, whole blocks, one or two pools | deterministic, batch invariant | none |

How the ops meet C1:

- **No allocation, no synchronisation inside a call.** `run()` of every op is kernel launches (and for Gemm and LmHead one `cublasLtMatmul`)
  on the caller's stream. Setup that allocates is in `roost_context_create` (the cuBLASLt handle) and in `plan()` (FlashInfer's scheduler and
  cuBLASLt's heuristic use host containers). `kernels/tests/test_kernel_tools.py` scans the op sources for the calls that would allocate or
  synchronise. FlashInfer's `ToVector`/`FromVector` plan conversion, which S2 found allocating a host vector per call, is not used: the plan
  record is the scheduler's plain struct, copied with `memcpy`.
- **Workspace queries.** PagedAttention's query reads the host copies of the batch tables (`q_indptr_host`, `kv_indptr_host`); its device
  workspace is `[int scheduler buffer | float split-KV buffer]` and `plan()` fills it through the pinned staging buffer. Gemm and LmHead need
  `plan()` once per shape bucket: it asks cuBLASLt's heuristic for M = `shape_bucket` (0 means the call's own M) and records the algorithm in
  `ws->plan`; `run()` accepts any `m` up to the bucket and returns `INVALID_ARGUMENT` above it, `NOT_PLANNED` without a plan.
- **Typed refusals instead of faults.** A wrong struct size or minor, a reserved field, a short span, a dtype or alignment the variant does not
  build, a head size, group size, sliding window or soft cap outside the specialisation: a status and a message, before any launch. Values that
  only the device can see are made harmless instead of checked: an Embedding id or LmHead row outside its range gives a zero row, a CacheWrite
  slot or CacheCopy pair outside the pool is skipped. CacheCopy pairs of one call must be independent (the CPU reference copies them in order).
- **Pinned Gemm algorithms.** S2 showed the default heuristic changes the split-K count with M, so rows differ bitwise between batch sizes. A plan
  fixes one algorithm per (N, K, dtype, bucket); the tests check that, with a plan for 64 rows, calls of 1, 2, 5, 17, 33 and 63 rows reproduce
  the 64-row result bit for bit for five Qwen3.5 shapes, three of them split-K (5, 8 and 9 ways). The algorithms observed for bucket 64 on SM 86 are recorded in the
  manifest (`gemm_algos`) with the digest of the cuBLASLt build; the plan record is private, so the check that a new cuBLASLt picks the same
  algorithm belongs to the conformance suite (R-019).

Findings that changed the design (evidence differs per finding: the Rope result is excerpted in `results/r017-rejected-configs-sm86.log` and the replacement is logged in `results/r017-ops-sm86.log`; the split-K figures have no log; the head_dim 256 attention result is in `results/r017-ops-sm86.log`):

- **FlashInfer's Rope is not accurate enough.** It forms `pos * freq` in fp32 and calls the `__sincosf` intrinsic, which does no range reduction.
  Against `CpuOps` its error grew with the position: 1.6e-2 absolute on outputs of RMS 1 (several bf16 ulps) for positions below 4096 and
  7.8e-2 for positions up to 2^18, in a first run of these tests (the FAIL lines are excerpted unedited in `results/r017-rejected-configs-sm86.log`; the
  full log was never committed and the harness has no flag that reruns the rejected kernel; the cause is in `pos_enc.cuh`: `__sincosf` and
  `__powf`). `rope-roost` forms frequency and angle in double, reduces the angle modulo 2 pi in double and calls `sincosf` on the reduced
  angle; with it the same cases, up to positions of 2^31 - 1, stay inside the elementwise tolerance below (the 2^31 - 1 cases peak at utilisation 0.436, and 0.491 is the largest of the 55 Rope checks in `results/r017-ops-sm86.log`).
- **cuBLASLt's default split-K reduction loses fp32 accumulation with a 16-bit output.** With the default reduction schemes, split-K plans kept
  partial sums in the output type and showed errors of 1.6e-2 to 3.1e-2 absolute on outputs of RMS 1 against `CpuOps` (about 4 bf16 ulps; the
  failing cases were the shapes whose plan chose split-K, 8 and 9 ways, in a first run). **These split-K figures have no committed or local log and are unreproduced**: treat them as the implementer's observation, not evidence. The plan now allows only
  `CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE`, which keeps fp32 partial sums and is deterministic.
- **FlashInfer attention at head_dim 256** builds for SM 86 with the unmodified headers (prefill chooses CTA tiles of 16 or 64 rows; 128 never
  occurs at this head size). Decode, prefill, chunked prefill over an existing cache, multi-sequence batches, non-causal prefill and split-KV
  (6000-token context) all match `CpuOps`; runs are bit-identical.

### Proposed tolerances (for R-019's protected file; nothing here edits one)

An element passes when `|gpu - ref| <= atol + rtol * |ref|` with `ref` from `CpuOps` (double, rounded once to the element type) and `atol`
a fraction of the reference's RMS (of max |V| for attention). Utilisation is the largest `|gpu - ref| / allowed` over all cases of the class in
`results/r017-ops-sm86.log` (1.0 is the limit).

| Class | rtol | atol | Cases | Largest utilisation | Largest abs error |
| --- | --- | --- | --- | --- | --- |
| exact (Embedding, CacheWrite, CacheCopy) | 0 | 0 | 26 | 0 (bit exact) | 0 |
| elementwise bf16 (RmsNorm, QkNorm, Rope, SiluAndMul) | 2^-6 | 2^-12 rms | 46 | 0.43 | 7.8e-3 |
| elementwise f16 | 2^-9 | 2^-15 rms | 43 | 0.49 | 3.9e-3 |
| gemm bf16 / f16 (fp32 accumulate, 16-bit result) | 2^-6 / 2^-9 | 2^-9 / 2^-12 rms | 40 / 21 | 0.46 / 0.47 | 1.6e-2 / 3.9e-3 |
| lm-head logits (fp32 out) | 2^-12 | 2^-13 rms | 6 | 0.02 | 1.6e-5 |
| attention bf16 | 2^-7 | 2^-8 max abs V | 20 | 0.49 | 1.6e-2 |

The attention bound comes from FlashInfer rounding the softmax weights to bf16 before the PV product: each output carries up to
2^-8 * sum(p |v|) <= 2^-8 * max|V| whatever its own size. A first proposal of 2^-6 relative to the output RMS failed two of 1.4M elements
of long prefills (1.56e-2 absolute at |V| up to about 4.5) and was replaced by this one; no kernel changed. **The criterion was changed after the code failed it**: the failing lines are in
`results/r017-rejected-configs-sm86.log` (utilisation 1.16 and 1.10 under the first criterion, 0.45 and 0.40 under the new one, same 1.56e-2 error). It is a
proposal for a maintainer to read (plan section 9), not a validated bound.

### Signal death during R-017

One run of the harness ended with a glibc abort: `tools/gpu-lease run --device 3090 --need-mib 6000 --stream C --wait 3600 -- dotnet
Roost.KernelOps.Tests.Cuda.dll libroost_kernels_sm86.so --filter attention,graph --log run4.log` printed `double free or corruption
(!prev)` after the two failures of the `graph` group, so the process died by SIGABRT (the pipe hid the exit code). The cause is in the test
harness, not the library: `GraphCase` assigned its own `q` buffer to the batch's `Q`, so `Batch.Dispose` and `using var q` freed it twice. The
assignment is removed. The `graph` group was then run once more, on the ABI 1.1 library (`results/r017-graph-sm86.log`, raw): the process exited
normally with code 1, no signal, and `coredumpctl list` held 1165 lines before and after. The five-op decode step (RmsNorm, Gemm, Rope,
CacheWrite, PagedAttention) captured into a CUDA graph without error (1 pass), but the replay did not equal the eager step bitwise and the second
replay did not equal the first (2 failures); the cause is unknown and was not investigated or rerun. CUDA graph replay is R-033's criterion, not R-017's.
The group stays opt-in (`--filter graph`), and no R-017 variant claims `graph_capturable`.

## Reproducibility and digests

Builds are byte-identical per SM when made from the same checkout path (the same path twice; the CI job compares two builds per SM at one path).
They are **not** path independent: a review build of this branch from another checkout gave sha256 14856fad... for SM 86 against 0f707ae0... in the
manifest. Sizes are equal and all 576 differing bytes are nvcc `_INTERNAL_<hash>_<file>_cu_...` symbol names, whose hash derives from the source
path (they come from `cuda::std` entities in the headers, so a named namespace in our sources does not remove them). The earlier statement that two
paths agree was wrong for this tree. The manifest digests are reproducible only from `/home/mark/Projects/roost-wave4/R-017`; making the build path
independent (for example a fixed build path in CI) is open. Linking needs no line info (it records source mtimes), no build id and no local symbols
(nvcc names them after the process id). The digest also depends on the toolchain (CUDA 13.3.73, gcc 16.2.1).
The manifest's SM 86 digests were re-recorded after the C1 ABI 1.1 header (one build, the development host's); the digests of SM 80, 89, 90, 100 and 120 are still those of the ABI 1.0 build and are stale until CI or the release workflow records them. Only SM 86 has run on hardware. The CI job
reports, and does not fail on, a difference from them; the release workflow (R-060) records the digests of the pinned
container build and publishes the libraries.

## Not done

- R-017: `graph_capturable` is not claimed for any M0 op (graph replay is R-033's; see "Signal death during R-017"); GeluAndMul, Argmax and Sample are not in R-017's list; FP8 and quantised KV (R-043), INT4 (R-077), sliding window (R-113) and RoPE scaling variants (R-098) are refused with `UNSUPPORTED`; no throughput number is claimed (the tests check numerical agreement only); SM 80, 89, 90, 100 and 120 compile (CI builds them) but no M0 op has run on them.

- MSVC kernel build: GitHub-hosted Windows runners ship no CUDA toolkit, so it is not built. The sources use `ROOST_API`
  from the C1 header and no POSIX types, and `stub_check.cpp` has a `LoadLibrary` path, but none of it has been compiled with
  MSVC.
- Triton AOT: the manifest pins the Python closure and the seven kernel configs from S3 (six in `pins.json`, the seventh from S3's capture log and report), and CMake fetches the
  flash-linear-attention source by SHA, but the AOT compile (which needs a visible CUDA device) is R-034's.
- CUTLASS and vLLM are fetched and verified by SHA; no op compiles against them (FlashInfer is used by R-017; cuBLASLt serves Gemm).

## Open before merge

- MSVC build: an unmet R-016 criterion that needs a maintainer decision (self-hosted Windows runner with CUDA, a CUDA install step, or relaxing the criterion and amending the plan).
- The `kernels` CI job has not run in GitHub Actions; its script passes locally for all six SMs outside the container.
- The patch-set check has never run on a real patch (there are none yet).
- `toolkit.min_driver = "580"` is unverified; only driver 610.57 was tested.
- SM 80, 89, 90, 100 and 120 are built and registration-checked only; none ran on hardware.

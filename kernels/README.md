# Kernel library (stream C)

CUDA C++ behind the C1 ABI (`kernels/include/roost_kernels.h`, frozen). One shared library is built per SM,
`libroost_kernels_sm<N>.so`, so the digest of every release asset is per SM and a missing SM is a missing file.
Each library exports `roost_abi_version`, `roost_last_error`, `roost_register` (the registration table),
`roost_context_create` and `roost_context_destroy`. The table holds the `residual-stub` variant of `Residual` (R-016: a
correct, unoptimised op that proves the build, registration and run path) and the hybrid ops of R-034 below. R-017 adds the
attention-side ops.

## Files

| Path | Role |
| --- | --- |
| `manifest.toml` | Contract C5: SM list with a status per SM, upstream pins (full SHA, licence per path, patches), ops and variants, binary digests, Triton AOT closure. The CMake build reads its SM list and pins. |
| `CMakeLists.txt`, `cmake/RoostUpstreams.cmake` | The build. Upstreams are fetched at configure time with `git fetch --depth 1 origin <sha>`, refused unless HEAD is that SHA, then patched from `upstream/<name>.patches/` (each patch checked against the manifest's SHA-256). |
| `src/ops/`, `cmake/RoostHybrid.cmake` | The hybrid ops of R-034 (below): one file per op family, their registration entries (`hybrid_entries.cu`) and their CMake hook. |
| `aot/` | Triton AOT, build time only (decision 23): `capture_specs.py` records the specialisations from live FLA launches, `gdn_specs.json` is its output, `build_gdn.py` compiles them to a C++ file of cubins per SM. |
| `tests/hybrid-conformance/` | Local .NET harness (not in `Roost.slnx`): the hybrid ops against `CpuOps` on random inputs, through the C ABI. |
| `src/`, `tests/stub_check.cpp` | The registration table, the Residual stub, and the host program that loads a library by its exports and runs the stub. |
| `tools/manifest_cmake.py` | Manifest to CMake variables. |
| `tools/record_digests.py` | SHA-256 of the built libraries; `--write` records them in the manifest, `--check` compares. |
| `tools/check_provenance.py` | Fails a pin or patch-set change that does not change `PROVENANCE.md` and `THIRD-PARTY-NOTICES.md` (and any tree where the manifest and those files disagree). |
| `ci/build-all-sm.sh` | What the CI `kernels` job runs: every SM, built twice, byte-compared, registration table checked without a GPU. |
| `results/` | Raw text logs of GPU runs. |

## Hybrid ops (R-034)

Qwen3.5-0.8B has 18 Gated DeltaNet layers (decision 15). Specialised for 16 key and value heads, head size 128, conv width
up to 16 (4 for Qwen3.5), bf16 activations and float32 recurrent state; anything else is `ROOST_STATUS_UNSUPPORTED`.

| Variant id | Op | Source | Flags |
| --- | --- | --- | --- |
| `gated-delta-rule-chunk-aot` | GatedDeltaRule, chunked prefill | S3 Triton AOT cubins of FLA at the manifest's SHA | deterministic, batch invariant, needs plan |
| `gated-delta-rule-recurrent-aot` | GatedDeltaRule, any length (decode, short continuations) | S3 AOT `fused_recurrent` cubin | graph capturable, deterministic, batch invariant |
| `causal-conv1d-cuda` | CausalConv1d | CUDA C++ (S3 did not cover it) | graph capturable, deterministic, batch invariant |
| `recurrent-state-copy-cuda`, `recurrent-state-snapshot-cuda` | RecurrentStateCopy, RecurrentStateSnapshot | CUDA C++ | graph capturable, deterministic, batch invariant |

- **Strided views.** q, k and v are read through their row strides (`q_stride`, `k_stride`, `v_stride`, 6144 for the
  conv1d output of Qwen3.5-0.8B). The stock FLA cubins read dense tensors, so each call first gathers the views into
  workspace (l2-normalising q and k on the chunk path, as FLA's `l2norm` kernel does) and the recurrent kernel normalises
  its own copy. The recurrent state is gathered from the pool slots into a dense tensor and scattered back. That costs
  about 2 MiB of state traffic per sequence and layer in each direction; a kernel patch that reads the views and slots
  directly is the follow-up, and the reason the AOT inputs are specs rather than a patch set.
- **Which variant.** The chunk variant is the prefill path (`plan()` builds FLA's chunk tables from `cu_seqlens_host`;
  run it on the host side of the step, never inside a graph). The recurrent variant takes any sequence lengths; the
  recurrent kernel is serial over tokens, so the loader should pick it for decode and short continuations. The choice is
  the caller's: the params carry no mode.
- **Workspace.** `workspace` gives the device bytes (`plan_bytes` 16 and `pinned_host_bytes` for the chunk tables on the
  chunk variant), bounded by shape only (chunks at most `tokens / 64 + batch`). The caller keeps the pinned buffer alive
  until the copy `plan()` queues has run. Nothing allocates inside a call; cubins load when the context is created.
- **Slots.** Slot indices are device data and are not range-checked. A sequence of zero tokens (or a negative slot) leaves
  its slot untouched, as `CpuOps` does.
- **Params of CausalConv1d and the state ops** are not in `roost_kernels.h` (contract C1, stream A). They are declared in
  `src/ops/hybrid_params.h`, mirroring the C3 argument structs, until a stream A contract-change PR lifts them into the
  header (ABI 1.2) with their C# mirrors. No loader can call these three ops before that.
- **Build.** The AOT step needs a visible CUDA device (S3; importing FLA queries `torch.cuda`), so it is opt-in and a
  GPU-less build compiles the library without `GatedDeltaRule` (it is then not registered, which the loader reports as a
  missing op). Pass the Python of the manifest's `[triton_aot]` closure and run the build through `tools/gpu-lease`:

```sh
cmake -S kernels -B <build> -G Ninja -DROOST_SM_LIST=86 -DROOST_FETCH_UPSTREAMS=OFF \
  -DROOST_TRITON_PYTHON=<venv>/bin/python -DROOST_FLA_DIR=<flash-linear-attention checkout at the manifest's SHA>
tools/gpu-lease run --device 3090 --need-mib 500 --stream C -- cmake --build <build> -j2
dotnet build -c Release kernels/tests/hybrid-conformance -o <harness dir>
tools/gpu-lease run --device 3090 --need-mib 6000 --stream C -- dotnet <harness dir>/Roost.HybridConformance.dll \
  <build>/libroost_kernels_sm86.so --log kernels/results/r034-hybrid-conformance-sm86.log
```

Conformance (`kernels/results/r034-*`) compares each op with `CpuOps` (double precision) on random inputs: the chunk
variant against S3's measured bf16 rounding of the chunk path, the recurrent variant and the CUDA ops much tighter.
The tolerances are development limits inside the harness, not a tolerance file; the maintainer sets the real ones (R-019).

## Build

```sh
# cmake >= 3.24, ninja, CUDA 13.x, Python >= 3.11, git. Throwaway output goes outside the repository or under kernels/build/.
cmake -S kernels -B kernels/build -G Ninja -DROOST_SM_LIST=86 -DROOST_UPSTREAM_DIR=<cache dir>
cmake --build kernels/build -j2
python3 -I kernels/tools/record_digests.py kernels/build --check
tools/gpu-lease run --device 3090 --need-mib 500 --stream C -- kernels/build/roost_stub_check kernels/build/libroost_kernels_sm86.so
```

Leave `ROOST_SM_LIST` out to build every SM in the manifest. `-DROOST_FETCH_UPSTREAMS=OFF` skips the fetch (no op uses
an upstream yet, so the stub builds without it).

## Reproducibility and digests

Builds of the same sources are byte-identical per SM, also from different checkout paths (checked for all six SMs at two
paths with CUDA 13.3.73 and gcc 16.2.1; the CI job also compares two builds per SM). That needs no line info (it records
source mtimes), no build id, no local symbols (nvcc names them after the process id) and no anonymous namespace in a `.cu`
file: nvcc derives that namespace's symbol hash from the source path, so kernels live in a named namespace. This is
verified on one toolchain only; the digest still depends on the toolchain, so another nvcc or gcc gives other digests.
The manifest's six digests are the development host's, one build each, and only SM 86 has run on hardware. The CI job
reports, and does not fail on, a difference from them; the release workflow (R-060) records the digests of the pinned
container build and publishes the libraries.

## Not done

- MSVC kernel build: GitHub-hosted Windows runners ship no CUDA toolkit, so it is not built. The sources use `ROOST_API`
  from the C1 header and no POSIX types, and `stub_check.cpp` has a `LoadLibrary` path, but none of it has been compiled with
  MSVC.
- Triton AOT: built by R-034 (above) for SM 80 to 120; only SM 86 has run on hardware, the others are `untested`.
  A GPU-less CI build does not compile the AOT cubins. Retiring Triton for a CUDA C++ port is an M3 item (decision 23).
- FlashInfer, CUTLASS and vLLM are fetched and verified by SHA; no op compiles against them yet (R-017).

## Open before merge

- MSVC build: an unmet R-016 criterion that needs a maintainer decision (self-hosted Windows runner with CUDA, a CUDA install step, or relaxing the criterion and amending the plan).
- The `kernels` CI job has not run in GitHub Actions; its script passes locally for all six SMs outside the container.
- The patch-set check has never run on a real patch (there are none yet).
- `toolkit.min_driver = "580"` is unverified; only driver 610.57 was tested.
- SM 80, 89, 90, 100 and 120 are built and registration-checked only; none ran on hardware.

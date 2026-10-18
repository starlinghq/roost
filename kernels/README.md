# Kernel library (stream C)

CUDA C++ behind the C1 ABI (`kernels/include/roost_kernels.h`, frozen). One shared library is built per SM,
`libroost_kernels_sm<N>.so`, so the digest of every release asset is per SM and a missing SM is a missing file.
Each library exports `roost_abi_version`, `roost_last_error`, `roost_register` (the registration table),
`roost_context_create` and `roost_context_destroy`. Today the table holds one entry, the `residual-stub` variant of
`Residual`: a correct, unoptimised op that proves the build, registration and run path. R-017 and R-034 add the real ops.

## Files

| Path | Role |
| --- | --- |
| `manifest.toml` | Contract C5: SM list with a status per SM, upstream pins (full SHA, licence per path, patches), ops and variants, binary digests, Triton AOT closure. The CMake build reads its SM list and pins. |
| `CMakeLists.txt`, `cmake/RoostUpstreams.cmake` | The build. Upstreams are fetched at configure time with `git fetch --depth 1 origin <sha>`, refused unless HEAD is that SHA, then patched from `upstream/<name>.patches/` (each patch checked against the manifest's SHA-256). |
| `src/`, `tests/stub_check.cpp` | The registration table, the Residual stub, and the host program that loads a library by its exports and runs the stub. |
| `tools/manifest_cmake.py` | Manifest to CMake variables. |
| `tools/record_digests.py` | SHA-256 of the built libraries; `--write` records them in the manifest, `--check` compares. |
| `tools/check_provenance.py` | Fails a pin or patch-set change that does not change `PROVENANCE.md` and `THIRD-PARTY-NOTICES.md` (and any tree where the manifest and those files disagree). |
| `ci/build-all-sm.sh` | What the CI `kernels` job runs: every SM, built twice, byte-compared, registration table checked without a GPU. |
| `results/` | Raw text logs of GPU runs. |

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
- Triton AOT: the manifest pins the Python closure and the seven kernel configs from S3 (six in `pins.json`, the seventh from S3's capture log and report), and CMake fetches the
  flash-linear-attention source by SHA, but the AOT compile (which needs a visible CUDA device) is R-034's.
- FlashInfer, CUTLASS and vLLM are fetched and verified by SHA; no op compiles against them yet (R-017).

## Open before merge

- MSVC build: an unmet R-016 criterion that needs a maintainer decision (self-hosted Windows runner with CUDA, a CUDA install step, or relaxing the criterion and amending the plan).
- The `kernels` CI job has not run in GitHub Actions; its script passes locally for all six SMs outside the container.
- The patch-set check has never run on a real patch (there are none yet).
- `toolkit.min_driver = "580"` is unverified; only driver 610.57 was tested.
- SM 80, 89, 90, 100 and 120 are built and registration-checked only; none ran on hardware.

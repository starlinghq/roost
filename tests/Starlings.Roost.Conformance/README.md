# Kernel conformance suite (R-019, stream D)

Checks every variant a kernel library registers against `CpuOps` (R-018) and against goldens made by the upstream definition of each
op. The CPU half runs in CI with `dotnet test --solution Roost.slnx`; the GPU half is local, goes through `tools/gpu-lease`, and
runs under `compute-sanitizer`.

| Path | What |
| --- | --- |
| `Suite/` | The runner, with no CUDA and no xunit dependency: case catalogue (`Cases.cs`), backends (`CpuBackend`, `OracleBackend`), comparison and tolerance classes (`Tolerances.cs`), goldens reader (`Goldens.cs`), runner (`Runner.cs`), results writer (`ResultsWriter.cs`) |
| `Tests/` | The CPU half (xunit v3, in `Roost.slnx`) |
| `../Starlings.Roost.Conformance.Cuda/` | The GPU half (local only, not in `Roost.slnx`): compiles `Suite/` and adds `CudaBackend`, which loads `libroost_kernels_sm<N>.so` through its C1 exports and runs every registered variant |
| `op-class-tolerances.proposal.json` | The float tolerance classes, **proposed** (see below) |
| `goldens/` | The golden generator (`generate_goldens.py`, a pinned uv project) and `goldens.lock.json`, which pins the asset by SHA-256. The goldens file itself is a release asset and is never committed |
| `sanitize.py` | Runs every variant under `compute-sanitizer` memcheck, racecheck and synccheck, one `gpu-lease` lease per run |
| `conformance-results.schema.json` | The results document (proposed; see "Results format") |
| `results/` | Raw text results of the runs reported in this file |

## How a case is judged

A case is one op call: named input buffers, parameters and the output buffers to compare. Cases are generated from a base seed
(`--seed`, default 20261019): fixed shapes of the Qwen3.5-0.8B full-attention layer (config.json at
`2fc06364715b967f1860aea9cf38778875588b17`: hidden 1024, 8 query heads over 2 KV heads, head_dim 256, 64 rotary dimensions,
theta 1e7, MLP 3584) plus randomized shapes and values (12 per op and dtype in the `full` profile, 4 in `ci`, 3 in `sanitizer`).
Every result records the case's seed, so a failure reproduces from its id and the run's base seed.

For each registered variant the runner runs every case inside the variant's registered constraints (a case outside them is
reported `not_run`, never launched) and compares the output with:

- `CpuOps` in double, rounded once to the element type (random and fixed cases), or
- the upstream golden for golden cases.

Integer and copy ops (Embedding, CacheWrite, CacheCopy) are compared byte for byte (class `exact`); the float classes are
`|got - ref| <= atol_rel * scale + rtol * |ref|`, `scale` being the reference's RMS (the largest `|V|` of the pool for attention).
A NaN or infinity in a candidate is never a pass. Beyond the comparison, per variant:

- **Determinism**: a variant registered `DETERMINISTIC` runs each case twice and must return identical bytes.
- **Batch invariance**: a variant registered `BATCH_INVARIANT` must return, for the first, middle and last row of a batch, the same
  bytes as the same row called alone (Gemm and LmHead with the plan bucket of the batch, as S2 found necessary).
- **ABI**: a call with a `struct_size` 8 bytes short must return `ABI_MISMATCH` before any launch.
- **Plan**: a `NEEDS_PLAN` variant run before `plan()` must return `NOT_PLANNED`.
- **Padding**: elements between strided rows (RmsNorm, QkNorm, LmHead) must be untouched.

Refusal paths that would have to launch something unsafe are not exercised here (R-017's harness has the typed-refusal checks).
The suite never injects a fault: it launches nothing outside a variant's registered constraints and passes only valid pointers.

## The CPU half (CI)

`dotnet test --solution Roost.slnx` runs the following (the timing was not measured in Release; in Debug the Conformance assembly took about 2.5 minutes in the review):

- the full `ci` profile with `OracleBackend` as the candidate. The oracle is a second implementation of the M0 ops written from the
  C3 definitions with no code shared with `CpuOps`; agreement checks the reference every GPU result is judged by;
- mutation tests: `CpuOps` damaged in one way at a time (a flipped bit in an exact op, a NaN, an error of four ulp, a determinism
  or batch-invariance break, a refusal, a failing backend check, an exception) must be reported as the matching failure, and an
  error inside the class must not;
- tolerance, catalogue (seeds reproduce, ids unique), results-writer and goldens-reader tests, including a synthetic goldens file;
- `RealGoldensMatchTheLockAndCpuOpsAgreesWithEveryUpstreamCase`: **skipped** unless `ROOST_CONFORMANCE_GOLDENS` names the goldens
  asset. A golden case that was not run is reported `not_run` and never counts as a pass.

The goldens asset is not in CI yet: fetching a release asset needs a step in `.github/workflows/ci.yml`, which is stream A's.

```sh
# CPU half with the goldens, full profile, and the stage-results files:
dotnet build tests/Starlings.Roost.Conformance.Cuda -c Release -o <out dir>
dotnet <out dir>/Starlings.Roost.Conformance.Cuda.dll --cpu --profile full --goldens <goldens file> --results <dir>
```

## Goldens

Upstream goldens are made by `goldens/generate_goldens.py`, pinned to transformers 5.20.1 and torch 2.9.1 (the versions the
evidence pipeline pins for its reference runner) and to numpy and safetensors by `uv.lock`. Each golden is the result of the
upstream definition of the op on seeded inputs, in float64:

| Op | Upstream definition used |
| --- | --- |
| RmsNorm, QkNorm | `Qwen3_5RMSNorm._norm` times the weight (the `1 + weight` convention is model definition, R-035) |
| Rope | `apply_rotary_pos_emb` and `rotate_half`, cos and sin formed in float64 (default theta 1e7, 64 of 256 dimensions) |
| PagedAttention | `eager_attention_forward` and `repeat_kv` (its softmax is float32, as upstream) |
| SiluAndMul | `ACT2FN["silu"]` on the gate half times the up half |
| Gemm, LmHead, Embedding | `torch.nn.functional.linear`, `embedding` |
| CacheWrite, CacheCopy, Residual | torch indexing and `add` |

57 cases. Llama 3 scaling and interleaved Rope, GeluAndMul, Argmax and Sample have no golden (the oracle covers the first two). The
file is deterministic (tensors are written sorted by name); the committed lock holds its SHA-256 (`99b07260…`), size, case count and
the SHA-256 of the generator.

```sh
export UV_PROJECT_ENVIRONMENT=<venv outside the repository>
uv run --project tests/Starlings.Roost.Conformance/goldens --locked roost-conformance-goldens --out <dir> --lock tests/Starlings.Roost.Conformance/goldens/goldens.lock.json
uv run --project tests/Starlings.Roost.Conformance/goldens --locked roost-conformance-goldens --verify <file> --lock tests/Starlings.Roost.Conformance/goldens/goldens.lock.json
```

Uploading the 122 MB file as a release asset, and the tag in the lock (`release.tag` is null), are not done: a release is not mine
to make. `GoldenSet.Open` refuses a file whose digest differs from the lock.

## The GPU half

```sh
cmake -S kernels -B <build> -G Ninja -DROOST_SM_LIST=86 -DROOST_UPSTREAM_DIR=<cache dir>   # kernels/README.md
dotnet build tests/Starlings.Roost.Conformance.Cuda -c Release -o <out dir>
tools/gpu-lease run --device 3090 --need-mib 8000 --stream D --wait 3600 -- \
  dotnet <out dir>/Starlings.Roost.Conformance.Cuda.dll <build>/libroost_kernels_sm86.so \
  --profile full --goldens <goldens file> --results <file>.json --commit <sha>
python3 -I tests/Starlings.Roost.Conformance/sanitize.py --runner <out dir> --library <build>/libroost_kernels_sm86.so \
  --out <log dir> --goldens <goldens file> --commit <sha>
```

Options: `--variants a,b`, `--ops Gemm`, `--case <text>`, `--seed N`, `--tolerances <file>`, `--no-determinism`, `--no-invariance`,
`--list-variants`. Exit code 0 only when every case passed. A variant added to a library is picked up from its registration table;
an op without cases is reported `not_run`.

## Tolerances (proposed)

`op-class-tolerances.proposal.json` holds the classes R-017 proposed (`kernels/README.md`), unchanged. It is **not** in
`evidence/protected/` and nothing here edits a protected file; a maintainer who approves classes puts them there and passes the
file with `--tolerances`. Results record the SHA-256 and `status` of the file used. Largest utilisation (1.0 is the limit) in the
R-019 SM 86 run, over passing and failing cases alike, `full` profile with goldens:

| Class | Cases | Largest utilisation | Largest abs error | Violations |
| --- | --- | --- | --- | --- |
| exact | 87 | 0 | 0 | 0 |
| elementwise-bf16 | 103 | 0.49 | 3.1e-2 | 0 |
| elementwise-f16 | 103 | 0.49 | 1.6e-2 | 0 |
| elementwise-f32 | 17 | 0 | 0 | 0 |
| gemm-f16 | 51 | 0.47 | 3.9e-3 | 0 |
| gemm-bf16 | 51 | **1.95** | 3.1e-2 | **44** (5 cases, below) |
| lm-head-f32 | 19 | 0.03 | 1.6e-5 | 0 |
| attention-bf16 | 38 | 0.47 | 1.6e-2 | 0 |

No class was widened to make a case pass.

## Results format

`ResultsWriter` writes the evaluator's stage-results envelope (`format: roost.stage-results`, `format_version: 3`, `profile`, `stage`,
`candidate`, `producer`, `environment`, `assets`, `cases`, `sections`) with `task: kernel-conformance` and stage `cpu` or `cuda`,
one case per op, variant and case id, and measurements only (`max_abs`, `max_utilisation`, `violations`, `nonfinite`, `ref_rms`,
`count`, the class name; no key the evaluator rejects as a tolerance). `conformance-results.schema.json` describes it and the
documents of the runs validate against it (checked with jsonschema 4.25.1).

**It does not validate against `evidence/schemas/stage-results.schema.json`**, which allows `task: generation`, stages `template` and
`dotnet`, and the sections of that task. This needs a stream B2 change (a `kernel-conformance` task, stages `cpu` and `cuda`, the
case and section shapes of the schema here); it is reported as a contract-change request and the proposed schema is the starting
point. The sanitizer driver writes the same envelope (`sanitizer.results.json`) with one case per variant and tool.

## Results of the SM 86 run (`results/`)

Library: `libroost_kernels_sm86.so`, sha256 `0f707ae0…` (the digest in `kernels/manifest.toml`; built from the R-017 checkout at
`5846bcd`, `kernels/README.md` explains why the digest depends on the build path). RTX 3090, driver 610.57, CUDA 13.3, `gpu-lease`
leases on device 3090. Suite commit `b18d0da`; base seed 20261019. The run documents record producer commit `b18d0da`, not the
branch head: the later commits changed only documentation, line wrapping in `sanitize.py`, and `generate_goldens.py` with its lock hash (the committed
goldens bytes still match the lock), with no change to the suite C# code. The GPU runs were not repeated after those commits.

- **Conformance, `full` profile, 464 passed and 5 failed** of 469 (240 random, 172 fixed, 57 golden; 0 not run). Determinism held for
  all 469 and batch invariance for the 275 cases that have a single-row form. Registered variants: 11, all with an `ABI_MISMATCH` pass.
- **Failures** (`results/r019-cuda-sm86.results.json`): five `gemm-cublaslt` cases, all bf16, all with `beta != 0` (a C input) and a
  bias together: `fixed-epilogue-BF16-39x256x200-tb1-bias1-ab1.1-bucket49` (utilisation 1.09, 3 of 9984 elements),
  `fixed-epilogue-…-ab0.5.1-bucket49` (1.10, 1), `random-2-BF16-39x856x832-…-bucket49` (1.95, 23 of 33384),
  `random-6-BF16-53x856x1152-…-bucket0` (1.48, 16 of 45368), `random-10-BF16-27x288x544-tb0-…-bucket0` (1.31, 1 of 7776). The f16
  versions of the fixed cases and the fixed bf16 cases with alpha, bias or C alone pass. The same inputs, `fixed-epilogue-BF16-39x256x200-tb1-bias1-ab1.1`, pass at bucket 0 and fail at bucket 49 (two random cases of other shapes fail at bucket 0), which points at the
  algorithm the plan pins for a shape (`kernels/src/gemm.cu`) more than at the epilogue formula. The error is 1 to 4 bf16 ulp of the
  result with partial sums of several units, which fits a rounding to the 16-bit output type between the C and bias parts of the
  epilogue; this is a hypothesis, not confirmed, and no log supports it. Reproduce: `--ops Gemm --case fixed-epilogue`. The kernel and the tolerance are not mine to
  change: the options are a kernel fix, or a maintainer-approved class for that epilogue.
- **compute-sanitizer** (`results/sanitizer/`, `sanitize.py`, profile `sanitizer`, every one of the 11 variants under memcheck with
  `--leak-check full`, racecheck and synccheck: 33 runs): **32 runs report zero** (memcheck 0 errors and 0 bytes leaked, synccheck 0
  errors, racecheck 0 hazards for 10 variants). **One does not: `paged-attention-flashinfer-fa2` under racecheck** reports
  hazards in `flashinfer::BatchPrefillWithPagedKVCacheKernel` (bf16, head_dim 256, 4 query heads per KV head, CTA tile 16): 23 errors
  and 370 warnings in the committed run (101 displayed); the count was not compared across runs, so whether it varies is unknown. All prefill-mode
  cases show it; decode-mode cases show none. Smallest reproducer: `--case prefill-one-token` (16 errors, 4 warnings). It is upstream
  FlashInfer code at the manifest's pinned SHA; whether it is a true race or a racecheck limitation with the kernel's `cp.async` and
  warp-synchronous code is not established. The same variant is clean under memcheck and synccheck, and its outputs match `CpuOps`.
- **CPU half** (`results/r019-cpu-oracle.results.json`, `r019-cpu-cpuops-goldens.results.json`, `r019-cpu.out`, `full` profile): the
  independent oracle against `CpuOps` passes 469 of 469 cases (largest utilisation 0.49, the elementwise classes), and `CpuOps`
  against the 57 upstream goldens passes 57 of 57 (largest utilisation 0.43, f16 elementwise). The goldens file digest is
  `99b07260…`, as in the lock.

## Not done

- Open decisions and requests (deferred, not guessed):
  - `paged-attention-flashinfer-fa2` is not racecheck-clean, so the criterion "zero reports" is not met. It needs a maintainer decision:
    a kernel or patch fix (stream D), a documented racecheck limitation, or an approved exception.
  - The five `gemm-cublaslt` bf16 failures (beta != 0 with bias) need a kernel fix (stream D) or a maintainer decision on the `gemm-bf16`
    class. Conformance is not green on SM 86 until then.
  - `evidence/schemas/stage-results.schema.json` does not accept `task: kernel-conformance`; stream B2 must extend it.
  - The maintainer must approve or amend `op-class-tolerances.proposal.json` and place it in `evidence/protected/`.
  - A filtered rerun (`--ops Gemm --case fixed-epilogue`) printed "1 not run" in the review; the full committed run has 0. Not investigated.

- The goldens file is not uploaded as a release asset (no release is made here) and CI does not fetch it; the real-goldens test is skipped there.
- SM 89 has not run (the RTX 4060 Ti had 15.7 of 16.4 GiB in use during the work); only SM 86 has hardware evidence. Other SMs stay `untested`.
- The recurrent ops (GatedDeltaRule, CausalConv1d, state copy and snapshot), Argmax, Sample, GeluAndMul and the fused ops have no GPU variant in this library,
  so they have no cases; `CaseCatalog` and the generator take a new op as one generator method each when R-034 registers its variants.
- No `graph` case: R-017 recorded a signal death in its graph test harness and has not rerun it, and capturing a graph is not a conformance property here.
- The suite does not run real Qwen3.5-0.8B weights (R-017's harness does); it uses seeded random data of the model's shapes.

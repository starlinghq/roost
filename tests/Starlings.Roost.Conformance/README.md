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
theta 1e7, MLP 3584; for the hybrid ops of R-034 the linear-attention layer: 16 key and value heads of 128, the conv1d over the
6144-wide fused q, k, v row with width 4, the 8 x 256 attention output gate) plus randomized shapes and values (12 per op and dtype in
the `full` profile, 4 in `ci`, 3 in `sanitizer`). The hybrid fixed shapes follow the R-034 harness (`kernels/tests/hybrid-conformance`).
Every result records the case's seed, so a failure reproduces from its id and the run's base seed.

For each registered variant the runner runs every case inside the variant's registered constraints (a case outside them is
reported `not_run`, never launched) and compares the output with:

- `CpuOps` in double, rounded once to the element type (random and fixed cases), or
- the upstream golden for golden cases.

Integer and copy ops (Embedding, CacheWrite, CacheCopy, RecurrentStateCopy, RecurrentStateSnapshot, and CausalConv1d's state) are
compared byte for byte (class `exact`); the float classes are `|got - ref| <= atol_rel * scale + rtol * |ref|`, `scale` being the
reference's RMS (the largest `|V|` of the pool for attention). A case may name a class per output (`OpCase.OutputClasses`): the state
outputs, the float32 outputs beside bf16 activations, and the chunked GatedDeltaRule, whose outputs have their own classes
(`output@variant`). Each output's class is recorded in the results.
A NaN or infinity in a candidate is never a pass. Beyond the comparison, per variant:

- **Determinism**: a variant registered `DETERMINISTIC` runs each case twice and must return identical bytes.
- **Batch invariance**: a variant registered `BATCH_INVARIANT` must return, for the first, middle and last row of a batch, the same
  bytes as the same row called alone (Gemm and LmHead with the plan bucket of the batch, as S2 found necessary). For the sequence ops
  (GatedDeltaRule, CausalConv1d) the unit is a sequence: run alone over the same state pool, its token rows and its state slot must
  equal those of the batched call, byte for byte.
- **ABI**: a call with a `struct_size` 8 bytes short must return `ABI_MISMATCH` before any launch.
- **Plan**: a `NEEDS_PLAN` variant run before `plan()` must return `NOT_PLANNED`.
- **Padding**: elements between strided rows (RmsNorm, QkNorm, LmHead) must be untouched.
- **Registered refusals** (`Suite/Refusals.cs`): a combination a variant is registered to refuse (today `gemm-cublaslt` with a C input, `beta != 0`, or a bias, in any dtype, R-019)
  must return the rule's typed status (`UNSUPPORTED`) from `plan()`; the outcome is `refused`, which is neither a pass nor a failure, is counted separately,
  and keeps the run green. Serving the combination, another status, or a refusal no rule predicts is a failure.
- **Prefill batch composition** (PagedAttention, prefill, more than one sequence): the first, middle and last sequence are run alone over the same K/V
  pool and compared, byte for byte, with the rows they have in the batch. This gates the case only for a variant registered `BATCH_INVARIANT`; for any other
  variant (the FA2 variant is not: decode split-KV depends on the batch) a mismatch is recorded as a failed check on a passing case and counted in the
  summary (`recorded_check_failures_in_passing_cases`), and a refusal of the one-sequence call still fails the case. It was written as the evidence a
  racecheck exception for the FA2 prefill kernel would need; it was gating in the first committed run (below) and was made record-only after it.

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

`op-class-tolerances.proposal.json` holds the classes R-017 proposed (`kernels/README.md`), unchanged, and five new classes for the hybrid
ops (`hybrid-f32`, `gdn-recurrent-bf16`, `gdn-recurrent-state-f32`, `gdn-chunk-bf16`, `gdn-chunk-state-f32`). It is **not** in
`evidence/protected/` and nothing here edits a protected file; a maintainer who approves classes puts them there and passes the file with
`--tolerances`. Results record the SHA-256 and `status` (`proposal`) of the file used.

The new classes were sized before the committed runs, from one preliminary run of the hybrid cases on SM 86 and SM 89 (the same maxima on
both): each is a pair of powers of two with the worst element at or below half the limit. The bf16 outputs of CausalConv1d,
GatedDeltaRuleGates (beta), GatedRmsNorm and SigmoidMul use the existing `elementwise-bf16` class (computed in float, rounded once);
their float32 outputs use `hybrid-f32`. The chunked GatedDeltaRule classes are loose (`rtol` 2^-4 and `atol_rel` 2^-3 of the RMS for `o`,
2^-5 and 2^-4 for the state) because FLA's chunk path rounds its intermediates to bf16 (S3); in the preliminary run the R-017-sized
values (2^-6 and 2^-5) left `o` at utilisation 1.29 and the state at 1.98. No existing class was widened. Largest utilisation (1.0 is the
limit) in the committed runs, `full` profile with goldens, over every compared output; SM 86 and SM 89 give the same figures except
`gemm-bf16` (0.448 on SM 86, 0.460 on SM 89):

| Class | Cases | Largest utilisation | Largest abs error | Violations |
| --- | --- | --- | --- | --- |
| exact | 139 | 0 | 0 | 0 |
| elementwise-bf16 | 147 | 0.49 | 3.1e-2 | 0 |
| elementwise-f16 | 103 | 0.49 | 1.6e-2 | 0 |
| elementwise-f32 | 17 | 0 | 0 | 0 |
| hybrid-f32 | 36 | 0.23 | 6.1e-5 | 0 |
| gemm-f16 | 32 | 0.47 | 3.9e-3 | 0 |
| gemm-bf16 | 31 | 0.45 / 0.46 | 1.6e-2 | 0 |
| lm-head-f32 | 19 | 0.03 | 1.6e-5 | 0 |
| attention-bf16 | 38 | 0.47 | 1.6e-2 | 0 |
| gdn-recurrent-bf16 | 23 | 0.45 | 2.4e-4 | 0 |
| gdn-recurrent-state-f32 | 23 | 0.07 | 3.0e-7 | 0 |
| gdn-chunk-bf16 | 23 | 0.32 | 9.8e-4 | 0 |
| gdn-chunk-state-f32 | 23 | 0.49 | 6.7e-3 | 0 |

## Results format

`ResultsWriter` writes the evaluator's stage-results envelope (`format: roost.stage-results`, `format_version: 3`, `profile`, `stage`,
`candidate`, `producer`, `environment`, `assets`, `cases`, `sections`) with `task: kernel-conformance` and stage `cpu` or `cuda`,
one case per op, variant and case id, and measurements only (`max_abs`, `max_utilisation`, `violations`, `nonfinite`, `ref_rms`,
`count`, the class name of the case and of each output; no key the evaluator rejects as a tolerance). `conformance-results.schema.json` describes it and the
documents of the runs validate against it (checked with jsonschema 4.25.1).

**It does not validate against `evidence/schemas/stage-results.schema.json`**, which allows `task: generation`, stages `template` and
`dotnet`, and the sections of that task. This needs a stream B2 change (a `kernel-conformance` task, stages `cpu` and `cuda`, the
case and section shapes of the schema here); it is reported as a contract-change request and the proposed schema is the starting
point. The sanitizer driver writes the same envelope (`sanitizer.results.json`) with one case per variant and tool.

## Results of the SM 86 and SM 89 runs (`results/`)

Library: `libroost_kernels_sm86.so` sha256 `d84e95be…` and `libroost_kernels_sm89.so` sha256 `eba3d599…`, the digests in
`kernels/manifest.toml` at the run commit. Both were built from this branch with the Triton AOT GatedDeltaRule cubins
(`ROOST_SM_LIST=86;89`, CUDA 13.3.73, gcc 16.2.1) at the worktree path the manifest names; `kernels/README.md` explains why the digest
depends on that path. RTX 3090 (SM 86) and RTX 4060 Ti (SM 89), driver 610.57, `gpu-lease` leases on devices 3090 and 4060. **Run commit
`7b2b2cf`** (the branch head when the results were produced; the later commits change only results, this README and `kernels/README.md`). Base seed 20261019. Goldens file `99b07260…`, as in the lock.

- **Conformance, `full` profile with goldens: 578 passed, 0 failed, 0 errors, 0 not run, 43 refused as registered, on each SM**
  (`r019-cuda-sm86.*`, `r019-cuda-sm89.*`; 621 case results: 228 fixed, 336 random, 57 golden). All 19 registered variants ran and passed
  the `ABI_MISMATCH` probe: the 11 M0 variants and the 8 hybrid ones (`gated-delta-rule-chunk-aot`, `gated-delta-rule-recurrent-aot`,
  `causal-conv1d-cuda`, `recurrent-state-copy-cuda`, `recurrent-state-snapshot-cuda`, `gated-delta-rule-gates-cuda`, `gated-rms-norm-cuda`,
  `sigmoid-mul-cuda`). Determinism held for all 578 compared cases and batch invariance for the 325 that have a single-row or
  single-sequence form. The integer and copy ops, the state copy and snapshot and the conv state are bit-exact.
- **Refused** (43 per SM, 22 bf16 and 21 f16): every `gemm-cublaslt` case with a C input or a bias, including 4 golden cases
  (`Gemm-<dtype>-bias` and `Gemm-<dtype>-alpha-beta-c-bias-bucket64`), which are now refused rather than compared. Alpha-only cases are
  served and pass. `kernels/results/r019-ops-gemm-sm86.log` is the stream C harness on the same SM 86 library (sha256 `d84e95be…`; the
  harness as committed in `8eaa795`, unchanged through `7b2b2cf`; the log's first lines record both): 81 passed, 0 failed,
  including the six refusal checks (bf16 and f16; C, bias, both: `UNSUPPORTED` from `plan()` and `run()`, D untouched).
- **Prefill batch composition (recorded)**: 7 multi-sequence prefill cases per SM (6 random and the golden `prefill-multi-seq`, plus the
  fixed `prefill-multi-seq`) return, for at least one sequence run alone, rows that differ in bytes from the batched call, while both are
  inside `attention-bf16` (utilisation at most 0.46) and each is deterministic. The FA2 batch prefill scheduler partitions the work by batch,
  so a sequence's rounding depends on who else is in the batch. Single-sequence prefill and decode are unaffected. **First committed run:**
  at `13c78e1` the check was gating and these 7 cases failed on both SMs (571 passed, 7 failed, 43 refused;
  `r019-cuda-sm86-first-run-13c78e1.out`, `r019-cuda-sm89-first-run-13c78e1.out`). Because the variant claims no batch invariance, the
  check was made record-only at `7b2b2cf` and both SMs were rerun; the result is itself evidence that FA2 prefill is not proven benign (below).
- **compute-sanitizer** (`results/sanitizer-sm86/`, `results/sanitizer-sm89/`, `sanitize.py`, profile `sanitizer`, each of the 19 variants
  under memcheck with `--leak-check full`, racecheck and synccheck: 57 runs per SM): **56 runs per SM report zero** (memcheck 0 errors and
  0 bytes leaked, synccheck 0 errors, racecheck 0 hazards), including the Triton AOT GatedDeltaRule cubins and every other hybrid variant.
  **One does not: `paged-attention-flashinfer-fa2` under racecheck**, in `flashinfer::BatchPrefillWithPagedKVCacheKernel` as before. The
  gate number is racecheck's own summary, errors plus warnings: **SM 86 48 errors and 412 warnings (460; 100 hazards displayed), SM 89 48
  errors and 413 warnings (461; 101 displayed)**. The first R-019 run on the old library (`0f707ae0…`, 11 variants, fewer prefill
  cases in the profile) had 23 errors and 370 warnings. All prefill-mode cases show hazards; decode-mode cases show none. The variant is
  clean under memcheck and synccheck, and its outputs match `CpuOps`.
- **CPU half** (`results/r019-cpu-oracle.results.json`, `r019-cpu-cpuops-goldens.results.json`, `r019-cpu.out`, `full` profile, at `7b2b2cf`):
  the independent oracle against `CpuOps` passes 473 of 473 cases of the M0 ops (largest utilisation 0.49, `elementwise-f16`), and `CpuOps` against the 57 upstream goldens passes 57 of 57 (largest utilisation 0.43, `elementwise-f16`). The oracle covers the M0 ops only; the hybrid ops are checked against `CpuOps` alone, which R-018 and R-022 tested. No upstream golden exists for them.

## R-019 decisions (maintainer ruling of 2026-10-20: decide each finding from data)

- **Gemm epilogues: refuse what no model uses.** The ruling is "refuse unused GEMM epilogues at plan time". No generated forward passes C,
  Beta or Bias to a Gemm node (`RefusalTests.NoGeneratedModelForwardUsesGemmWithBetaOrBias`; every node passes Alpha), so `gemm-cublaslt`
  refuses every epilogue with a C input or a bias, in bf16 and f16, with `UNSUPPORTED` from `plan()` and `run()` before any launch
  (`kernels/src/gemm.cu`, `ValidateGemm`), and the suite's rule `RefusalRules.GemmEpilogueWithCOrBias` expects it. Alpha stays served.
  This also removes the five bf16 C-and-bias cases that were outside `gemm-bf16` in the first run, without widening the class.
- **`paged-attention-flashinfer-fa2` prefill is experimental.** The ruling is "FA2 prefill experimental unless proven benign", and it is not
  proven benign, on two independent grounds: (1) racecheck reports hazards on SM 86 and SM 89 (above), and the SASS reading below does not
  put them on a known false-positive pattern; (2) a sequence's prefill output is not byte-identical across batch compositions (above).
  The kernel manifest (C5) has no per-variant status and the C1 table no flag for one, so the marking is in `kernels/README.md` (the
  PagedAttention row) and here; a `status` field on a manifest variant (or a C1 flag) is a stream A request. R-025 uses the prefill path
  (`CudaOps.PagedAttention` selects prefill whenever a step has more tokens than sequences, every prompt step), so a profile built on it
  inherits the experimental status until the hazards are resolved. Still open: the `-lineinfo` racecheck run that maps the hazards to
  source lines, and then a patch set in `kernels/upstream/flashinfer.patches/` (with `PROVENANCE.md`) or a maintainer exception.
  What the machine code says (`cuobjdump -sass -arch sm_86` of the first R-019 SM 86 library, sha256 `0f707ae0…`, kernel
  `BatchPrefillWithPagedKVCacheKernel<MaskMode 1, CTA_TILE_Q 16, warps 1x4, head_dim 256>`): the read site in the racecheck log (`+0x5300`) is a plain `LDS.128` and the
  write sites (`+0x9670` to `+0x9bf0`, 32 hazards each) are plain 32-bit `STS`. They are not `LDGSTS` (cp.async) or `LDSM` (ldmatrix), the patterns the
  maintainer's rule names; all 80 `LDGSTS` in the function sit before `+0x4330`, behind `DEPBAR`/`BAR.SYNC`. Read and writes lie in one barrier interval (the single
  `BAR.SYNC.DEFER_BLOCKING` at `+0x4d40` to the `BAR.SYNC` at `+0xaa20`). Reading `include/flashinfer/attention/prefill.cuh` at the manifest's SHA, the merge of the four
  KV warps (`threadblock_sync_mdo_states`, `cta_sync_o_smem` reused across the head-dim 256 VO tiles) is the code that writes and reads that region, with `__syncthreads()`
  after the loop and at the end of each tile, which does not by itself explain a hazard. The SASS was not re-read for the new library (the FlashInfer sources and pin are
  unchanged; only `gemm.cu` changed).
- **Hybrid ops**: every variant the current library registers has cases and ran on both SMs (above); none needed a refusal or an exception.

## Not done

- Open decisions and requests (deferred, not guessed):
  - The maintainer must approve or amend `op-class-tolerances.proposal.json` (including the five new classes and the per-output class
    mechanism) and place it in `evidence/protected/`.
  - A variant-level capability status in the kernel manifest (or a C1 flag), so that "experimental" for the FA2 prefill path is machine
    readable (stream A).
  - `evidence/schemas/stage-results.schema.json` does not accept `task: kernel-conformance`; stream B2 must extend it.
  - The `-lineinfo` racecheck attribution of the FA2 prefill hazards.
  - All six manifest digests are the R-019 build (SM 80, 90, 100 and 120 were rebuilt build-only from the same source and path after the
    runs, so the manifest names one build). That path is a worktree that will not exist once this branch is merged, so the digests
    reproduce only by rebuilding there; the release pipeline (R-060) replaces them.
- The goldens file is not uploaded as a release asset (no release is made here) and CI does not fetch it; the real-goldens test is skipped there.
- Only SM 86 and SM 89 have hardware evidence; other SMs stay `untested`. The manifest still lists SM 89 `untested`: setting it `experimental` with this evidence is the maintainer's (or stream C's) call, and `ManifestAndNamesTests` (stream F) asserts the current value.
- Argmax, Sample, GeluAndMul and the fused ops have no GPU variant in this library, so they have no cases.
- No `graph` case: capturing a graph is not a conformance property here; the R-034 harness replays the graph-capturable hybrid ops once.
- The suite does not run real Qwen3.5-0.8B weights (R-017's harness does); it uses seeded random data of the model's shapes.
- CI timing of the Conformance assembly in Release: the CPU half took 67 to 135 s locally in Release (`dotnet test` reported 77 s for 51
  tests and 67 s for 52), not measured on the CI runners.

# Roost evidence

Roost's own evidence pipeline (build plan section 2.2). It never ships in the runtime or the image, and Python here is test- and tool-time only.

| Path | Holds | Stream |
| --- | --- | --- |
| `pipeline/` | The `roost-evidence` Python package: manifest validation (R-012), the generation evaluator and the fp32 CPU reference runner (R-010); per-layer capture and evaluation (R-011); later calibration | B2, B3 |
| `schemas/` | `profile-manifest.schema.json` (B3) and `stage-results.schema.json` (B2) | B2, B3 |
| `registry/` | Profile manifests, admissions and asset locks | B3 |
| `protected/` | Fixtures and tolerances. Maintainer-only: agents propose changes and never edit it. The maintainer creates this directory | maintainer |

## Validate a profile manifest

```sh
uv run --project evidence/pipeline --locked roost-evidence validate-manifest evidence/registry --registry
```

Exit 0 when every manifest is valid; otherwise one line per problem, `path: location: message`. `--registry` also rejects `synthetic: true` manifests, which exist only as test fixtures. Manifests are `.json`, `.yaml` or `.yml`; duplicate keys are errors. Quote `min_driver` and `min_toolkit` in YAML (`13.0` unquoted is a float).

## Evaluate a generation run

Stage results format v3, task `generation`, is `schemas/stage-results.schema.json`: a run directory holds `template.results.json` (rendering and prompt ids), `dotnet.results.json` (generation, streaming, usage, and the `cached_vs_uncached` and `concurrency` sections) and an optional `step0.f32` sidecar of whole step-0 log-softmax vectors. Version 3 is the number Starling's published p016 section 10 gives the format; Roost has no versions 1 or 2.

```sh
uv run --project evidence/pipeline --locked roost-evidence evaluate <results-dir> \
    --reference <reference.jsonl> --tolerances <tolerances file> [--reference-vectors <dir>] [--report report.json]
```

Exit 0 when every check passed; 1 when any check failed or was not run; 2 when the inputs cannot be evaluated. Each check reports `passed`, `failed`, `not_run` (no tolerance key, or inputs absent: no sidecar, no reference vectors) or `not_applicable` (the tolerance file says so). `not_run` never counts as a pass. The report lists every greedy divergence with its step and the reference margin, allowed or not. Tolerances are an input: the evaluator never reads or writes `protected/`, and results may not carry tolerance keys. The tolerance keys are in `evaluate_generation.TOLERANCE_KEYS`.

The reference is a `reference.jsonl` (format in `pipeline/src/roost_evidence/generation_reference.py`) that the reference runner writes:

```sh
UV_PROJECT_ENVIRONMENT=<venv outside the repository> \
uv run --project evidence/pipeline --locked --extra reference roost-evidence reference-run \
    <model directory, or hub id with --revision <40-character SHA>> prompts.jsonl --out <run artifacts directory>
```

The runner is model-agnostic: transformers `AutoModelForCausalLM`, float32, CPU, greedy with a KV cache. Its output (reference rows and step-0 vectors) is a run artifact and stays out of Git. The `reference` extra (torch from the CPU wheel index, transformers) is tool-time only; the default gate does not install it.

## Capture and compare per-layer hidden states

Per-layer stage (R-011). The reference runner records, for every case, forward-hook captures of the token embedding, every decoder layer and the final norm at every position of every forward (the prefill, then each generated token that is fed back), and, for hybrid models, the recurrent and convolution state of every Gated DeltaNet layer. That state is not in what a forward hook returns: it lives in the model's cache object, so the runner reads it from the cache after each forward and copies it before the next forward updates it in place.

```sh
UV_PROJECT_ENVIRONMENT=<venv outside the repository> \
uv run --project evidence/pipeline --locked --extra reference roost-evidence reference-run \
    <model directory, or hub id with --revision <40-character SHA>> prompts.jsonl --out <run artifacts directory> \
    --capture-layers [--state-forwards 2]
```

This adds `layers/<id>.safetensors` per case to the run artifacts (outside Git) and `layers_sha256` to each `reference.jsonl` line. The file layout, tensor names and metadata are documented in `pipeline/src/roost_evidence/layer_captures.py`; the evaluator reads them with the standard library, so the default gate needs no torch. Hidden states are captured for every forward. States are large (one Qwen3.5-0.8B layer holds 262144 recurrent floats, 18 such layers per forward), so they are kept for the first `--state-forwards` forwards only (default 2: the prefill and the first decode).

transformers is pinned to 5.20.1: it is the version whose `qwen3_5` implementation runs on the CPU in float32 with its PyTorch fallbacks (no `flash-linear-attention` or `causal-conv1d`), and 4.57.3 has no Qwen3.5 model. The runner hooks the module list named `layers`, so it works on the text-only class and on the checkpoint's multimodal wrapper alike.

A candidate (`CpuOps`, the GPU worker) must be teacher-forced on the reference's prompt ids and generated ids, and write the same layout under `<results-dir>/layers/<id>.safetensors`:

```sh
uv run --project evidence/pipeline --locked roost-evidence evaluate-layers <results-dir> \
    --reference <reference.jsonl> --reference-captures <reference run directory> --tolerances <tolerances file> [--report report.json]
```

The model revision of a real capture is whatever `--revision` the operator passes; this branch records it in `environment.json` but does not decide which revision is admitted (that is R-013's). The R-011 development run used the hub's then-current main, 2fc06364, as a spike only.

Exit codes and statuses are those of `evaluate`. Four checks, bound by the tolerance keys `float.layer_hidden_max_abs`, `float.recurrent_state_max_abs`, `float.conv_state_max_abs` and `float.nan_or_inf`. The report holds, per case and per stage, the maximum absolute difference, and `first_divergent`: the earliest forward in which a stage exceeds the hidden-state bound, the first stage in model order within it, and the first position. Recurrent and convolution state are attributed separately. A candidate capture for another token sequence, another layer count or other layer types is an input error (exit 2); a case with no candidate capture, or a candidate without state tensors, is `not_run`.

## Test the pipeline

```sh
uv run --project evidence/pipeline --locked pytest -q evidence/pipeline/tests
```

Tests that need torch (`test_reference_runner.py`, `test_layer_capture_runner.py`) skip unless the `reference` extra is installed: add `--extra reference` and point `UV_PROJECT_ENVIRONMENT` outside the repository to run them.

The root `ruff.toml` governs this directory; the package has no ruff settings of its own.

## Changing the profile manifest schema

The schema is a contract (build plan section 5). A change bumps `schema_version`, updates every manifest in `registry/` and the validator tests in the same PR, and goes through the contract-version rule.

## Changing the stage-results schema

It is a contract (build plan section 5, owned by stream B2). A change bumps `format_version` and updates the evaluator, its tests and every producer in the same PR.

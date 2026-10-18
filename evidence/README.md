# Roost evidence

Roost's own evidence pipeline (build plan section 2.2). It never ships in the runtime or the image, and Python here is test- and tool-time only.

| Path | Holds | Stream |
| --- | --- | --- |
| `pipeline/` | The `roost-evidence` Python package: manifest validation (R-012), the generation evaluator and the fp32 CPU reference runner (R-010); later per-layer capture and calibration (R-011) | B2, B3 |
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

## Test the pipeline

```sh
uv run --project evidence/pipeline --locked pytest -q evidence/pipeline/tests
```

Tests that need torch (`test_reference_runner.py`) skip unless the `reference` extra is installed: add `--extra reference` and point `UV_PROJECT_ENVIRONMENT` outside the repository to run them.

The root `ruff.toml` governs this directory; the package has no ruff settings of its own.

## Changing the profile manifest schema

The schema is a contract (build plan section 5). A change bumps `schema_version`, updates every manifest in `registry/` and the validator tests in the same PR, and goes through the contract-version rule.

## Changing the stage-results schema

It is a contract (build plan section 5, owned by stream B2). A change bumps `format_version` and updates the evaluator, its tests and every producer in the same PR.

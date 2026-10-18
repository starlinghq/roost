# Roost evidence

Roost's own evidence pipeline (build plan section 2.2). It never ships in the runtime or the image, and Python here is test- and tool-time only.

| Path | Holds | Stream |
| --- | --- | --- |
| `pipeline/` | The `roost-evidence` Python package: manifest validation (R-012), later the evaluator, reference runners and calibration (R-010, R-011) | B2, B3 |
| `schemas/` | `profile-manifest.schema.json` (B3) and, from R-010, `stage-results.schema.json` (B2) | B2, B3 |
| `registry/` | Profile manifests, admissions and asset locks | B3 |
| `protected/` | Fixtures and tolerances. Maintainer-only: agents propose changes and never edit it. The maintainer creates this directory | maintainer |

## Validate a profile manifest

```sh
uv run --project evidence/pipeline --locked roost-evidence validate-manifest evidence/registry --registry
```

Exit 0 when every manifest is valid; otherwise one line per problem, `path: location: message`. `--registry` also rejects `synthetic: true` manifests, which exist only as test fixtures. Manifests are `.json`, `.yaml` or `.yml`; duplicate keys are errors. Quote `min_driver` and `min_toolkit` in YAML (`13.0` unquoted is a float).

## Test the pipeline

```sh
uv run --project evidence/pipeline --locked pytest -q evidence/pipeline/tests
```

The root `ruff.toml` governs this directory; the package has no ruff settings of its own.

## Changing the profile manifest schema

The schema is a contract (build plan section 5). A change bumps `schema_version`, updates every manifest in `registry/` and the validator tests in the same PR, and goes through the contract-version rule.

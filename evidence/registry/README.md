# Registry

Stream B3 puts here the profile manifests (`evidence/schemas/profile-manifest.schema.json`), admission records and asset locks (R-013). Manifests are validated with:

```sh
uv run --project evidence/pipeline --locked roost-evidence validate-manifest evidence/registry --registry
```

Weights and other model assets are never committed; a lock records full SHAs, sizes and SHA-256 digests. Admissions and locks are `.toml` (the manifest validator reads `.json`, `.yaml` and `.yml` here as profile manifests).

| Model | Admission | Lock | State |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | `qwen3.5-0.8b.admission.toml` | `qwen3.5-0.8b.lock.toml` | admitted at `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`; no profile manifest yet (the schema needs kernel-manifest, variant and per-SM fields that do not exist) |

Note: `validate-manifest evidence/registry --registry` currently prints "no manifest files found" and exits 0. It validates nothing here, so it is not evidence for the admission or lock. `tests/tools/test_registry_locks.py` guards the lock and admission files instead. The tokenizer-asset difference is recorded in the admission (`[comparison.tokenizer_json]`) and needs no report to Starling (`report_to_starling.status = "not needed"`): Starling's asset is what transformers itself loads for this checkpoint, so there is nothing to raise.

# Registry

Stream B3 puts here the profile manifests (`evidence/schemas/profile-manifest.schema.json`), admission records and asset locks (R-013). Manifests are validated with:

```sh
uv run --project evidence/pipeline --locked roost-evidence validate-manifest evidence/registry --registry
```

Weights and other model assets are never committed; a lock records full SHAs, sizes and SHA-256 digests. Nothing is admitted yet.

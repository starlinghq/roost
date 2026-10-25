# Proposals awaiting the maintainer

Proposed tolerances and fixtures that agents produce for `evidence/protected/`, which only the maintainer edits (build plan section 3).
Nothing here is binding and no gate reads this directory. Each proposal names its item, its profile and the evidence it comes from.

| File | Item | Profile | State |
| --- | --- | --- | --- |
| `r022-tolerances.toml`, `r022-tolerances.md` | R-022 | `qwen3.5-0.8b/cpu-fp64-reference` (CpuOps against transformers fp32) | proposed; the bf16 GPU profile is pending a GPU measurement |
| `r014-fixtures.toml`, `r014-fixtures.md` | R-014 | `qwen3.5-0.8b/cpu-fp64-reference` and `qwen3.5-0.8b/cuda-bf16-eager` | proposed; fixture data pinned by SHA-256, storage and the near-tie rule are maintainer decisions; GPU against the fixtures not yet measured |

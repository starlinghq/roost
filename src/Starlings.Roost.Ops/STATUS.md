# R-098 status: partial (CPU groundwork only)

R-098 (build plan section 10, streams C and D, M2): "YaRN, longrope and dynamic NTK in `RopeScaled` with CPU references;
long-context parity for one Qwen profile at its rated length." This change is stream D's CPU half. R-098 stays open.

## Delivered

- `CpuOps.RopeScaled` (`CpuOps.RopeScaled.cs`), against the v0 `RopeScaledArgs` unchanged: applies a precomputed
  `[MaxPosition, RotaryDim]` cache (R/2 cosines then R/2 sines per row, attention factor multiplied in) to Q and K in place, NeoX
  halves, in double, rounded once. Positions outside the cache, an unknown `ScalingKind`, a non-positive or non-finite `Factor`
  and a bad `RotaryDim` throw before anything is written. `CpuOps.Supports(OpId.RopeScaled)` is now true.
- `RopeScaling` (`RopeScaling.cs`): double-precision builders of the frequencies and cache for YaRN (with the mscale /
  mscale_all_dim attention factor, beta_fast/beta_slow, truncate), longrope (short and long factors, the sequence length picks
  them, attention factor from `factor` or max/original positions), dynamic NTK (base from the sequence length) and llama3,
  each the formula of transformers 5.20.1 `modeling_rope_utils.py` (sha256 6a0727073e080e163d0287e15fc1acb90035e9a9df9cb19a13b632a40abfac24).
  Llama3 was already in `CpuOps.Rope`; the builder gives the same frequencies and `RopeLlama3MatchesTransformers` now checks
  `CpuOps.Rope` against the reference too.
- References: `tests/Starlings.Roost.Ops.Tests/Reference/make_rope_scaling_cases.py` writes `rope-scaling-cases.json` from
  transformers' own functions (float32) and from the same formulas in numpy float64 (asserted to agree), plus rotations by
  transformers' `apply_rotary_pos_emb` in float64. 15 parameter sets, synthetic parameters only (no checkpoint).
  `RopeScalingTests`: frequencies to 1e-13 relative of the float64 values and 2e-6 of transformers' float32 values, cache rows
  to 1e-9, rotations to 1e-9 (float64 buffers) and 4e-6 (float32).
- Conformance, CPU half (`tests/Starlings.Roost.Conformance`): RopeScaled cases (`Suite/RopeScaledCases.cs`, bf16 and f16, f32
  cache, YaRN / longrope short and long / dynamic NTK, Qwen3.5 full-attention shape and a 128-wide head), run by `CpuBackend` and by
  an independent implementation in `OracleBackend`; `RopeScaledTests` adds a discrimination check (an unscaled cache fails).

## Capability status

| Capability | Status | Evidence |
| --- | --- | --- |
| `CpuOps.RopeScaled` and the `RopeScaling` builders (YaRN, longrope, dynamic NTK, llama3) | experimental | CPU tests against transformers 5.20.1 reference values above; no model has run with them |
| RopeScaled on CUDA | unsupported | no kernel library registers a RopeScaled variant; untested on every SM. The conformance runner reports it as `not_run` with that reason (`CaseCatalog.GpuHalfOpen`) |
| Long-context parity for a Qwen profile at its rated length | untested | not run |

## Still open (do not mark R-098 done)

- CUDA kernel for RopeScaled and its GPU conformance run (stream F / D GPU half); no GPU evidence exists.
- Long-context parity for one Qwen profile at its rated length (stream C), which also needs the model definitions to read
  `rope_scaling` from the config and build the cache through `RopeScaling`.
- Where the cache for a length-dependent kind (dynamic NTK, longrope) is rebuilt during serving: transformers rebuilds it per
  forward from the largest position; the engine-side policy is not written.
- `RopeScaledArgs` has no interleaved (GPT-J) form and no field for the cache's element type beyond the span's own Dtype; neither is
  needed by the formulas above, so no contract change was made.

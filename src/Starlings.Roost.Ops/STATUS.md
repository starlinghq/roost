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

# Quantisation in `CpuOps` (R-077, R-043): CPU groundwork only

R-077 (INT4 weight-only and FP8 KV cache, #93) and R-043 (FP8 profiles, #65) stay open. This file records the CPU reference
half that stream D landed for both, and what is still missing. Nothing here has GPU evidence.

## Delivered

| Capability | Where | Status | Evidence |
| --- | --- | --- | --- |
| FP8 E4M3 and E5M2 conversion: round to nearest even (subnormals included), saturating (CUDA `satfinite`) and OFP8 overflow modes, NaN and infinity rules | `Fp8.cs`, `CpuElements.Load`/`Store` | CPU reference only; GPU `untested` | `Fp8Tests`: decode of all 256 codes and encode of every binary16 and bfloat16 value, every midpoint and its double neighbours, and 200,000 sampled binary32 values, against a table and a nearest-code search written in the test from the bit fields. A one-off check against ml_dtypes 0.5.1 agreed on every binary16 input (NaN sign aside); it is not part of the suite |
| `Quantize`: symmetric per-row (per-tensor as one row, per-channel as one scale per row) or per-group scales into F8E4M3, F8E5M2, I8 or packed I4 | `CpuOps.Quant.cs` | CPU reference only; GPU `untested` | `QuantOpsTests` |
| `Dequantize`: `(q - z) * s` from F8E4M3, F8E5M2, I8, U8, I4, U4, with optional zero points (packed 4-bit, integer or float) and the `uint4b8` / `uint8b128` defaults | `CpuOps.Quant.cs` | CPU reference only; GPU `untested` | `QuantOpsTests` |
| `GemmFp8`: dequantise-then-Gemm reference, per-tensor or per-token A scales, per-tensor or per-channel B scales | `CpuOps.Quant.cs` | CPU reference only; GPU `untested` (no FP8 tensor-core path on SM 86, S6) | `QuantOpsTests` (hand case and agreement with Dequantize + Gemm) |
| `GemmW4A16`: dequantise (rounded to the activation type) then Gemm, B in the plain [N, K] U4/I4 layout | `CpuOps.Quant.cs` | CPU reference only; GPU `untested` | `QuantOpsTests` (bit-for-bit equal to Dequantize then Gemm) |
| AWQ and GPTQ scale and zero-point rules (round to nearest), checkpoint packing (AWQ interleaved order, GPTQ along K, GPTQ v1 `z - 1`), U4 packing, scale transposition | `Int4Weights.cs` | CPU reference only | `Int4WeightsTests`: hand-computed groups and words, round trips, act-order refusal |
| GPTQ act order (`g_idx` not `k / group`) | `Int4Weights.UnpackGptq` | `unsupported` | refused with `NotSupportedException` |
| FP8 KV cache write: `CacheWrite` with an FP8 `KvDType` divides by `KScale`/`VScale` and saturates | `CpuOps.CacheWrite` | CPU reference only; GPU `untested` | `QuantOpsTests.Fp8CacheWriteScalesAndSaturatesAndAttentionReadsTheCodes` |
| FP8 KV cache read: `PagedAttention` over an FP8 pool | `CpuOps.PagedAttention` | CPU reference at unit scale only; GPU `untested` | same test: bit-for-bit equal to attention over float32 pools holding the decoded codes |
| Conformance cases for the four ops (CPU half) | `tests/Starlings.Roost.Conformance/Suite/QuantCases.cs` | CPU half runs in CI; GPU half `untested` (no variant registered) | `CatalogTests`, `ToleranceTests`, `OracleTests.CpuOpsDoesNotMutateTheCaseItIsGiven` |

## Not done, and why

- **Contract gap, FP8 KV read with scales.** `PagedAttentionArgs` (C3) has no `KScale` or `VScale`, so a pool written with
  scales other than 1 cannot be dequantised by `PagedAttention`. A key scale can be folded into `Scale`; a value scale cannot.
  Fixing it is a stream A contract change (add the two fields, bump `ContractVersion.Current`, and the C1 `k_scale`/`v_scale`
  of the attention params if absent). FP8 KV "per state group" (R-043) also needs the scales per group in C2.
- **Marlin layout.** `GemmW4A16Args.B` says "the kernel's repacked layout" and `IRoostOps.GemmW4A16` says "Marlin layout", but
  C3 does not define the layout, so there is no repacker and no round-trip test. The CPU reference reads the plain layout above;
  a GPU result is compared after repacking the same weights. Defining the layout (or adding a layout field) is stream A's.
- **Real checkpoints.** The AWQ and GPTQ packing and rules are checked against hand-computed words written from the format
  description, not against a real AWQ or GPTQ checkpoint and its upstream dequantisation. The R-077 fidelity profile is where
  that check belongs.
- **Act-order GPTQ.** `GemmW4A16Args` has no input permutation, so `g_idx` checkpoints are `unsupported`.
- **Quantize with a static scale.** `QuantizeArgs` always computes its scales; a precomputed per-tensor activation scale (static
  FP8 activation quantisation) is not expressible except through `CacheWrite`'s KV scales.
- **Conformance tolerances.** Quantize is proposed as `exact`, Dequantize as the elementwise class of its output type, and the two
  GEMMs as the existing `gemm-bf16`/`gemm-f16` classes. These are guesses for kernels that do not exist yet; the first kernel
  run sizes them (no tolerance file was changed).
- **Kernels, fidelity profiles and admissions.** No CUDA kernel (Marlin W4A16, FP8 GEMM, FP8 KV attention), no fidelity profile
  against the dequantised reference and no admission exists. These are stream C's and the evidence pipeline's.

# R-042 status: partial (CPU half only)

R-042 (build plan section 10, streams F and D, M2): "Build the fused GPU sampler. All V1 sampling parameters, Philox per-sequence
counters, top-k logprobs, stop flags, grammar bitmask applied on the device; masks and probabilities bit-exact against the CPU
sampler in fp32; same seed gives the same draw in any batch." This change is the CPU half and the contract change it needs
(contract version 4, C1 ABI 1.3). R-042 stays open.

## Delivered

- `CpuOps.Sample` (`CpuOps.Sampling.cs`): penalties (repetition over prompt and output, frequency, presence, in vLLM's order)
  over `TokenCounts` and `PromptMask`; grammar bitmask; temperature, top-k, top-p, min-p; Philox4x32-10 draws keyed by the row's
  seed and offset only; the chosen token's logprob and top-k logprobs of the raw logits; stop flags against per-row stop ids. Two
  arithmetics: `SampleMath.Reference` (double) and `SampleMath.Fp32`, every step a correctly rounded binary32 operation in a stated
  order, with exp and log from `Fp32Math` (series coefficients, no platform `expf`/`logf`) and a 32-lane row sum.
- `HostSampler` (`src/Starlings.Roost.Worker/Eager/HostSampler.cs`, stream F): the worker-side CPU sampler over `CpuOps.Sample`,
  with per-sequence token history across steps (prompt boundary kept across preemption), bounded history cache, logprob rows for the
  C2 result.

## Capability status

| Capability | Status | Evidence |
| --- | --- | --- |
| `CpuOps.Sample` with penalties, stop flags, top-k logprobs, both math modes | untested (CPU reference; no GPU to compare with) | `SamplingTests`, `SamplingV4Tests`, `Fp32MathTests` (exp and log within 2 ulp of double over 200,000 points each; the float32 logprob equals the documented operation sequence bit for bit; per-row draws identical in any batch order) |
| Sampling, penalties, logprobs through `RoostChatClient` with `RoostSampler.Cpu` | untested | `tests/Starlings.Generation.Roost.Tests/SamplingTests.cs`, `tests/Starlings.Roost.Worker.Tests/HostSamplerTests.cs` |
| GPU sampler (`CudaOps.Sample`) | unsupported | no kernel; `CudaOps.Sample` refuses every call |
| Bit-exactness of a device sampler against `SampleMath.Fp32` | untested | no kernel exists |

## Still open (do not mark R-042 done)

- The fused GPU sampler kernel (stream C/F), registered for `ROOST_OP_SAMPLE` at ABI 1.3, its conformance cases against
  `SampleMath.Fp32` and hardware evidence; then the CUDA eager worker sets `PlanLimits.Sampling` and `RoostSampler` needs a GPU value.
- Rebuilding the kernel library and `kernels/manifest.toml` at ABI 1.3 (stream C): until then the 1.3 loader refuses the released
  1.2 assets.
- Grammar masks in plans (R-051): the eager worker still refuses `grammar_slot`; `CpuOps.Sample` applies a mask when given one.
- The server's `logprobs` and `top_logprobs` (stream H): `RoostChatClient` returns logprobs, but the OpenAI response bodies do not
  carry them yet, so the server still refuses both.
- `min_tokens` (C6 `SamplingParameters.MinTokens`) has no plan field and is not applied.

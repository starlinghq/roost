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

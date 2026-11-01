# R-071 status: partial (CPU groundwork only)

R-071 (streams L and E): AddRmsNorm, AddRmsNormQuant, SiluMulQuant and QkvQkNormRopeCacheWrite as graph passes with CUDA
variants; each fused output checked against the unfused graph on `CpuOps`; step time before and after for Qwen3 and Qwen3.5.

## Delivered

- `Passes/FuseAddRmsNormPass.cs`: a C7 fusion pass. A `Residual` and the first later `RmsNorm` of its sum become one
  `AddRmsNorm`, placed where the Residual was, when the result is the same computation (both addends are intermediates no
  later node reads, one shape and element type, a weight as the norm weight, no column views, the same activation type, no
  partial sums). It emits the op for the CPU reference (SM 0) and for a GPU target only when `GraphTarget.AvailableVariants`
  lists a conformance-passed variant of `AddRmsNorm`.
- `CpuOps.AddRmsNorm` and `CpuOps.QkvQkNormRopeCacheWrite` (`src/Starlings.Roost.Ops/CpuOps.Fused.cs`), with
  `SupportsAddRmsNorm`, `SupportsQkvQkNormRopeCacheWrite` and `Supports(OpId)` true. Each is its unfused parts called in
  order on `CpuOps`, so fused and unfused give the same bytes (`tests/Starlings.Roost.Ops.Tests/FusedOpsTests.cs`, F32 and
  BF16, BF16 activations into an F32 cache).
- Graph checks (`tests/Starlings.Roost.Graph.Tests`): the pass fuses every residual of roost-tiny (8), roost-tiny-hybrid (16)
  and the Qwen3.5-0.8B definition (48); the C7 checker on `CpuOps` with zero tolerance passes for the whole roost-tiny and
  roost-tiny-hybrid graphs and for a slice of the Qwen3.5-0.8B graph at its real widths (layers 2 and 3 and the input norm
  of layer 4: one Gated DeltaNet and one gated full-attention layer, four fusions), with seeded weights and random KV and
  recurrent state. A pass with a wrong epsilon is caught.

## Capability status

| Capability | Status | Evidence |
| --- | --- | --- |
| `FuseAddRmsNormPass` on the CPU reference | experimental | CPU tests above; no profile runs a pass output yet |
| `CpuOps.AddRmsNorm` | experimental (CPU reference only) | bitwise equal to Residual then RmsNorm |
| `CpuOps.QkvQkNormRopeCacheWrite` | experimental (CPU reference only) | bitwise equal to its unfused parts |
| AddRmsNorm, QkvQkNormRopeCacheWrite CUDA variants | untested (not built) | none |
| QkvQkNormRopeCacheWrite fusion pass | unsupported | blocked, see below |
| AddRmsNormQuant, SiluMulQuant (pass and `CpuOps`) | unsupported | `CpuOps.Quantize` does not exist (reserved for M2) |

## Still open (do not mark R-071 done)

- **QkvQkNormRopeCacheWrite pass.** Not written, because no reference graph can contain its pattern under the v0 contracts:
  the op reads one `[tokens, (QHeads + 2 * KvHeads) * HeadDim]` buffer, while QkNorm, Rope and CacheWrite (C3) take
  contiguous q, k and v with no row stride and C3 has no split op, so the unfused graph must project q, k and v with three
  Gemms into three values. A pass would have to add a concatenated weight and a new intermediate, and the C7 checker
  (rightly) fails a node that writes a value the reference lacks. The argument struct also has no rotary width, rope kind,
  pair layout, KV layout or KV scales; `CpuOps` runs it with full rotary width, default NeoX rope, `PagedNhd` and unit scales,
  so it cannot express roost-tiny-hybrid or Qwen3.5 (rotary width 4 of 16 and 64 of 256). Both need a stream A contract
  change: a strided or split form for the unfused reference (or a derived-weight value in C7), and `RotaryDim` (with the
  other rope and cache fields) in `QkvQkNormRopeCacheWriteArgs`.
- **Quantising fusions.** AddRmsNormQuant and SiluMulQuant need `Quantize` on `CpuOps` first (M2).
- **Lowering.** The generator lowers `Forward<TOps>` from the unfused definition; it does not run passes, so no generated
  forward calls a pass-emitted fused op yet.
- **CUDA variants** (stream C/L, `kernels/src/fused/`) and **step time before and after** for Qwen3 and Qwen3.5 (bet X12):
  need a GPU and are not started. No throughput number is claimed.

# R-021 tiny models: measured tolerances and mutation checks

Recorded at commit 2c2edfb on CpuOps (one machine, one libm build). The full diagnostic lines of one run are in
`r021-tiny-measurements.log`; they were captured by running the test executable with `-diagnostics -verbose`
(`TestContext.SendDiagnosticMessage` output). Tolerances live in `tests/Starlings.Roost.Models.Tiny.Tests/Support/Tolerances.cs`.
Scaled difference is |got - want| / (1 + |want|), worst over a logits row.

| Comparison | Measured | Bound |
| --- | --- | --- |
| roost-tiny vs double transcription, 28 rows | worst 3.561e-7 | 1e-5 |
| roost-tiny, one call vs chunked/batched | exactly 0 | 1e-6 |
| roost-tiny-hybrid vs double transcription (bf16 rounded), 28 rows | 26 rows exactly 0; position 20: worst 0.105, median 5.7e-3; 20-token prefill: worst 0.071, median 8.0e-3 | worst 0.25, median 0.03 |
| roost-tiny-hybrid vs unrounded double, 21 decode positions | worst 1.600 | 3.0 |
| roost-tiny-hybrid, one call vs chunked/batched | exactly 0 in every row | worst 0.25, median 0.03 |
| roost-tiny-hybrid greedy gap (top two logits), 8 steps | smallest 0.742 | margin 0.5 |

The two non-zero hybrid rows are consistent with bf16 rounding-tie flips; no analysis in this change proves that they are.

## Mutation checks (each applied to RoostTinyHybrid.cs, Models.Tiny.Tests rerun, then reverted)

| Mutation | Failing tests of 37 |
| --- | --- |
| K view start moved to 0 | 6 |
| GatedDeltaRule Scale 0.25 to 0.3 | 4 |
| PagedAttention Scale 0.25 to 0.3 | 3 |
| o_proj reads the attention output, bypassing SigmoidMul | 5 |
| GatedDeltaRule Beta bound to the raw b input | 6 |
| GatedDeltaRule UseInitialState=false | 5 |

## Open for the maintainer

The hybrid self-consistency bound (0.25) is loose against a measured 0. A bound near 1e-6 would hold on this machine
but is not justified across libm builds, so it is left unchanged.

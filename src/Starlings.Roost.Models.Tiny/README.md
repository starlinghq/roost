# Starlings.Roost.Models.Tiny

The tiny CI models (R-021). Both are written as model-compiler definitions (R-020) and run on `CpuOps` (R-018) in CI.
Nothing here is packed or shipped, and no weight file is committed: `TinyWeights` generates each weight slot from the
model's seed and the slot's name (SplitMix64 and IEEE double arithmetic, so the values are the same on every host).

| Model | Shape | State groups | Status |
| --- | --- | --- | --- |
| `roost-tiny` (`RoostTiny`) | Qwen3-shaped: 4 layers of GQA attention with per-head Q and K norms, NeoX rope, SwiGLU; f32; separate head | one full-attention group | Generates and runs on `CpuOps`; prefill, decode, chunked prefill and mixed batches match an independent double transcription |
| `roost-tiny-hybrid` (`RoostTinyHybrid`) | Qwen3.5-shaped, layer structure and slot names as R-035: 8 layers, full attention at `i % 4 == 3`, Gated DeltaNet elsewhere; bf16 with float32 `A_log` and gated-norm weights; head tied to the embedding; fused `in_proj_qkv` and a convolution over all its channels, read by `GatedDeltaRule` as three column views (`Q=l{i}.qkvc[0:key_dim]`, ...); `GatedDeltaRuleGates`, `GatedRmsNorm` with `in_proj_z`, and in attention a separate `q_gate_proj` with `SigmoidMul` | paged K and V; recurrent state and convolution state | Generates and runs on `CpuOps` (contract v3); prefill, decode, chunked prefill and mixed batches match an independent transcription that rounds to bfloat16 where the backend stores |

Self-consistency tests run on both models in CI: chunked prefill against one call, and several sequences in one batch
against each alone. The tolerances and what they were measured against are in `tests/.../Support/Tolerances.cs`.
`TinyHybridReference` is an independent double-precision transcription of the hybrid layer (conv with SiLU, L2-normalised q/k,
the delta rule, the gated norm, the sigmoid gate); with `roundActivations` it rounds each op's output to bfloat16 the way the
definition's dtype says, which makes the comparison with `CpuOps` tight (most rows are identical); without it, the difference is the
bfloat16 noise of the activations, which is large for these random weights.

`TinyConfig` repeats each model's widths and seed for tests and tools; a test compares them with the generated slots.
Not modelled (as in R-035): multimodal rope sections (partial rotary, a quarter of the head, is) and the zero-centred norm
weights (tiny norm weights are applied as stored).

## Open follow-ups

1. `PagedAttention.Mode` is left at its default for prefill (inherited from R-020); a GPU backend may mishandle it.
   Track it with the GPU backend item.
2. Maintainer decisions pending: the fused-slot `Source` convention (`a+b`), the split of `q_proj` into `q_proj` and
   `q_gate_proj` (both slots name the same checkpoint tensor; the loader splits the interleaved rows), the view-binding syntax
   (taken unchanged from R-035's branch), and the placement of this project (the plan layout lists no Tiny project).

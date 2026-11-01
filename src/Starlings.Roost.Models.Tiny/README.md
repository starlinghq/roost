# Starlings.Roost.Models.Tiny

The tiny CI models (R-021, and `roost-tiny-moe` from R-080). All three are written as model-compiler definitions (R-020) and run on `CpuOps` (R-018) in CI.
Nothing here is packed or shipped, and no weight file is committed: `TinyWeights` generates each weight slot from the
model's seed and the slot's name (SplitMix64 and IEEE double arithmetic, so the values are the same on every host).

| Model | Shape | State groups | Status |
| --- | --- | --- | --- |
| `roost-tiny` (`RoostTiny`) | Qwen3-shaped: 4 layers of GQA attention with per-head Q and K norms, NeoX rope, SwiGLU; f32; separate head | one full-attention group | Generates and runs on `CpuOps`; prefill, decode, chunked prefill and mixed batches match an independent double transcription |
| `roost-tiny-hybrid` (`RoostTinyHybrid`) | Qwen3.5-shaped, layer structure and slot names as R-035: 8 layers, full attention at `i % 4 == 3`, Gated DeltaNet elsewhere; bf16 with float32 `A_log` and gated-norm weights; head tied to the embedding; fused `in_proj_qkv` and a convolution over all its channels, read by `GatedDeltaRule` as three column views (`Q=l{i}.qkvc[0:key_dim]`, ...); `GatedDeltaRuleGates`, `GatedRmsNorm` with `in_proj_z`, and in attention a separate `q_gate_proj` with `SigmoidMul` | paged K and V; recurrent state and convolution state | Generates and runs on `CpuOps` (contract v3); prefill, decode, chunked prefill and mixed batches match an independent transcription that rounds to bfloat16 where the backend stores |
| `roost-tiny-moe` (`RoostTinyMoe`) | Qwen3-MoE-shaped (R-080): roost-tiny's attention (4 layers, GQA, Q and K norms, NeoX rope at theta 1e6) and in every layer a routed-expert MLP as in Qwen3-30B-A3B: a router `Gemm` to `[tokens, 8]` logits and `FusedMoe` with 8 experts, top 2, renormalised weights, expert width 32, no shared expert; bf16 weights and activations; separate head; expert weights stacked per layer (`experts_gate_up` `[8, 64, 64]`, `experts_down` `[8, 64, 32]`) | one full-attention group | Generates and runs on `CpuOps` (`experimental`: no independent transcription and no checkpoint parity yet). Chunked prefill, batched sequences and forked sequences give the same bytes as one call; greedy ids are pinned |

Self-consistency tests run on all three models in CI: chunked prefill against one call, and several sequences in one batch
against each alone; for `roost-tiny-moe` also sequences forked from a cached prefix (full blocks shared, the partial block copied with
`CacheCopy`) against fresh ones. Every `CpuOps` op computes a row independently of the rest of the call, so for `roost-tiny-moe` these
comparisons assert identical bytes rather than a tolerance (measured difference 0 in every comparison). A further test zeroes one expert of
layer 0 at a time and counts the experts whose removal moves the logits of a 40-token prompt: 8 of 8, so the router spreads tokens. The tolerances and what they were measured against are in `tests/.../Support/Tolerances.cs`.
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
3. `roost-tiny-moe` uses the one-node `FusedMoe`. The unfused chain (MoeRouteTopK, MoePermute, GroupedGemm, SiluAndMul, GroupedGemm,
   MoeUnpermute) cannot be written as a definition yet: its permuted rows are `[tokens * top_k, hidden]` and its expert counts and offsets
   `[experts]` and `[experts + 1]` int32, and the model compiler lays out only token-major activations (leading dimension `tokens` or
   `selected`). Expressing them needs a model-compiler extension and possibly a C7 shape change; it is not done here.
4. The `Source` of a stacked expert slot uses `*` for "every expert index, ascending" (`model.layers.{i}.mlp.experts.*.down_proj.weight`);
   no loader reads it yet, and the convention awaits the same maintainer decision as `+`. FP8 and INT4 expert weights are `unsupported`.

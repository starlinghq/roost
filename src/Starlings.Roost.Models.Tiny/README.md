# Starlings.Roost.Models.Tiny

The tiny CI models (R-021). Both are written as model-compiler definitions (R-020) and run on `CpuOps` (R-018) in CI.
Nothing here is packed or shipped, and no weight file is committed: `TinyWeights` generates each weight slot from the
model's seed and the slot's name (SplitMix64 and IEEE double arithmetic, so the values are the same on every host).

| Model | Shape | State groups | Status |
| --- | --- | --- | --- |
| `roost-tiny` (`RoostTiny`) | Qwen3-shaped: 4 layers of GQA attention with per-head Q and K norms, NeoX rope, SwiGLU; f32; separate head | one full-attention group | Generates and runs on `CpuOps`; prefill, decode, chunked prefill and mixed batches match an independent double transcription |
| `roost-tiny-hybrid` (`RoostTinyHybrid`) | Qwen3.5-shaped: 8 layers, full attention at `i % 4 == 3`, Gated DeltaNet elsewhere; bf16; head tied to the embedding; fused `in_proj_qkv` and a convolution over all its channels | paged K and V; recurrent state and convolution state | Structure tests only. Its numeric run waits for the C3-qkv item (views), because `GatedDeltaRule` cannot yet read q, k and v from the fused convolution output |

`TinyConfig` repeats each model's widths and seed for tests and tools; a test compares them with the generated slots.
What the hybrid leaves out of Qwen3.5 (the attention output gate, the z projection with its gated norm, the `A_log` and
`dt_bias` transforms) is listed in `RoostTinyHybrid.cs`; those belong to the real definition (R-035).

## Open follow-ups

R-021 is partial until the first item below is done.

1. After the C3-qkv item lands (Q, K and V as views of the `CausalConv1d` output): bind the three `delta` arguments Q, K
   and V to those views, delete `GatedDeltaRuleReadsThePlaceholderUntilC3ViewsLand`, enable `NumericRunWaitsForViews`
   and compare against a double transcription as `RoostTinyTests` does.
2. `PagedAttention.Mode` is left at its default for prefill (inherited from R-020); a GPU backend may mishandle it.
   Track it with the GPU backend item.
3. The hybrid omits `z`, `A_log`, `dt_bias` and the attention output gate; R-035 owns them.
4. Maintainer decisions pending: the fused-slot `Source` convention (`a+b`), the placement of this project (the plan
   layout lists no Tiny project), and the C3-qkv contract-change request.

# Starlings.Roost.Models.Qwen35

The Qwen3.5-0.8B definition (R-035, stream E): `Qwen35Model`, written as a model-compiler definition (R-020) and compiled into this
assembly. The package ships the generated code (`Forward<TOps>`, `WeightSlots`, `StateSlots`, `CreateStateGroups`, `PlanInputs`,
`ForwardLayout`, `BuildGraph`), so a consumer needs no generator; the generator is a build-time reference of this project only.

Checkpoint: `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` (locked in `evidence/registry/qwen3.5-0.8b.lock.toml`).

## Shape

| | |
| --- | --- |
| Layers | 24: full attention at `i % 4 == 3` (6 layers: 3, 7, 11, 15, 19, 23), Gated DeltaNet elsewhere (18) |
| Widths | hidden 1024, MLP 3584, vocabulary 248320, head tied to the embedding |
| Attention | 8 query heads, 2 key/value heads, head size 256, rotary on the first 64 dimensions, theta 1e7, per-head q/k norm, output gate |
| Gated DeltaNet | 16 key heads and 16 value heads of 128, convolution width 4 over 6144 channels |
| State groups | `attn` (Full, paged K and V, block 16) and `linear` (Recurrent: float32 state [16, 128, 128] and conv state [6144, 3] per layer in one slot) |
| Dtypes | bfloat16 weights and activations, float32 recurrent state |

The linear block reads the fused `in_proj_qkv` projection through the convolution and then q, k and v as row-strided column
views of the convolution output (`Q=l{i}.c[0:key_dim]`, `GatedDeltaRuleArgs.QStride` = 6144), as transformers does with `torch.split`.

## Weight map (`Qwen35WeightMap`)

Slot sources are plain checkpoint tensor names (`a+b` joins two). What a loader does to them is `WeightTransform`:

| Slot | Checkpoint tensor | Transform |
| --- | --- | --- |
| `embed` | `model.language_model.embed_tokens.weight` | none (also the LM head) |
| `l{i}.input_norm`, `post_norm`, `q_norm`, `k_norm`, `final_norm` | `...input_layernorm.weight` and the like | `OnePlus`: the checkpoint is zero-centred, the slot holds `1 + w` |
| `l{i}.q_proj`, `q_gate_proj` (attention) | `...self_attn.q_proj.weight` [4096, 1024] | `QueryRows` / `GateRows`: per head, 256 query rows then 256 gate rows |
| `l{i}.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` | `...linear_attn.*.weight` | none |
| `l{i}.conv` | `...linear_attn.conv1d.weight` [6144, 1, 4] | `SqueezeConv`: slot [6144, 4] |
| `l{i}.A_log` (F32), `dt_bias`, `linear_norm` (F32) | `...linear_attn.A_log`, `dt_bias`, `norm.weight` | none; read by `GatedDeltaRuleGates` and `GatedRmsNorm` |
| `l{i}.gate_up` | `...mlp.gate_proj.weight+...mlp.up_proj.weight` | `ConcatRows`: gate rows then up rows |
| everything else | `...mlp.down_proj.weight`, `self_attn.{k,v,o}_proj.weight` | none |

Not slots: the 15 `mtp.*` tensors (`Qwen35WeightMap.Mtp`, the MTP head, mapped and unused in M0) and the 153 `model.visual.*` tensors.
`tests/Starlings.Roost.Models.Qwen35.Tests/Data/qwen35-0.8b.safetensors-header.txt` lists every text and MTP tensor with its dtype and shape
from the locked file's header, and a test checks the map against it.

## Gating steps (contract version 3)

The forward runs the whole Qwen3.5 block on `CpuOps` (`Qwen35Model.NumericallyComplete`):

- linear layers: `GatedDeltaRuleGates` turns the raw `in_proj_a` and `in_proj_b` outputs into `g = -exp(A_log) * softplus(a + dt_bias)`
  (float32 value) and `beta = sigmoid(b)`; the delta rule reads those;
- linear layers: `GatedRmsNorm` computes `rmsnorm(core_out) * linear_norm.weight * silu(z)` per value head (Rows = tokens * 16,
  Dim = 128, F32 weight, not zero-centred) and `out_proj` reads its output;
- attention layers: `q_proj` interleaves each head's query and gate rows; the loader splits them (`QueryRows` / `GateRows`) and
  `SigmoidMul` computes `attn_out * sigmoid(gate)` before `o_proj`.

Zero-centred norms are not in v3, so every decoder, q/k and final norm slot holds `bfloat16(1 + w)` (the ADR's loading rule;
`WeightTransform.OnePlus`). transformers computes `1 + w` in float32, so those norms agree with it only to bfloat16 weight rounding.

## Attention window and logits

`PagedAttentionArgs.WindowLeft` is a sliding window when it is not negative, and an attribute left unbound is 0: a window of one
token. The full-attention node binds `WindowLeft=-1`. Until R-022 it did not, and every query attended to itself only; position 0 and
the linear layers were unaffected, so the smoke test below still passed. `LmHead` writes float32 logits (C3), and the definition
declares the `logits` value `F32` so that `Output_logits` and the arena size agree with what the op writes
(`ForwardLayout.PerTokenBytes` 498688 before, 995328 after: the generator sizes the logits per token of capacity).

## Development evidence (not a fixture)

One CPU forward on the locked checkpoint (Release, `CpuOps`) for the prompt "The capital of France is" (ids 760 6511 314 9338 369):
the top logits are 11751 " Paris" 14.875, 279 " the" 14.596, 7172 " located" 14.273, 25 14.062, 198 13.982 (transformers fp32 on
the same ids: 14.871, 14.587, 14.285, 14.086, 14.022). Before the gating ops the argmax was 105607, a nonsense token; with the
one-token attention window it was " the". Prefill 10 to 19 s, decode step 2.7 s. Logits are finite, and prefill plus a decode step
gives the same logits (cosine 1.000000) as one prefill.

Per-layer parity against transformers fp32 (hidden states of every layer, Gated DeltaNet recurrent and convolution state, step-0
log-probabilities, five chat prompts, 24 forwards each) is R-022's: `docs/m0/roost-r022-cpu-parity.md`. At attention layers 3
and 15 a bf16 emulation accounts for the remaining difference as bfloat16 storage of activations; for the 18 Gated DeltaNet layers
that is inferred, as is the reduction of the multimodal rope to plain rotary embedding for text-only prompts (not tested directly).

## Unverified

- Package restore: the nupkg lists Abstractions, Graph and Ops as dependencies, but those projects are not packable yet, so it cannot be restored standalone.

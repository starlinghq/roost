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
| `l{i}.A_log` (F32), `dt_bias`, `linear_norm` (F32) | `...linear_attn.A_log`, `dt_bias`, `norm.weight` | none; reserved slots (see below) |
| `l{i}.gate_up` | `...mlp.gate_proj.weight+...mlp.up_proj.weight` | `ConcatRows`: gate rows then up rows |
| everything else | `...mlp.down_proj.weight`, `self_attn.{k,v,o}_proj.weight` | none |

Not slots: the 15 `mtp.*` tensors (`Qwen35WeightMap.Mtp`, the MTP head, mapped and unused in M0) and the 153 `model.visual.*` tensors.
`tests/Starlings.Roost.Models.Qwen35.Tests/Data/qwen35-0.8b.safetensors-header.txt` lists every text and MTP tensor with its dtype and shape
from the locked file's header, and a test checks the map against it.

## Not yet numerically complete

C3 has no op for four steps of the Qwen3.5 blocks, so `Qwen35Model.NumericallyComplete` is false and the logits are not the
checkpoint's (`Qwen35Model.PendingOps` lists the steps). The definition already holds the projections and weights each step needs
(`in_proj_z`, `in_proj_a`, `in_proj_b`, `q_gate_proj`, and the reserved `A_log`, `dt_bias`, `linear_norm` slots), so each lands as
one node when its op exists:

1. `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)` before the delta rule (today `a` and `b` go in raw);
2. the gated output norm `rmsnorm(core_out) * linear_norm.weight * silu(z)` per value head after it (this order, norm before the
   gate, is a proposal that no code or reference check on this branch has verified; the op's author must confirm it against transformers);
3. the attention output gate `attn_out * sigmoid(gate)`;
4. a zero-centred RMS norm form, so `1 + w` need not be rounded to bfloat16 at load.

`NumericsAreIncompleteUntilTheContractHasTheOps` fails when `IRoostOps` gains any of the candidate ops, as a reminder to put the nodes in.

## Unverified

- A text-only prompt reducing the interleaved mrope (sections `[11,11,10]`) to plain rotary embedding is not tested here; R-022 must check it.
- Package restore: the nupkg lists Abstractions, Graph and Ops as dependencies, but those projects are not packable yet, so it cannot be restored standalone.

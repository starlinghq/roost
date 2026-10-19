# Proposed tolerances from R-022 (Qwen3.5-0.8B, CPU reference profile)

**Status: proposed, not binding.** The maintainer sets binding tolerances in `evidence/protected/` (decision 8, R-014); this
directory holds proposals awaiting that decision, and no gate reads it. Values: [`r022-tolerances.toml`](r022-tolerances.toml).
Procedure, revisions and full tables: [`docs/m0/roost-r022-cpu-parity.md`](../../docs/m0/roost-r022-cpu-parity.md).

## Decision order: the metric first

**The maintainer is being asked to choose the hidden-state metric before any of the values below.** The values are unchanged from the
first version of this proposal; they are the answer to the question "how large an absolute error is acceptable", and that question is
the weaker one. The data supports a row-relative metric (`||c - r|| / ||r||` per position), paired with cosine similarity, over the
absolute `float.layer_hidden_max_abs` bound of 2.0:

- Row-relative separates the fixed definition from every defect by a wide margin: 0.067 for the fixed definition, at least 1.41 for
  each of the four defects (1.62 window, 5.82 attention gate, 11.3 `silu(z)`, 1.41 eps), so a bound near 0.1 passes the fixed definition
  and fails each defect.
- Cosine similarity was not measured in this run; only the row-relative figures above are data. Row-relative L2 bounds the
  angle as well, so it is a pairing choice, not a separate measurement.
- It also attributes each defect to the layer where it enters (layer 3 for the window defect in all five cases; layer 3, 0 and 0 for
  the attention-gate, `silu(z)` and eps runs; `docs/m0/r022/relative-attribution.txt`).
- The absolute bound of 2.0 must be at least the final norm's 1.33 (ten times the layers' scale), so it cannot attribute: it
  attributes the window defect to layers 18-23, not layer 3, and a layers-only absolute bound misses short prompts.

The evaluator reads only the absolute key today; a row-relative or cosine check is a request to stream B2 (see the R-022 report,
Requests). If the maintainer picks the row-relative metric, the other proposed keys (recurrent, conv, step-0, near-tie) stand as
proposed and the hidden key is replaced by the metric's own bound; if the absolute metric is kept, 2.0 is the value, with the
attribution limit above.

## Which profile these are for

Tolerances are per profile (decision 8). **Every value below belongs to one profile: `qwen3.5-0.8b/cpu-fp64-reference`**, the
generated forward on `CpuOps` against transformers 5.20.1 float32 on the CPU. It checks the model definition.

What "fp64" means here: `CpuOps` computes every op in double, but the definition stores activations, K/V and convolution state as
bfloat16 and `CpuOps` rounds each op's result once into that type; weights are the checkpoint's bfloat16 (the zero-centred norm slots
hold bfloat16(1 + w)), the recurrent state is float32 and the logits float32. The differences these bounds cover are therefore
bfloat16 storage rounding, about 2^-9 relative per stored value, carried through 24 layers. They are not fp64-against-fp32
accumulation differences, which are about 1e-6 relative: an exact double recomputation of attention layers 3 and 15 reproduces
transformers to 1e-6, and the same recomputation with the definition's bfloat16 roundings reproduces the candidate to 0.0021
(`docs/m0/r022/emulate-bf16.txt`).

**The M0 gate's profile (the bf16 GPU worker against protected fixtures, R-026) is PENDING a GPU measurement.** No value here is
extrapolated to it. The GPU worker stores the same bfloat16 activations, but its accumulation order, fused kernels and fp32
accumulators differ from `CpuOps`; its bounds come from a run of the worker (R-025), not from this one.

## Data

5 chat prompts rendered by Starling's `Qwen35ChatTemplate` (25 to 134 tokens), 24 greedy tokens each from the reference, the
candidate teacher-forced on them: 120 forwards (5 prefills, 115 decode steps). Hidden states of embeddings, 24 layers and the final
norm at every position; recurrent and convolution state of the 18 Gated DeltaNet layers after every forward; step-0 log-softmax over
248320 tokens.

## Values

| Key | Proposed | Measured max | Set by | Margin |
| --- | --- | --- | --- | --- |
| `float.layer_hidden_max_abs` | 2.0 | 1.33 | final_norm, multi-turn, prefill position 33, channel 687 (reference 5.73) | 1.5x |
| `float.recurrent_state_max_abs` | 0.15 | 0.0729 | layer 16, capital, forward 0 | 2x |
| `float.conv_state_max_abs` | 0.5 | 0.306 | layer 21, code, forward 0 | 1.6x |
| `float.nan_or_inf` | `fail` | none | | |
| `float.step0_logprob_max_abs` | 0.4 | 0.257 | code, a tail token (reference log-probability -26.6) | 1.6x |
| `float.step0_top5_logprob_max_abs` | 0.25 | 0.159 | code | 1.6x |
| `discrete.step0_argmax` | `equal` | 5 of 5 equal | | |
| `discrete.greedy_ids` | `exact_except_listed_margin_below_0.2` | 120 of 120 argmax equal; gap error at most 0.137 | long-context, forward 10 (reference gap 8.85) | 1.5x |

### `float.layer_hidden_max_abs` = 2.0

Distribution (max abs over the 5 cases and 24 forwards, per stage): embeddings 0; layers 0-2 at most 0.016; layers 3-14 at most
0.052; layers 15-22 at most 0.071; layer 23 0.138; final norm 1.33. Per case the maximum is always the final norm in the prefill
(0.76, 1.10, 1.33, 0.76, 0.78); decode forwards reach at most 0.41. Relative to the row, the worst error is 0.067 everywhere.

Reasoning: one stored bfloat16 value has a rounding error up to 2^-9 of its magnitude, so a stage whose values reach 4 to 8 (layers
15-23) carries 0.008 to 0.016 per rounding, and the 24-layer residual stream accumulates and amplifies them. The final norm multiplies
by its weight (values up to 42.2), which is why it is ten times the layers. A single key covers every stage, so the bound must pass
the final norm; 2.0 is 1.5x its measured maximum. This is the weakest of the proposals because of what the margin hides: it is 15x
the layers' maximum, so a defect that changes a layer by 0.2 to 1.9 passes this key. The recurrent and conv state keys catch such a
defect in any linear layer (below), and a row-relative check would catch it anywhere (requested from stream B2).

### `float.recurrent_state_max_abs` = 0.15

Per layer: at most 0.017 in layers 0-2, 0.011 to 0.073 in later layers (reference magnitude 1.3 to 13.9, relative L2 at most 0.016).
Across forwards: 0.073 after the prefill and 0.065 to 0.072 after each of the 23 decode steps, so no growth. The state is float32 and
the delta rule runs in double; its difference comes from the bfloat16 q, k, v (convolution output) and gates it reads. Qwen3.5's
recurrence multiplies the state by exp(g) < 1 every step, so old error decays at the rate old state does; the measured flat profile
agrees. Margin 2x for longer decodes than the 24 forwards measured.

### `float.conv_state_max_abs` = 0.5

Per layer: at most 0.083 in layers 0-2, 0.11 to 0.31 later; reference magnitude up to 38.4, relative L2 at most 0.011. The state holds
the last inputs of the convolution, which are bfloat16 projections: one rounding of a value between 32 and 64 is up to 0.125. Column 0
of the compared state is reconstructed (transformers keeps 4 inputs, Roost 3) and agrees with columns 1-3. Margin 1.6x.

### Step-0 log-probabilities, step-0 argmax and the near-tie bound

`float.step0_logprob_max_abs` = 0.4: the whole-vocabulary maximum is set by tail tokens (reference log-probability -17.9 to -26.6
in every case), 0.134 to 0.257 per case. `float.step0_top5_logprob_max_abs` = 0.25: 0.030 to 0.159 per case; top-1 at most 0.041.

Near-tie: the greedy rule allows a divergence only where the reference's top-1/top-2 gap is below the bound. Measured directly, the
candidate's gap differs from the reference's by at most 0.137 over all 120 forwards and by at most 0.103 over the 20 forwards whose
reference gap is below 0.5. All 120 argmax agree, including the smallest reference gaps 0.0096 (code, forward 23) and 0.048
(multi-turn, forward 9). A flip needs the gap error to exceed the reference gap, so 0.2 (1.5x the largest gap error) allows every
divergence this profile can produce. This was measured teacher-forced; a free-running greedy decode equals the reference for these
prompts because no forward's argmax differed.

## What a real defect looks like against these values

| Run | hidden (bound 2.0) | recurrent (0.15) | conv (0.5) | step-0 top-5 (0.25) |
| --- | --- | --- | --- | --- |
| fixed definition | 1.33 passed | 0.073 passed | 0.306 passed | 0.159 |
| attention window 0 (the real defect R-022 found) | 27.1 failed, attributed to layers 18-23 | 2.18 failed, layer 4 | 15.8 failed, layer 4 | 8.19 |
| attention output gate dropped | 58.5 failed, layer 3 | 11.5 failed, layer 4 | 27.0 failed, layer 4 | 19.5 |
| `silu(z)` dropped from the gated norm | 41.5 failed, layer 0 | 12.3 failed, layer 1 | 45.3 failed, layer 1 | 34.9 |
| RMS norm eps 1e-5 instead of 1e-6 | 37.7 failed, layer 6 | 4.47 failed, layer 1 | 18.2 failed, layer 1 | 7.41 |

Every defect exceeds the hidden, recurrent and conv bounds by 13x or more (the maximum over cases, not every case: `nan_or_inf` passes in every negative run, and the window defect's capital case still has an equal step-0 argmax; the drop-attention-gate, drop-z-gate and norm-eps runs covered 2 of 5 cases, capital and long-context). The attribution is right for the gate perturbations and wrong for the window defect
on the hidden key (the defect enters at layer 3, but its absolute size there, 0.61 at most, is below 2.0); the state keys point at
the first linear layer after the defect. A row-relative hidden bound of 0.1 (fixed definition 0.067; defects 1.41 to 11.3) attributes
every one to the layer where it enters (`docs/m0/r022/relative-attribution.txt`).

## Not proposed

- `discrete.template_rendering`, `discrete.prompt_ids`: the template comparison was exact for all five prompts, but this run used no
  `dotnet` generation stage, so the generation evaluator was not run.
- `float.cached_vs_uncached_logprob_max_abs`, `float.chunked_vs_one_call_prefill_logprob_max_abs`: not measured here.
- Anything for the bf16 GPU profile: pending a GPU measurement (R-025, R-026).

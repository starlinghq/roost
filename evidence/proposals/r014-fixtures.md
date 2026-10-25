# R-014: proposed Qwen3.5-0.8B reference fixtures

Status: **proposed**, for the maintainer. Nothing here is binding. `r014-fixtures.toml` pins the fixture data by SHA-256.

## What the fixtures are

They are the transformers float32 CPU reference for the five `R022PromptSet` chat prompts, at the locked checkpoint:
- greedy output ids (24 tokens per case);
- step-0 logits (float32, the full 248320-entry vocabulary);
- per-layer hidden states for all 24 layers;
- recurrent (Gated DeltaNet) and convolution state for the 18 linear-attention layers, over 24 forwards.

They come from the R-022 reference run, which depends only on transformers, the checkpoint and Starling's template and tokenizer, not on Roost's code. The run's snapshot matches `evidence/registry/qwen3.5-0.8b.lock.toml` for the weights, config, chat template, `tokenizer.json` and `tokenizer_config.json`.

This proposal reuses that run rather than recomputing it. A rerun costs about 2.5 GB of captures and several CPU-minutes per case, and it would only reproduce the same transformers output.

## Where the bytes should live (maintainer decision)

The data is 2.5 GB, so it cannot enter Git. It is on the development host at `/home/mark/Projects/roost-evidence/R-014/ref`, hardlinked from the R-022 run so that cleaning up that run does not lose it. `SHA256SUMS` sits beside it.

Proposal: publish the directory as assets of a GitHub release named for the fixture set (for example `fixtures-qwen3.5-0.8b-r014`), the way R-019 stores its upstream goldens. `evidence/protected/` would then carry only this digest table. Gates would fetch by digest and refuse a mismatch.

## Near ties: the greedy rule needs a margin exemption

Four of the five cases have a greedy step whose top-2 logit margin is below 0.1:

| Case | Step | Margin |
| --- | --- | --- |
| code | 23 | 0.0096 |
| multi-turn | 9 | 0.0479 |
| capital | 15 | 0.0554 |
| system-haiku | 5 | 0.0711 |

R-025 measured GPU-vs-CpuOps logit error up to 0.23 on SM 86, so a GPU candidate can legitimately flip any of these. An unconditional "greedy ids equal" check would be flaky. The options:
- **(a)** Keep the R-022 rule form, `exact_except_listed_margin_below_<bound>`, with the bound set per profile (decision 8). The CPU proposal uses 0.2. The GPU profile needs its own value once it is measured against these fixtures.
- **(b)** Compare teacher-forced: feed the reference's tokens and check per-step argmax only where the margin clears the bound. This is how R-022 and R-025 already compare.
- **(c)** Replace or trim the cases so that no step falls below the bound. This loses realistic prompts and makes the gate weaker.

Recommended: (a) together with (b), which is what the evaluator already supports.

## Tolerances per profile

- **`qwen3.5-0.8b/cpu-fp64-reference`:** values proposed in `r022-tolerances.toml`, measured against exactly these fixtures.
- **`qwen3.5-0.8b/cuda-bf16-eager` (the M0 gate):** not measured against these fixtures yet. The only GPU numbers so far are R-025's GPU-vs-CpuOps bounds (`tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-tolerance-proposal-sm86.txt`), which are a floor, not the profile's tolerance. The next step is a GPU run on SM 86 and SM 89 against these five cases, teacher-forced, with per-layer, state and step-0 comparisons. Its proposal follows in a separate PR, together with the SM 89 evidence the M0 gate needs.

## Open questions for the maintainer

1. Accept these five cases as the protected fixture set for both profiles, or ask for more (for example a case longer than 134 tokens, so that chunked prefill is exercised)?
2. Approve publishing the bytes as release assets, under which tag?
3. Choose the near-tie rule (a, b or c above) and its bound per profile.

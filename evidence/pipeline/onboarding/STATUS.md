# R-055 status: partial (inspect and recipes built; onboarding trials open)

R-055 (stream I, M3): build the `roost-v1` onboarding plugin and recipes. Acceptance criteria: inspect maps config to the op
vocabulary and state groups and reports composition or missing ops; recipes in the strict schema.

## Delivered

- `roost-onboard inspect` (`ConfigInspector`): config.json to C3 ops and C4 state groups, typed JSON report with a verdict,
  per-op status (available, declared, gap, missing), layer map, MoE shape, state groups, load transforms, notes and warnings.
  Covers full, sliding-window, Gated DeltaNet and MLA token mixers; dense and MoE MLPs (Qwen and DeepSeek layer rules, shared
  experts); default, llama3, YaRN, longrope, dynamic NTK, linear and multimodal rope; partial rotary; FP8, AWQ and GPTQ
  quantisation configs; soft-capping; unknown layer types, activations and norms.
- The roost-v1 recipe schema (`Schemas/roost-recipe-v1.schema.json`, strict) and `roost-onboard validate-recipe`
  (`RecipeValidator`), with the rules beyond the schema.
- `evidence/recipes/qwen3.5-0.8b.json`: the first recipe, for the existing M0 profile. It is not one of the M3 gate's unseen
  models.
- Tests (`tests/Starlings.Roost.Onboarding.Tests`): Qwen3 (composes), Qwen3.5 (composes; state groups equal the generated
  definition's), Qwen3-MoE (composes, optional expert group), Gemma 3 sliding window (composes; soft-capping variant is a gap),
  DeepSeek-V3 MLA (`missing_ops`: MoE routing, latent cache write and block FP8 scales are gaps, `MlaAttention` declared
  without a reference), malformed configs, every schema and rule violation, the command line.

What inspect found for the five test configs, at contract version 6:

| Config | Verdict | Missing or gap | Pending (declared, no `CpuOps`) |
| --- | --- | --- | --- |
| Qwen3-0.6B | composes | none | none |
| Qwen3.5-0.8B | composes | none | none |
| Qwen3-30B-A3B | composes | none | none |
| Gemma 3 1B | composes | none | none |
| DeepSeek-V3 | missing_ops | CacheWrite (MLA latent write), MoeRouteTopK (sigmoid, group-limited, bias), GemmFp8 (block scales) | MlaAttention |

"Composes" is a statement about the C3 vocabulary and the CPU reference only. Only Qwen3.5-0.8B has a model definition and
GPU evidence; the others have neither.

## Open

- **Three unseen compositional models onboarded in under one engineer-day each** (M3 gate evidence, R-057). Not started: no
  recipe beyond Qwen3.5-0.8B exists, and inspect does not write model definitions (stream E's inspect-to-definition mapping).
- Inspect reads config.json only. Weight names, tokenizer and chat template are not inspected, and the family table covers
  Llama, Mistral, Qwen2, Qwen2-MoE, Qwen3, Qwen3-MoE, Qwen3-Next, Qwen3.5, Gemma 1 to 3 and DeepSeek V2 and V3.

## Capability status

| Capability | Status | Evidence |
| --- | --- | --- |
| `roost-onboard inspect` | untested (CPU tool, tested on hand-written configs; no onboarding trial yet) | `tests/Starlings.Roost.Onboarding.Tests` |
| roost-v1 recipe schema and validator | untested (CPU tool) | `tests/Starlings.Roost.Onboarding.Tests` |
| Onboarding an unseen model | unsupported | none (R-057) |

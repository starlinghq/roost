# roost-v1 onboarding plugin

Stream I, build plan R-055. `Starlings.Roost.Onboarding` builds the `roost-onboard` tool. It runs at onboarding time only and
never ships in the runtime, the worker or the image. It references `Starlings.Roost.Abstractions` (C1 `OpId` and `OpCatalog`, C4
`StateGroupDescriptor`) and `Starlings.Roost.Ops` (C3 and `CpuOps.Supports`), and nothing else: no third-party package, no
network, no Python.

## inspect

```sh
dotnet run --project evidence/pipeline/onboarding/Starlings.Roost.Onboarding -- inspect <config.json> \
    [--block-size 16] [--kv-dtype BF16|F16|F32|F8E4M3|F8E5M2] [--snapshot-slots 0] [--out report.json]
```

Reads one Hugging Face `config.json` (a local file; a multimodal wrapper's `text_config` is inspected) and writes a typed JSON
report (`InspectReport`, snake_case):

- `verdict`: `composes` when every op the model needs is in C3, implemented by `CpuOps`, and able to express what the config
  asks; `composes_pending_reference` when every op is in C3 but at least one has no CPU reference yet; `missing_ops` otherwise.
- `ops`: one entry per op, in first-use order, with its C3 phase, the parts of the model that use it and a status:
  `available`, `declared` (reserved in C3, no `CpuOps` implementation), `gap` (in C3, but its arguments cannot express the
  config, for example sigmoid MoE routing or attention logit soft-capping) or `missing` (not in C3, for example `LayerNorm`).
  `missing_ops` lists the gap and missing ops by name, `pending_ops` the declared ones.
- `layers`: which layers are full attention, sliding-window attention, linear attention (Gated DeltaNet) or MLA, and which have
  a dense or MoE MLP. Layer types come from `layer_types`, else from `full_attention_interval`, `sliding_window_pattern`,
  `use_sliding_window` with `max_window_layers`, or a bare `sliding_window` (every layer, Mistral's rule).
- `state_groups`: the C4 groups, built as `StateGroupDescriptor`s and reported with their layers: `attn` (Full), `sliding`
  (Sliding, with the window), `mla` (MlaLatent, one `kv_lora_rank + qk_rope_head_dim` row per token), `linear` (Recurrent, the
  float32 delta-rule state plus the convolution state, laid out as the model compiler lays out a recurrent slot) and `experts`
  (Expert, marked optional: paging experts is a placement choice).
- `load_transforms`: weight changes a loader makes so the model maps onto C3 without a new op (zero-centred norms stored as
  `1 + w`, Gemma's embedding scale, DeepSeek's `routed_scaling_factor`, MLA weight absorption).
- `notes` and `warnings`: equivalences used and assumptions made where config.json says nothing.

Exit 0 when the model composes, 1 for `composes_pending_reference` or `missing_ops`, 2 when the config cannot be read.

What a config does not state (whether a family norms q and k per head, zero-centres its norms, gates the attention output or
scales the embedding) comes from a family table keyed by `model_type` in `ConfigInspector.cs`. An unknown `model_type` is
inspected from its fields alone with a warning. `available` means a model can be defined against C3; whether a GPU kernel
exists for the dtype, head size and SM is the worker's startup check (R-023), and nothing here claims a GPU result.

## validate-recipe

```sh
dotnet run --project evidence/pipeline/onboarding/Starlings.Roost.Onboarding -- validate-recipe --root . evidence/recipes
dotnet run --project evidence/pipeline/onboarding/Starlings.Roost.Onboarding -- schema    # prints the schema
```

Validates recipes against the strict schema `Starlings.Roost.Onboarding/Schemas/roost-recipe-v1.schema.json` (embedded in the
tool; every object sets `additionalProperties: false`) and then against the rules a schema cannot express, listed in
`RecipeValidator`'s remarks: file named after the id, unique ids, status needs evidence, `supported` needs
`evidence/protected/`, `experimental` needs an SM entry with hardware evidence, a consistent composition, a model that does not
compose is `unsupported`, and every path exists without a `..` segment. Exit 0 when every recipe is valid, 1 with one
`path: location: message` line per problem.

The schema lives next to the tool because build plan section 4 gives stream I `evidence/pipeline/onboarding/` and
`evidence/recipes/`, not `evidence/schemas/` (streams B2 and B3). The tool evaluates it with `StrictSchema`, a JSON Schema
subset evaluator that refuses any keyword it does not implement, so the schema file stays the source of truth.

## Tests

`tests/Starlings.Roost.Onboarding.Tests` (in `Roost.slnx`, CPU only). The configs in its `TestData/` are written by hand from
the public `config.json` fields of Qwen3-0.6B, Qwen3.5-0.8B, Qwen3-30B-A3B, Gemma 3 1B and DeepSeek-V3; nothing is downloaded.
The Qwen3.5-0.8B state groups inspect builds are compared with the generated `Qwen35Model.CreateStateGroups()`.

Status: [STATUS.md](STATUS.md).

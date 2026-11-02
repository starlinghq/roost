# Starlings.Roost.EngineGates.Tests

The engine gate harnesses of R-039 (stream I), CPU half. Each gate runs a variant of the engine and compares it with a
baseline: every case alone, its prompt prefilled in one call, no prefix cache, no preemption. The engine is the real one
(`StepPlanner` over a `BlockManager`, with its `PrefixCache` when configured, from `Starlings.Roost.Engine`; for streaming,
`RoostChatClient` and its driver on the in-process transport) and the model is `roost-tiny` or `roost-tiny-hybrid`
(R-021) running its generated forward on `CpuOps`. The run writes stage results, format v3
(`evidence/schemas/stage-results.schema.json`), that the evaluator reads.

```sh
dotnet test --project tests/Starlings.Roost.EngineGates.Tests
ROOST_ENGINE_GATES_OUT=<dir> dotnet test --project tests/Starlings.Roost.EngineGates.Tests   # also keep the run directories
uv run --project evidence/pipeline --locked roost-evidence evaluate <dir>/roost-tiny --reference <reference.jsonl> --tolerances <file>
```

Without `ROOST_ENGINE_GATES_OUT` the run directories go to `engine-gates/<model>/` beside the test assembly.

## Gates

| Gate | Variant | Baseline | Asserted on CpuOps | Format v3 field |
| --- | --- | --- | --- | --- |
| Chunked prefill | prefill chunks capped at 1, 5 and 16 tokens (`MaxChunkTokens`), each below the prompt length | one call | same tokens; roost-tiny: bit-identical logits | `sections.chunked_prefill["<case>/<chunk>"]` |
| Cached vs uncached | `shared-b` after `shared-a` with the prefix cache on (32 prompt tokens from the cache; on the hybrid model one snapshot restore), decoding from the KV cache and recurrent state | a fresh, uncached one-call prefill of the prompt plus the tokens before each step | every fresh prefill selects the cached token; roost-tiny: difference 0 | `sections.cached_vs_uncached` |
| Prefix cache on vs off | every case after the primer, cache on | each alone, cache off | same tokens; roost-tiny: bit-identical | (test only) |
| Recurrent restore | `shared-b` after `shared-a`, cache on: its state at the shared boundary is a restored snapshot | `shared-b` alone, cache on (same chunk boundaries, state computed) | same tokens and bit-identical logits (hybrid only) | folded into `sections.cached_vs_uncached` of the hybrid run |
| Concurrency | every case in one wave, up to 4 rows per step, 1 and 2 plans in flight | each alone | same tokens and bit-identical logits | `sections.concurrency.results` |
| Preemption (recompute) | every case in one wave on a 6-block pool; the planner preempts once | each alone, uninterrupted | same tokens; roost-tiny: bit-identical | folded into `sections.concurrency.results` |
| Streaming | `GetStreamingCompletionAsync` through `RoostChatClient` | `GetCompletionAsync`, and the tokenizer's decode of the emitted ids | same text, same ids, text equals decode; ids equal the engine baseline; usage | `cases.<id>.stream_*`, `cases.<id>.usage` |

Format v3 (contract owner: stream B2) has no section for preemption or for recurrent snapshot restore. Until it has one,
a case's `concurrency` flag is true only when its tokens equal the single, uninterrupted run in the batched run, the
pipelined batched run and the preempted run, and the hybrid run's `cached_vs_uncached` case is one that starts from a
restored snapshot. The separate checks are the tests in `EngineGateTests`.

The cases (`Support/GateCases.cs`) are five raw text prompts of 13 to 82 tokens under the 4 KB byte-level tokenizer of the
generation tests, chosen to encode inside the tiny models' 256 ids; two share a 32-token prefix (two attention blocks).
Neither tiny model has a stop token in its vocabulary, so every case ends at its length limit: the `eos` finish path is
not exercised here.

## Measured (CpuOps, this branch)

From the diagnostics of `EngineGateTests` (`MEASURED` lines) and the sample run in
`evidence/pipeline/tests/data/engine-gates/`:

- roost-tiny: every gate is bit-identical to its baseline (largest log-probability difference 0).
- roost-tiny-hybrid: concurrency, preemption, prefix cache, recurrent restore and cached vs uncached are bit-identical.
  Chunked prefill is bit-identical at chunks of 5 and 16 tokens for every case and at 1 token for every case except
  `long` (82 tokens), where the step-0 log-probabilities differ by up to 0.155 and later steps by up to 0.534; the tokens
  are the same. Only the tokens are asserted for the hybrid model outside batching.

## Status

| Capability | roost-tiny on CpuOps | roost-tiny-hybrid on CpuOps | GPU (any SM) |
| --- | --- | --- | --- |
| Chunked vs one-call gate | supported (CI) | supported (CI) | untested |
| Cached vs uncached gate | supported (CI) | supported (CI) | untested |
| Concurrency gate | supported (CI) | supported (CI) | untested |
| Preemption gate | supported (CI) | supported (CI) | untested |
| Recurrent restore gate | not applicable (no recurrent group) | supported (CI) | untested |
| Streaming gate | supported (CI) | supported (CI) | untested |

"Supported" here means the harness runs and asserts in CI on the CPU reference backend; it says nothing about the GPU
worker. No GPU run of these gates has been made: the GPU half of R-039 (the same gates against the eager worker on the
RTX 3090, through `tools/gpu-lease`, with evidence) is open. Nothing here is fidelity evidence for a real model.

## Open

- GPU runs with evidence (R-039's second half).
- Format v3 sections for preemption and recurrent restore, and an evaluator check that reads `chunked_prefill`
  (`evaluate_generation` reports it `not_run`): stream B2's contract and evaluator.
- No `step0.f32` sidecar is written (the whole step-0 vector), so the evaluator's full-vocabulary step-0 checks are
  `not_run` on these rows.

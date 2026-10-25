# R-014: the GPU worker against the proposed Qwen3.5-0.8B fixtures, and proposed tolerances for `cuda-bf16-eager`

Status: **proposed**, for the maintainer. Nothing here is binding. The values are in `r014-gpu-tolerances.toml`.

## What ran

| | |
| --- | --- |
| Reference | The R-014 fixtures (`r014-fixtures.toml`): transformers 5.20.1, float32, CPU, five chat cases, 24 greedy tokens each. The bytes at `/home/mark/Projects/roost-evidence/R-014/ref` match `SHA256SUMS` (12 of 12 files). |
| Candidate | The eager worker as R-025 runs it: `EagerStepExecutor`, `Qwen35EagerModel` over `CudaOps`, bf16 weights and activations, f32 recurrent state, f32 logits. It is teacher-forced: the prompt in one forward, then the reference's first 23 generated tokens, one per forward (24 forwards per case). |
| Capture | `tests/Starlings.Roost.Worker.Tests.Cuda capture` (`CaptureGpu.cs`, new). It records the probe's op outputs in the R-011 layout that R-022's CPU recorder writes, with the same stage order: embeddings, every second Residual output as `layer.i`, the 49th RmsNorm as `final_norm`, and the recurrent and conv state of slot 0 for each Gated DeltaNet layer. |
| Comparison | `roost-evidence evaluate-layers` (`evidence/pipeline` at `d6c6fe6`, with PR #134's relative and cosine metrics) for hidden states, recurrent state and conv state. Step-0 log-probabilities and greedy margins are compared with the pipeline's `metrics.compare` (mean-centred for step 0, as the generation evaluator does). |
| Devices | RTX 3090 (SM 86), library `fdf421a9…`. RTX 4060 Ti (SM 89), library `e4b8798c…`. These are the manifest's R-034 build. Leases through `tools/gpu-lease`, stream B3. |
| Checkpoint | The locked snapshot. All 13 files match `evidence/registry/qwen3.5-0.8b.lock.toml`. |

The evaluator reports, summary, per-SM run records and analysis scripts are committed in [`docs/m0/r014-gpu/`](../../docs/m0/r014-gpu/README.md), with SHA-256 digests of the GPU captures. The large captures stay outside Git, under `/home/mark/Projects/roost-evidence/R-014/`:

- captures in `gpu-sm86/` and `gpu-sm89/` (`layers/*.safetensors`, `step0/*.f32`, `candidate.jsonl`, `run.json`, `capture.log`);
- evaluator reports in `eval/layers-sm86.json` and `eval/layers-sm89.json`, and the summary in `eval/summary.json`.

The evaluator ran with a measurement-only tolerance file (every bound loose), so its report holds the measured values. Every check
reported `passed`; no value was non-finite.

**Conv state.** Roost's slot keeps 3 inputs per channel and the reference keeps 4. The capture builds column 0 from the conv input,
which is the `in_proj_qkv` Gemm output, for a forward of 4 or more tokens, and from the previous forward otherwise, as R-022's CPU
recorder does. On the prompt forward (4 or more tokens) the harness checked that columns 1 to 3 equal the last rows of that Gemm
output; the check held for every case on both SMs. On the 23 decode forwards there is no such check: column 0 comes from the previous
forward's capture, unchecked.

## Results (worst over 5 cases x 24 forwards)

| Metric | SM 86 | SM 89 | Proposed bound |
| --- | --- | --- | --- |
| hidden, row-relative L2 | 0.0551 (final norm; decoder layers 0.0541) | 0.0457 | 0.1 |
| hidden, cosine (min) | 0.99848 | 0.99897 | 0.995 |
| hidden, max abs / max ref per tensor | 0.0668 | 0.0668 | 0.12 |
| hidden, max abs | 0.738 (final norm; decoder layers 0.132) | 0.923 (decoder layers 0.105) | 1.5 (absolute alternative) |
| recurrent state, relative L2 | 0.0148 | 0.0141 | 0.03 |
| recurrent state, cosine (min) | 0.99989 | 0.99990 | 0.9995 |
| recurrent state, max abs / max ref | 0.0215 | 0.0338 | 0.06 |
| recurrent state, max abs | 0.0506 | 0.0704 | 0.12 (absolute alternative) |
| conv state, relative L2 | 0.0108 | 0.0111 | 0.02 |
| conv state, cosine (min) | 0.99994 | 0.99994 | 0.9997 |
| conv state, max abs / max ref | 0.0127 | 0.0143 | 0.03 |
| conv state, max abs | 0.284 | 0.250 | 0.45 (absolute alternative) |
| step-0 log-prob, relative L2 (centred) | 0.0163 | 0.0150 | 0.03 |
| step-0 log-prob, cosine (centred, min) | 0.99987 | 0.99989 | 0.9995 |
| step-0 log-prob, max abs / max ref | 0.0093 | 0.0059 | 0.02 |
| step-0 log-prob, max abs (whole vocabulary) | 0.300 | 0.210 | 0.5 (absolute alternative) |
| step-0 log-prob, max abs over the reference top 5 | 0.0712 | 0.0815 | 0.15 |
| step-0 argmax equal | 5 of 5 | 5 of 5 | equal |
| forwards whose argmax equals the reference greedy token | 120 of 120 | 119 of 120 | near-tie rule below |
| largest margin error where the argmax agrees | 0.121 | 0.137 | |

Rule for the bounds: about 1.5 to 2 times the worst value over both SMs, rounded to a short number. A cosine bound leaves 3 to 5 times
the measured `1 - cosine`. The reports carry the values per forward and layer. This proposal uses only the maxima and does not
analyse how the state error evolves over the 24 forwards.

Compared with the CPU reference profile (`r022-tolerances.toml`, `CpuOps` in double with bf16 storage): the GPU's hidden-state error
is of the same order. The GPU's decoder-layer max abs is 0.132 against R-022's 0.138, and its row-relative L2 is 0.055 against
R-022's 0.067. So the GPU kernels add little beyond the bf16 storage rounding that both profiles share. The step-0 maximum is larger on
the GPU (0.300 against 0.257), in tail tokens; the top-5 maximum is smaller (0.082 against 0.159).

## The near-tie rule

The one greedy disagreement is on SM 89, case `code`, forward 23: the GPU picks 13 and the reference 5199. The reference's top-2 margin
there is 0.0096, the smallest of all 120 forwards. The GPU's own margin is 0.0034 the other way. SM 86 agrees at that step.

Where the argmax agrees, the GPU's top-2 margin differs from the reference's by at most 0.137 (SM 89; SM 86 0.121). A step whose
reference margin is above that error cannot flip, so the bound must sit above about 0.14.

Proposed: `exact_except_listed_margin_below_0.2`, compared teacher-forced, which is option (a) with (b) of `r014-fixtures.md`. That is
the same bound the CPU profile proposes, and 1.46 times the largest measured margin error. It lists 7 of the 120 steps:

| Case | Forward | Reference margin |
| --- | --- | --- |
| capital | 15 | 0.0554 |
| system-haiku | 5, 7, 10 | 0.0711, 0.1256, 0.1663 |
| multi-turn | 9 | 0.0479 |
| code | 23, 17 | 0.0096, 0.1903 |

R-025 measured GPU against `CpuOps` logit errors up to 0.23. That is a different pair of numerics, but it suggests 0.2 is not
generous. If the maintainer wants more headroom, 0.25 lists 3 more steps (the counts per bound are in `r014-fixtures.md`).

## Caveats

- **Kernel library.** The libraries are the manifest's R-034 build. R-019 (#137) replaces them with a build whose only source change
  is `gemm.cu` refusing Gemm epilogues with C or bias, which no Qwen3.5 node uses. So the numbers should not change. This was not
  rerun against #137's libraries.
- **FA2 prefill.** The prompt forward goes through FA2 prefill, which R-019 marks experimental: racecheck hazards, and a sequence's
  result depends on the batch composition. With one sequence per step, as here, the composition question does not arise.
- **Coverage.** Five cases, one sequence per step, prompts of at most 134 tokens, so prefill is never split across steps. Batching and
  chunked prefill are not covered.
- **Code in another stream's path.** The capture code is new test code in `tests/Starlings.Roost.Worker.Tests.Cuda`, which belongs to
  stream F. It reuses R-022's `LayerCaptureFile.cs` (stream E's test project) by linking it. No product code changed.

## Reproduce

```sh
tests/Starlings.Roost.Worker.Tests.Cuda/run-capture.sh <snapshot> <dir with libroost_kernels_sm<N>.so> <R-014 ref dir> <out dir> [3090|4060]
# with ROOST_R014_TOLERANCES pointing at a JSON copy of [tolerances] (or a measurement-only file)
```

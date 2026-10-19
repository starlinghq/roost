# R-022: Qwen3.5-0.8B end to end on the CPU reference, per layer against transformers fp32

Streams E and D, M0. Development evidence, not a fixture and not the M0 gate: it checks that the generated Qwen3.5-0.8B forward
(`Qwen35Model.Forward<TOps>`, R-035) computes the checkpoint's function, and it measures what the proposed tolerances in
[`evidence/proposals/r022-tolerances.toml`](../../evidence/proposals/r022-tolerances.toml) are chosen from. Run artifacts (captures,
step-0 vectors) stay outside Git; the summaries, gate reports and digests are in [`r022/`](r022/).

## Result

1. **R-022 found a definition defect, and this branch fixes it.** The full-attention node left `PagedAttentionArgs.WindowLeft`
   unbound. An unbound field is 0, and `CpuOps.PagedAttention` reads a non-negative `WindowLeft` as a sliding window, so every query
   attended only to itself. Position 0 (whose only key is itself) and all 18 Gated DeltaNet layers were unaffected, which is why the
   R-035 smoke test passed; the per-layer comparison put the first divergence at layer 3 (the first attention layer), position 1,
   in every case. The definition now binds `WindowLeft=-1`, and a definition test checks it on all six attention nodes.
2. **With the fix, every stage agrees with transformers to bfloat16 storage precision.** Over 5 chat prompts x 24 forwards (prefill
   plus 23 decode steps), teacher-forced: hidden states of every layer at most 0.138 (layer.23) and 1.33 at the final norm, worst row
   relative L2 0.067; recurrent state at most 0.073 with no growth over 24 forwards; convolution state at most 0.306; step-0
   log-probabilities at most 0.257 over the whole vocabulary and 0.159 over the reference's top 5; the argmax of all 120 forwards
   equals the reference's greedy token, including reference margins of 0.0096 and 0.048.
3. **The remaining difference is storage precision, not the definition.** The definition stores activations, K/V and convolution
   state as bfloat16 (`[RoostModel(Dtype = "BF16")]`); `CpuOps` computes each op in double and rounds once into that type. A
   double-precision recomputation of an attention layer with the same bfloat16 roundings, fed the candidate's captured input,
   reproduces the candidate's output to 0.0021 row-relative (and the exact recomputation reproduces transformers to 1e-6)
   ([`r022/emulate-bf16.txt`](r022/emulate-bf16.txt)). So the "CPU fp64 reference" profile measures bfloat16 storage rounding, not
   fp64 against fp32 accumulation.
4. **Multimodal rope reduces to plain rotary embedding for text-only prompts** (the R-035 README asked R-022 to check it): layers 3,
   7, 11, 15, 19 and 23 agree like the linear layers around them at every prefill position and decode step.
5. **The evaluator's single hidden-state bound cannot attribute a defect to its layer** when the bound also has to pass the final
   norm (10x the layers' scale): with the proposed 2.0, the window defect is attributed to layers 18 to 23, not layer 3. A
   row-relative bound of 0.1 attributes it to layer 3 in all five cases and passes the fixed definition. This is a request to stream
   B2 (below); the proposal still gives the single value the evaluator reads today.
6. **Two smaller fixes in the definition.** `LmHead` writes float32 logits (contract C3), but the definition declared the `logits` value
   with the model dtype (bf16), so `Output_logits` reported BF16 and the arena reserved half the bytes the op writes. The value is now
   `F32`; `ForwardLayout.PerTokenBytes` rises from 498688 to 995328 because the generator sizes the logits per token of capacity.
   And the smoke test's regression guard was built on the window defect (" the" won); it now expects " Paris" first and " the"
   second, as transformers gives them (14.871 and 14.587; `CpuOps` 14.875 and 14.596).

## What was compared

| | Reference | Candidate |
| --- | --- | --- |
| Code | `roost-evidence reference-run --capture-layers --state-forwards 24` (R-011, `evidence/pipeline`) | `Qwen35CpuParityRun.TeacherForcedCapture` (`tests/Starlings.Roost.Models.Qwen35.Tests`): `Qwen35Model.Forward<RecordingOps>`, which runs every op on `CpuOps` and copies out each stage as it is produced |
| Numerics | transformers 5.20.1, torch 2.9.1+cpu, float32, CPU, PyTorch fallbacks for the Gated DeltaNet kernels | double per op; bf16 weights (decoder, q/k and final norm slots hold bf16(1 + w)); bf16 activations, K/V and conv state; f32 recurrent state; f32 logits |
| Weights | `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` (the locked snapshot; SHA-256 of config, tokenizer, template and safetensors equal to `evidence/registry/qwen3.5-0.8b.lock.toml`) | the same file, memory-mapped |
| Tokens | greedy, 24 new tokens per case, no case stopped early | teacher-forced on the reference's prompt ids and its first 23 generated tokens |
| Captured | per R-011 layout: embeddings, 24 layer outputs, final norm at every position of every forward; recurrent [16,128,128] and conv [6144,4] state of the 18 linear layers after each of the 24 forwards; step-0 log-softmax | the same names and shapes |

The prompts are `R022PromptSet` (5 chat conversations: one user turn, a system prompt, a three-turn conversation, a code request, and
a 134-token note so that transformers' chunked delta-rule prefill crosses its 64-token chunk boundary), rendered by Starling's
`Qwen35ChatTemplate` (through `Qwen35RoostChatTemplate`, thinking off) and tokenised by `Starlings.Tokenizers` 0.3.0 over the
snapshot's `tokenizer.json`. The checkpoint's own Jinja template and the transformers tokenizer give the same text and ids for all
five ([`r022/template-check.txt`](r022/template-check.txt)). Prompt lengths: 25, 44, 69, 31, 134 tokens.

The convolution state: transformers keeps the last `conv_kernel_size` = 4 inputs per channel, the Roost slot keeps 3. The recorder
reads the slot before each `CausalConv1d` and emits the last 4 inputs of [slot, this step's inputs], so column 0 is reconstructed
from the candidate's own inputs, not stored by Roost; the analysis reports columns 1-3 and column 0 separately (they agree).

## Measurements (fixed definition)

Full tables: [`r022/analysis-baseline.md`](r022/analysis-baseline.md); gate report:
[`r022/evaluate-layers-baseline.json`](r022/evaluate-layers-baseline.json) (all four checks `passed` with the proposed values).

Hidden states, max abs over cases, forwards and positions, and the worst row's relative L2 (`||c - r|| / ||r||`):

| Stages | max abs | max rel L2 | Note |
| --- | --- | --- | --- |
| embeddings | 0 | 0 | the bf16 table, exact in both |
| layer.0 - layer.2 (linear) | 0.016 | 0.016 | |
| layer.3 - layer.14 | 0.052 | 0.026 | reference magnitude up to 4.6 |
| layer.15 - layer.22 | 0.071 | 0.063 | position 0 (`<\|im_start\|>`) sets the relative maximum; bf16 rounding from the exact layer input alone gives 0.0185 there |
| layer.23 | 0.138 | 0.052 | |
| final_norm | 1.33 | 0.067 | multi-turn, prefill position 33, channel 687, reference 5.73; stage maximum 42.2 |

| Forwards | hidden max abs | recurrent state max abs | conv state max abs |
| --- | --- | --- | --- |
| 0 (prefill) | 1.33 | 0.073 | 0.306 |
| 1 - 23 (decode), range of per-forward maxima | 0.227 - 0.407 | 0.065 - 0.072 | 0.135 - 0.306 |

The recurrent state does not grow across the 23 decode steps (0.073 after the prefill, 0.067 after forward 23): the per-step decay
`exp(g)` < 1 forgets old error at the rate it forgets old state. Per layer it is at most 0.017 in layers 0-2 and 0.073 in layer 16
(reference magnitude 5.1), relative L2 at most 0.016. The conv state is at most 0.306 (layer 21, reference magnitude 38.4), relative
L2 at most 0.011.

Step 0 and greedy: [`r022/analysis-baseline.md`](r022/analysis-baseline.md), last table. Whole-vocabulary log-probability error is set
by tail tokens (reference log-probability -17.9 to -26.6); top-5 at most 0.159 and top-1 at most 0.041. The top-1/top-2 gap differs
from the reference's by at most 0.137 over all 120 forwards, and by at most 0.103 over the 20 forwards whose reference gap is below 0.5.

## The defect and the negative runs

Each negative run changes one op inside `RecordingOps` (`ROOST_R022_PERTURB`), never the definition, on two cases (capital,
long-context) for perturbations and on all five for the window defect (the definition as R-035 left it). Gate reports are
`r022/evaluate-layers-*.json`; tables `r022/analysis-*.md`; relative attribution [`r022/relative-attribution.txt`](r022/relative-attribution.txt).

| Run | hidden max abs (worst rel L2) | `evaluate-layers` first divergent, hidden (bound 2.0) | first divergent, recurrent / conv state | row-relative 0.1 attributes | step-0 top-5 max abs; argmax differs |
| --- | --- | --- | --- | --- | --- |
| fixed definition | 1.33 (0.067) | none (passed) | none / none | none | 0.159; 0 of 120 |
| window defect (`WindowLeft` 0) | 27.1 (1.62) | forward 0, layer.18 to layer.23 by case: wrong layer | layer.4 / layer.4 | layer.3, position 2, all 5 cases | 8.19; 48 of 120 |
| attention output gate dropped | 58.5 (5.82) | forward 0, layer.3, position 0 | layer.4 / layer.4 | layer.3 | 19.5; 47 of 48 |
| `silu(z)` dropped from the gated norm | 41.5 (11.3) | forward 0, layer.0, position 2 | layer.1 / layer.1 | layer.0 | 34.9; 48 of 48 |
| every RMS norm eps 1e-5 instead of 1e-6 | 37.7 (1.41) | forward 0, layer.6, position 0 | layer.1 / layer.1 | layer.0 | 7.41; 24 of 48 |

Every defect fails all three state and hidden checks by a factor of 13 or more. The recurrent and conv checks attribute to the first
linear layer downstream of the defect (the state of layer k depends on layer k - 1's output). The wrong eps was meant as the subtle
case and is not subtle here: the gated norm normalises the delta-rule output per 128-wide head, whose mean square is small enough
that eps 1e-5 against 1e-6 changes it materially.

With a scratch hidden bound of 0.25 (which the fixed definition fails at the final norm) the window defect is attributed to layer.3
in four cases and layer.11 in one ([`r022/evaluate-layers-window0-defect-hidden-0.25.json`](r022/evaluate-layers-window0-defect-hidden-0.25.json)):
an absolute bound cannot both pass the final norm and attribute, and even a layers-only absolute bound misses short prompts.

## Requests

- **Stream B2 (evaluator):** add a row-relative hidden-state check (`||c - r|| / ||r||` per position; measured 0.067 for the fixed
  definition and at least 1.41 for every defect, so 0.1 separates them and attributes correctly), or at least a separate key for
  the final norm, so that attribution works with a bound that passes the final norm. No `evidence/pipeline` file is changed here.
- **Stream A (contract C3), contract-change request:** a zeroed `PagedAttentionArgs` is a one-token sliding window. Either make 0
  mean "no window" (a sliding window of 0 is never useful) or have the model compiler fill `WindowLeft = -1` when it is unbound. The
  ModelCompiler test models `HybridModel` and `Hybrid24Model` leave it unbound too; they build graphs only and are not run.
- **Stream F:** the logits value is now `F32` and the arena per token of capacity doubles (498688 to 995328 bytes) because the
  generator sizes `[selected, vocab]` values per token of capacity; the worker allocates from `ForwardLayout.ArenaBytes`.

## Reproduce

Run artifacts under `envs/R-022/` (outside the repository). Every command that loads the checkpoint or runs the fp64 forward ran
under the wave's heavy-work lock (`flock .../envs/heavy.lock -c '...'`).

```sh
SNAP=<snapshot of Qwen/Qwen3.5-0.8B at 2fc06364>; E=<run artifacts>
export UV_PROJECT_ENVIRONMENT=$E/venv
uv sync --project evidence/pipeline --locked --extra reference
dotnet build tests/Starlings.Roost.Models.Qwen35.Tests -c Release
# 1. prompt set: Starling's template and tokenizer (writes prompts.jsonl, rendered.jsonl)
ROOST_QWEN35_CHECKPOINT=$SNAP ROOST_R022_OUT=$E/run dotnet test --project tests/Starlings.Roost.Models.Qwen35.Tests -c Release --no-build -- --filter-method '*WritesPromptSet'
$E/venv/bin/python -I tests/Starlings.Roost.Models.Qwen35.Tests/Parity/scripts/check_template.py $SNAP $E/run
# 2. reference: transformers fp32 with hook capture (39 s, 5 x ~500 MB of captures)
OMP_NUM_THREADS=4 uv run --project evidence/pipeline --locked --extra reference roost-evidence reference-run $SNAP $E/run/prompts.jsonl --out $E/ref --capture-layers --state-forwards 24
# 3. candidate: CpuOps, teacher-forced (about 20 minutes for 5 cases); ROOST_R022_PERTURB and ROOST_R022_CASES for negative runs
ROOST_QWEN35_CHECKPOINT=$SNAP ROOST_R022_REFERENCE=$E/ref ROOST_R022_OUT=$E/cand dotnet test --project tests/Starlings.Roost.Models.Qwen35.Tests -c Release --no-build -- --filter-method '*TeacherForcedCapture'
# 4. the gate (tolerances.json rendered from the proposal's [tolerances], command in the .toml) and the distributions
uv run --project evidence/pipeline --locked roost-evidence evaluate-layers $E/cand --reference $E/ref/reference.jsonl --reference-captures $E/ref --tolerances $E/tolerances.json --report report.json
$E/venv/bin/python -I tests/Starlings.Roost.Models.Qwen35.Tests/Parity/scripts/analyze_parity.py $E/ref $E/cand
$E/venv/bin/python -I tests/Starlings.Roost.Models.Qwen35.Tests/Parity/scripts/relative_attribution.py $E/ref $E/cand 0.1
$E/venv/bin/python -I tests/Starlings.Roost.Models.Qwen35.Tests/Parity/scripts/emulate_bf16_attention_layer.py $SNAP $E/ref $E/cand 15 capital
```

Revisions: branch `evidence/r022-cpu-parity` from `models/r035-qwen35` at `25c24c94febae08a053c15b5f50b4bd2a1be65ea`; .NET 10.0.12;
transformers 5.20.1, torch 2.9.1+cpu, safetensors 0.8.0 (`evidence/pipeline/uv.lock`); `Starlings.Models.Qwen35` 0.2.0,
`Starlings.Tokenizers` 0.3.0. Digests of the snapshot files, the reference and candidate captures, step-0 vectors and prompt files:
[`r022/sha256.txt`](r022/sha256.txt). Candidate capture digests are also in [`r022/candidate-run.json`](r022/candidate-run.json),
the reference environment in [`r022/reference-environment.json`](r022/reference-environment.json).

## Not done

- No GPU run; the bf16 GPU worker's tolerances (R-026) are pending a GPU measurement and are not extrapolated from these.
- Five prompts, 24 forwards each; longer decodes (hundreds of steps) were not run, so "no growth" is measured over 24 forwards only.
- No fp32-activation variant of the definition was run; the bf16 emulation of attention layers 3 and 15 stands in for it.

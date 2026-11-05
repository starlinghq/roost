# R-015: proposed Qwen3-1.7B reference fixtures

Status: **proposed**, for the maintainer. Nothing here is binding, and this proposal sets no tolerance. `r015-fixtures.toml` pins the fixture data by SHA-256.

## What the fixtures are

They are the transformers float32 CPU reference for the five `R022PromptSet` chat prompts, at the locked checkpoint `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`:
- greedy output ids (24 tokens per case; no case reached a stop id);
- step-0 log-probabilities (float32, log-softmax over the full 151936-entry vocabulary);
- per-layer hidden states at every position of every forward: the embeddings, all 28 layers and the final norm, 720 tensors per case.

**No recurrent state.** Every Qwen3-1.7B layer is `full_attention`, so there are no recurrent or convolution captures. The run used `--state-forwards 0`, and each capture was checked to hold zero state tensors. This is the one difference in content from R-014.

The reference depends only on transformers, the checkpoint and Starling's template and tokenizer. Roost has no Qwen3-1.7B model definition (`docs/capabilities.md`: `unsupported`), so no candidate has been compared with these fixtures and no error has been measured.

This covers the CPU fp32 reference only. Nothing here touches a GPU.

## Checkpoint verification

All 12 files in `evidence/registry/qwen3-1.7b.lock.toml` were downloaded at the locked revision and checked before use: none missing, none extra, each equal to the lock by size and computed SHA-256. That includes the two safetensors shards, which the lock records as Hub LFS digests that had not been hashed locally. The computed values equal those digests.

The run recorded `model: Qwen/Qwen3-1.7B` and the full revision in `environment.json`, which closes the `revision: null` gap R-014 noted. It loaded the verified snapshot from a local cache with `HF_HUB_OFFLINE=1`.

The checkpoint stores `lm_head.weight` separately in the second shard even though `tie_word_embeddings` is true. It is bit-identical to `model.embed_tokens.weight` (both bfloat16, [151936, 2048]), so tying loses nothing.

## Prompts and template

The conversations are `R022PromptSet.Cases`, compiled in by link from `tests/Starlings.Roost.Models.Qwen35.Tests/Parity/R022PromptSet.cs`, so they are the same source as R-014's. They are rendered by `Qwen3RoostChatTemplate.Instance.Render(messages, enableThinking: false)`, which calls Starling's `QwenChatTemplate` with `QwenChatCheckpoint.Qwen3` (`Starlings.Models.QwenChat` 0.5.0), and tokenised by `Tokenizer.FromFile(<snapshot>/tokenizer.json).Encode(text, false)` (`Starlings.Tokenizers` 0.3.4).

`QwenChatCheckpoint.Qwen3` reports `TemplateSha256` `a55ee1b1...` and `TokenizerSha256` `aeb13307...`, which equal the admission's digests. `check_template.py` found the text and the ids equal to the checkpoint's own Jinja template and the transformers tokenizer for all five cases.

Prompt lengths are 25, 44, 69, 31 and 133 tokens. `long-context` is 133 here against 134 for Qwen3.5-0.8B. R-022 made that case long to cross the Gated DeltaNet 64-token chunk boundary, which does not apply to this model.

## Reproducibility: the digests depend on the thread count

| Comparison | Files | Result |
| --- | --- | --- |
| `OMP_NUM_THREADS=4`, run twice | every data file | byte-identical |
| 4 threads against 2 threads | `reference.jsonl`, all `step0/*.f32`, all `layers/*.safetensors` | **differ** |

Between 4 and 2 threads, on the same host and the same day:
- greedy ids are equal in all five cases;
- top-2 margins differ by at most 4.6e-05;
- step-0 log-probabilities differ by at most 4.2e-05;
- hidden states differ by at most 8.5e-03 absolute (always at `forward0/layer.27`), and at most 2.2e-06 in relative L2 over a tensor.

So a SHA-256 here identifies one exact set of bytes. It proves the fixture was not altered; it does not prove a rerun was correct. A rerun reproduces these digests only with 4 threads, and that is shown on this host only (Intel Core i7-5820K). Another CPU or a rebuilt torch wheel was not tried, and could well differ. Gates should fetch by digest and then compare numerically; they should not expect a fresh run to hash the same. The thread count is not in `environment.json`, so it is recorded here and in the TOML.

The differences above are an observed noise floor for this reference between thread counts. They are not a proposed tolerance.

## Where the bytes should live (maintainer decision)

The data is 106 MB (105,864,041 bytes), far smaller than R-014's 2.5 GB because there is no recurrent state, but it is still not for Git. It is on the development host at `/home/mark/Projects/roost-evidence/R-015/ref`, with `SHA256SUMS` beside that directory. The weights are in `/home/mark/Projects/roost-evidence/R-015/hf` and are not part of the fixture set.

Proposal: handle it the way R-014 proposes, as assets of a GitHub release named for the fixture set (for example `fixtures-qwen3-1.7b-r015`), with `evidence/protected/` carrying only the digest table. This should follow whatever the maintainer decides for R-014, so the two sets are stored alike.

## Near ties

No greedy step in any case has a top-2 margin below 0.1. The smallest is 0.1136. The smallest margin per case:

| Case | Step | Margin |
| --- | --- | --- |
| system-haiku | 2 | 0.1136 |
| long-context | 7 | 0.2036 |
| code | 0 | 0.2616 |
| capital | 14 | 0.2663 |
| multi-turn | 7 | 0.5503 |

How many of each case's 24 greedy steps fall below a given margin bound:

| Case | < 0.1 | < 0.2 | < 0.5 | < 1.0 |
| --- | --- | --- | --- | --- |
| capital | 0 | 0 | 1 | 2 |
| system-haiku | 0 | 2 | 6 | 7 |
| multi-turn | 0 | 0 | 0 | 3 |
| code | 0 | 0 | 1 | 3 |
| long-context | 0 | 0 | 2 | 2 |

Margins are differences of log-probabilities, which equal logit differences. The `code` case is at risk from its first token (step 0, margin 0.2616). No thread-count difference comes near any of these margins.

These counts say which steps are at risk for a given bound. They do not choose one: no candidate error exists for this model yet. The same options as R-014 apply: **(a)** `exact_except_listed_margin_below_<bound>` with the bound set per profile (decision 8), **(b)** teacher-forced comparison, checking the argmax only where the margin clears the bound, or **(c)** changing the cases.

## Tolerances per profile

None are proposed. No Qwen3-1.7B profile exists, so `[profiles]` in the TOML is deliberately empty. The maintainer sets the binding tolerances and the near-tie bound once a model definition exists and a candidate has been measured against these fixtures.

## How they were produced

Versions: transformers 5.20.1, torch 2.9.1+cpu, safetensors 0.8.0, huggingface_hub 2.2.0, Python 3.13.15, uv 0.12.19, .NET SDK 10.0.403, `roost-evidence` at `main` 6ba40c7, with `evidence/pipeline/uv.lock` unchanged.

```sh
E=/home/mark/Projects/roost-evidence/R-015
SNAP=$E/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export UV_PROJECT_ENVIRONMENT=$E/venv HF_HOME=$E/hf
uv sync --project evidence/pipeline --locked --extra reference
# 1. fetch each path in evidence/registry/qwen3-1.7b.lock.toml:
#    huggingface_hub.hf_hub_download("Qwen/Qwen3-1.7B", path, revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
# 2. compare the file list, each size and each SHA-256 in $SNAP with the lock; stop on any difference
# 3. render and tokenise as under "Prompts and template", writing prompts.jsonl (id, prompt_ids, max_new_tokens 24)
#    and rendered.jsonl in the shape of Qwen35CpuParityRun.WritesPromptSet, into $E/prompts
# 4. cross-check against the checkpoint's Jinja template
$E/venv/bin/python -I tests/Starlings.Roost.Models.Qwen35.Tests/Parity/scripts/check_template.py $SNAP $E/prompts
# 5. the reference (46.5 s wall, 2 min 41 s CPU, 7.9 GB peak, on the shared host)
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run --project evidence/pipeline --locked --extra reference \
    roost-evidence reference-run Qwen/Qwen3-1.7B $E/prompts/prompts.jsonl \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --out $E/ref --capture-layers --state-forwards 0
```

Steps 1 to 3 used small throwaway programs that are not in Git, because no Roost command does them for this model; the calls that matter are stated above. Input digests: `prompts.jsonl` `4f5a2a24bd1d15c6050f872b2252cec3d12e0abc4aeedc98220fa40fccd5e924`, `rendered.jsonl` `3d142296e8b816cac116479a3a24829e9ec412ef803ef2574c68d207c1f735e3`.

Every file was then checked: `load_reference` and `load_reference_vectors` accept the run, `parse_capture` accepts every capture, each capture's digest, prompt ids and decode ids match `reference.jsonl`, and no value is non-finite.

## What is not verified

- No candidate has run against these fixtures, so `evaluate` and `evaluate-layers` have not compared anything real for this model.
- Reproducibility was tried on one host only, at 2 and 4 threads.
- The template check covers these five conversations, not the template in general.
- Admission stays `proposed`: no Starling package pins this revision.

## Open questions for the maintainer

1. Accept these five cases as the protected fixture set for Qwen3-1.7B, or ask for more (for example a prompt long enough that Roost's chunked prefill splits it across steps, the same question R-014 asks)?
2. Approve publishing the bytes as release assets, under which tag?
3. Should fixture regeneration be pinned to 4 threads, and should gates treat the digest as an integrity check only?
4. Choose the near-tie rule and its bound per profile, once a candidate exists.
5. Accept the revision in `evidence/registry/qwen3-1.7b.admission.toml`? It is the Hub's main on 2026-11-01, not a Starling pin.

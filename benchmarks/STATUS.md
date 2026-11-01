# R-047 status: partial (groundwork; CPU-tested only; no measurement)

R-047 builds the section 14.3 scorecard harness and competitor runners (build plan section 10). This change is
groundwork. Every capability below is `untested` against a real server: the harness has only run against the in-process
fake server in `tests/tools/benchmarks/test_bench_client.py`.

## Done

| Capability | Status | Evidence |
| --- | --- | --- |
| Metric math: TTFT, ITL, TPOT, latency percentiles, throughput, goodput under SLO, saturation and knee | untested (CPU unit tests on synthetic traces) | `tests/tools/benchmarks/test_bench_metrics.py` |
| Workload files W1 to W7 (`roost-bench-workload/0`), deterministic expansion per seed | untested (no server run) | `tests/tools/benchmarks/test_bench_workloads.py` |
| Streaming OpenAI-compatible client (chat, completions, embeddings), closed and open loop runner, CLI | untested (fake loopback server only) | `tests/tools/benchmarks/test_bench_client.py` |
| Proposed SLO file | proposed, pending decision 25 | `slo.toml` |
| Runner recipes for vLLM v0.31.0, SGLang and llama.cpp, with the dtype and TF32 policy | draft; image digests, SGLang version and llama.cpp commit are TODO-maintainer placeholders | `recipes/` |

## Open

- **Raw-results schema.** The acceptance criterion asks for a Roost raw-results schema in `evidence/schemas/` that follows
  Starling's published `starlings-bench-raw/1` design (build plan section 2, "adds a generation `kind` to the raw
  schema"). That design was not available to this change, so no schema was written; inventing one would fork from
  Starling. Until it exists, `trace.jsonl` and `summary.json` use the interim `roost-bench-trace/0` layout, which no gate
  reads and which is not for publication. `evidence/schemas/` is also owned by streams B2 and B3 (section 4), so the
  schema needs their agreement.
- **Decision 25.** The SLOs, the workload set and shapes, the competitor pins and the gate thresholds are the maintainer's.
  Every value here is a proposal and the harness must not measure for a gate before the decision is recorded.
- **Competitor pins.** vLLM v0.31.0 and SGLang pinned by image digest in reference environments outside the repository;
  the recipes leave the digests as TODO-maintainer placeholders because they were not verified here.
- **Rented-pod runbook.** Not written.
- **Metrics that need server control.** Cold start to first token, host RSS, and recovery time and in-flight completion
  after an injected worker fault (section 14.3) need a launcher around the server process; not built.
- **W5 on Roost.** `w5-kev-decisions.toml` is the vLLM-equivalent form (one `max_tokens` 1 request per question). Roost's
  decision route (R-086) is not defined yet, so the Roost side of W5 cannot be driven.
- **Roost server coverage.** Roost's server accepts `response_format` text, `tool_choice` `"none"` and `logprobs` false only
  (src/Starlings.Roost.Server/STATUS.md), and has no `/v1/embeddings` route in R-037, so W4, W5 and W6 cannot run against
  Roost today.
- **ShareGPT dataset.** W1 defaults to synthetic prompts. A ShareGPT file can be passed at run time; which dataset, pinned
  by SHA-256, is part of decision 25.

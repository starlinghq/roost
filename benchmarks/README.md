# Roost scorecard harness

Groundwork for R-047 (stream I): a serving-load harness for the build plan's section 14.3 scorecard. It drives any
OpenAI-compatible endpoint (Roost's `roost` server, vLLM, SGLang, llama.cpp's `llama-server`) with the section 14.3
workloads and computes TTFT and ITL at p50 and p99, TPOT, end-to-end latency, throughput, goodput under an SLO and
saturation. What is and is not done is in [STATUS.md](STATUS.md).

Nothing here is a measurement. No harness run has been made against any GPU server, the SLO and workload shapes are
proposals pending decision 25 (build plan section 11), and no number from this harness may be published until the
profile has passed its fidelity gates (section 14.3, "Rules").

| Path | Holds |
| --- | --- |
| `bench.py` | Entry point: `run`, `summarize`, `validate` |
| `roost_bench/` | The harness: `metrics.py` (metric math), `workloads.py` (workload files), `client.py` (streaming client), `runner.py` (closed and open loop), `cli.py` |
| `workloads/` | W1 to W7 as `roost-bench-workload/0` TOML files |
| `slo.toml` | The proposed goodput SLO (p99 TTFT at most 1 s, p99 ITL at most 50 ms), `status = "proposed"` |
| `recipes/` | Competitor runner recipes: [vLLM v0.31.0](recipes/vllm-v0.31.0.md), [SGLang](recipes/sglang.md), [llama.cpp](recipes/llama-cpp.md), and the shared [dtype and TF32 policy](recipes/README.md) |

The harness is standard-library Python (3.11 or later), so it runs on a client host or a rented pod with nothing
installed. It is tool-time code outside `src/` and outside `Roost.slnx`; its tests are in `tests/tools/benchmarks/` and
run in CI with the other tool tests (`pytest -q tests/tools`).

## Run

```sh
python3 -I benchmarks/bench.py validate benchmarks/workloads/*.toml

python3 -I benchmarks/bench.py run \
    --workload benchmarks/workloads/w1-chat.toml \
    --base-url http://127.0.0.1:8000/v1 --model <served model name> \
    --label vllm-0.31.0 --slo benchmarks/slo.toml \
    --out <directory outside Git> [--levels 1,2,4] [--seed 0] [--warmup 8]

python3 -I benchmarks/bench.py summarize <out>/trace.jsonl --slo benchmarks/slo.toml
```

`run` writes `trace.jsonl` (a header line, then one line per request with its chunk arrival times) and `summary.json`
(per level, then saturation and the highest level that meets the SLO). Exit 0 when every request completed, 1 when any
failed. Both files are run artifacts: keep them out of Git. Their format, `roost-bench-trace/0`, is interim and is **not**
the Roost raw-results schema (see STATUS.md).

- `--extra-body '{"ignore_eos": true}'` merges backend-specific fields into generation requests. Nothing beyond the
  OpenAI fields is sent by default, because Roost's server refuses unknown fields. Use the same extra fields on every
  backend that accepts them, and record the choice; a run with `ignore_eos` on one backend and not another is not a
  comparison.
- `--embed-model` names the embedding model for W6's `/v1/embeddings` requests.
- `--dataset sharegpt.json` supplies prompts for a workload whose `prompt.source` is `sharegpt`. The dataset is not
  committed; record its SHA-256 next to the results.
- Run the client on cores the server does not use. `max_client_lag_s` in the summary is how far the open-loop client
  fell behind its Poisson schedule; a lag above a few milliseconds means the client, not the server, limited the run.

## Workloads

Each file gives the traffic shape, not a fixed request list: the same file, seed and dataset expand into the same
sessions on every backend. Synthetic prompts are common English words, about one token per word for the catalogue's BPE
tokenizers; the metrics use the server's `usage` counts, not the target lengths.

| File | Section 14.3 | Load | Shape (proposed) |
| --- | --- | --- | --- |
| `w1-chat.toml` | W1 chat (ShareGPT-style) | Poisson rate 1 to 32 req/s | prompt median 256, output median 192 (lognormal) |
| `w2-long-context-rag.toml` | W2 long-context RAG | Poisson rate 0.05 to 1 req/s | prompt 30k to 32k, output median 256 |
| `w3-agentic-multiturn.toml` | W3 agentic multi-turn | 1 to 32 concurrent sessions | 4 shared 4k-token prefixes, 6 turns carrying replies |
| `w4-structured-tools.toml` | W4 structured JSON and tools | Poisson rate 1 to 32 req/s | half strict `json_schema`, half one tool with `tool_choice` `required` |
| `w5-kev-decisions.toml` | W5 Kev decisions | 1 to 16 concurrent sessions | 3 to 16 parallel `max_tokens` 1 questions per shared context, with logprobs (the vLLM-equivalent form of bet X3) |
| `w6-mixed-embed-generate.toml` | W6 mixed embed-and-generate | Poisson rate 2 to 64 req/s | half `/v1/embeddings`, half chat |
| `w7-single-user.toml` | W7 single user, batch 1 | concurrency 1 | prompt median 512, 512 output tokens |

## Metric definitions

From `roost_bench/metrics.py`, which the tests in `tests/tools/benchmarks/test_bench_metrics.py` check on synthetic
traces:

- **TTFT**: send to the first SSE chunk that carries output (content, reasoning content, a tool-call delta or text).
  Role-only and usage-only chunks do not count.
- **ITL**: the gap between successive output-bearing chunks; a chunk the client marks as k tokens splits its gap into k
  samples. Run-level ITL percentiles pool every request's samples, as vLLM's `benchmark_serving` does. A server that
  sends several tokens per chunk without saying so shows a larger ITL and a smaller sample count; TPOT, from `usage`, is
  the cross-check.
- **TPOT**: (last chunk - first chunk) / (output tokens - 1), per request.
- **Percentiles**: linear interpolation between closest ranks (NumPy's default).
- **Throughput**: requests, output tokens and prompt plus output tokens per second, over the window from the first send
  to the last completion of the level.
- **Goodput**: completed requests per second whose own TTFT is within the TTFT SLO and whose own p99 ITL is within the
  ITL SLO (embeddings: latency within the TTFT SLO). `run_meets_slo` is true when the level's pooled p99 TTFT and p99
  ITL are within the SLO and no request failed; `max_goodput_level` is the highest such level. Which of the two is the
  scorecard's "goodput" is part of decision 25; the harness reports both.
- **Saturation**: the highest output-token throughput in the sweep, and the knee, the first level after which
  throughput grows by less than 5 %. `knee_level` is null when throughput was still growing at the last level.

Cold start to first token, host RSS and recovery after an injected worker fault (section 14.3 metrics) are not measured
by this client; they need control of the server process and are listed as open in STATUS.md.

# R-014 GPU evidence: eager worker against the reference fixtures

Committed evidence for `evidence/proposals/r014-gpu-tolerances.toml`. The proposal's `[measured]` values come from these files.

| Path | What it is |
| --- | --- |
| `eval/layers-sm86.json`, `layers-sm89.json` (+ `.out`) | Evaluator reports (`roost-evidence evaluate-layers`, row-relative and cosine metrics from #134) per SM |
| `eval/measure-tolerances.json` | The measurement-only tolerance file the evaluator ran with (every check reports, nothing gates) |
| `eval/summary.json`, `summary.txt` | `scripts/analyze.py` output: step-0 metrics, greedy agreement and margins, per-layer maxima |
| `sm86/`, `sm89/` | Per SM: `run.json` (device, driver, library and weights digests), `capture.log`, `candidate.jsonl` (teacher-forced greedy ids and margins) |
| `captures.sha256` | SHA-256 of the GPU step-0 and per-layer captures (about 5 GB, outside Git) |
| `scripts/` | The capture, evaluation and analysis commands as run |

The absolute paths in `run.json` and `scripts/*.sh` are on the development host, where the run artifacts live:
- the reference fixtures under `/home/mark/Projects/roost-evidence/R-014/ref`, pinned by `evidence/proposals/r014-fixtures.toml`;
- the GPU captures under `gpu-sm86/` and `gpu-sm89/` beside them.

The runs used the R-034 kernel libraries named in `run.json`, not the rebuilt libraries of #137 (see the proposal's caveats).

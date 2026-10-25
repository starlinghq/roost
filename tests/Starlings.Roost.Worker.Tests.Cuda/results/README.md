# GPU evidence of the eager worker (R-024, R-025)

Raw logs of `run-eager.sh` (R-025: greedy decode of Qwen3.5-0.8B on the eager worker, then the same prompts on `CpuOps` in double,
teacher-forced with the GPU's tokens) and of the R-024 weight check. The checkpoint is the locked snapshot
(`evidence/registry/qwen3.5-0.8b.lock.toml`, all 13 files verified by size and SHA-256 before the SM 89 run). Kernel libraries are the
ones `kernels/manifest.toml` names for each SM.

| SM | Device | Kernel library (sha256) | Files | Greedy tokens, GPU vs `CpuOps` | Worst step-0 logits max\|d\| |
| --- | --- | --- | --- | --- | --- |
| 86 | RTX 3090 | `fdf421a9…` | `r025-eager-decode-sm86.log`, `r025-cpu-compare-sm86.txt` (and `r025-cpu-window-zero-sm86.txt`, `r025-tolerance-proposal-sm86.txt`, `r024-weights-sm86.log`) | 24 of 24 | 0.2284 |
| 89 | RTX 4060 Ti | `e4b8798c…` | `r025-eager-decode-sm89.log`, `r025-cpu-compare-sm89.txt` | 24 of 24 | 0.1824 |

SM 89 run: from `plan/build-plan` at `51b06cf` (no change under `src/`, `kernels/` or the worker tests up to `d6c6fe6`). Command:
`ROOST_GPU_DEVICE=4060 tests/Starlings.Roost.Worker.Tests.Cuda/run-eager.sh <snapshot> <dir holding libroost_kernels_sm89.so> <scratch> sm89`.
The GPU phase passed: three prompts, eight greedy tokens each, the same tokens as on SM 86; the second run of the first prompt was
bit-identical; the two refused plans were refused with the expected codes. The CPU comparison agrees on all 24 greedy tokens. Its worst
step-0 logit difference is 0.1824 (step-0 cosine at least 0.99997) and its worst over all steps 0.2372; the SM 86 run's worst step-0
difference was 0.2284.
The manifest lists SM 89 as `untested`; `run-eager.sh` always passes `--allow-untested`, which lets the loader use it.

Caveats: the prompt steps go through the FA2 prefill path of `paged-attention-flashinfer-fa2`, which R-019 marks experimental
(racecheck hazards and batch-composition dependence, not proven benign). Forward times in the logs are eager and synchronised, on a
shared machine, and are not throughput figures.

# GPU evidence of the eager worker (R-024, R-025)

These are the raw logs of two harnesses:
- `run-eager.sh` (R-025): greedy decode of Qwen3.5-0.8B on the eager worker, then the same prompts on `CpuOps` in double, teacher-forced
  with the GPU's tokens;
- the R-024 weight check.

The checkpoint is the locked snapshot: all 13 files match `evidence/registry/qwen3.5-0.8b.lock.toml` by size and SHA-256. Each
`r025-eager-decode-*.log` starts with the leased GPU (name, driver, compute capability) and the CUDA versions, read inside the lease.
Then come the harness's identity line and the SHA-256 of the kernel library it loaded.

| SM | Device | Kernel library (sha256) | Files | Greedy tokens, GPU vs `CpuOps` | Worst step-0 logits max\|d\| | Min step-0 cosine |
| --- | --- | --- | --- | --- | --- | --- |
| 86 | RTX 3090, driver 610.57.04 | `d84e95be…` | `r025-eager-decode-sm86.log`, `r025-cpu-compare-sm86.txt` | 24 of 24 | 0.2284 | 0.9999458 |
| 89 | RTX 4060 Ti, driver 610.57.04 | `eba3d599…` | `r025-eager-decode-sm89.log`, `r025-cpu-compare-sm89.txt` | 24 of 24 | 0.1824 | 0.9999479 |

**Libraries.** Both runs use the R-019 libraries that `kernels/manifest.toml` names on this branch, which merges #137
(`conformance/r019`). The run tree was an export of `conformance/r019` at `97e5f71` with this branch's `run-eager.sh`. Its manifest (harness
identity `76420225…`) has the same SM 86 and SM 89 entries as the branch head. #137 later re-recorded SM 80, 90, 100 and 120 from the same
build. The worker code is the same as on `plan/build-plan`.

**Command.** `[ROOST_GPU_DEVICE=4060] tests/Starlings.Roost.Worker.Tests.Cuda/run-eager.sh <snapshot> <dir with libroost_kernels_sm<N>.so> <scratch> sm86|sm89`.

**Results.** Both GPU phases pass:
- the greedy tokens of all three prompts are identical on SM 86 and SM 89;
- the second run of the first prompt is bit-identical;
- the out-of-vocabulary and sampling plans are refused with codes 1002 and 1003.

The CPU comparison agrees on all 24 greedy tokens on each SM. The worst logit difference over all steps is 0.2335 on SM 86 and 0.2372 on SM 89. The worst
step-0 differences are in the table. Tokens and worst step-0 differences equal those of the earlier runs on the R-034 libraries (`fdf421a9…`, `e4b8798c…`;
in git history), as expected: the R-019 build differs only in a Gemm epilogue refusal that Qwen3.5 does not reach.

`r025-cpu-window-zero-sm86.txt`, `r025-tolerance-proposal-sm86.txt` and `r024-weights-sm86.log` are from the earlier SM 86 runs on the
R-034 library and were not repeated.

The manifest lists SM 89 as `untested`; `run-eager.sh` always passes `--allow-untested`, which lets the loader use it.

**Caveats.** The prompt steps go through the FA2 prefill path of `paged-attention-flashinfer-fa2`, which R-019 marks experimental:
racecheck hazards and batch-composition dependence, so it is not proven benign. Forward times in the logs are eager and synchronised,
on a shared machine, and are not throughput figures.

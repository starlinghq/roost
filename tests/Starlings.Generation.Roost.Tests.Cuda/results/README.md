# R-036 GPU runs

Raw output of `run.sh`: Qwen3.5-0.8B through `RoostChatClient` with the eager worker in a worker process.

- **Code:** branch `frontend/r036-generation-roost` at `2ab9ceb`, run from an export of that commit.
- **Checkpoint:** the locked R-013 snapshot. All 13 files match `evidence/registry/qwen3.5-0.8b.lock.toml` by size and SHA-256.
  The tokenizer is its `tokenizer.json`.
- **Kernel libraries:** the ones the branch's `kernels/manifest.toml` names, `libroost_kernels_sm86.so` `fdf421a9…` and
  `libroost_kernels_sm89.so` `e4b8798c…` (R-034 build).
- **Leases:** `tools/gpu-lease`, stream H.

| File | Device | Command | Result |
| --- | --- | --- | --- |
| `r036-chat-sm86.log` | RTX 3090 (SM 86) | `run.sh <snapshot> <kernels> <snapshot>/tokenizer.json sm86` | PASS: 3 prompts, a mid-stream cancel, and a request after the cancel |
| `r036-chat-sm89.log` | RTX 4060 Ti (SM 89) | the same executable and arguments as `run.sh`, leased with `--device 4060` (`run.sh` hardcodes 3090) | PASS |

The two devices give the same text except one word in the second prompt ("stretches endlessly across the horizon" on SM 86, "toward the
horizon" on SM 89). Greedy decode on different SMs need not be bit-identical, because the pinned cuBLASLt algorithms differ per SM. The
manifest lists SM 89 as `untested`, and `run.sh` passes `--allow-untested`. The times in the logs come from a shared machine and are not
throughput figures.

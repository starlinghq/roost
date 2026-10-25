# R-036 GPU runs

Raw output of `run.sh`: Qwen3.5-0.8B through `RoostChatClient` with the eager worker in a worker process. Each log records its own
provenance. `run.sh` prints the leased GPU's name, driver and compute capability, the CUDA versions and the exact command. The harness
then prints the SHA-256 of the kernel libraries, manifest and weights it loads.

- **Code.** The harness of `frontend/r036-generation-roost` at `2ab9ceb` plus this branch's changes to `Program.cs` and `run.sh`. The
  run tree was an export of `2ab9ceb` with those changes applied, identical to this branch's files.
- **Checkpoint.** The locked R-013 snapshot (13 of 13 files match `evidence/registry/qwen3.5-0.8b.lock.toml`). Its `tokenizer.json`
  is the tokenizer.
- **Kernel libraries.** The R-019 build that #137 records: SM 86 `d84e95be…`, SM 89 `eba3d599…`. The branch's own manifest still
  names the R-034 build, so the runs pass `ROOST_KERNEL_MANIFEST` with the manifest of `conformance/r019` at `97e5f71` (sha256
  `76420225…`). That manifest has the same SM 86 and SM 89 entries as #137's head. #137 later re-recorded SM 80, 90, 100 and 120 from
  the same build.
- **Leases.** `tools/gpu-lease`, stream H. Driver 610.57.04, CUDA 13.3.

| File | Device | Command | Result |
| --- | --- | --- | --- |
| `r036-chat-sm86.log` | RTX 3090 (SM 86) | `ROOST_KERNEL_MANIFEST=<that manifest> run.sh <snapshot> <kernels> <snapshot>/tokenizer.json sm86` | PASS |
| `r036-chat-sm89.log` | RTX 4060 Ti (SM 89) | `ROOST_GPU_DEVICE=4060 ROOST_KERNEL_MANIFEST=<that manifest> run.sh <snapshot> <kernels> <snapshot>/tokenizer.json sm89` | PASS |
| `r036-chat-sm86-roost-worker.log` | RTX 3090 (SM 86) | `ROOST_WORKER_EXE=<extracted>/runtimes/linux-x64/native/roost-worker run.sh <snapshot> <kernels> <snapshot>/tokenizer.json sm86-roost-worker` | PASS |

**The packaged worker (R-062).** `r036-chat-sm86-roost-worker.log` starts `roost-worker` through `RoostWorkerLaunch` instead of re-running the harness as the worker. The binary was taken from the package: `dotnet pack src/Starlings.Roost.Worker.Runtime` (version `0.0.1-drill.2`), then `unzip` of the `.nupkg`. The log records the binary's SHA-256 (`38cb0605…`).
- **Execute bit:** `unzip` dropped it, as NuGet extraction can, and the host restored it before starting the worker.
- **Manifest:** the worker read the one shipped beside it in the package (the default; nothing passed it a manifest path). The C8 handshake checked its digest (`35103ee8…`) against the profile's, which comes from this branch's `kernels/manifest.toml`.
- **Kernel library:** `--kernels` names the R-034 SM 86 library that manifest pins (`fdf421a9…`). The package ships no library until R-060 fetches it by digest.

Text, finishes and usage are identical to the harness-as-worker run. The worker was ready in 11.7 s, against 6.4 s for the harness-as-worker run the same day. These are single runs on a shared machine, not compared further.

`PASS` now requires all of the following:
- each of the three prompts streams the same text and finish reason as the non-streamed call;
- the mid-stream cancel ends the stream with `OperationCanceledException`;
- the request after the cancel returns text.

Both logs show all three. The finishes are `stop`, `stop`, `length` (usage 24+2, 19+23, 17+48).

**Cross-SM difference.** The two devices give the same text except one word in the second prompt: "stretches endlessly across the
horizon" on SM 86 and "toward the horizon" on SM 89. The same difference appeared with the R-034 libraries. Its cause is an inference,
not measured here: the SM 86 and SM 89 libraries pin cuBLASLt algorithms per SM, so their bf16 rounding differs, and a greedy step
with a small top-2 margin can flip. This is a smoke run, not a parity check: nothing compares the text with a reference. The R-014
near-tie rule (#141) applies to exactly this kind of flip.

The manifest lists SM 89 as `untested`, and `run.sh` passes `--allow-untested`. The times in the logs come from a shared machine and are
not throughput figures.

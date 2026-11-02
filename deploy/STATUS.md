# R-046 status: partial (CPU-side groundwork; no image built, no GPU evidence)

R-046's acceptance criteria: AOT binary plus kernel library in an image under 4 GB; `roost doctor`; cold-start budget per
stage; host RSS measured and its target proposed. This change lands what can be done without a GPU or Docker; the rest needs
a run through `tools/gpu-lease`.

| Item | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Container image (`deploy/Dockerfile`, `deploy/stage-image.sh`) | unsupported | — | Written, never built: no Docker on the development host, and no image is pushed. The base is pinned by a digest read from the registry API (README). `stage-image.sh` passes `bash -n`; it has not run end to end (it needs kernel release assets). |
| Image under 4 GB | unknown | — | Not measured. Known parts: base 1,705,907,451 compressed bytes (registry), `roost` 13,795,048 bytes, kernel libraries 22,290,048 bytes (README). |
| `roost doctor` | untested | `tests/Starlings.Roost.Server.Tests/DoctorTests.cs` | Ten typed checks (driver, NVML, GPU, manifest, SM, kernel digests, weights lock, tokenizer, disk, memory), each tested with fake probes; the real file, tokenizer and manifest probes are tested on the CPU. The AOT `roost` ran `doctor` on a GPU-less host (driver and GPU fail, manifest passes). Never run on a GPU host. |
| Cold-start stage timing | untested | `tests/Starlings.Roost.Server.Tests/ColdStartTests.cs` | Stages `runtime`, `arguments`, `profile`, `worker`, `listen`, logged and on `/metrics`. Tested on an injected clock and over the in-memory server; never timed on a real start. |
| Cold-start budget per stage | proposed | `deploy/README.md` | Proposals with no measurement behind them. |
| Host RSS | not measured | `deploy/README.md` | Procedure written; proposed target 256 MiB host VmHWM for Qwen3.5-0.8B, a proposal. |

## Still open

- Build the image from the release drill's exact artifacts (R-060, M2), measure its uncompressed size, and record it under
  `deploy/results/`.
- On the 3090 and the 4060 Ti through `tools/gpu-lease`: run `roost doctor` in the image, time a real cold start, and take
  the RSS readings; then replace the proposals with measured budgets for the maintainer to accept.
- Sub-stages of the `worker` stage (CUDA init, kernel load, weight streaming) are visible only inside the worker; reporting
  them to the host needs a C8 field (stream A).
- The decision 5 licence review must cover the cuBLASLt the image carries before any image is published.
- Kubernetes probes and drain are R-101.

# Deployables (stream J)

Roost's container image, `roost doctor`, cold-start staging and the host RSS procedure (build plan R-046). This directory
holds the CPU-side groundwork only: no image has been built or pushed, and nothing here has run on a GPU. Status per item is in
[STATUS.md](STATUS.md); the capability rows are in [docs/capabilities.md](../docs/capabilities.md).

| File | What it is |
| --- | --- |
| `Dockerfile` | The image: the Native AOT `roost` and `roost-worker`, the kernel manifest and the per-SM kernel libraries on NVIDIA's CUDA runtime image, pinned by digest. |
| `stage-image.sh` | Writes the image's build context: publishes both executables, checks the kernel libraries against `kernels/manifest.toml`, and lists every staged file's SHA-256. |

## The image

```sh
deploy/stage-image.sh --kernel-assets <dir with libroost_kernels_sm<N>.so>    # writes artifacts/image-context/
docker build -f deploy/Dockerfile -t roost:local artifacts/image-context
docker run --rm --gpus all roost:local                                       # roost doctor
docker run --rm --gpus all -p 8000:8000 -v /models:/models:ro roost:local \
    serve --model qwen3.5-0.8b --weights /models/model.safetensors --weights-sha256 <hex> \
    --tokenizer /models/tokenizer.json --host 0.0.0.0 --allow-non-loopback
```

Layout: everything is in `/opt/roost/` (`roost`, `roost-worker`, `roost-kernels.manifest.toml`, `libroost_kernels_sm{80,86,89,90,100,120}.so`,
`SHA256SUMS`). `roost` finds `roost-worker` beside itself, and the worker reads the manifest and loads the library from its own
directory (`RoostWorkerRuntime`), so no flag points at them. The image runs as uid 10001 and carries no weights: they are
mounted read-only. The server has no authentication, so listening on `0.0.0.0` stays an explicit `--allow-non-loopback`.

The build checks itself: `sha256sum --check` over the staged files, then `roost doctor --no-gpu`, which compares each kernel
library's SHA-256 with the manifest's. Either failure fails the build.

**Base image.** `nvidia/cuda:13.3.1-runtime-ubuntu24.04@sha256:63da350831208559df18c7b8f3e0d5d1c984eaa815cdde690aacd6606cd0cb11`,
the tag's multi-platform index digest as the Docker Hub registry returned it on 2026-11-02 (`Docker-Content-Digest` of
`GET /v2/nvidia/cuda/manifests/13.3.1-runtime-ubuntu24.04`). Its linux/amd64 image config sets `CUDA_VERSION=13.3.1` and
`NV_LIBCUBLAS_VERSION=13.6.0.2-1`, the cuBLASLt version that `kernels/manifest.toml` pins under `[[external_libraries]]`. The
13.3.0 runtime image ships cuBLAS 13.5.1.27, which is why 13.3.1 is the base. Raising the base changes the digest here, in
`Dockerfile`, and is checked against the manifest's cuBLASLt pin.

**Licence.** The kernel library links cuBLASLt and does not redistribute it; the image does carry it (in NVIDIA's base layers).
Publishing an image is therefore part of the decision 5 licence review of the CUDA toolkit EULA, which is open. Nothing here
publishes one.

### Image size

Acceptance criterion (build plan R-046): under 4 GB. Not measured: no image has been built. What is known:

| Part | Bytes | Source |
| --- | --- | --- |
| Base image, linux/amd64, compressed layers | 1,705,907,451 | Sum of `layers[].size` in the registry manifest `sha256:6155abf1…` (amd64 entry of the index above), 2026-11-02 |
| `roost` (Native AOT, linux-x64, Release) | 13,795,048 | One local `dotnet publish -c Release -r linux-x64` on 2026-11-02, symbols in a separate file; zero trim or AOT warnings |
| Kernel libraries, six SMs | 22,290,048 | Sum of the `bytes` fields in `kernels/manifest.toml` (the R-019 build; a release build has other bytes) |
| `roost-worker` | not measured | |

The uncompressed size, which the 4 GB criterion is read against here (`docker image inspect --format '{{.Size}}'`), is not
in the registry manifest and is unknown until a build. The base dominates. A smaller image is possible: `nvidia/cuda:*-base` plus
the `libcublas-13-3=13.6.0.2-1` package alone would drop NCCL, cuSPARSE, NPP and the other runtime libraries Roost does not
load today. That is a proposal, not done here, because an apt install pins less reproducibly than a digest.

Procedure, once Docker and the staged context exist:

```sh
docker image inspect roost:local --format '{{.Size}}'          # uncompressed bytes: the number compared with 4 GB
docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' roost:local   # per layer
docker save roost:local | gzip -1 | wc -c                       # compressed transfer size (approximate)
```

Record the three outputs, the base digest and the staged `SHA256SUMS` under `deploy/results/` with the date.

## `roost doctor`

`roost doctor` checks what `roost serve` needs before it starts a worker, one line per check, and exits 1 when any check
fails (0 otherwise, 2 for a usage error). It loads no CUDA or NVML library into its own process (decision 21): it reads files
and runs `nvidia-smi`. It reports; the worker's loader and the C8 handshake remain the gates that refuse a mismatched asset.

```text
roost doctor [--model qwen3.5-0.8b] [--weights <model.safetensors> --weights-sha256 <hex>] [--tokenizer <tokenizer.json>]
             [--kernels <dir>] [--worker <roost-worker>] [--manifest <manifest.toml>] [--allow-untested-sm] [--no-gpu]
             [--min-free-disk-gib <n>] [--min-free-memory-gib <n>]
```

| Check | Fails when | Warns when |
| --- | --- | --- |
| `driver` | no driver (`/proc/driver/nvidia/version`, else `nvidia-smi`), or its major is below the manifest's `toolkit.min_driver` | the version cannot be compared |
| `nvml` | — | `libnvidia-ml.so.1` (or `nvml.dll`) is not found on disk |
| `gpu` | `nvidia-smi --query-gpu` lists no GPU or cannot run | it lists some GPUs and then fails |
| `manifest` | the manifest (`--manifest`, else beside `--worker`, else beside the packaged worker) is missing or not C5-shaped | — |
| `sm` | a GPU's SM is `untested` without `--allow-untested-sm`, `unsupported`, or not in the manifest | the SM is `experimental`, or `untested` with the opt-in |
| `kernels` | the library for a GPU's SM (every library present, with `--no-gpu`) is missing, or its SHA-256 differs from the manifest's | — |
| `weights` | the file is missing or its SHA-256 differs from `--weights-sha256` (the profile's weights lock) | — |
| `tokenizer` | it does not load, or lacks the profile's stop tokens (`<\|im_end\|>`, `<\|endoftext\|>`) | — |
| `disk` | — | free space on the weights' file system is below `--min-free-disk-gib` (default 2, proposed) or unreadable |
| `memory` | — | `MemAvailable` is below `--min-free-memory-gib` (default 2, proposed) or unreadable |

A check whose input is not given (`--weights`, `--tokenizer`) is `skip`, never `pass`. `--no-gpu` skips `driver`, `nvml`,
`gpu` and `sm`; the image build uses it. `SM supported` is the only SM result that passes, and no SM is `supported` in the
manifest today.

## Cold start

`roost serve` times each start stage on a monotonic clock (the `runtime` stage from the OS process start time), writes `roost: cold start: <stage> <seconds> s` to standard error as
each ends and a `total` line once listening, and exports the stages on `GET /metrics` as
`roost_cold_start_stage_seconds{model_name,stage}` and their sum as `roost_cold_start_seconds` (gauges).

| Stage | From, to | Proposed budget |
| --- | --- | --- |
| `runtime` | process creation to `Main` (OS process start time) | 0.1 s |
| `arguments` | parsing and checking the command line | 0.01 s |
| `profile` | tokenizer, chat template, manifest SHA-256 | 0.5 s |
| `worker` | spawning `roost-worker` to a ready worker: CUDA context, kernel library load, weight streaming with digest check, C8 handshake | 2 s plus weights bytes / read throughput |
| `listen` | ASP.NET Core host built and bound | 0.2 s |

Every budget is a proposal with no Roost measurement behind it. The only related figures are from the S1 spike on the 3090
(`docs/m0/roost-s1-aot-cuda.md`): the CUDA primary context, not .NET, dominated a small AOT process's RSS, and the AOT
ASP.NET host published with zero trim warnings. Time to first token is not a stage: the server sends no warm-up request, and build plan
X9 measures process start to first token from outside, against vLLM's fastest supported start on the same model and disk. The
`worker` stage is one number from the host; its sub-stages are inside the worker and would need a C8 field to report them,
which is a stream A contract change. The R-046 GPU run measures these through `tools/gpu-lease` and replaces the proposals.

## Host RSS

Roost runs two processes per GPU: the host (`roost`: HTTP, tokenizer, engine) and the worker (`roost-worker`: CUDA context,
kernel library, weights staging). The host never creates a CUDA context (decision 21), so its RSS should sit near the S1 spike's
AOT ASP.NET process without CUDA: about 11 MiB runtime plus about 8 MiB web host (`docs/m0/roost-s1-aot-cuda.md`), plus the
tokenizer and the engine's host-side tables. The worker carries the CUDA context, which S1 measured at about 123 MiB beyond
`cuInit`.

Procedure (run through `tools/gpu-lease`, charged to stream J):

1. Start `roost serve` and wait for the `roost: cold start: total` line.
2. Read both processes: `grep -E 'VmRSS|VmHWM|RssAnon|RssFile' /proc/$(pgrep -x roost)/status` and the same for
   `pgrep -x roost-worker`. Record as "idle after start".
3. Send one chat completion; read again ("after first request").
4. Send 1,000 requests at the workload's concurrency (for example with the R-040 Python client or `benchmarks/` when it
   exists); read again, including `VmHWM` (peak).
5. Commit the readings, the command lines, the GPU, driver and the image or commit under `deploy/results/` with the date.

**Proposed targets** (proposals only, for the maintainer to set after the measurement): host `roost` VmHWM at most 256 MiB
after step 4 for Qwen3.5-0.8B; worker RSS reported, not targeted, because it is dominated by the driver's mappings.

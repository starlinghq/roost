# S1 (R-005): Native AOT with CUDA interop and an AOT ASP.NET host

Status: measured on the RTX 3090 only. Verdict: **go, provisional** until the SM 89 re-run on the RTX 4060 Ti (plan section 6 names that card; it has about 278 MiB free and was not attempted). Go for Native AOT on linux-x64 with CUDA interop and an AOT ASP.NET host. The JIT fallback is not needed on the evidence below.

Every number below comes from a raw log committed under `spikes/s1-aot-cuda/results/` (see Raw logs). The logs are from a re-run on 2026-10-17, taken because the first run's output was not saved; tables are the re-run's figures.

Spike code is in `spikes/s1-aot-cuda/`. It is throwaway evidence, outside `Roost.slnx`, and does not import the repository build props.

## Question (plan section 6)

Can a Native AOT linux-x64 binary load generated CUDA bindings (this spike uses hand-written bindings; generated bindings are deferred to R-023) and a C ABI `.so`, launch a kernel, capture and replay a CUDA graph, and serve SSE from an AOT ASP.NET minimal API? What are the P/Invoke and replay overheads and the baseline RSS?

## Setup

| Item | Value |
| --- | --- |
| Device | RTX 3090 (SM 86), through `tools/gpu-lease run --device 3090 --stream F`. The plan names the 4060 Ti, but it was occupied by other work at the time, so these numbers are **not** 4060 Ti (SM 89) numbers. Host-side overhead should transfer; kernel and RSS figures need a re-run on the 4060 Ti when it is free. The native library is built for `sm_86` only (`build-native.sh`); a 4060 Ti re-run must either add `-gencode arch=compute_89,code=sm_89` or state that it runs the sm_86 cubin (NVIDIA's binary compatibility rule says an sm_86 cubin runs on 8.9, but that was not shown here). |
| CUDA | toolkit 13.3.73, `cudart` 13.3 (`libcudart.so.13`), driver 610.57 (reported as 13030) |
| .NET | SDK 10.0.401, runtime 10.0.12, ILCompiler 10.0.12, `PublishAot`, `OptimizationPreference=Speed`, linux-x64 |
| Bindings | Hand-written `[LibraryImport]` (about 20 functions: cudart, driver `cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`). The full generator is R-023. |
| Native library | `native/s1_kernels.cu`, built by `build-native.sh`: `nvcc -arch=sm_86 -cudart shared -Xcompiler -fPIC -shared`. Exports `s1_noop`, `s1_add`, `s1_launch_inc`. |
| Host | one binary, `s1app`, with subcommands `pinvoke`, `graph`, `rss`, `serve`; each measurement is its own short `gpu-lease` run |

The `.so` is linked with **shared** cudart. nvcc's default is a static cudart; the AOT host binds `libcudart.so.13` itself, and two runtime copies would not share streams or graphs. Library names are resolved by `NativeLibrary.SetDllImportResolver`, so no CUDA path is baked into the binary.

Machine: shared, 12 cores, other agents active, so tails include noise from neighbours.

## Results

### Per-call P/Invoke overhead (1,000,000 calls each, one timestamp pair per call)

Log: `spikes/s1-aot-cuda/results/pinvoke.txt` (the run printed `AOT=True`). Stopwatch is 1 GHz. The timer pair itself costs p50 17 ns, p99 19 ns; the "minus timer" columns subtract the p50 baseline from both percentiles.

| Mechanism | p50 ns | p99 ns | p50 minus timer | p99 minus timer | Amortised, 10M-call loop |
| --- | --- | --- | --- | --- | --- |
| `LibraryImport` no-op | 22 | 37 | 5 | 20 | 6.6 ns |
| `LibraryImport` + `SuppressGCTransition` | 19 | 20 | 2 | 3 | 3.0 ns |
| `delegate* unmanaged` | 21 | 27 | 4 | 10 | 4.4 ns |
| `delegate* unmanaged` + `SuppressGCTransition` | 19 | 25 | 2 | 8 | 1.8 ns |
| `cudaGetDevice` (a real runtime call) | 76 | 86 | 59 | 69 | n/a |
| Eager kernel launch, host side (`s1_launch_inc`, 1024 elements, no sync; 100,000 calls) | 1,956 | 3,403 | 1,939 | 3,386 | n/a |

Reading: the managed-to-native transition costs about 7 ns amortised for `LibraryImport` (2 to 4 ns with `SuppressGCTransition` or function pointers). Per-call timing at this scale sits near the timer floor, so the amortised column is the trustworthy one and the p50/p99 columns bound it. The CUDA call itself (about 60 to 80 ns for a trivial runtime call, about 2 us for a kernel launch) dominates, so the P/Invoke boundary is not a cost driver for the worker. `SuppressGCTransition` is safe only for calls that cannot block or re-enter managed code; it is not needed to meet any budget here. Max values (up to tens of us) are scheduler or GC noise on a shared box (two gen0 GCs in the whole run).

### CUDA graph capture and replay (3090, 5,000 replays per row, one increment kernel per node)

Logs: `results/graph-256.txt` (`S1_ELEMS=256`) and `results/graph-1048576.txt` (`S1_ELEMS=1048576`). `graph replay+sync` is `cudaGraphLaunch` plus `cudaStreamSynchronize`, timed from the managed side. `eager` is k separate launches plus sync. Capture and instantiate used 20 repeats; with n=20, "p99" is the maximum, and it is the cold first iteration (up to 0.7 ms), so use p50 for steady state.

Tiny kernels (256 elements, so launch cost dominates):

| Kernels k | Capture + instantiate p50 | Replay + sync p50 / p99 | Graph launch call only p50 / p99 | Eager k launches + sync p50 / p99 |
| --- | --- | --- | --- | --- |
| 1 | 3.9 us | 4.3 / 5.6 us | 1.6 / 2.8 us | 4.9 / 6.9 us |
| 8 | 17 us | 10.8 / 12.6 us | 1.5 / 2.9 us | 18.1 / 29.9 us |
| 32 | 66 us | 32.1 / 35.6 us | 1.6 / 3.9 us | 64.2 / 81.4 us |
| 128 | 267 us | 117.3 / 124.3 us | 1.7 / 18.9 us | 257.4 / 496.9 us |

Larger kernels (1,048,576 elements, so the GPU is the bottleneck): replay is 12 to 14 percent faster than eager (k=32: 132.5 us vs 151.6 us p50; k=128: 520.7 us vs 595.0 us), as expected when kernel time dominates.

Correctness: in both runs the buffer element equalled the number of executed kernel launches (2,568,800 expected, `OK`), so replay executed every node and capture executed none.

Reading: for launch-bound work (small decode kernels), replay is about 2x faster than eager at k=32 and k=128 (1.7x at k=8) and the CPU cost of a replay call is about 1.6 us regardless of k. That is what F's graph plan needs, and it works unchanged from AOT. An earlier, unlogged run on a busier machine gave slower absolute figures (for example 34.8 us replay and 119.5 us eager at k=32); this report uses only the logged run.

### RSS split (AOT binary, `rss` subcommand, VmRSS at checkpoints)

Log: `results/rss.txt`, including the smaps dump.

| Checkpoint | VmRSS MiB | anon | file-backed |
| --- | --- | --- | --- |
| Process start (AOT, before CUDA) | 10.9 | 3.8 | 7.1 |
| After `cuInit` (driver loaded) | 17.9 | 5.8 | 12.1 |
| After `cudaFree(0)` (primary context) | 141.0 | 23.5 | 109.5 |
| After 64 MiB device alloc and a stream | 141.1 | 23.5 | 109.6 |
| After first kernel launch | 141.4 | 23.6 | 109.9 |
| After graph capture and 100 replays | 142.0 | 23.9 | 110.1 |
| After `GC.Collect()` | 143.1 | 24.5 | 110.6 |

By mapping at the end (MiB): `nvidiactl` mapping 61.3, `libnvidia-gpucomp` 20.9, `libcuda` 10.6, `[heap]` 9.6, anonymous 8.7, deleted zero-file mapping 8.0, `libnvidia-nvvm` 7.6, AOT image 5.9, `libnvidia-ptxjitcompiler` 4.3, other shared libraries 4.1, `nvidia-uvm` 2.0, `libcudart` 0.6, `libs1kernels` about 0.

Split: managed heap is 0.2 to 0.3 MiB in this binary. The runtime and AOT image account for roughly 11 MiB. **The CUDA context (driver libraries and the driver's own mappings) costs about 123 MiB beyond `cuInit` (130 MiB from process start) and is the dominant item**, and it is paid on first context creation, not by .NET. The "kernel images" bucket is about zero because the spike has one trivial SM 86 cubin; it says nothing about the real kernel library, which R-016 and later must measure. The GPU-side context (device memory) is not in this table.

### AOT ASP.NET host (`serve`, `WebApplication.CreateSlimBuilder`, Kestrel, SSE)

Logs: `results/serve-stderr.txt` (server stderr with RSS checkpoints and request log) and `results/serve-curl.txt` (curl output), produced by `serve-harness.sh`.

- Published `s1app` is 10.1 MB (symbols are in a separate 24.6 MB `.dbg` file). `results/ldd-s1app.txt` shows it links only libc and libm; all CUDA libraries are loaded at run time.
- `GET /sse?n=8` returned `Content-Type: text/event-stream` and eight events, each produced by a real GPU step (launch, sync, copy back; the value counted 1 to 8). The endpoint uses `TypedResults.ServerSentEvents` over an `IAsyncEnumerable<string>`.
- `GET /health` returns JSON through a source-generated `JsonSerializerContext`.
- Time to first byte for a 64-event stream: 0.80 ms (curl, loopback); total 335 ms (a 5 ms delay per event is in the endpoint, so this is not a throughput figure). One run, one request.
- RSS with the server: 11.1 MiB at start, 141.3 after the CUDA context, 148.1 after the web host is built, 149.6 once listening, 151.5 after 2 health requests, 1 + 200 `n=8` SSE requests (all 200 returned HTTP 200) and one `n=64` stream (3.4 MiB managed heap, 5.2 committed). The ASP.NET host adds about 8 MiB over the CUDA-only process, and the short load added about 2 MiB.

### Trimming and AOT warnings

Log: `results/publish.txt`. `dotnet publish -c Release -r linux-x64` with `PublishAot`, `IsAotCompatible`, `EnableTrimAnalyzer` and `EnableAotAnalyzer` produced **zero** IL2xxx, IL3xxx or compiler warnings (`ILLinkTreatWarningsAsErrors=false`, so any warning would have been reported; the log ends `0 Warning(s)`). That publish took 21 s wall clock on a build tree that already existed, with two build jobs. The reasons are the patterns chosen: `[LibraryImport]` (source-generated marshalling, no runtime IL stubs), `delegate* unmanaged` instead of `Marshal.GetDelegateForFunctionPointer`, `NativeLibrary.SetDllImportResolver`, and a `JsonSerializerContext`. The spike surface is small, so this is evidence for those patterns, not for the whole engine. Rules the real code should keep: no reflection-based JSON, no `DllImport` with non-blittable types, no `Marshal.GetDelegateForFunctionPointer`, no dynamic assembly loading.

## Verdict against the plan's criteria

- Load generated bindings and a C ABI `.so` from a Native AOT linux-x64 binary: **partly met**. Hand-written bindings and the C ABI `.so` load and run; generated bindings are not covered and are deferred to R-023.
- Launch a kernel: yes. Capture and replay a CUDA graph: yes, with the expected speed-up for launch-bound work.
- Serve SSE from an AOT ASP.NET minimal API with GPU work in the loop: yes.
- Overheads (3090): transition about 7 ns amortised, runtime call about 76 ns, kernel launch about 2 us, graph replay call about 1.6 us.
- Baseline RSS: about 11 MiB runtime, about 123 MiB CUDA context beyond `cuInit`, about 8 MiB ASP.NET host, and about 2 MiB after a short load; about 150 MiB total.
- Trimming warnings: none.
- The stop condition (AOT trimming or native loading blocks after two attempts) did not occur. One compile repair was needed (async/unsafe split) and no AOT repair.

## Limits and open items

- **Provisional on the 4060 Ti (SM 89) re-run**, which section 6 names and which was not done: the card had about 278 MiB free. Re-run `graph`, `rss`, `pinvoke` and the serve harness there when it has free memory. The harness is not unchanged: `build-native.sh` hard-codes `-arch=sm_86`, so the re-run must add an `sm_89` target (`-gencode`) or say it runs the sm_86 cubin, and label the kernel figures accordingly. Change the device to `--device 4060` in each lease command and in `serve-harness.sh`.
- Generated bindings (section 6) are not met; deferred to R-023.
- Path ownership: `spikes/s1-aot-cuda/` is not a path in plan sections 3 or 4 for streams F and H. The maintainer should confirm it or move it.
- No JIT baseline RSS or startup was measured, so the "JIT versus AOT" delta is not quantified here.
- The kernel library has one tiny cubin; kernel-image RSS for the real library is unmeasured.
- Throughput and long-run soak of the SSE path were not tested; it is a smoke test, not a load test.
- Replay latency includes a synchronise round trip; an async engine loop (stream F) will pay only the launch call.
- Single GPU, single process; no claim about multi-process or co-tenant behaviour.

## Raw logs

All under `spikes/s1-aot-cuda/results/`, from the 2026-10-17 re-run on the 3090 (one `gpu-lease` run per file, stream F; first line of each file is the UTC start time and the last line is the exit code):

| File | Produced by |
| --- | --- |
| `pinvoke.txt` | `publish/s1app pinvoke` |
| `graph-256.txt` | `S1_ELEMS=256 publish/s1app graph` |
| `graph-1048576.txt` | `S1_ELEMS=1048576 publish/s1app graph` |
| `rss.txt` | `publish/s1app rss` (includes the smaps dump) |
| `serve-stderr.txt`, `serve-curl.txt` | `serve-harness.sh` |
| `ldd-s1app.txt` | `ldd publish/s1app` |
| `publish.txt` | `dotnet publish` (warning count and wall time) |

## Reproduce

```sh
cd spikes/s1-aot-cuda
./build-native.sh
dotnet publish app/S1App.csproj -c Release -r linux-x64 -o publish/
cp build/libs1kernels.so publish/
# each measurement is its own short lease
S1_ELEMS=256 ../../tools/gpu-lease run --device 3090 --need-mib 500 --wait 3600 --stream F -- publish/s1app graph
S1_ELEMS=1048576 ../../tools/gpu-lease run --device 3090 --need-mib 500 --wait 3600 --stream F -- publish/s1app graph
../../tools/gpu-lease run --device 3090 --need-mib 500 --wait 3600 --stream F -- publish/s1app pinvoke
../../tools/gpu-lease run --device 3090 --need-mib 500 --wait 3600 --stream F -- publish/s1app rss
# serve: starts the server under one lease, curls /health and /sse, sends 200 more SSE requests, stops it
./serve-harness.sh   # override the address with S1_URL
```

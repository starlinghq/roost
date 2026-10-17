# S1 (R-005): Native AOT with CUDA interop and an AOT ASP.NET host

Status: measured. Verdict: **go** for Native AOT on linux-x64 with CUDA interop and an AOT ASP.NET host. The JIT fallback is not needed on the evidence below.

Spike code is in `spikes/s1-aot-cuda/`. It is throwaway evidence, outside `Roost.slnx`, and does not import the repository build props.

## Question (plan section 6)

Can a Native AOT linux-x64 binary load generated CUDA bindings and a C ABI `.so`, launch a kernel, capture and replay a CUDA graph, and serve SSE from an AOT ASP.NET minimal API? What are the P/Invoke and replay overheads and the baseline RSS?

## Setup

| Item | Value |
| --- | --- |
| Device | RTX 3090 (SM 86), through `tools/gpu-lease run --device 3090 --stream F`. The plan names the 4060 Ti, but it was occupied by other work at the time, so these numbers are **not** 4060 Ti (SM 89) numbers. Host-side overhead should transfer; kernel and RSS figures need a re-run on the 4060 Ti when it is free. |
| CUDA | toolkit 13.3.73, `cudart` 13.3 (`libcudart.so.13`), driver 610.57 (reported as 13030) |
| .NET | SDK 10.0.401, runtime 10.0.12, ILCompiler 10.0.12, `PublishAot`, `OptimizationPreference=Speed`, linux-x64 |
| Bindings | Hand-written `[LibraryImport]` (about 20 functions: cudart, driver `cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`). The full generator is R-023. |
| Native library | `native/s1_kernels.cu`, built by `build-native.sh`: `nvcc -arch=sm_86 -cudart shared -Xcompiler -fPIC -shared`. Exports `s1_noop`, `s1_add`, `s1_launch_inc`. |
| Host | one binary, `s1app`, with subcommands `pinvoke`, `graph`, `rss`, `serve`; each measurement is its own short `gpu-lease` run |

The `.so` is linked with **shared** cudart. nvcc's default is a static cudart; the AOT host binds `libcudart.so.13` itself, and two runtime copies would not share streams or graphs. Library names are resolved by `NativeLibrary.SetDllImportResolver`, so no CUDA path is baked into the binary.

Machine: shared, 12 cores, other agents active, so tails include noise from neighbours.

## Results

### Per-call P/Invoke overhead (1,000,000 calls each, one timestamp pair per call)

Stopwatch is 1 GHz. The timer pair itself costs p50 21 ns, p99 41 ns; the last two columns subtract the p50 baseline from both percentiles.

| Mechanism | p50 ns | p99 ns | p50 minus timer | p99 minus timer | Amortised, 10M-call loop |
| --- | --- | --- | --- | --- | --- |
| `LibraryImport` no-op | 25 | 67 | 4 | 46 | 9.8 ns |
| `LibraryImport` + `SuppressGCTransition` | 29 | 38 | 8 | 17 | 5.7 ns |
| `delegate* unmanaged` | 28 | 54 | 7 | 33 | 6.9 ns |
| `delegate* unmanaged` + `SuppressGCTransition` | 23 | 53 | 2 | 32 | 3.3 ns |
| `cudaGetDevice` (a real runtime call) | 115 | 272 | 94 | 251 | n/a |
| Eager kernel launch, host side (`s1_launch_inc`, 1024 elements, no sync) | 3,552 | 7,449 | 3,531 | 7,428 | n/a |

Reading: the managed-to-native transition costs about 10 ns amortised (3 to 7 ns with `SuppressGCTransition` or function pointers). Per-call timing at this scale sits near the timer floor, so the amortised column is the trustworthy one and the p50/p99 columns bound it. The CUDA call itself (about 100 ns for a trivial runtime call, about 3.5 us for a kernel launch) dominates, so the P/Invoke boundary is not a cost driver for the worker. `SuppressGCTransition` is safe only for calls that cannot block or re-enter managed code; it is not needed to meet any budget here. Max values (up to ms) are scheduler or GC noise on a shared box (one gen0 GC in the whole run).

### CUDA graph capture and replay (3090, 5,000 replays per row, one increment kernel per node)

`graph replay+sync` is `cudaGraphLaunch` plus `cudaStreamSynchronize`, timed from the managed side. `eager` is k separate launches plus sync. Capture and instantiate used 20 repeats; with n=20, "p99" is the maximum, and it is the cold first iteration (up to 4.5 ms), so use p50 for steady state.

Tiny kernels (256 elements, so launch cost dominates):

| Kernels k | Capture + instantiate p50 | Replay + sync p50 / p99 | Graph launch call only p50 / p99 | Eager k launches + sync p50 / p99 |
| --- | --- | --- | --- | --- |
| 1 | 7.9 us | 5.7 / 7.9 us | 2.6 / 3.8 us | 6.9 / 11.7 us |
| 8 | 35 us | 13.6 / 19.4 us | 2.7 / 15.1 us | 36.1 / 101.9 us |
| 32 | 111 us | 34.8 / 47.7 us | 3.0 / 22.1 us | 119.5 / 143.9 us |
| 128 | 511 us | 124 / 144 us | 3.1 / 31.6 us | 449 / 1,945 us |

Larger kernels (1,048,576 elements, so the GPU is the bottleneck): replay and eager are within 10 percent of each other (k=32: 138 us vs 156 us p50; k=128: 541 us vs 611 us), as expected when kernel time dominates.

Correctness: after every run the buffer element equalled the number of executed kernel launches (2,568,800, matching the expected count), so replay executed every node and capture executed none.

Reading: for launch-bound work (small decode kernels), replay is about 3.5x faster than eager at k=32 and the CPU cost of a replay call is about 3 us regardless of k. That is what F's graph plan needs, and it works unchanged from AOT.

### RSS split (AOT binary, `rss` subcommand, VmRSS at checkpoints)

| Checkpoint | VmRSS MiB | anon | file-backed |
| --- | --- | --- | --- |
| Process start (AOT, before CUDA) | 10.9 | 3.8 | 7.1 |
| After `cuInit` (driver loaded) | 17.6 | 5.8 | 11.8 |
| After `cudaFree(0)` (primary context) | 140.4 | 23.5 | 108.9 |
| After 64 MiB device alloc and a stream | 140.5 | 23.5 | 109.0 |
| After first kernel launch | 140.9 | 23.6 | 109.3 |
| After graph capture and 100 replays | 141.4 | 23.9 | 109.5 |
| After `GC.Collect()` | 142.7 | 24.5 | 110.2 |

By mapping at the end (MiB): `nvidiactl` mapping 61.3, `libnvidia-gpucomp` 20.6, `libcuda` 10.6, `[heap]` 9.6, anonymous 8.8, `libnvidia-nvvm` 7.6, AOT image 5.9, `libnvidia-ptxjitcompiler` 4.3, other shared libraries 4.1, `libcudart` 0.6, `libs1kernels` about 0.

Split: managed heap is 0.2 to 0.3 MiB in this binary. The runtime and AOT image account for roughly 11 MiB. **The CUDA context (driver libraries and the driver's own mappings) costs about 123 MiB and is the dominant item**, and it is paid on first context creation, not by .NET. The "kernel images" bucket is about zero because the spike has one trivial SM 86 cubin; it says nothing about the real kernel library, which R-016 and later must measure. The GPU-side context (device memory) is not in this table.

### AOT ASP.NET host (`serve`, `WebApplication.CreateSlimBuilder`, Kestrel, SSE)

- Published `s1app` is 10.1 MB (symbols stripped to a separate file). It links only libc and libm; all CUDA libraries are loaded at run time.
- `GET /sse?n=8` returned `Content-Type: text/event-stream` and eight events, each produced by a real GPU step (launch, sync, copy back; the value counted 1 to 8). The endpoint uses `TypedResults.ServerSentEvents` over an `IAsyncEnumerable<string>`.
- `GET /health` returns JSON through a source-generated `JsonSerializerContext`.
- Time to first byte for a 64-event stream: 0.83 ms (curl, loopback); total 347 ms (a 5 ms delay per event is in the endpoint, so this is not a throughput figure).
- RSS with the server: 10.5 MiB at start, 140.4 after the CUDA context, 147.9 after the web host is built, 149.1 once listening, 151.6 after 1 + 1 + 200 SSE requests (4.3 MiB managed heap, 5.7 committed). The ASP.NET host adds about 9 MiB over the CUDA-only process, and a short load added about 2.5 MiB.

### Trimming and AOT warnings

`dotnet publish -c Release -r linux-x64` with `PublishAot`, `IsAotCompatible`, `EnableTrimAnalyzer` and `EnableAotAnalyzer` produced **zero** IL2xxx, IL3xxx or compiler warnings (`ILLinkTreatWarningsAsErrors=false`, so any warning would have been reported). Publish took 28 s wall clock. The reasons are the patterns chosen: `[LibraryImport]` (source-generated marshalling, no runtime IL stubs), `delegate* unmanaged` instead of `Marshal.GetDelegateForFunctionPointer`, `NativeLibrary.SetDllImportResolver`, and a `JsonSerializerContext`. The spike surface is small, so this is evidence for those patterns, not for the whole engine. Rules the real code should keep: no reflection-based JSON, no `DllImport` with non-blittable types, no `Marshal.GetDelegateForFunctionPointer`, no dynamic assembly loading.

## Verdict against the plan's criteria

- Load generated-style bindings and a C ABI `.so` from a Native AOT linux-x64 binary: yes.
- Launch a kernel: yes. Capture and replay a CUDA graph: yes, with the expected speed-up for launch-bound work.
- Serve SSE from an AOT ASP.NET minimal API with GPU work in the loop: yes.
- Overheads: transition about 10 ns amortised, runtime call about 100 ns, kernel launch about 3.5 us, graph replay call about 3 us.
- Baseline RSS: about 11 MiB runtime, about 123 MiB CUDA context, about 9 MiB ASP.NET host, and about 2.5 MiB after a short load; 150 MiB total.
- Trimming warnings: none.
- The stop condition (AOT trimming or native loading blocks after two attempts) did not occur. One compile repair was needed (async/unsafe split) and no AOT repair.

## Limits and open items

- Not run on the 4060 Ti (SM 89). Re-run `graph`, `rss` and `pinvoke` there when it has free memory; the harness is unchanged.
- No JIT baseline RSS or startup was measured, so the "JIT versus AOT" delta is not quantified here.
- The kernel library has one tiny cubin; kernel-image RSS for the real library is unmeasured.
- Throughput and long-run soak of the SSE path were not tested; it is a smoke test, not a load test.
- Replay latency includes a synchronise round trip; an async engine loop (stream F) will pay only the launch call.
- Single GPU, single process; no claim about multi-process or co-tenant behaviour.

## Reproduce

```sh
cd spikes/s1-aot-cuda
./build-native.sh
dotnet publish app/S1App.csproj -c Release -r linux-x64 -o publish/
cp build/libs1kernels.so publish/
# each measurement is its own short lease
S1_ELEMS=256 ../../tools/gpu-lease run --device 3090 --need-mib 500 --wait 3600 --stream F -- publish/s1app graph
../../tools/gpu-lease run --device 3090 --need-mib 500 --wait 3600 --stream F -- publish/s1app pinvoke
../../tools/gpu-lease run --device 3090 --need-mib 500 --wait 3600 --stream F -- publish/s1app rss
# serve: start `publish/s1app serve` under a lease, then curl http://127.0.0.1:5199/sse?n=8 (override with S1_URL)
```

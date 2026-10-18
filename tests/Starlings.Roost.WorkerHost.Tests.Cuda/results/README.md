# R-069: per-step overhead of the .NET worker host (repeat of S8, decision 21)

Measured 2026-10-18 on the RTX 3090 (SM 86, CUDA 13.3, one `tools/gpu-lease` lease per run, stream F), .NET 10.0.12
Release build, shared box (i7-5820K, 12 threads, other agents building). The step is S8's: 4 KiB plan H2D from ring
memory, a spin kernel for 4500 us, 1 KiB D2H into ring memory, stream sync with the spin schedule, the segment
registered with `cudaHostRegister(portable)` (`native/roost_spin.cu`, a port of `spikes/s8-worker-process/s8.cu`).
The 4500 us is S8's derived Qwen3-1.7B batch-1 step, not a measured one (R-025 does not exist yet).

Modes: `direct` (the host thread calls the step, no rings), `inproc` (the real `WorkerRuntime` on a thread over the same
segment and rings), `proc` (the real `WorkerRuntime` in a child process). Host loop: `CanPublish`, `TryPublish`,
`TryTakeResult`, `Poll`, `ReleaseResult`, as an engine thread would. Latency is publish to result seen, on
`Stopwatch.GetTimestamp` (CLOCK_MONOTONIC). 8000 steps after 200 warm-up steps, three interleaved rounds
(direct, inproc, proc) per round. Percentiles are nearest-rank over the pooled steps.

## Result (depth 1, lock-step, `bench_4500us_d1.txt`, `pool_d1.txt`)

Load average at the start of each of the nine runs was 1.0 to 3.9 for seven, 6.2 for one (`proc`, round 3) and 9.3 for one (`inproc`, round 3); it
was not a quiet box throughout, and the first attempt (`bench_4500us_d1_busy_first_attempt.txt`, load 3 to 18) is
kept only to show how the tails behave there. No number from that attempt is used below.

| Mode | p50 (us) | p99 (us) | p99.9 (us) |
| --- | --- | --- | --- |
| direct | 4526.8 | 5143.7 | 7487.6 |
| in-process | 4528.2 | 4580.8 | 6159.2 |
| process | 4527.0 | 4579.5 | 5418.4 |
| process minus in-process | -1.2 | -1.3 | |

Rounds 1 and 2 only (every run at load 4 or below): process minus in-process is -1.3 us at p50 and -1.7 us at p99
(direct 4525.3 / 4578.8, in-process 4529.0 / 4585.0, process 4527.8 / 4583.3).

Per round, process p99 minus in-process p99 was -316.1, +15.0 and -5.3 us (4565.4 vs 4881.2, 4589.9 vs 4574.9, 4553.8 vs 4559.1).
Read against S8's +1.2 us p50 and +21 us p99 (C++ host): this run finds no process-boundary cost above the run-to-run
noise, and the largest per-round p99 gap in favour of in-process is +15 us. The threshold is 2% of 4.5 ms = 90 us. The
measurement does not show the boundary to be free: it shows that its cost is smaller than what one round of 8000 steps
can resolve on this box (the direct baseline's own p99 swings by 1.5 ms between rounds). The p99.9 and the direct p99
are scheduler effects of a shared machine and are not attributed to any mode.

Host thread allocation in the measured loop was 0 bytes and 0 collections (gen0, gen1, gen2) in all nine runs.
The worker's done-to-host-seen time (worker `PublishedNs` to the host's read) had a p50 of 0.5 to 0.6 us in every run.

## Depth 2 (two plans in flight, 4000 steps, 2 rounds, `bench_4500us_d2.txt`)

Wall time per step (4500 us GPU step): in-process 4528.5 and 4530.5 us, process 4545.7 and 4528.4 us. The worst
difference is +17.2 us per step (0.38%), in the round whose process run also had a 13.8 ms
p99.9 (load average 3.6 to 4.3, other agents active); the other round is -2.1 us. No allocation on the host thread.

## What this does not show

- A synthetic spin step, not a model step: one kernel and two copies, not hundreds of kernels. Host launch cost
  common to both modes cancels in the difference; it is not measured here.
- The 4500 us step is derived (S8), not measured; the 2% gate must be re-applied to a measured step once R-025 lands.
- The box was shared, load 1 to 9 during the retained runs. Three rounds per mode pool 24000 steps, but the p99 of any
  one round moves by hundreds of microseconds. A dedicated machine would resolve a smaller difference.
- Busy-wait on both sides (two cores). A blocking or hybrid wait was not measured.
- Release build with default tiered JIT and 200 warm-up steps; ReadyToRun or AOT was not tried. No collection occurred in the measured loops.
- One GPU, one machine. Respawn timing and fault timing with CUDA faults are not repeated here (S8's figures stand);
  the fault paths are tested without a GPU in `tests/Starlings.Roost.WorkerHost.Tests` (`FaultStressTests`).

## Reproduce

```sh
tests/Starlings.Roost.WorkerHost.Tests.Cuda/native/build.sh            # libroostspin.so into ../envs/R-069/native (never committed)
dotnet build tests/Starlings.Roost.WorkerHost.Tests.Cuda -c Release
export ROOST_SPIN_LIB=$PWD/../envs/R-069/native/libroostspin.so
tests/Starlings.Roost.WorkerHost.Tests.Cuda/run_bench.sh 4500 8000 3 1  # depth 1, three rounds; raw latencies in ../envs/R-069/bench
tests/Starlings.Roost.WorkerHost.Tests.Cuda/run_bench.sh 4500 4000 2 2  # depth 2
BIN=tests/Starlings.Roost.WorkerHost.Tests.Cuda/bin/Release/net10.0/Starlings.Roost.WorkerHost.Tests.Cuda
$BIN pool direct <files> inproc <files> proc <files>                  # pooled percentiles, process minus in-process
```

The per-step raw latency files (about 40 KB per run) are not committed; the logs and the pooled output are.

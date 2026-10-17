# S8 (R-065): worker process and shared-memory rings

Status: measured on one machine, one GPU, a C++ host. **The recommendation for decision 21 is provisional** until the measurement is repeated with the real .NET host (R-069): the host here is C++, so the measured +21 us p99 is a lower bound for the .NET host that decision 21 is about. Raw run output is committed under `spikes/s8-worker-process/results/` (see Evidence). Code: `spikes/s8-worker-process/` (throwaway evidence, outside `Roost.slnx`).

## Question

Plan section 6, S8: what does the process boundary cost per step compared with in-process rings, how fast are kill, respawn, re-handshake and weight reload, and how do sticky faults and hangs present? Kill criterion: overhead above 2% of a Qwen3-1.7B decode step at batch 1 that cannot be removed.

## Answer in four lines

- **Process boundary cost.** On a quiet box (load average 3.5 on 12 cores) the worker process adds **+1.2 us at p50 and +21 us at p99** to a lock-step 4.5 ms step, against in-process rings (0.03% and 0.47% of the step). The threshold is 2%, which is 90 us. With two steps in flight the process mode is not slower than in-process.
- **Respawn.** Kill to reaped about 50 to 67 ms; a new worker is ready (context, ring registration, handshake) in about 130 ms; with 3.44 GB of weights in page cache the first step completes about 0.8 s after spawn, of which about 0.64 to 0.68 s is the weight upload. These are the retained rerun figures (box load 3 to 5); the first, unretained runs on a busier box were slower (see Kill, respawn, re-handshake).
- **Faults.** Illegal address is classified sticky in 8 to 9 ms and poisons the process for good; a device hang is detected by the step deadline while the heartbeat still advances; a worker crash is detected by `waitpid`. In process mode all three end in a respawn and a good step 131 to 156 ms later (no weights; fault table below, both runs). In process, a sticky fault takes the host with it and a hang cannot be cancelled.
- **Decision 21.** Recommend option (a), out-of-process by default. The kill criterion is not met on the C++ host. **Provisional:** the host here is C++, so +21 us p99 is a lower bound for a .NET host; repeat with the real .NET host (R-069) before treating this as settled.

## Setup

| Item | Value |
| --- | --- |
| GPU | RTX 3090 (SM 86), driver 610.57.04, CUDA 13.3 toolkit, shared through `tools/gpu-lease` (stream F), one short lease per measurement |
| CPU | Intel i7-5820K (12 threads), shared with other agents |
| Host | C++17 and CUDA runtime, one binary `s8` with subcommands. Not .NET: see limits |
| Rings | SPSC plan ring and result ring, 8 slots each, plan slot 4096 B, result slot 1024 B, cursors on their own cache lines, in a `MAP_SHARED` file under `/dev/shm`. The worker calls `cudaHostRegister(portable)` on the whole segment and copies plan slot to device and result back from device directly to and from ring memory (no staging buffer) |
| In-process mode | The same ring code and the same `worker_run`, with the worker on a thread of the host process |
| Direct mode | Baseline: the host thread calls the same H2D, kernel, D2H, sync sequence with no rings |
| Step | `cudaMemcpyAsync` 4 KiB plan H2D, one kernel that spins on `%globaltimer` for the stated GPU time, `cudaMemcpyAsync` 1 KiB result D2H, `cudaStreamSynchronize` (spin schedule). Host and worker both busy-wait on the ring cursors |
| Timing | `CLOCK_MONOTONIC`, stamped by the host on publish and on result seen, and by the worker on plan seen and result published. This clock is shared across processes |

### Synthetic step time: 4.5 ms (derived, one input measured)

No Qwen3-1.7B step has been measured by Roost yet (S5 is optional, R-025 is not built), so the figure is derived and should be replaced when R-025 lands:

- Batch-1 decode of Qwen3-1.7B in bf16 reads every weight once. 1.72 B parameters (28 layers; the 151,936 x 2,048 tied embedding is read in full as the LM head) is about 3.44 GB. This is arithmetic from the published architecture, not a measurement.
- Measured here: `s8 calib` reads 3.44 GB of device memory in **3.91 ms** at p50 (880 GB/s; the 3090 peak is 936 GB/s) in the original, unretained run. The retained rerun (`results/calib.txt`) gives min 3.870, p50 3.875, max 3.911 ms (888 GB/s), so the floor is 3.88 to 3.91 ms; 15% on top is 4.46 to 4.50 ms, and 4.5 ms stands.
- Add about 15% for attention over a short KV, norms, sampling, and launch gaps: **4.5 ms**. The direct baseline measures 4.53 ms with its own H2D, launch, D2H and sync costs (the GPU event time H2D to D2H is 4.52 ms).
- 2% of 4.5 ms is 90 us. Sensitivity runs use 2.0 ms (40 us) and 50 us (transport only).

## Per-step overhead

Lock-step, depth 1 (the host waits for each result before publishing the next plan, the worst case for latency). Pooled percentiles in us.

Quiet run (load average 3.2 to 3.8; two interleaved rounds of 8,000 steps per mode, 16,000 pooled; the figure the recommendation rests on):

| Mode | p50 | p99 | p99.9 |
| --- | --- | --- | --- |
| direct | 4526.2 | 4557.1 | 4582.4 |
| in-process rings | 4526.2 | 4561.5 | 4581.8 |
| process rings | 4527.4 | 4582.4 | 6112.8 |
| process minus in-process | +1.2 (0.03%) | +21.0 (0.47%) | not meaningful |

Where the time goes (process mode, median of the quiet run): plan published to worker saw it 0.1 us, result published to host saw it 0.1 us. Ring and cross-process cost is about 0.2 us at p50; the +1.2 us p50 difference is about the size of the run-to-run variation between the direct and in-process runs (0.0 to 1.2 us), so it is not shown to be a real cost. The p99 gap is one of the two rounds: round 1 had p99 4554.8 us (below in-process), round 2 had 4763.5 us with the excess in "worker busy", i.e. the worker thread was descheduled or its copy was delayed, not the ring.

Earlier run on a busy box (load average 11.8 to 15.6 on 12 cores, `results/run1/`, three rounds of 5,000 steps per mode, 15,000 pooled), kept because it shows how the tails behave when spinning threads compete with other agents:

| Mode | p50 | p99 | p99.9 |
| --- | --- | --- | --- |
| direct | 4591.0 | 5338.5 | 7515.8 |
| in-process rings | 4586.5 | 6390.1 | 10749.2 |
| process rings | 4598.2 | 8207.1 | 15440.7 |
| process minus in-process | +11.7 (0.26%) | +1817 (40%) | |

Under load, every mode's p99 is 0.7 to 3.7 ms above its p50, and the order of the modes differs between rounds (per-round p99 for process: 7378, 6146, 9900 us; in-process: 6468, 6529, 6063 us). Two busy-wait threads on a loaded host are preempted by the scheduler, whichever mode they run in. The tail is a scheduling effect of this machine; the data does not let us attribute it to the process boundary. The p50 gap stays at or below 12 us in both runs.

Sensitivity (busy box, load average 17.5 to 20.5, two rounds each, pooled; `results/bench_2000us_d1.txt`, `bench_50us_d1.txt`): at a 2.0 ms step the process minus in-process p50 is +3.6 us (0.18%); at a 50 us step (transport only) it is +5.8 us at p50, with the direct baseline itself at 74 us (the H2D, launch, D2H and sync cost). Process p99 in the 50 us run was lower than in-process (1068 vs 1510 us), which again shows that the busy-box tails are not about the transport.

Pipelined, depth 2 (the engine publishes plan n+1 while step n runs, as an asynchronous scheduler does; load average 10.7 to 18.1 across the four rounds, with the first in-process round at 18.07; `results/bench_4500us_d2.txt`): wall time per step is 4536.8 and 4544.0 us for the process, 4548.0 and 4580.5 us in-process, for a 4500 us GPU step. Throughput overhead is under 1% in both and the process is no slower.

## Kill, respawn, re-handshake

`s8 respawn`, six rounds each, SIGKILL of the worker after one step, the weights file read through the page cache (a 3,440,377,856-byte file of zeros under `envs/R-065`; `cat` to warm it first). Milliseconds. Source: `results/respawn_weights.txt` and `results/respawn_noweights.txt`, rerun on the RTX 3090 at load average 3.2 and 4.7.

| Phase | With 3.44 GB weights (6 rounds) | No weights (6 rounds) |
| --- | --- | --- |
| SIGKILL to reaped (`waitpid`) | 60 to 68 | 52 to 66 |
| fork+exec to worker `main` | 1.9 to 2.5 | 1.8 to 3.4 |
| CUDA context creation (`cudaFree(0)`) | 144 to 170 | 127 to 130 |
| `cudaHostRegister` of the ring segment (45 KB) | 0.13 to 0.22 | 0.13 to 0.14 |
| Weights: mmap and 64 MiB chunked `cudaMemcpy` | 639 to 683 | 0.1 |
| Handshake publish to state ready | 0.05 to 0.08 | 0.04 to 0.05 |
| Ready to first result | 1.25 to 1.38 | 1.20 to 1.31 |
| **Spawn to first result** | **786 to 842** | **131 to 134** |

Earlier figures. The first respawn runs of this spike were not saved. They were taken on a busier box and were higher: kill to reaped 75 to 83 ms (63 to 86 without weights), context creation 190 to 225 ms (165 to 195), `cudaHostRegister` 0.3 to 0.4 ms, weight upload 776 to 1105 ms, spawn to first result 971 to 1313 ms (170 to 201 without weights), and a round 0 of 3,233 ms with part of the file evicted from the page cache (free memory 2 GB at the time). Those earlier numbers have no retained log and the rerun above replaces them. Run-to-run spread on this shared box is therefore at least 1.3x for the with-weights case; take 0.8 to 1.3 s as the range and not a precise figure. In the rerun, the weight upload runs at about 5 GB/s through pageable memory (3.44 GB in 0.64 to 0.68 s); staging through pinned buffers has not been tried. A cold page cache was not characterised.

Handshake: the worker compares the ABI word and a 32-byte manifest digest against what the host wrote, before touching CUDA. A wrong manifest digest was rejected with state "rejected" (state 3, reason 2) and the worker process gone after 2.3 ms (no weights) and 3.1 ms (with weights) in the rerun; the unretained first runs showed 3.2 and 7.6 ms. A weights-size mismatch is also checked, after the load. A digest of the weights themselves was not part of the handshake here: a full SHA-256 of the 3.44 GB file took 9.4 s on this box with the file in page cache (`results/sha256sum.txt`; an unretained first run measured 13.1 s), so the plan's "weight digest" check has to compare a digest recorded in the manifest, not hash the weights at each respawn.

Kill to ready is therefore about 0.19 s without weights and about 0.9 s with them in the rerun (first, unretained runs: about 0.27 s and 1.1 to 1.4 s). Requests in flight would be re-queued by R-070; this spike does not exercise that.

## Fault injection

`s8 fault` uses the plan's `kind` word. Each case runs five good steps first. Steps use a 2 ms synthetic GPU time, the step deadline is 300 ms. All eight cases (four kinds, both modes) were rerun under `tools/gpu-lease` on the RTX 3090 at load average 4.6 to 4.7 and the output is committed as `results/fault_<mode>_<kind>.txt`. One run per case; the earlier, unretained runs had slightly different timings, so the figures below carry the spread of two runs where both exist.

| Injection | What the kernel or worker does | Process mode | In-process mode |
| --- | --- | --- | --- |
| Recoverable | Launch with 2048 threads per block, over the 1024 limit. The runtime returns `cudaErrorInvalidValue`, error 1, printed as "invalid argument" (read from the log; this is not error 9, `cudaErrorInvalidConfiguration`, which the earlier code comment and notes said) | Result ring reports `recoverable` after 0.04 ms (first run 0.08 ms); the next step succeeds in the same worker | Same: `recoverable` in 0.06 ms, next step succeeds |
| Sticky | Store to unmapped address 0x10: `cudaErrorIllegalAddress` (700) | Result ring reports `sticky` after 9.1 ms (first run 8 to 9.5 ms). The classification is a probe: `cudaGetLastError` then a memset and sync; a second failure means sticky. The worker publishes the result, marks state faulted and exits. Respawn: ready 138.3 ms, good first step at +140.4 ms (first run 143.5 and +145.6 ms) | Result reports `sticky` after 7.8 ms (first run 8.1 ms). Every following step in the same process fails with 700 (three tried). The context is lost for the process lifetime; only a restart of the host application recovers |
| Hang | Kernel spins on a flag that is never set | Step deadline fires at 300 ms; heartbeat advanced 283 times (1 ms heartbeat thread; first run 274) in that time, so the worker is alive but the step does not finish: hang. `SIGKILL` to reaped 67.5 ms (first run 63). Respawn ready 135.4 ms, good first step at +137.6 ms (first run 154 and +156 ms). The respawned worker created a context and ran a good step, so the GPU was usable again; steady-state speed after the respawn was not measured | Deadline fires at 300 ms, heartbeat advanced 281 (first run 279): hang detected, but the stepping thread is stuck inside the driver and cannot be cancelled or killed. The host application has to exit |
| Crash | `abort()` in the worker (core dumps disabled by `RLIMIT_CORE=0`) | `waitpid` reports exit 148 ms after injection (first run 168 ms), with no result and before the deadline. Respawn ready 129.2 ms, good first step at +131.4 ms (first run 151 and +153 ms) | Not meaningful: the harness has no worker to kill, and the host process itself terminated (`terminate called without an active exception`, exit 250). It is recorded in `results/fault_inproc_crash.txt` as a host death, not a measurement |

Good step after the fault, process mode, both runs: +131.4 to +156 ms (recoverable needs no respawn). The earlier answer text said 145 to 175 ms; the fault tables support only 131 to 156 ms.

Detection classes for C8: `recoverable` and `sticky` are reported by the worker through the result ring; `hang` is the host's conclusion from a missed deadline with a live heartbeat; a dead worker is reported by process exit; a missed deadline with a stopped heartbeat is "stalled". "Stalled" was seen once, in an early run where the abort took more than 300 ms, which the implementer attributed to a core dump of the CUDA process being written (hence `RLIMIT_CORE=0` in the worker). That observation has no retained log and was not reproduced (reproducing it means writing a multi-GB core dump), so treat the core-dump explanation as a hypothesis behind condition (3) below, not as a measured result.

## Limits of this evidence

- The host is C++, so the measured +21 us p99 is a lower bound for the .NET host that decision 21 is about, and the decision 21 recommendation is provisional until the measurement is repeated with the real .NET host (R-069). A .NET host adds its own ring-polling and P/Invoke costs and GC pauses; the transport is a few hundred nanoseconds here, so it is unlikely to approach 90 us, but R-027 and R-069 must measure it, including the allocation-tick test.
- The synthetic step is a spin kernel, not a model step: no real memory traffic, one kernel and one H2D and D2H copy rather than a graph of hundreds of kernels. The host-side cost of launching a real step is in both modes alike and cancels in the difference.
- Both sides busy-wait, which uses two cores. A blocking or hybrid wait would save CPU but was not measured; it will add wake-up latency of tens of microseconds.
- One GPU, one machine, one 4.5 ms step. 3,440 MB of weights were zeros; the upload path does not depend on content.
- The busy-box tails cannot be attributed. Two quiet rounds per mode are few; p99 of the quiet run for the process (4582 us) includes one round with 4764 us.
- Weights were not loaded through pinned staging; cold-cache respawn was not characterised beyond one observation.
- Multi-GPU, NCCL bootstrap, MPSC request queue and R-070 request resumption were not exercised. The allocation-tick and racecheck checks belong to R-027.
- RTX 4060 Ti was not used (about 15.6 of 16 GB in use by other work).

## Recommendation for decision 21

Choose **(a), out-of-process for everything with in-process for tests and development**, as the plan recommends. **This is provisional**: the host here is C++, so +21 us p99 is a lower bound for the .NET host that decision 21 is about; confirm with the real .NET host (R-069) before the decision is final.

- The overhead criterion is cleared by a wide margin: +1.2 us at p50 and +21 us at p99 against a 90 us allowance on the quiet box, and no sign of a systematic cost on the busy box beyond scheduler noise. Ring traffic costs about 0.2 us each way. With two steps in flight the process mode is not slower than in-process.
- What the process buys is concrete. In-process, a sticky illegal-address fault leaves the context failing every call until the host exits, and a hung kernel cannot be cancelled. In process mode both are contained to a worker that is killed and replaced, and the engine is serving again 130 to 156 ms after the fault without weights, and (not measured as a fault, from the respawn table) about 0.9 s with weights in page cache.
- Preview 1 runs inside a consumer's application, which is the case where losing the host to a GPU fault is least acceptable.
- Conditions to carry into R-004 and R-069: (1) the deadline and heartbeat are in C8 as drafted, and hang is defined as a missed deadline with a live heartbeat; (2) the worker classifies recoverable and sticky with a probe after any error and exits on sticky; (3) workers disable core dumps (the implementer saw one early abort run past a 300 ms deadline, which was not logged or reproduced, so this is a precaution); (4) the handshake compares manifest-recorded digests, never hashes weights at respawn; (5) keep at least two plans in flight so that host scheduling jitter does not reach the GPU; (6) re-measure with the .NET host and the real step once R-025 exists, and treat 2% of the measured step as the gate.
- If that re-measurement shows more than 2% of a real step, the plan's fallback (option (b), in-process library with an out-of-process server) remains available without a contract change, because both transports sit behind one interface.

## Evidence

Raw output is committed as text under `spikes/s8-worker-process/results/`:

- `bench_4500us_d1.txt`, `bench_4500us_d2.txt`, `bench_2000us_d1.txt`, `bench_50us_d1.txt`: the bench logs (the originals were git-ignored `bench_*.log` files). `pool-d1.txt` and `pool-d2.txt`: `pool.py` output over the CSVs. `run1/`: the busy-box run (log and pool output).
- `calib.txt`, `sha256sum.txt`, `respawn_weights.txt`, `respawn_noweights.txt` (including the bad-manifest case), and `fault_{proc,inproc}_{recoverable,sticky,hang,crash}.txt`: rerun on 2026-10-17 on the RTX 3090 through `tools/gpu-lease`, one lease per run. The first runs of these were not saved, and where the rerun differs, the tables above say so.
- The per-round latency CSVs (1.9 MB) are not committed; the p50/p99 tables can be checked from the logs and the pooled output.

## Reproduce

```sh
cd spikes/s8-worker-process
./build.sh                                   # nvcc, SM 86 only, output in build/ (git-ignored)
cd ../..
LEASE="tools/gpu-lease run --device 3090 --need-mib 6000 --wait 3600 --stream F --"
S=spikes/s8-worker-process/build/s8
$LEASE $S calib
spikes/s8-worker-process/run_bench.sh 4500 8000 2          # quiet lock-step A/B/C rounds; csv and logs in out/
spikes/s8-worker-process/run_bench.sh 2000 5000 2          # sensitivity, 2.0 ms step
spikes/s8-worker-process/run_bench.sh 50 20000 2           # sensitivity, transport only
spikes/s8-worker-process/run_bench.sh 4500 4000 2 2        # depth 2 (process and in-process only)
for us in 4500 2000 50; do python3 -I spikes/s8-worker-process/pool.py spikes/s8-worker-process/out $us 1; done
python3 -I spikes/s8-worker-process/pool.py spikes/s8-worker-process/out 4500 2
for m in proc inproc; do for k in recoverable sticky hang crash; do
  (ulimit -c 0; tools/gpu-lease run --device 3090 --need-mib 1000 --wait 3600 --stream F -- $S fault --mode $m --kind $k)
done; done
dd if=/dev/zero of=$WEIGHTS bs=1M count=3281; cat $WEIGHTS > /dev/null
$LEASE $S respawn --weights $WEIGHTS --rounds 6            # also prints the bad-manifest case
$LEASE $S respawn --rounds 6                               # no weights
sha256sum $WEIGHTS                                         # about 9 s with the file in page cache
```

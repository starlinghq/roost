# Starlings.Roost.Engine status

## R-032 async scheduling: partial (engine and CPU path only)

Build plan section 10, R-032 (streams F and G). Acceptance criteria: plan N+1 built during step N; sampled ids patched on
the device; stop ids checked on the device with one-step lag hidden by the frontend; identical outputs to synchronous mode;
a preemption-storm stress test with racecheck.

### Delivered (stream G, and the CPU worker path; CPU-tested)

- Contract version 5, shared-memory layout 2 (stream A change made for this item): `RoostStepPlanHeader.input_step_id`
  (appended) and `ROOST_SEQ_FLAG_SAMPLED_INPUT`, a one-token decode row whose input is the id the worker sampled for the
  sequence in step `input_step_id`; the plan holds -1 (`SharedLayout.PendingTokenId`) there.
- `Scheduling/StepPlanner.cs` with `PlannerOptions.AsyncScheduling` (off by default; off plans exactly as before and keeps
  the committed recordings' digests): a sequence whose token is being sampled by the plan built last gets a chained decode
  row in the next plan. Its position, Philox offset and tokens-left count include the tokens in flight, so it is the row a
  synchronous planner builds once the token returns. Length limits are applied at plan time and never overrun; stop ids
  are seen when the result is applied, one step late, and the row already planned for a finished sequence is ignored (its
  blocks stay parked until that plan completes). A sequence left out of the last plan falls back to the synchronous rule.
  A sampling row that returns no token now ends its sequence with `FinishReason.Error` instead of leaving it waiting.
- `Starlings.Roost.Worker`: `StepPlanReader` checks sampled-input rows (refused with `Unsupported` unless
  `PlanLimits.SampledInputs`), and `SampledInputPatcher` patches them on the host from the ids of the step the worker ran
  last, refusing a plan that names another step. The CUDA eager worker leaves `SampledInputs` false and refuses such plans,
  naming stream F's device-side patching.
- `Starlings.Generation.Roost` (the M0 driver, stream H): `RoostWorkerOptions.AsyncScheduling` keeps two steps published
  and applies results in order; the token of a row planned after a stop id never reaches the channel.

### Evidence (CPU only)

- `tests/Starlings.Roost.Engine.Tests/Scheduling/AsyncSchedulingTests.cs`: the chained row equals the synchronous row
  field by field; length limit; stop-id lag; restart; 40 random workloads under a preemption storm with the recorded
  patching worker, async token streams and finish reasons equal to synchronous ones; plan determinism and replay of the
  recorded inputs.
- `tests/Starlings.Roost.Engine.Tests/Scheduling/TinyModelAsyncTests.cs`: roost-tiny and roost-tiny-hybrid on `CpuOps`
  (`TinyPlanWorker`, patching from its last sampled ids): async and synchronous token streams equal, including a
  preemption storm (6 blocks of 16 tokens, or 3 recurrent slots, for 10 requests); the recorded async run replays to the
  same plans with no worker.
- `tests/Starlings.Roost.Worker.Tests/SampledInputTests.cs` and `tests/Starlings.Generation.Roost.Tests/AsyncSchedulingTests.cs`
  (in-process and process transport).

### Still open (do not mark R-032 done)

- **Device-side patching and stop checks, stream F: not built.** The CUDA worker refuses sampled-input plans; nothing here
  ran on a GPU and there is no `compute-sanitizer racecheck` run. Async scheduling is `untested` on the CPU path and
  `unsupported` on the CUDA worker.
- **No host-gap measurement.** The point of the item on a GPU is hiding the host gap between steps; no number exists.
- **A recoverable fault in step N also fails step N+1's requests** (the worker refuses a plan whose input step failed).
  Replanning N+1's other rows is not built.
- The prefix cache adds a chained row's block only once its token has returned, so a sequence that finishes with a chained
  row in flight does not cache its last block.

## R-094 host-memory KV tier: partial (stream G side only)

Build plan section 10, R-094 (streams G and F). Acceptance criteria: blocks evicted to pinned host memory and restored
ahead of use; prefix hits from host memory on agentic multi-turn traces; consumer-card TTFT recorded.

### Delivered (stream G, CPU-tested)

- `Blocks/HostTier.cs`: the host tier as a second `BlockManager` with `BlockTier.Host` groups, sized from a byte budget
  (`PrefixCacheOptions.HostTierBytes`; 0, the default, turns the tier off). Only the paged attention kinds the prefix
  cache serves (`Full`, `MlaLatent`) get host blocks; recurrent snapshots stay in the device snapshot pool.
- `Caching/PrefixCache.cs`: under device pressure the cache demotes the least recently used cache-only node with no
  device child to the host tier with a `CacheOpKind.TierMove` op (C2) instead of dropping it. Host blocks are dropped
  only when the host tier is full (least recently used host leaf first). A lookup that reaches host nodes allocates
  device blocks for them and queues host-to-device `TierMove` ops.
- `Scheduling/StepPlanner.cs`: tier moves ride in the next plan ahead of its other ops. A demotion pins its host block
  until that plan completes; a restore pins both sides. The sequence that triggered a restore holds the restored blocks
  and is planned from the plan after the one carrying the restore (one step ahead of use), or in the same plan, behind
  the ops, when nothing else could run. Plans stay a function of the inputs: replay is deterministic.
- No contract changed: `BlockTier.Host`, `CacheOpKind.TierMove` and `PlanCacheOp.SrcTier`/`DstTier` were already in
  C2 and C4.

### Evidence (CPU only)

`tests/Starlings.Roost.Engine.Tests/Caching/HostTierTests.cs`:

- demote and restore round trips through the planner against `ContentWorker` (attention and hybrid with recurrent
  snapshots), outputs equal to an uncached run; host pressure; restore one plan ahead of first use; worker restart;
  40 random workloads with cancels and restarts, invariants checked after every input and replayed identically;
- a recorded multi-turn agentic trace (four sessions, four turns, shared tool preamble), committed under
  `Caching/Recordings/agentic-host-tier.*.txt`, replaying to the committed plan digests, with host-tier prefix hits and
  more reused tokens than the device-only cache;
- the CpuOps model (`CpuOpsTinyWorker`): bit-identical logits with the host tier through the planner, and a demote and
  restore through `InProcessBlockTransfer` that preserves logits bit for bit (and changes them when the restore is
  skipped).

### Still open (do not mark R-094 done)

- **GPU execution, stream F: untested.** The CUDA worker does not allocate pinned host memory for the tier or execute
  `TierMove` ops; nothing here has run on a GPU. Until it does, the host tier is for the CPU worker and planner tests
  only and stays off by default.
- **Consumer-card TTFT: not recorded.** No GPU run was made for this item, so there is no TTFT number with or without
  the host tier.
- The restore is scheduled one plan ahead, but overlapping the host-to-device copy with the previous forward (a side
  stream) is a worker concern and is not built.

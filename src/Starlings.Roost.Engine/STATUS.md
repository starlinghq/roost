# Starlings.Roost.Engine status

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

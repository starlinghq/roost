# Second adversarial review: Roost as a vLLM replacement

Written 2026-10-17 against `docs/plan/build-plan.md` at `03457fa` (the 2026-10-16 refresh), `docs/plan/review.md` and the README. The question asked: does the plan build a full replacement for vLLM that is better, faster and more resilient, and does it leave room for multi-LoRA and for patching and controlling the kernels? The first review asked a different question (will anyone outside the project use it) and is not repeated here.

The findings are ranked by how badly each undercuts that goal. Changes marked **applied** are in the revised build plan. Changes marked **decision** are rows in its section 11.

## Verdict

The engineering is careful, but the plan as reviewed no longer builds a vLLM replacement and does not say so. The first review moved the product toward in-process .NET inference with fidelity evidence on consumer cards; the week-12 checkpoint can drop the vLLM claim entirely; and the one throughput target left is 0.97x vLLM, which allows Roost to be 3% slower. Each of the stated goals runs into a design choice that the C0 contract freeze fixes in week 2, so most of this review is cheap to act on before C0 and expensive after it.

The maintainer has since set the goal explicitly: a full replacement for vLLM on NVIDIA GPUs (build plan decision 18).

## Findings

### 1. A full replacement with no MoE, no tensor parallelism and no speculative decoding (applied)

**Evidence.** The C3 op vocabulary has no router, grouped GEMM or fused MoE. The V1 non-goals include parallelism; the only multi-GPU item is whole-request replicas (R-054). C2 reserves `draft count` and `verify flag` but no backlog item implements speculative decoding. The model roadmap is Qwen3 (dense), Qwen3.5-0.8B (hybrid) and Gemma 4.

**Why it matters.** Most open models people serve in 2026 are MoE, and several ship MTP heads that vLLM uses for speculative decoding. Without MoE or a way to split a model across GPUs, Roost replaces vLLM only for small dense models, where llama.cpp and Ollama already win on ease of use. C2 is laid out in pinned double buffers that streams F and G both read, so adding TP ranks, expert routing and per-device plans later is a rewrite, not a version bump.

**Change.** The build plan gains a parity matrix (section 13) that defines "full", reserved C3 names for MoE, collectives, MLA and tree attention, a device-mesh contract (C9), and phases M4 (MoE at scale and parallelism) and M5 (disaggregation, KV tiers, multimodal, long context). Speculative decoding is R-083 to R-085.

### 2. "Faster" has no mechanism (applied)

**Evidence.** Kernels are vendored from vLLM `csrc/` and FlashInfer with patch sets capped near 300 lines. C3 is a set of unfused primitives. The generator emits straight-line `Forward<TOps>()` calls with no graph between the model definition and the backend.

**Why it matters.** At saturation, throughput comes from GPU kernels. With the same kernels as vLLM, the best result is parity, which is presumably why the gate was 0.97x. vLLM's current lead comes from compiler fusion passes and autotuned kernels, and a fixed list of unfused calls has no place to run a fusion pass. Devirtualising host dispatch optimises a layer that CUDA graphs already make nearly free.

**Change.** An op graph contract (C7) sits between the generator and the backend; fusion, sharding, variant selection and persistent-kernel lowering are passes over it, and `CpuOps` still runs the unfused reference. Section 14.2 lists the specific, falsifiable ways Roost intends to be faster, each with a kill criterion.

### 3. Kernel control conflicts with C1 and the manifest (applied)

**Evidence.** C1 has one `roost_<op>_v1` per op in one library. The loader refuses any digest mismatch and PTX is refused by default. Triton runs at AOT build time only.

**Why it matters.** Patching and controlling kernels means several implementations of an op, chosen per shape, SM, model or policy, and sometimes supplied by the user. The ABI can name "the kernel for op X" but not "variant k of op X", and the only way to add a kernel is a full library release.

**Change.** C1 gains a registration table with variant ids and constraints, and more than one kernel library may load. R-072 adds the variant selector, a persisted autotune cache, a kernel-candidate path that turns a user kernel into a digest-pinned `local-evidence` profile after conformance, and conformance-gated NVRTC JIT.

### 4. In-process fault isolation cannot work for CUDA's worst failures (applied, plus decision)

**Evidence.** The README describes a single-process engine; R-045 promises worker fault isolation with queues preserved; Preview 1 runs the engine inside the consumer's process.

**Why it matters.** Sticky faults (illegal address, misaligned access, Xid errors) leave the CUDA context unusable for the rest of the process, and a hung kernel raises no fault at all. In Preview 1 one bad kernel takes down the consumer's application, and user-supplied kernels (finding 3) make faults likelier.

**Change.** A worker transport contract (C8): one worker process per GPU, plan and result rings in shared memory, a handshake, heartbeat and per-step deadline, and respawn with queues held by the engine (R-069, R-070). `compute-sanitizer` joins the conformance gate (R-019). Decision 21 sets out-of-process as the default.

### 5. Multi-LoRA needs contract space now (applied)

**Evidence.** R-050 sits in M3. The C3 `Gemm` has no adapter input, C4 has no adapter pool, and admission reserves nothing for adapters.

**Why it matters.** Multi-LoRA touches every linear layer of the generated forward, the memory planner, admission and the CUDA graph buckets. Freezing C3, C4 and the generator without it means reopening all three in M3.

**Change.** Reserved `LoraShrink` and `LoraExpand` in C3, an `adapter` state-group kind in C4 so adapters page from the KV budget, and R-050's acceptance extended to dynamic load and unload, graphs with adapters, and LoRA over an INT4 base.

### 6. The budget cannot produce "faster than vLLM" on datacenter hardware (decision)

**Evidence.** 40 GPU-hours a month on two shared consumer cards, about 16 H100 hours per release cycle, 20 maintainer review hours a week, CUDA role possibly unfilled.

**Why it matters.** Kernels cannot be tuned for a GPU the project sees 16 hours a cycle. A one-time 0.97x on H100 in mid-2027 would describe the previous generation.

**Change.** The scorecard (section 14.3) has two hardware classes. The consumer class is gated at M2 on hardware the project owns. The datacenter class is gated at M6 and needs decisions 19 (staffing) and 20 (multi-GPU cloud budget).

### 7. The benchmark measured the one metric Roost is least likely to win (applied)

**Evidence.** The M2 gate was "at least 0.97x at saturation on the stated sweep".

**Change.** The gate is now a scorecard: goodput under an SLO, TTFT and inter-token latency percentiles, cold start, host memory and recovery time, across chat, long-context, agentic, structured-output, decision and mixed embed-and-generate workloads, against pinned vLLM and SGLang (and llama.cpp for single-user consumer runs).

### 8. Tool calling and structured output arrived too late (applied)

R-051 and R-053 move to M1, so the first server can serve agents. Structured outputs use llguidance, a Rust library with a C API, behind the native ABI (decision 22).

### 9. The contract freeze came before the spikes that inform it (applied)

R-004 now depends on the S1, S2 and S8 reports and freezes the contracts as v0; they are promoted to v1 at the M0 gate.

### 10. The hardest concurrency code had thin review (applied)

R-027, R-028, R-032 and R-069 always get a human reviewer (section 9), with deterministic replay, `compute-sanitizer racecheck` stress runs and a preemption-storm test in M1.

### 11. Smaller points (applied)

- One upstream update drill moves to M1 (R-058).
- A host-memory KV tier lands in M3 for consumer cards (R-094).
- The fp64 CPU reference holds bf16 weights and upcasts per op; tolerances are per op class, not per kernel variant.
- The two local GPUs differ in SM and have no NVLink, so multi-GPU evidence comes from cloud pods (decision 20).

## What this review does not change

The typed state groups, the step plan as the hot-loop contract, the C ABI kernel boundary, a CPU reference for every op, fidelity evidence as the product's edge, `gpu-lease`, and the decision to build an engine rather than a host over llama.cpp or ONNX Runtime GenAI.

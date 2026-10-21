# Charter

Status: adopted 2026-10-18 (R-001); consumer, kernel-layer owner and budget recorded by the maintainer on 2026-10-20.

## Purpose

Roost is a whole and complete replacement for vLLM, without compromises, and faster and better than vLLM in the categories listed under Targets (build plan decision 18, set by the maintainer on 2026-10-17 and confirmed on 2026-10-18). It is a serving engine written in .NET for NVIDIA GPUs first and AMD GPUs (ROCm) next, with kernels in CUDA C++ (and HIP C++ for ROCm) behind a C ABI and a few host-side native libraries in Rust. There is no Python at runtime. ONNX Runtime may be used in the host process, never in the GPU worker process (plan decision 30).

The plan is [docs/plan/build-plan.md](../docs/plan/build-plan.md).

## Scope

- Every vLLM serving capability on NVIDIA GPUs, as the parity matrix in plan section 13.1 lists it: APIs, scheduling, sampling, speculation, quantisation, MoE, parallelism, KV tiers, disaggregated serving, multi-LoRA, multimodal and audio input, operations and hardware coverage.
- Distributed serving (tensor, pipeline, data and expert parallelism, multi-node, disaggregated prefill and decode) and bespoke GPU kernels (fused, persistent and collective kernels written by Roost, beside vendored upstream kernels pinned by full commit SHA).
- vLLM's Python plugin and custom-op APIs are covered by Roost model definitions and kernel libraries, and Ray by Roost's .NET launcher, because Roost has no Python at runtime (plan section 13.2).
- Training and fine-tuning are not vLLM capabilities and are out of scope. In-place weight update for reinforcement-learning loops is in scope.
- Backends beyond NVIDIA (plan decision 29, decided 2026-10-18): NVIDIA first; ROCm is a definite target, delivered after the replacement gate. TPU, Gaudi, Neuron, XPU and CPU serving are not planned.

## Targets

Where Roost aims to be faster or better than vLLM. These are targets, not claims: each is published only when the scorecard (plan section 14.3) measures it against a pinned vLLM release, and a speed target is dropped if it meets its kill criterion (plan section 14.2).

- Low-batch decode latency, through persistent decode kernels specialised per profile.
- Exact prefix caching and state forks for hybrid models such as the Qwen3.5 family.
- Decision serving for Kev: one prefill per request, no decode.
- MoE models on consumer cards, through dynamic CPU and GPU expert placement.
- Precision chosen per layer from fidelity evidence.
- Embedding and extraction models scheduled into gaps between decode steps on the same GPU.
- Cold start with nothing compiled at start, and recovery from GPU worker faults that keeps in-flight requests.
- Published fidelity evidence for every profile, and a batch-invariant mode on consumer GPUs.

## Relationship to Starling

[Starling](https://github.com/starlinghq/starling) works out the models; Roost runs on top of it. Roost consumes Starling's published NuGet packages, pinned exactly, and nothing else (plan section 2.2). Roost makes no changes to Starling's repository; what Roost needs from Starling is requested as an issue on Starling's roadmap. It owns its own evidence pipeline, profiles, fixtures and tolerances in `evidence/`, and no throughput number is published for a Roost profile before it passes Roost's fidelity gates.

## Roles

| Area | Owner |
| --- | --- |
| Technical lead, maintainer, final release reviewer | Mark |
| Named consumer and use case | starlinghq (Mark), serving Starling's models from .NET apps on local NVIDIA GPUs. First use cases: Qwen3.5-0.8B chat through `IChatClient`, then Kev decision serving. |
| Kernel-layer owner | Mark, until the CUDA roles in plan decision 19 are filled. |
| Second release owner | Open |

## Release authority

The maintainer approves releases. On 2026-10-20 the maintainer gave the orchestrating agent permission to cut releases (tags, GitHub release assets, NuGet packages) once a release drill passes; implementation agents still open draft changes and produce evidence, cannot approve their own change, and cannot change protected fixtures, tolerances or release evaluation sets in `evidence/protected/`.

## Maturity states

Roost uses the same states as Starling's charter (Experimental, Preview, Stable), and every capability carries an explicit status: supported, unsupported, experimental or untested. Missing never means supported.

## Budget

| Line | Value |
| --- | --- |
| Local GPU hours per milestone | 40 GPU-hours a month (the recorded cap), measured by `tools/gpu-lease`; the maintainer may stop other local GPU work (for example Strata on the RTX 4060 Ti) for evidence runs. |
| Rented single-GPU hours per milestone | RunPod as needed (plan decision 16), amounts set by the maintainer per milestone. |
| Rented multi-GPU hours per milestone (NVLink pods, and a multi-node rental) | Set by the maintainer per milestone when M4 starts (plan decision 20). |
| ROCm hardware (rented AMD Instinct pods; one consumer RDNA card) | *For the maintainer.* Plan decision 29, M7. |

Amounts are recorded in [docs/kickoff.md](../docs/kickoff.md) when the maintainer sets them. This charter approves no spend.

# Charter

Status: proposed (R-001), for the maintainer. Nothing here is adopted until the maintainer approves the pull request that adds it. Entries marked *for the maintainer* are deliberately blank.

## Purpose

Roost is a whole and complete replacement for vLLM, without compromises, and faster and better than vLLM in the categories listed under Targets (build plan decision 18, set by the maintainer on 2026-10-17 and confirmed on 2026-10-18). It is a serving engine for NVIDIA GPUs written in .NET, with kernels in CUDA C++ behind a C ABI and a few host-side native libraries in Rust. There is no Python at runtime.

The plan is [docs/plan/build-plan.md](../docs/plan/build-plan.md).

## Scope

- Every vLLM serving capability on NVIDIA GPUs, as the parity matrix in plan section 13.1 lists it: APIs, scheduling, sampling, speculation, quantisation, MoE, parallelism, KV tiers, disaggregated serving, multi-LoRA, multimodal and audio input, operations and hardware coverage.
- Distributed serving (tensor, pipeline, data and expert parallelism, multi-node, disaggregated prefill and decode) and bespoke GPU kernels (fused, persistent and collective kernels written by Roost, beside vendored upstream kernels pinned by full commit SHA).
- vLLM's Python plugin and custom-op APIs are covered by Roost model definitions and kernel libraries, and Ray by Roost's .NET launcher, because Roost has no Python at runtime (plan section 13.2).
- Training and fine-tuning are not vLLM capabilities and are out of scope. In-place weight update for reinforcement-learning loops is in scope.
- Backends beyond NVIDIA (ROCm, TPU, Gaudi, Neuron, XPU, CPU serving) are open: plan decision 29, *for the maintainer*.

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

[Starling](https://github.com/starlinghq/starling) works out the models; Roost runs on top of it. Roost consumes Starling's published NuGet packages, pinned exactly, and nothing else (plan section 2.2). Roost makes no changes to Starling's repository. It owns its own evidence pipeline, profiles, fixtures and tolerances in `evidence/`, and no throughput number is published for a Roost profile before it passes Roost's fidelity gates.

## Roles

| Area | Owner |
| --- | --- |
| Technical lead, maintainer, final release reviewer | Mark |
| Named consumer and use case | *For the maintainer.* |
| Kernel-layer owner | *For the maintainer.* The plan proposes two CUDA roles (kernels and worker; acceleration) and, from M4, a distributed-systems role (decision 19). |
| Second release owner | Open |

## Release authority

Only a human maintainer approves a release. Implementation agents open draft changes and produce evidence. They cannot approve their own change, hold publishing credentials, or change protected fixtures, tolerances or release evaluation sets in `evidence/protected/`.

## Maturity states

Roost uses the same states as Starling's charter (Experimental, Preview, Stable), and every capability carries an explicit status: supported, unsupported, experimental or untested. Missing never means supported.

## Budget

| Line | Value |
| --- | --- |
| Local GPU hours per milestone | *For the maintainer.* Measured by `tools/gpu-lease`; the recorded cap is 40 GPU-hours a month. |
| Rented single-GPU hours per milestone | *For the maintainer.* Plan decision 16. |
| Rented multi-GPU hours per milestone (NVLink pods, and a multi-node rental) | *For the maintainer.* Plan decision 20. |

Amounts are recorded in [docs/kickoff.md](../docs/kickoff.md) when the maintainer sets them. This charter approves no spend.

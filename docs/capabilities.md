# Capability table

What Roost can do today, one row per capability, each with an explicit status. The rows follow the parity matrix in build plan
section 13.1, plus the hardware, OS, op and profile rows that the PRD and AGENTS.md require. This is groundwork for R-026 (the
M0 gate publishes the table) and R-062 (Preview 1 links it from the package README); it records the state of `main` on
2026-11-01 and is not the gate's verdict.

## Statuses

| Status | Meaning |
| --- | --- |
| `supported` | Passed its gate with protected fixtures and tolerances in `evidence/protected/`. |
| `experimental` | Built and run on hardware (SM 86 or SM 89), with committed evidence and known caveats. |
| `untested` | Built and tested on the CPU path, or built only, without hardware evidence. |
| `unsupported` | Not built, or refused with a typed error naming the option. |

Rules:

- **Nothing is `supported`.** `evidence/protected/` does not exist yet: the fixtures and tolerances that R-014, R-019 and R-022
  measured are proposals (`evidence/proposals/`), and the maintainer has not placed them. R-026 promotes rows once they are.
- **Missing never means supported.** A capability not in this table is `unsupported`.
- **An SM without hardware evidence is `untested`.** Only an RTX 3090 (SM 86) and an RTX 4060 Ti (SM 89) have run anything.
- **Where code holds a status, this table copies it.** `src/Starlings.Generation.Roost/RoostCapabilities.cs` holds the
  `RoostChatClient` rows and `kernels/manifest.toml` the SM rows; `tests/tools/test_capabilities.py` fails if this table and
  either file disagree. Where the evidence would justify a different status, the row says so and the disagreement is listed
  under [Disagreements](#disagreements); the owning stream changes the code, then this table.
- Every row that is not `unsupported` names its evidence: committed logs and reports under `docs/m0/`, `kernels/results/`,
  `tests/*/results/` and `evidence/proposals/`, or, for CPU-only rows, the test project and status file. The paths are checked
  to exist. Times in those logs come from a shared machine and are not throughput figures (AGENTS.md: no throughput claim
  without fidelity evidence).

## Hardware

Status per SM, as `kernels/manifest.toml` lists it. Every SM library is built from one source; the CI kernels job builds all six.

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| SM 80 | `untested` | `kernels/manifest.toml` | Built only (digest in the manifest); never run. |
| SM 86 | `experimental` | `kernels/results/r017-ops-sm86.log`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/sanitizer-sm86/` | RTX 3090, driver 610.57. Conformance 578 passed, 0 failed, 43 refused as registered (R-019). FA2 prefill racecheck hazards (see `PagedAttention`). |
| SM 89 | `untested` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out`<br>`tests/Starlings.Roost.Conformance/results/sanitizer-sm89/`<br>`tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm89.log` | RTX 4060 Ti. The manifest says `untested`, so the loader needs `AllowUntested`; every SM 89 run passed it. The evidence equals SM 86's (conformance 578 passed, 0 failed): see [Disagreements](#disagreements). |
| SM 90 | `untested` | `kernels/manifest.toml` | Built only; never run. |
| SM 100 | `untested` | `kernels/manifest.toml` | Built only; never run. |
| SM 120 | `untested` | `kernels/manifest.toml` | Built only; never run. |
| SM 75 and older | `unsupported` | — | No library is built; the loader refuses the profile (`tests/Starlings.Roost.Kernels.Tests.Cuda/results/r023-loader-sm86.log` shows the refusal for SM 75). |
| ROCm (AMD Instinct and RDNA) | `unsupported` | — | M7 (R-110 to R-112, decision 29). |
| TPU, Gaudi, Neuron, XPU, CPU serving | `unsupported` | — | Not planned (decision 29). `CpuOps` is a reference, not a serving backend. |

## Operating systems

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| linux-x64 | `experimental` | `tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86.log`<br>`tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm86.log` | Every GPU run is on one Linux host. |
| win-x64 | `untested` | `.github/workflows/ci.yml` | CI builds and runs the CPU test suite on `windows-latest` and compiles the C headers with MSVC. No Windows kernel library is built (no CUDA toolkit on the hosted runner) and nothing has run on a GPU under Windows. |
| WSL2 | `untested` | `docs/plan/build-plan.md` | The plan (section 3) sets `untested` until someone runs the R-062 drill on a Windows host. No WSL2 run and no WSL2 document exist yet. |
| linux-arm64, macOS | `unsupported` | — | Not built. |

## Kernel ops (contract C3)

Status of each op on CUDA through the kernel library. Every registered variant passed conformance on SM 86 and SM 89
(R-019: `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`, `r019-cuda-sm89.out`), so a registered op is
`experimental`, the best status an op can have before R-026. `CpuOps` (stream D) is the reference for every op marked
"CPU reference"; it is not a serving backend.

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| `RmsNorm` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `rms-norm-flashinfer`, bf16 and f16. |
| `QkNorm` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `qk-norm-flashinfer`, head_dim 256 only. |
| `Rope` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `rope-roost`. |
| `PagedAttention` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out`<br>`tests/Starlings.Roost.Conformance/results/sanitizer-sm86/` | `paged-attention-flashinfer-fa2`. Prefill is experimental by the R-019 ruling: racecheck reports hazards in `BatchPrefillWithPagedKVCacheKernel` on both SMs, and prefill output depends on batch composition. Decode is clean. Sliding window (`WindowLeft != -1`) is refused (R-113). |
| `SiluAndMul` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `silu-and-mul-flashinfer`. |
| `Gemm` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`kernels/results/r019-ops-gemm-sm86.log` | `gemm-cublaslt`. Epilogues with a C input or a bias are refused (`UNSUPPORTED`); alpha is served. |
| `Embedding` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `embedding-roost`. |
| `LmHead` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `lm-head-cublaslt`. |
| `Residual` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `residual-stub` (R-016), correct but unoptimised. |
| `CacheWrite` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `cache-write-roost`. |
| `CacheCopy` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `cache-copy-roost`. |
| `Argmax` | `experimental` | `tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm86.log`<br>`tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm89.log` | No kernel: the eager worker copies the logits to the host and runs `CpuOps.Argmax`. Not in the conformance suite. The GPU sampler is R-042. |
| `Sample` | `unsupported` | — | No kernel; the worker refuses sampling plans (code 1003 in the R-025 logs). R-042. |
| `GeluAndMul` | `unsupported` | — | CPU reference only; no kernel. |
| `GatedDeltaRule` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out`<br>`kernels/results/r034-hybrid-conformance-sm86.log` | `gated-delta-rule-chunk-aot` and `gated-delta-rule-recurrent-aot`, Triton AOT cubins from flash-linear-attention. |
| `CausalConv1d` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `causal-conv1d-cuda`. |
| `RecurrentStateCopy` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `recurrent-state-copy-cuda`, bit-exact. |
| `RecurrentStateSnapshot` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `recurrent-state-snapshot-cuda`, bit-exact. |
| `GatedDeltaRuleGates` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `gated-delta-rule-gates-cuda`. |
| `GatedRmsNorm` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `gated-rms-norm-cuda`. |
| `SigmoidMul` | `experimental` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out`<br>`tests/Starlings.Roost.Conformance/results/r019-cuda-sm89.out` | `sigmoid-mul-cuda`. |
| `RopeScaled` | `unsupported` | — | No kernel. The CPU reference (YaRN, longrope, dynamic NTK, llama3) is experimental on the CPU (`src/Starlings.Roost.Ops/STATUS.md`, R-098). |
| `SlidingWindowAttention` | `unsupported` | — | No kernel. CPU reference for `PagedAttention` with a window exists (R-113). |
| Fused ops (`AddRmsNorm`, `AddRmsNormQuant`, `SiluMulQuant`, `QkvQkNormRopeCacheWrite`, `GemmLoraEpilogue`) | `unsupported` | — | Reserved op ids; no kernel and no CPU reference. R-071. |
| Quantisation ops (`Quantize`, `Dequantize`, `GemmW4A16`, `GemmFp8`, `DequantizeGguf`) | `unsupported` | — | No kernel. `CpuOps` has references for the first four (R-043, R-077; `src/Starlings.Roost.Ops/STATUS.md`); `DequantizeGguf` is a reserved op id (R-108). |
| Speculation ops (`TreeAttention`, `RejectSample`) | `unsupported` | — | Reserved op ids. R-083 to R-085. |
| MoE ops (`MoeRouteTopK`, `MoePermute`, `MoeUnpermute`, `GroupedGemm`, `FusedMoe`) | `unsupported` | — | Reserved op ids. S6 measured candidates on SM 86 (`docs/m0/roost-s6-moe-kernels.md`); nothing is in the library. `CpuOps` has bf16 references and conformance CPU cases (R-080). R-080, R-081. |
| LoRA ops (`LoraShrink`, `LoraExpand`) | `unsupported` | — | Reserved op ids. R-050. |
| Collectives (`AllReduce`, `AllGather`, `ReduceScatter`, `AllToAll`) | `unsupported` | — | Reserved op ids. R-089, R-091. |
| `MlaAttention` | `unsupported` | — | Reserved op id. R-093. |
| Multimodal ops (`PatchEmbed`, `BidirectionalAttention`, `MelSpectrogram`, `AudioConvSubsample`) | `unsupported` | — | Reserved op ids. R-099, R-107. |
| User kernel libraries registered through C1 | `unsupported` | — | R-072. |

## Model profiles

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| `qwen3.5-0.8b/cuda-bf16-eager` (Qwen3.5-0.8B, eager worker, bf16) | `experimental` | `tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm86.log`<br>`tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm89.log`<br>`docs/m0/r014-gpu/README.md`<br>`evidence/proposals/r014-gpu-tolerances.md` | Greedy tokens agree with `CpuOps` (24 of 24 per SM) and are measured against the R-014 reference fixtures, under proposed tolerances only. Inherits the FA2 prefill caveat. Checkpoint locked by `evidence/registry/qwen3.5-0.8b.lock.toml`. Greedy parity against protected fixtures is the R-026 condition. |
| `qwen3.5-0.8b/cpu-fp64-reference` (`CpuOps`, double) | `experimental` | `docs/m0/roost-r022-cpu-parity.md`<br>`evidence/proposals/r022-tolerances.md` | A reference profile for evaluation, not served. Per-layer comparison against transformers fp32 under proposed tolerances. |
| Qwen3 (R-015, Qwen3-1.7B first) | `unsupported` | — | No model definition or profile. The Qwen3 chat template is built and experimental on the CPU (`src/Starlings.Roost.Generation/STATUS.md`). |
| Roost tiny test models (R-021) | `untested` | `docs/models/r021-tiny-measurements.md` | Synthetic CPU-only models for the compiler and engine tests; not served. |

## Model families and formats

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Hybrid recurrent families (Qwen3.5, Gated DeltaNet) | `experimental` | `tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm86.log` | Qwen3.5-0.8B only (row above). Mamba-style models: `unsupported`. |
| Dense decoder families (Qwen, Llama, Mistral, Gemma) | `unsupported` | — | R-021, R-049, R-055. |
| MoE families, including MLA models | `unsupported` | — | R-081, R-093. |
| Long context: RoPE scaling | `unsupported` | — | CPU reference only (`RopeScaled`, R-098). |
| Long context: sliding window | `unsupported` | — | CPU reference only (R-113); the CUDA attention refuses a window. |
| Vision-language input | `unsupported` | — | R-099. |
| Audio input and output | `unsupported` | — | R-107. |
| GGUF checkpoints | `unsupported` | — | R-108. |
| safetensors checkpoints (bf16) | `experimental` | `tests/Starlings.Roost.Worker.Tests.Cuda/results/r024-weights-sm86.log` | The locked Qwen3.5-0.8B snapshot. |

## `RoostChatClient` (`IChatClient`)

These rows are the table in `src/Starlings.Generation.Roost/RoostCapabilities.cs`, row for row and status for status (R-036);
the capability column is the constant's value. Options for an `unsupported` row are refused with a typed `InvalidOption` error
naming the option. The GPU smoke runs (`run.sh`: three prompts, streamed and whole, one mid-stream cancel) passed on SM 86 and
SM 89 and are listed as evidence, but the code keeps these rows `untested` until the Preview 1 drill (R-062).

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| `chat` | `untested` | `tests/Starlings.Generation.Roost.Tests/`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86.log`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm89.log` | `GetResponseAsync`, in a worker process by default (decision 21). |
| `raw_completion` | `untested` | `tests/Starlings.Generation.Roost.Tests/` | `GetStreamingCompletionAsync`, no chat template. CPU only. |
| `streaming` | `untested` | `tests/Starlings.Generation.Roost.Tests/`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86.log`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm89.log` | `GetStreamingResponseAsync`. |
| `cancellation` | `untested` | `tests/Starlings.Generation.Roost.Tests/`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86.log`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm89.log` | Token or ending the enumeration; the worker drops the sequence from its next step. |
| `usage` | `untested` | `tests/Starlings.Generation.Roost.Tests/`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86.log` | Input and output token counts. |
| `greedy_decoding` | `untested` | `tests/Starlings.Generation.Roost.Tests/`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86.log` | `Temperature` null or 0. |
| `stop_token_ids` | `untested` | `tests/Starlings.Generation.Roost.Tests/`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86.log` | Ends with `ChatFinishReason.Stop`. |
| `max_output_tokens` | `untested` | `tests/Starlings.Generation.Roost.Tests/`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86.log` | Ends with `ChatFinishReason.Length`. |
| `instructions` | `untested` | `tests/Starlings.Generation.Roost.Tests/` | A leading system message. CPU only. |
| `stop_sequences` | `untested` | `tests/Starlings.Generation.Roost.Tests/`<br>`tests/Starlings.Roost.Generation.Tests/` | Matched on detokenized text in the host (Starling's `StopSequenceMatcher`). CPU only. |
| `sampling` | `unsupported` | — | Temperature, top-p, top-k. A seed is accepted under greedy decoding, where it changes nothing. R-042. |
| `penalties` | `unsupported` | — | Frequency and presence. R-042. |
| `tools` | `untested` | `tests/Starlings.Generation.Roost.Tests/Parsing/`<br>`src/Starlings.Roost.Server/STATUS.md` | R-053: `ChatOptions.Tools` (`AIFunction` declarations) rendered by the chat template; calls parsed incrementally into `FunctionCallContent`, finish reason `ToolCalls`. Qwen3.5 format through Starling's `Qwen35ToolCallParser`; Hermes (`<tool_call>` JSON) for a profile that declares it. `ToolMode` auto and none; a required or named call and `AllowMultipleToolCalls` false are refused (nothing constrains decoding). The Qwen3 template still refuses tools. CPU only, on a local corpus of template renderings (P-029 is not in this repository); no model has run with tools on Roost. |
| `response_format` | `unsupported` | — | JSON and JSON-schema structured output. Refused with `InvalidOption` naming `response_format`: without `RoostChatClientOptions.Grammar`, or, with it, after the format is compiled (an invalid or unsupported schema is refused with the grammar engine's reason), because the worker takes no token masks until contract C2 carries them. The engine and the CPU sampler path are the `untested` row under [Structured output](#structured-output-tools-and-reasoning). R-051, R-042. |
| `logprobs` | `unsupported` | — | R-042. |
| `reasoning` | `untested` | `tests/Starlings.Generation.Roost.Tests/Parsing/`<br>`src/Starlings.Roost.Server/STATUS.md` | R-053: `ChatOptions.AdditionalProperties["enable_thinking"]` opens the template's think block, and the completion is split incrementally into `TextReasoningContent` and the answer. Qwen3.5 only (the Qwen3 template still refuses thinking). CPU only, on a local corpus; no model has run with thinking on Roost. |
| `multimodal_input` | `unsupported` | — | R-099, R-107. |
| `conversation_state` | `unsupported` | — | `ConversationId`. |
| `continuous_batching` | `unsupported` | — | M0 runs one sequence per step; concurrent requests take turns. R-029 wiring, M1. |
| `worker_recovery` | `unsupported` | — | A worker fault fails the requests it carried. R-070. |
| `co_tenant_admission` | `untested` | `tests/Starlings.Roost.Engine.Tests/Admission/` | Refuses new requests with `InsufficientDeviceMemory` while free device memory is below `MinFreeDeviceBytes` (R-044). CPU only: the co-tenant GPU test is open and no NVML source is the default yet. |
| `telemetry` | `untested` | `tests/Starlings.Roost.Server.Tests/`<br>`src/Starlings.Roost.Server/STATUS.md` | `RoostChatClient.Telemetry` (R-038): a `Meter` per client (request counts by finish reason, token counters and histograms, time to first token, inter-token and end-to-end latency, running and waiting requests, KV block usage) and one `roost.request` activity per request tagged with its id, token counts and finish reason, never its text. CPU only; no GPU run has been scraped. |

## OpenAI-compatible server (`roost serve`, R-037, R-074)

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| `POST /v1/chat/completions` (whole and SSE) | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/` | CPU tests over a scripted in-process worker; no request has reached a GPU through the server. |
| `POST /v1/completions` (whole and SSE) | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/` | As above. |
| `GET /v1/models` | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/` | As above. |
| Native AOT `roost` binary (linux-x64) | `untested` | `src/Starlings.Roost.Server/STATUS.md` | One local publish with zero trim or AOT warnings; a build check, not serving evidence. S1 measured AOT with CUDA interop on SM 86 (`docs/m0/roost-s1-aot-cuda.md`). |
| Loopback-only listening by default | `untested` | `tests/Starlings.Roost.Server.Tests/` | Non-loopback needs `--allow-non-loopback`; there is no authentication. |
| `POST /v1/responses`, `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel` | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/` | Whole, SSE and background, over an in-memory bounded store with expiry; text input only. CPU tests over a scripted in-process worker (R-074). |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` (Anthropic Messages API) | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/` | Whole and SSE with Anthropic's events and error body; text blocks only; `stop_reason` `stop_sequence` is not reported (R-074). |
| `POST /tokenize`, `POST /detokenize` (vLLM bodies) | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/` | Prompt and chat-message forms over the model's tokenizer and template (R-074). |
| `GET /health`, `/ping`, `GET /version`, `GET /ready` | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/` | Liveness (vLLM's names) and readiness (worker started, host not stopping); no worker liveness probe (R-074). |
| Prompt-embeddings input | `unsupported` | — | `prompt_embeds` is refused by name: `RoostChatClient` exposes no embedding input for contract C2's `Embeddings` part (R-074). |
| `POST /v1/embeddings` (Starling's E5 and EmbeddingGemma 2 packages) | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/PoolingRoutesTests.cs`<br>`tests/Starlings.Roost.ApiCompat.Tests/EmbeddingsCompatTests.cs` | R-052: OpenAI's request and response (`float` and `base64`, `dimensions` equal to the served length) over `Starlings.Models.E5` and `Starlings.Models.EmbeddingGemma2` 0.3.0 on ONNX Runtime's CPU provider in the server process, loaded from verified local snapshots with `roost serve --pooling-e5` and `--pooling-embeddinggemma2`. CPU tests use fakes behind `IRoostEmbeddingModel`; no model has run through the route here. 400 `profile_unsupported` when no embedding model is configured. |
| `POST /v1/extract` (Starling's GLiNER2 package) | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/PoolingRoutesTests.cs` | R-052: Roost's own shape (labels and threshold in, labelled spans with scalar offsets out) over `Starlings.Models.Gliner2` 0.3.0 on ONNX Runtime's CPU provider, `roost serve --pooling-gliner2`. CPU tests use fakes behind `IRoostExtractionModel`; no model has run through the route here. |
| `POST /v1/rerank` | `unsupported` | — | No Starling package provides a reranking model (nuget.org, 2026-11-02): the route answers 400 `profile_unsupported` naming `rerank`. |
| `POST /v1/classify` | `unsupported` | — | No Starling package provides a classification model (nuget.org, 2026-11-02): the route answers 400 `profile_unsupported` naming `classify`. |
| Score, pooling, Cohere- and Jina-style routes | `unsupported` | — | vLLM's `/v1/score`, `/pooling` and rerank aliases are not built (R-052, R-087). |
| Pooling models on the GPU within the device budget | `unsupported` | — | The pooling routes run ONNX Runtime on the CPU only; the CUDA provider is not offered, and native pooling models in the worker are R-087. |
| Runtime LoRA load and unload | `unsupported` | — | R-050, R-074. |
| `GET /metrics` (Prometheus text format) | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/` | R-038: every family under a `roost_` name and under vLLM's `vllm:` names (`request_success_total`, `prompt_tokens_total`, `generation_tokens_total`, `time_to_first_token_seconds`, `inter_token_latency_seconds`, `time_per_output_token_seconds`, `e2e_request_latency_seconds`, `request_prompt_tokens`, `request_generation_tokens`, `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`, `gpu_cache_usage_perc`) with the `model_name` label. No exporter package. Prefix-cache, preemption and queue-time metrics are not exported (the M0 driver has no prefix cache). |
| One activity per request | `untested` | `tests/Starlings.Roost.Server.Tests/` | `roost.request` on the `Starlings.Roost` `ActivitySource`: request id, model, input and output tokens, finish reason, and the C6 code name on failure. No exporter is wired; a host adds its own `ActivityListener` or OpenTelemetry SDK. |
| No prompt or output text in logs | `untested` | `tests/Starlings.Roost.Server.Tests/` | A test captures every log category at `Trace` over every route, refusals included, and finds neither the prompt nor the reply. |
| vLLM CLI flags and environment | `unsupported` | — | R-075. |
| OpenAI client library compatibility suite | `untested` | `tests/Starlings.Roost.ApiCompat.Tests/`<br>`tests/Starlings.Roost.ApiCompat.Tests/STATUS.md` | R-040: the official OpenAI .NET client (2.14.0, in CI) drives chat completions, tool calls, reasoning, the Responses API, models and error bodies on Kestrel over a scripted worker, and asserts every unsupported option it can set is a 400 naming it; the official openai Python package (3.35.1, `tests/tools/api_compat/openai_compat.py`) is an opt-in run that also covers `/v1/completions`. No client has run against a GPU. |

## Scheduling and engine

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Chunked prefill | `untested` | `tests/Starlings.Roost.Engine.Tests/Scheduling/` | Always on in the step planner (R-029), which `RoostChatClient` drives. The GPU runs never split a prompt (prompts of 17 to 24 tokens under a 64-token step budget). |
| Recompute preemption | `untested` | `tests/Starlings.Roost.Engine.Tests/Scheduling/` | In the planner; unreachable with one sequence per step. |
| Request priority | `unsupported` | — | The planner orders by priority, but neither `RoostChatClient` nor the server sets one. |
| Continuous batching | `unsupported` | — | See `continuous_batching` above. |
| Admission control (worst-case reservation) | `untested` | `tests/Starlings.Roost.Engine.Tests/Admission/` | `Starlings.Roost.Engine/Admission`, CPU tests only. |
| Prefix caching, attention | `unsupported` | — | `Starlings.Roost.Engine/Caching/PrefixCache` is built and CPU-tested (`tests/Starlings.Roost.Engine.Tests/Caching/`), but the M0 driver does not use it. R-030. |
| Prefix caching, hybrid (recurrent state) | `unsupported` | — | R-094. |
| Async scheduling | `unsupported` | — | R-032. |
| CUDA graphs, full and piecewise | `unsupported` | — | The worker is eager. The one R-017 capture on SM 86 replayed with a mismatch (`kernels/results/r017-graph-sm86.log`). R-032, R-033. |
| Compiler fusion passes | `unsupported` | — | No CUDA fused variant. The AddRmsNorm and QkvQkNormRopeCacheWrite passes and their `CpuOps` definitions exist and are checked against the unfused graph on CPU (R-071). R-089. |
| Batch-invariant mode | `unsupported` | — | R-106. FA2 prefill output depends on batch composition (R-019). |

## Worker process and transports (contract C8)

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Worker process over shared-memory rings (default) | `experimental` | `tests/Starlings.Roost.WorkerHost.Tests.Cuda/results/README.md`<br>`tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86-roost-worker.log` | One worker per GPU (R-069); handshake checks ABI, manifest and weight digests; heartbeat. Measured on SM 86. |
| In-process transport | `experimental` | `tests/Starlings.Roost.WorkerHost.Tests.Cuda/results/README.md`<br>`tests/Starlings.Roost.WorkerHost.Tests/` | For tests only (decision 21, decision 30). |
| Worker fault classification (recoverable, sticky, hang) | `untested` | `tests/Starlings.Roost.WorkerHost.Tests/results/stress-runs.txt` | CPU fault-injection stress runs; recovery itself is `unsupported` (`worker_recovery`). |
| KV block transport between processes and hosts | `unsupported` | — | `native/roost-kvx`, R-095. |

## Sampling

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Greedy decoding | `untested` | `tests/Starlings.Generation.Roost.Tests.Cuda/results/r036-chat-sm86.log` | See `greedy_decoding`. |
| All sampling parameters, logprobs, seeds | `unsupported` | — | R-042. |
| n > 1, best_of, beam search | `unsupported` | — | R-076. The server accepts `n` 1 and `best_of` 1 only. |
| Speculative decoding (n-gram, suffix, draft model, MTP, EAGLE, newer proposers) | `unsupported` | — | R-083 to R-085. |

## Structured output, tools and reasoning

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Structured output (JSON schema, regex, grammar) | `unsupported` | — | Not served: see `response_format`. The server reads `response_format` and `text.format` (`text`, `json_object`, `json_schema`) and refuses every other type, regular expressions and grammars included. R-051, R-042. |
| Grammar engine and masked CPU sampling (`native/roost-grammar`, `Starlings.Roost.Native`) | `untested` | `native/roost-grammar/tests/`<br>`tests/Starlings.Roost.Native.Tests/`<br>`src/Starlings.Roost.Native/STATUS.md` | R-051: llguidance (pinned by SHA) behind a C ABI; JSON schema, any JSON object, regex and Lark matchers; masks per C2 `grammar_slot` in `GrammarSlots`, computed on thread-pool threads; `GrammarSampler` applies them in `CpuOps.Sample` and commits each token. CPU tests: a masked token is never sampled under adversarial logits, and every greedy and sampled output of roost-tiny and roost-tiny-moe under three JSON schemas parses and satisfies its schema. Not reachable from `RoostChatClient` or the server yet; the library is not built in CI (the tests skip). No mask latency measured here; S10 measured 17 to 27 us p50 per token on Qwen3's vocabulary. |
| Tool calling parsers | `untested` | `tests/Starlings.Generation.Roost.Tests/Parsing/` | See `tools`. Qwen3.5 (Starling's parser) and Hermes; both streaming, every split of a corpus completion giving the whole parse. Other families: `unsupported`. R-053. |
| Reasoning parsers | `untested` | `tests/Starlings.Generation.Roost.Tests/Parsing/` | See `reasoning`. Qwen think blocks (`</think>` after an opened block). Other families: `unsupported`. R-053. |
| OpenAI `tools`, `tool_choice`, `tool_calls` and vLLM `reasoning_content` | `untested` | `tests/Starlings.Roost.Server.Tests/`<br>`src/Starlings.Roost.Server/STATUS.md` | `POST /v1/chat/completions` only, whole and SSE (one `tool_calls` delta per call); vLLM's `chat_template_kwargs.enable_thinking`. The Responses, Messages and tokenize routes still refuse tools. R-053. |

## Quantisation and data types

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| bf16 weights, activations and KV cache | `experimental` | `tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm86.log`<br>`tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm89.log` | The one served profile. |
| f16 activations | `untested` | `tests/Starlings.Roost.Conformance/results/r019-cuda-sm86.out` | Kernels pass f16 conformance on SM 86 and SM 89; no model profile runs f16. |
| FP8 weights and activations, FP8 KV | `unsupported` | — | No kernel; CPU references only (R-043, R-077). |
| AWQ and GPTQ INT4 (Marlin) | `unsupported` | — | R-077. S6 has compile evidence only. |
| MXFP4 and NVFP4 | `unsupported` | — | R-078. |
| AutoRound weight formats, 2-bit KV cache | `unsupported` | — | R-109. |

## MoE and parallelism

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Fused MoE (bf16, FP8, INT4) | `unsupported` | — | R-080, R-081. |
| Expert parallelism, all-to-all, expert load balancing | `unsupported` | — | R-091. |
| Elastic expert parallelism | `unsupported` | — | R-092. |
| CPU expert offload | `unsupported` | — | R-082. |
| Multi-GPU: tensor and pipeline parallelism | `unsupported` | — | R-089, R-090. One worker drives one GPU; nothing spans GPUs. |
| Data-parallel attention | `unsupported` | — | R-091. |
| Multi-node | `unsupported` | — | R-092. |
| Context parallelism | `unsupported` | — | R-097. |

## KV cache

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Paged KV cache with recurrent state pool | `experimental` | `tests/Starlings.Roost.Worker.Tests.Cuda/results/r025-eager-decode-sm86.log` | Block manager (R-028) and recurrent slots, one sequence per step. |
| CPU, disk and remote KV offload | `unsupported` | — | The host-memory tier is planned and cache-managed in the engine and CPU-tested (R-094, off by default); the CUDA worker does not execute tier moves. Disk and remote: R-096. |
| Disaggregated prefill and decode | `unsupported` | — | R-095. |
| Encoder disaggregation | `unsupported` | — | R-099. |

## Adapters, operations and extension

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Multi-LoRA (with graphs and prefix-cache awareness) | `unsupported` | — | R-050. |
| LoRA on MoE experts and multimodal backbones | `unsupported` | — | R-050, R-099. |
| Sleep and wake, in-place weight update | `unsupported` | — | R-100. |
| Per-GPU weight cache and engine snapshots | `unsupported` | — | R-046, R-100. |
| Metrics and traces | `untested` | `src/Starlings.Roost.Server/STATUS.md`<br>`tests/Starlings.Roost.Server.Tests/` | See `GET /metrics` and one activity per request above (R-038). CPU only. |
| Container image | `unsupported` | — | R-046: `deploy/Dockerfile` (AOT `roost` and `roost-worker`, kernel manifest and libraries on `nvidia/cuda:13.3.1-runtime-ubuntu24.04` pinned by digest) and `deploy/stage-image.sh` are written but no image has been built, measured against the 4 GB criterion or published (`deploy/STATUS.md`). |
| `roost doctor` | `untested` | `tests/Starlings.Roost.Server.Tests/DoctorTests.cs`<br>`deploy/STATUS.md` | R-046: driver and its minimum, NVML, GPUs and their SMs against the manifest's statuses, kernel library digests against the manifest, weights lock, tokenizer stop tokens, free disk and host memory, each a typed pass, warn, fail or skip; exit 1 on a failure. CPU-tested with fake probes; never run on a GPU host. |
| Cold-start stage timing | `untested` | `tests/Starlings.Roost.Server.Tests/ColdStartTests.cs`<br>`deploy/README.md` | R-046: `roost serve` logs each start stage (`runtime`, `arguments`, `profile`, `worker`, `listen`) and exports them as `roost_cold_start_stage_seconds` on `GET /metrics`. Per-stage budgets are proposals; no real start has been timed. |
| Kubernetes, router integration | `unsupported` | — | R-101. |
| Model definitions as the plugin equivalent | `unsupported` | — | R-055 (section 13.2). |
| .NET launcher as the Ray equivalent | `unsupported` | — | R-092 (section 13.2). |
| NuGet packages on nuget.org | `unsupported` | — | Nothing is released. The R-062 drill packed `Starlings.Roost.Worker.Runtime` locally (`tests/Starlings.Generation.Roost.Tests.Cuda/results/README.md`). |

## Disagreements

Reported, not changed here: each is owned by another stream or needs a maintainer decision.

1. **SM 89 in `kernels/manifest.toml` is `untested` although it has the same evidence as SM 86.** R-019 conformance (578 passed, 0
   failed), the sanitizer runs, R-025 greedy decode and R-036 chat all ran on the RTX 4060 Ti, with `AllowUntested`. By the rule
   "SM 86/89 evidence is experimental" SM 89 would be `experimental`. Setting it is stream C's change (the manifest) and stream F's
   (`ManifestAndNamesTests` asserts the current value); `tests/Starlings.Roost.Conformance/README.md` already leaves it to the
   maintainer. This table follows the manifest.
2. **`RoostCapabilities` keeps `chat`, `streaming`, `cancellation`, `usage`, `greedy_decoding`, `stop_token_ids` and
   `max_output_tokens` at `untested` although the R-036 GPU smoke runs passed on SM 86 and SM 89.** The class documents that they
   move to `experimental` when they pass the Preview 1 drill (R-062), so this is by design, but it is stricter than the hardware-
   evidence rule used elsewhere in this table (for example the worker process transport is `experimental` on the same runs). Stream H
   owns the class; this table follows it.
3. **The kernel manifest (C5) has no per-variant status.** The FA2 prefill path of `paged-attention-flashinfer-fa2` is
   experimental by the R-019 ruling, but only `kernels/README.md`, the conformance README and this table say so; nothing machine
   readable does. A `status` field on a manifest variant (or a C1 flag) is a contract change for stream A at R-026.

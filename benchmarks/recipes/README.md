# Competitor runner recipes

Section 14.3 compares Roost with vLLM v0.31.0 and SGLang, pinned at each gate and run with their published recipes for
the model, and with llama.cpp for W7 and MoE offload on consumer cards. These recipes are drafts for R-047: they describe
how each reference environment is set up outside the repository. Nothing here has been run.

Every value marked `TODO-maintainer` was not verified while writing the recipe and must be filled in from the upstream
release before a run. Never fill a digest or SHA from memory or from a third-party page: pull the image or check out the
commit and record what the tool reports.

## Rules shared by every recipe

- **Same model, same weights.** Load the same Hugging Face repository at the same full commit SHA that the Roost profile
  locks (`evidence/registry/`). Pass the revision explicitly; never a branch name.
- **Same dtype.** bf16 weights and bf16 activations unless the profile under test says otherwise. Quantised profiles are
  compared only with the same quantisation format on the competitor, or not at all.
- **TF32 policy: off for fp32 matmuls, recorded.** Roost's Gemm runs bf16 inputs with fp32 accumulation
  (`CUBLAS_COMPUTE_32F`, docs/m0/roost-s2-kernel-extraction.md), which has no TF32 path. Competitors run with
  `NVIDIA_TF32_OVERRIDE=0` in the server environment, NVIDIA's documented switch that stops cuBLAS and cuDNN from using
  TF32. It does not reach custom kernels (for example Triton kernels whose fp32 dots default to TF32), so each recipe
  also records the engine's own setting. If an engine cannot run with TF32 off, the run is reported with its TF32 state
  and no "faster" claim is made from it.
- **Same KV cache dtype** as the Roost profile (bf16 unless the profile says FP8).
- **Same context limit and sampling.** `--max-model-len` (or the engine's equivalent) equal to the Roost profile's
  context; the harness sends `temperature` 0 by default.
- **Record the environment.** Image digest or commit SHA, driver and CUDA versions (`nvidia-smi`), GPU model and clock
  state, the full server command line, and the harness command line, next to the raw results. Raw results are release
  assets, never committed.
- **GPU leases.** On the shared development GPUs, every server runs under `tools/gpu-lease run --stream I ...`.

## Recipes

- [vLLM v0.31.0](vllm-v0.31.0.md)
- [SGLang](sglang.md)
- [llama.cpp](llama-cpp.md)

# vLLM v0.31.0 reference runner (draft, not run)

The section 14.3 baseline: [vLLM v0.31.0](https://github.com/vllm-project/vllm/releases) (released 2026-10-05, build
plan section 1). The shared dtype and TF32 rules are in [README.md](README.md).

## Pin

| Item | Value |
| --- | --- |
| Release | v0.31.0 |
| Source commit | `TODO-maintainer`: full SHA of the `v0.31.0` tag, from `git rev-parse v0.31.0^{commit}` in a clone |
| Image | `vllm/vllm-openai:v0.31.0` |
| Image digest | `TODO-maintainer`: `sha256:...` from `docker buildx imagetools inspect vllm/vllm-openai:v0.31.0` (or `docker inspect --format '{{index .RepoDigests 0}}'` after a pull); run by digest, not tag |
| Model recipe | `TODO-maintainer`: the published vLLM recipe for the model under test (vllm-project/recipes), pinned by full commit SHA |

## Server

Run by digest, with TF32 off for cuBLAS and cuDNN:

```sh
docker run --rm --gpus '"device=<index>"' --ipc=host -p 8000:8000 \
    -e NVIDIA_TF32_OVERRIDE=0 \
    -v <hf cache outside Git>:/root/.cache/huggingface \
    vllm/vllm-openai@sha256:<TODO-maintainer digest> \
    --model <hf repo> --revision <40-character SHA from the Roost profile lock> \
    --served-model-name <name> \
    --dtype bfloat16 \
    --max-model-len <Roost profile context> \
    --seed 0
```

Add the flags the published model recipe gives (for example the KV cache dtype, tool-call parser and reasoning parser
for W4) and record them. Prefix caching and chunked prefill are left at the release's defaults unless the recipe says
otherwise; record the values the server logs at start.

## TF32 check

Record the PyTorch settings inside the same image and environment, next to the results:

```sh
docker run --rm --gpus '"device=<index>"' -e NVIDIA_TF32_OVERRIDE=0 --entrypoint python3 \
    vllm/vllm-openai@sha256:<TODO-maintainer digest> -c \
    "import torch; print(torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())"
```

Whether vLLM v0.31.0 changes these at server start, and whether any of its Triton kernels used by the model run fp32
dots in TF32, is `TODO-maintainer` to confirm from the v0.31.0 source at the pinned SHA.

## Harness

```sh
python3 -I benchmarks/bench.py run --workload benchmarks/workloads/w1-chat.toml \
    --base-url http://<host>:8000/v1 --model <name> --label vllm-0.31.0 \
    --slo benchmarks/slo.toml --out <results directory outside Git>
```

vLLM accepts `ignore_eos` and `min_tokens` as extra sampling fields; pass them with `--extra-body` only when every
backend in the comparison gets the same fields.

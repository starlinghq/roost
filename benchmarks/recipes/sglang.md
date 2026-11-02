# SGLang reference runner (draft, not run)

Section 14.3 names SGLang as the second competitor, pinned at each gate. The build plan does not fix a version; the
version is part of decision 25. The shared dtype and TF32 rules are in [README.md](README.md).

## Pin

| Item | Value |
| --- | --- |
| Release | `TODO-maintainer`: the SGLang release chosen under decision 25 |
| Source commit | `TODO-maintainer`: full SHA of that release tag |
| Image | `lmsysorg/sglang:<TODO-maintainer tag>` |
| Image digest | `TODO-maintainer`: `sha256:...` from `docker buildx imagetools inspect lmsysorg/sglang:<tag>`; run by digest, not tag |
| Model recipe | `TODO-maintainer`: SGLang's published recipe for the model under test, pinned by full commit SHA |

## Server

```sh
docker run --rm --gpus '"device=<index>"' --ipc=host -p 30000:30000 \
    -e NVIDIA_TF32_OVERRIDE=0 \
    -v <hf cache outside Git>:/root/.cache/huggingface \
    lmsysorg/sglang@sha256:<TODO-maintainer digest> \
    python3 -m sglang.launch_server \
        --model-path <hf repo> --revision <40-character SHA from the Roost profile lock> \
        --served-model-name <name> \
        --dtype bfloat16 \
        --context-length <Roost profile context> \
        --host 0.0.0.0 --port 30000 \
        --random-seed 0
```

Add the flags SGLang's published recipe gives for the model (KV cache dtype, tool-call and reasoning parsers for W4) and
record them. The radix prefix cache is left on (SGLang's default), since W3 and W5 measure prefix reuse; record the
server's start-up log. The flag spellings above must be checked against the pinned release's `--help`
(`TODO-maintainer`).

## TF32 check

As for vLLM: record `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32` and
`torch.get_float32_matmul_precision()` inside the pinned image with `NVIDIA_TF32_OVERRIDE=0`. Whether SGLang or its
attention backend changes them is `TODO-maintainer` to confirm from the pinned source.

## Harness

```sh
python3 -I benchmarks/bench.py run --workload benchmarks/workloads/w3-agentic-multiturn.toml \
    --base-url http://<host>:30000/v1 --model <name> --label sglang-<version> \
    --slo benchmarks/slo.toml --out <results directory outside Git>
```

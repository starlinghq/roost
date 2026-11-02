# llama.cpp reference runner (draft, not run)

Section 14.3 uses llama.cpp for W7 (single user, batch 1) and for MoE offload on consumer cards, not for the batched
workloads. The shared rules are in [README.md](README.md).

## Pin

| Item | Value |
| --- | --- |
| Source commit | `TODO-maintainer`: full SHA of the llama.cpp commit (ggml-org/llama.cpp) chosen under decision 25 |
| Build | from source at that SHA: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<86, 89 or 120> && cmake --build build --config Release -j` |
| Converter | `convert_hf_to_gguf.py` from the same SHA |
| GGUF file | converted from the Roost profile's locked Hugging Face revision; record its SHA-256 |

## Dtype

Convert at bf16 so the weights match the Roost bf16 profile:

```sh
python3 convert_hf_to_gguf.py <model directory at the locked revision> --outtype bf16 --outfile <model>.bf16.gguf
```

A quantised GGUF (Q4_K_M and similar) is a different format from any Roost profile; compare it only with a Roost profile
of matching precision, and say so with the result. llama.cpp's CUDA backend chooses its own matmul paths and compute
types, and `NVIDIA_TF32_OVERRIDE=0` covers only its cuBLAS calls; which paths a bf16 model takes on each SM is
`TODO-maintainer` to confirm from the pinned source and record.

## W7 server (single user)

```sh
NVIDIA_TF32_OVERRIDE=0 build/bin/llama-server -m <model>.bf16.gguf \
    --alias <name> -c <Roost profile context> -ngl 999 -np 1 \
    --host 0.0.0.0 --port 8080 --seed 0
```

## MoE offload server (consumer cards)

Expert tensors on the CPU, the rest on the GPU. The flag that places expert weights on the CPU (`--n-cpu-moe` or an
`--override-tensor` pattern for the expert tensors) and its exact spelling at the pinned SHA are `TODO-maintainer` to
confirm from `llama-server --help`. Record the CPU model, core count, `-t` thread count and host memory bandwidth with
the results, since they bound decode speed.

## Harness

llama.cpp's server speaks `/v1/chat/completions` and `/v1/completions`. W7:

```sh
python3 -I benchmarks/bench.py run --workload benchmarks/workloads/w7-single-user.toml \
    --base-url http://<host>:8080/v1 --model <name> --label llama.cpp-<short sha> \
    --slo benchmarks/slo.toml --out <results directory outside Git>
```

W7 asks for a fixed 512 output tokens; a model that stops early at an end-of-sequence token produces shorter replies.
llama.cpp, vLLM and SGLang accept `ignore_eos`; Roost's server does not, so a W7 comparison with Roost uses the same
prompts without it and compares ITL and TPOT, which do not depend on reply length.

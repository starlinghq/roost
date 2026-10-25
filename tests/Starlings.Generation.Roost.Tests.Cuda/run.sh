#!/usr/bin/env bash
# R-036 GPU run: Qwen3.5-0.8B through RoostChatClient with the eager worker in a worker process (decision 21), on a GPU leased
# through tools/gpu-lease and charged to stream H. The checkpoint, tokenizer and kernel library stay outside the repository.
# The log starts with the leased GPU's name, driver and compute capability and the CUDA versions, then the harness prints the
# digests of the kernel libraries, manifest and weights it loads.
# usage: run.sh <checkpoint-dir> <kernel-build-dir> <tokenizer.json> [label]
# ROOST_GPU_DEVICE picks the gpu-lease device (default 3090; 4060 for the RTX 4060 Ti, SM 89, with label sm89).
set -euo pipefail
cd "$(dirname "$0")/../.."
HUB=${1:?checkpoint directory holding model.safetensors-00001-of-00001.safetensors}
KERNELS=${2:?directory holding libroost_kernels_sm<N>.so for the device, built from kernels/CMakeLists.txt at this commit}
TOKENIZER=${3:?Qwen3.5 tokenizer.json (Starlings.Tokenizers profile qwen35-bpe-chat)}
LABEL=${4:-sm86}
DEVICE=${ROOST_GPU_DEVICE:-3090}
# ROOST_KERNEL_MANIFEST names another manifest, e.g. one that records the digests of a newer kernel build.
MANIFEST=${ROOST_KERNEL_MANIFEST:-kernels/manifest.toml}
EXE=tests/Starlings.Generation.Roost.Tests.Cuda/bin/Release/net10.0/Starlings.Generation.Roost.Tests.Cuda
RESULTS=tests/Starlings.Generation.Roost.Tests.Cuda/results
# Printed inside the lease, where CUDA_VISIBLE_DEVICES is the leased GPU's UUID.
IDENTIFY='nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,driver_version,compute_cap --format=csv,noheader | sed "s/^/gpu: /"
echo "cuda: driver UMD $(nvidia-smi --version | grep -o "CUDA UMD version *: *[0-9.]*" | grep -o "[0-9.]*$"); toolkit $(nvcc --version 2>/dev/null | grep -o "release [0-9.]*" || echo unknown)"
echo "command: $*"
exec "$@"'
dotnet build tests/Starlings.Generation.Roost.Tests.Cuda -c Release -v q
mkdir -p "$RESULTS"
tools/gpu-lease run --device "$DEVICE" --need-mib 6000 --wait 3600 --stream H -- \
  bash -c "$IDENTIFY" identify "$EXE" --ckpt "$HUB" --lock evidence/registry/qwen3.5-0.8b.lock.toml --manifest "$MANIFEST" \
  --kernels "$KERNELS" --tokenizer "$TOKENIZER" --allow-untested 2>&1 | tee "$RESULTS/r036-chat-$LABEL.log"

#!/usr/bin/env bash
# R-036 GPU run: Qwen3.5-0.8B through RoostChatClient with the eager worker in a worker process (decision 21), on a GPU leased
# through tools/gpu-lease and charged to stream H. The checkpoint, tokenizer and kernel library stay outside the repository.
# usage: run.sh <checkpoint-dir> <kernel-build-dir> <tokenizer.json> [label]
set -euo pipefail
cd "$(dirname "$0")/../.."
HUB=${1:?checkpoint directory holding model.safetensors-00001-of-00001.safetensors}
KERNELS=${2:?directory holding libroost_kernels_sm86.so built from kernels/CMakeLists.txt at this commit}
TOKENIZER=${3:?Qwen3.5 tokenizer.json (Starlings.Tokenizers profile qwen35-bpe-chat)}
LABEL=${4:-sm86}
EXE=tests/Starlings.Generation.Roost.Tests.Cuda/bin/Release/net10.0/Starlings.Generation.Roost.Tests.Cuda
RESULTS=tests/Starlings.Generation.Roost.Tests.Cuda/results
dotnet build tests/Starlings.Generation.Roost.Tests.Cuda -c Release -v q
mkdir -p "$RESULTS"
tools/gpu-lease run --device 3090 --need-mib 6000 --wait 3600 --stream H -- \
  "$EXE" --ckpt "$HUB" --lock evidence/registry/qwen3.5-0.8b.lock.toml --manifest kernels/manifest.toml \
  --kernels "$KERNELS" --tokenizer "$TOKENIZER" --allow-untested 2>&1 | tee "$RESULTS/r036-chat-$LABEL.log"

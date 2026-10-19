#!/usr/bin/env bash
# R-025 GPU evidence: greedy decode of Qwen3.5-0.8B on the eager worker (SM 86 through gpu-lease), then the same prompts on
# CpuOps (double precision, teacher-forced with the GPU's tokens) and a text report of the differences. Two processes so the
# GPU lease is held only while the GPU is used. Every heavy step takes the wave's heavy lock. The checkpoint, the kernel
# library and the dump stay outside the repository.
# usage: run-eager.sh <checkpoint-dir> <kernel-build-dir> <scratch-dir> [label]
set -euo pipefail
cd "$(dirname "$0")/../.."
HUB=${1:?checkpoint directory holding model.safetensors-00001-of-00001.safetensors}
KERNELS=${2:?directory holding libroost_kernels_sm86.so built from kernels/CMakeLists.txt at this commit}
SCRATCH=${3:?scratch directory outside the repository}
LABEL=${4:-sm86}
HEAVY=${ROOST_HEAVY_LOCK:-/dev/null}
LOCK=evidence/registry/qwen3.5-0.8b.lock.toml
MANIFEST=kernels/manifest.toml
EXE=tests/Starlings.Roost.Worker.Tests.Cuda/bin/Release/net10.0/Starlings.Roost.Worker.Tests.Cuda
RESULTS=tests/Starlings.Roost.Worker.Tests.Cuda/results
dotnet build tests/Starlings.Roost.Worker.Tests.Cuda -c Release -v q
mkdir -p "$SCRATCH"
flock "$HEAVY" -c "tools/gpu-lease run --device 3090 --need-mib 6000 --wait 3600 --stream F -- $EXE gpu --ckpt $HUB --lock $LOCK --manifest $MANIFEST --kernels $KERNELS --out $SCRATCH/gpu.dump --allow-untested" 2>&1 | tee "$RESULTS/r025-eager-decode-$LABEL.log"
flock "$HEAVY" -c "$EXE cpu --ckpt $HUB --lock $LOCK --dump $SCRATCH/gpu.dump --report $RESULTS/r025-cpu-compare-$LABEL.txt" 2>&1 | tail -n 5

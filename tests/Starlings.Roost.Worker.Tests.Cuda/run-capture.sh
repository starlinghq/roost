#!/usr/bin/env bash
# R-014 GPU check: the eager worker teacher-forced on the R-014 reference cases with per-layer captures (CaptureGpu.cs), then the
# per-layer evaluator (evidence/pipeline). Captures and reports are run artifacts and stay outside the repository.
# usage: run-capture.sh <checkpoint-dir> <kernel-dir> <reference-run-dir> <out-dir> [gpu-lease device, default 3090]
set -euo pipefail
cd "$(dirname "$0")/../.."
HUB=${1:?checkpoint directory holding model.safetensors-00001-of-00001.safetensors}
KERNELS=${2:?directory holding libroost_kernels_sm<N>.so for the device}
REF=${3:?reference run directory (reference.jsonl, layers/, step0/)}
OUT=${4:?output directory outside the repository}
DEVICE=${5:-3090}
EXE=tests/Starlings.Roost.Worker.Tests.Cuda/bin/Release/net10.0/Starlings.Roost.Worker.Tests.Cuda
dotnet build tests/Starlings.Roost.Worker.Tests.Cuda -c Release -v q
mkdir -p "$OUT"
tools/gpu-lease run --device "$DEVICE" --need-mib 8000 --wait 3600 --stream B3 -- \
  "$EXE" capture --ckpt "$HUB" --lock evidence/registry/qwen3.5-0.8b.lock.toml --manifest kernels/manifest.toml \
  --kernels "$KERNELS" --reference "$REF" --out "$OUT" --allow-untested 2>&1 | tee "$OUT/capture.log"
# Measurement only: every bound loose, so the report carries the measured values (see evidence/proposals/r014-gpu-tolerances.md).
uv run --project evidence/pipeline --locked roost-evidence evaluate-layers "$OUT" --reference "$REF/reference.jsonl" \
  --reference-captures "$REF" --tolerances "${ROOST_R014_TOLERANCES:?a tolerance file (json)}" --report "$OUT/layers.json"

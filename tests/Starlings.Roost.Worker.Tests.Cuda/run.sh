#!/usr/bin/env bash
# R-024 GPU evidence: the locked Qwen3.5-0.8B checkpoint streamed to the RTX 3090 and verified as read, then a
# one-byte-substituted copy refused. One short lease per run. The checkpoint stays outside the repository.
# usage: run.sh <checkpoint-dir> <scratch-dir> [log]
set -euo pipefail
cd "$(dirname "$0")/../.."
HUB=${1:?checkpoint directory holding model.safetensors-00001-of-00001.safetensors}
SCRATCH=${2:?scratch directory outside the repository}
LOG=${3:-tests/Starlings.Roost.Worker.Tests.Cuda/results/r024-weights-sm86.log}
NAME=model.safetensors-00001-of-00001.safetensors
LOCK=evidence/registry/qwen3.5-0.8b.lock.toml
EXE=tests/Starlings.Roost.Worker.Tests.Cuda/bin/Release/net10.0/Starlings.Roost.Worker.Tests.Cuda
dotnet build tests/Starlings.Roost.Worker.Tests.Cuda -c Release -v q
mkdir -p "$SCRATCH"
{
  echo "== locked checkpoint =="
  tools/gpu-lease run --device 3090 --need-mib 3000 --wait 3600 --stream F -- "$EXE" "$HUB/$NAME" "$LOCK"
  echo "== substituted copy (one byte flipped in the last tensor) =="
  cp "$HUB/$NAME" "$SCRATCH/$NAME"
  python3 -I -c '
import os, sys
p = sys.argv[1]
with open(p, "r+b") as f:
    f.seek(os.path.getsize(p) - 7)
    b = f.read(1)
    f.seek(-1, 1)
    f.write(bytes([b[0] ^ 1]))
' "$SCRATCH/$NAME"
  tools/gpu-lease run --device 3090 --need-mib 3000 --wait 3600 --stream F -- "$EXE" "$SCRATCH/$NAME" "$LOCK" --expect-mismatch
  rm -f "${SCRATCH:?}/${NAME:?}"
} 2>&1 | tee "$LOG"

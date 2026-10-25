#!/bin/bash
# R-014 GPU capture: the eager worker teacher-forced on the five reference cases, per-layer captures in the R-011 layout.
# usage: capture.sh <3090|4060> <label> [--cases a,b]
export PATH=$HOME/.dotnet:$PATH
S=/home/mark/Projects/roost-evidence/R-014/src
dev=$1; label=$2; shift 2
cd "$S"
agent-budget level
mkdir -p "/home/mark/Projects/roost-evidence/R-014/$label"
tools/gpu-lease run --device "$dev" --need-mib 8000 --wait 3600 --stream B3 -- \
  nice -n10 tests/Starlings.Roost.Worker.Tests.Cuda/bin/Release/net10.0/Starlings.Roost.Worker.Tests.Cuda capture \
  --ckpt /home/mark/Projects/roost-wave3/envs/R-013/hub --lock evidence/registry/qwen3.5-0.8b.lock.toml --manifest kernels/manifest.toml \
  --kernels /home/mark/Projects/roost-evidence/R-036-gpu/kernels --reference /home/mark/Projects/roost-evidence/R-014/ref \
  --out "/home/mark/Projects/roost-evidence/R-014/$label" --allow-untested "$@" > "/home/mark/Projects/roost-evidence/R-014/$label/capture.log" 2>&1
echo "rc=$?"

#!/usr/bin/env bash
# Interleaved direct / in-process / process rounds so machine drift hits every mode alike. Each measurement is its own
# gpu-lease run (stream F, RTX 3090). Raw per-step latencies go to $OUT (outside the repository); the logs are text.
# usage: run_bench.sh <gpu-us> <steps> <rounds> [depth]     (needs ROOST_SPIN_LIB, see native/build.sh)
set -euo pipefail
cd "$(dirname "$0")/../.."
US=$1; STEPS=$2; ROUNDS=$3; DEPTH=${4:-1}
OUT="${OUT:-../envs/R-069/bench}"
BIN=tests/Starlings.Roost.WorkerHost.Tests.Cuda/bin/Release/net10.0/Starlings.Roost.WorkerHost.Tests.Cuda
LOG="$OUT/bench_${US}us_d${DEPTH}.log"
mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for m in direct inproc proc; do
    [ "$DEPTH" -gt 1 ] && [ "$m" = direct ] && continue
    echo "uptime: $(uptime)" | tee -a "$LOG"
    tools/gpu-lease run --device 3090 --need-mib 1000 --wait 3600 --stream F -- \
      "$BIN" bench --mode "$m" --steps "$STEPS" --warmup 200 --gpu-us "$US" --depth "$DEPTH" \
      --raw "$OUT/lat_${m}_${US}us_d${DEPTH}_r${r}.txt" 2>&1 | grep -v '^\[' | tee -a "$LOG"
  done
done

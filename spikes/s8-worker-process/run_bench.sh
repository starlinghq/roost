#!/usr/bin/env bash
# Interleaved A/B/C rounds so machine drift hits every mode alike. Each measurement is its own gpu-lease run.
# usage: run_bench.sh <gpu-us> <steps> <rounds> [depth]
set -euo pipefail
cd "$(dirname "$0")/../.."
US=$1; STEPS=$2; ROUNDS=$3; DEPTH=${4:-1}
mkdir -p spikes/s8-worker-process/out
for r in $(seq 1 "$ROUNDS"); do
  for m in direct inproc proc; do
    [ "$DEPTH" -gt 1 ] && [ "$m" = direct ] && continue
    tools/gpu-lease run --device 3090 --need-mib 1000 --wait 3600 --stream F -- \
      spikes/s8-worker-process/build/s8 bench --mode "$m" --steps "$STEPS" --warmup 200 --gpu-us "$US" --depth "$DEPTH" \
      --csv "spikes/s8-worker-process/out/lat_${m}_${US}us_d${DEPTH}_r${r}.csv" 2>&1 | grep -v '^\[' \
      | tee -a "spikes/s8-worker-process/out/bench_${US}us_d${DEPTH}.log"
  done
done

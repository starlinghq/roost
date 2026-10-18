#!/usr/bin/env bash
# Build the Gemm comparison (cuBLASLt vs one CUTLASS tile config). SM 86 only. Outputs under $ENVS/build.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS="${ROOST_S2_ENVS:-/home/mark/Projects/roost-wave1/envs/R-006}"
SM="${1:-86}"; OUT="$ENVS/build/sm$SM"; mkdir -p "$OUT"
s=$(date +%s)
nvcc -std=c++17 -O2 -DNDEBUG --expt-relaxed-constexpr -gencode arch=compute_${SM},code=sm_${SM} \
  -I"$ENVS/src/cutlass/include" -I"$HERE/gemm" -c "$HERE/gemm/gemm_cutlass.cu" -o "$OUT/gemm_cutlass.o"
e=$(date +%s)
echo "cutlass (2 tile configs) compile: $((e-s)) s, $(stat -c %s "$OUT/gemm_cutlass.o") bytes"
nvcc -std=c++17 -O1 -I"$HERE/gemm" "$HERE/gemm/gemm_bench.cpp" "$OUT/gemm_cutlass.o" -o "$OUT/gemm_bench" -lcublasLt -lcudart

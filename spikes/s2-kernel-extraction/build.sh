#!/usr/bin/env bash
# Build the S2 attention wrapper and its C ABI test. Sources are fetched by full SHA into $ENVS.
# Usage: build.sh [sm]   (default 86). Outputs go to $ENVS/build, never into the repository.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS="${ROOST_S2_ENVS:-/home/mark/Projects/roost-wave1/envs/R-006}"
SM="${1:-86}"
SRC="$ENVS/src"
OUT="$ENVS/build/sm$SM"
mkdir -p "$OUT"
FI="$SRC/flashinfer/include"
NVCC="${NVCC:-nvcc}"
GEN="-gencode arch=compute_${SM},code=sm_${SM}"
FLAGS="-std=c++17 -O2 -DNDEBUG --expt-relaxed-constexpr -Xcompiler -fPIC -lineinfo"
s=$(date +%s)
$NVCC $FLAGS $GEN -I"$FI" -I"$HERE/src" -c "$HERE/src/roost_s2_attn.cu" -o "$OUT/roost_s2_attn.o"
e=$(date +%s)
echo "wrapper compile: $((e-s)) s, $(stat -c %s "$OUT/roost_s2_attn.o") bytes"
$NVCC $GEN -shared -o "$OUT/libroost_s2_attn.so" "$OUT/roost_s2_attn.o" -Xcompiler -fPIC
$NVCC -std=c++17 -O1 -I"$HERE/src" "$HERE/test/test_paged_decode.cpp" -o "$OUT/test_paged_decode" \
  -L"$OUT" -lroost_s2_attn -lcudart -Xlinker -rpath,"$OUT"
ls -l "$OUT/libroost_s2_attn.so"

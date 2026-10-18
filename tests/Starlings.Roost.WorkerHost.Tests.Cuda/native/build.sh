#!/usr/bin/env bash
# Builds libroostspin.so for SM 86 into the directory given (default: envs/R-069/native next to the checkout).
# The output is a build product and is never committed.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../../../../envs/R-069/native}"
mkdir -p "$OUT"
nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -fPIC -shared -ccbin "${ROOST_HOSTCC:-g++}" -o "$OUT/libroostspin.so" roost_spin.cu
echo "$OUT/libroostspin.so"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
/opt/cuda/bin/nvcc -arch=sm_86 -cudart shared -Xcompiler -fPIC -shared -O2 native/s1_kernels.cu -o build/libs1kernels.so
ldd build/libs1kernels.so | grep -E "cudart|not found" || true

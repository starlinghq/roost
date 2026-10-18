#!/usr/bin/env bash
# Build the S8 spike for SM 86 only. Output goes to build/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
HOSTCC="${S8_HOSTCC:-g++}"
nvcc -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCC" ${S8_NVCC_FLAGS:-} -o build/s8 s8.cu

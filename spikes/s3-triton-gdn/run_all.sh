#!/usr/bin/env bash
# Reproduces the S3 numbers. Spike code, not part of Roost.slnx. Run from this directory.
# ENV is a uv venv with torch 2.14.1+cu130, triton 3.8.0, einops, numpy, transformers 5.20.1 and the flash-linear-attention
# checkout at 329b534245d3d7bb27cc03a99e87bdc51fb28d06 installed with `uv pip install --no-deps -e`.
set -euo pipefail
ENV=${ENV:-/home/mark/Projects/roost-wave1/envs/R-007}
source "$ENV/venv/bin/activate"
LEASE=../../tools/gpu-lease
mkdir -p out
# pins.json is the committed, frozen set of configs that the committed results were built with. Autotune is not
# reproducible (see the report), so step 1 writes its winners to out/pins_autotuned.json and never overwrites pins.json;
# later steps use pins.json. To adopt new pins, copy that file over pins.json by hand and rerun everything.
# 1. autotuned capture on a cold Triton cache (the on-device build step), then pick one config per kernel
export TRITON_CACHE_DIR=$ENV/triton-cache-jit; rm -rf "$TRITON_CACHE_DIR"
$LEASE run --device 3090 --need-mib 8000 --wait 3600 --stream C -- python capture.py out/specs_sm86.json
python make_pins.py out/specs_sm86.json out/pins_autotuned.json
diff -q out/pins_autotuned.json pins.json || echo "autotune picked different pins than the committed pins.json (expected); continuing with pins.json"
# 2. capture again with the committed pins forced: these are the specialisations that are compiled ahead of time
$LEASE run --device 3090 --need-mib 8000 --wait 3600 --stream C -- python capture.py out/specs_pinned.json pins.json
# 3. AOT compile per SM (cold cache each; needs a visible device, see the report)
for sm in 80 86 89 90 100 120; do
  export TRITON_CACHE_DIR=$ENV/triton-cache-aot$sm; rm -rf "$TRITON_CACHE_DIR"
  $LEASE run --device 3090 --need-mib 500 --wait 3600 --stream C -- python build_aot.py out/specs_pinned.json $sm out/aot$sm
done
# 4. C ABI library and numeric comparison against Python FLA (same pinned configs)
for sm in 86 89; do
  gcc -O2 -Wall -shared -fPIC -I/opt/cuda/include -Iout/aot$sm gdn_api.c out/aot$sm/gdn_kernels.c \
      -o out/libgdn_aot_sm$sm.so -L/opt/cuda/lib64/stubs -lcuda
done
export TRITON_CACHE_DIR=$ENV/triton-cache-jit
$LEASE run --device 3090 --need-mib 8000 --wait 3600 --stream C -- python test_aot.py out/libgdn_aot_sm86.so pins.json
# SM 89 (RTX 4060 Ti) only if the lease reports room:
# $LEASE run --device 4060 --need-mib 1500 --wait 3600 --stream C -- python test_aot.py out/libgdn_aot_sm89.so pins.json

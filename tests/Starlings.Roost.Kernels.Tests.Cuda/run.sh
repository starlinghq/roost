#!/usr/bin/env bash
# R-023 GPU evidence: the R-016 stub library through the loader on the RTX 3090, one short lease.
# usage: run.sh <kernel-build-dir> [log]      (build the stub first: see kernels/README.md; the library stays outside the repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
BUILD=${1:?kernel build directory}
LOG=${2:-tests/Starlings.Roost.Kernels.Tests.Cuda/results/r023-loader-sm86.log}
dotnet build tests/Starlings.Roost.Kernels.Tests.Cuda -c Release -v q
tools/gpu-lease run --device 3090 --need-mib 500 --wait 3600 --stream F -- \
  tests/Starlings.Roost.Kernels.Tests.Cuda/bin/Release/net10.0/Starlings.Roost.Kernels.Tests.Cuda \
  "$BUILD" kernels/manifest.toml 2>&1 | tee "$LOG"

#!/usr/bin/env bash
# Build-only CI step (R-016): compile the kernel library for every SM in kernels/manifest.toml, twice into separate
# directories, and fail unless the two builds are byte-identical per SM. Runs inside the pinned CUDA container; needs
# no GPU. It also loads each library and checks its registration table (stub_check --no-gpu). The digests go to
# $ROOST_DIGESTS_OUT as JSON; binaries are never uploaded or committed.
#
#   kernels/ci/build-all-sm.sh <work-dir> [sm-list]     # sm-list defaults to the manifest's, as 80;86;...
set -euo pipefail

work=${1:?usage: build-all-sm.sh <work-dir> [sm-list]}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
sms=${2:-$(python3 -I "$root/kernels/tools/manifest_cmake.py" "$root/kernels/manifest.toml" --sm-list)}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-2}
mkdir -p "$work"

echo "SM list: $sms"
nvcc --version | tail -2

# The first configure fetches the upstream sources by SHA into the shared cache; the second reuses them.
cmake -S "$root/kernels" -B "$work/a" -G Ninja -DROOST_SM_LIST="$sms" -DROOST_UPSTREAM_DIR="$work/upstream"
cmake --build "$work/a"
cmake -S "$root/kernels" -B "$work/b" -G Ninja -DROOST_SM_LIST="$sms" -DROOST_UPSTREAM_DIR="$work/upstream"
cmake --build "$work/b"

status=0
IFS=';' read -ra list <<<"$sms"
for sm in "${list[@]}"; do
  lib="libroost_kernels_sm${sm}.so"
  "$work/a/roost_stub_check" "$work/a/$lib" --no-gpu
  if ! cmp -s "$work/a/$lib" "$work/b/$lib"; then
    echo "sm_${sm}: two builds of the same sources differ" >&2
    status=1
  fi
done

python3 -I "$root/kernels/tools/record_digests.py" "$work/a" --json "${ROOST_DIGESTS_OUT:-$work/digests.json}"
# The manifest's digests come from the development host until the release workflow records the container build's:
# a difference here is reported, not a failure.
python3 -I "$root/kernels/tools/record_digests.py" "$work/a" --check || echo "note: digests differ from the manifest's (see kernels/manifest.toml)"
exit $status

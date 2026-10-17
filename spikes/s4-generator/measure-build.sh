#!/usr/bin/env bash
# Build-time measurement for the S4 spike. CPU only, two build jobs. The machine is shared, so each case runs N times
# and the script prints min and median wall time plus the median CPU (user+sys) time, which is less load-sensitive.
# Usage: spikes/s4-generator/measure-build.sh [N]   (run from anywhere; default N=5)
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-5}
B=(dotnet build Toy/Toy.csproj -c Release -m:2 --no-restore -nologo -v:q)
cp Toy/ToyModel.cs /tmp/s4-ToyModel.cs.bak
cp Toy/Program.cs /tmp/s4-Program.cs.bak
cp Generator/Graph.cs /tmp/s4-Graph.cs.bak
restore() { cp /tmp/s4-ToyModel.cs.bak Toy/ToyModel.cs; cp /tmp/s4-Program.cs.bak Toy/Program.cs; cp /tmp/s4-Graph.cs.bak Generator/Graph.cs; }
trap restore EXIT

# run_case <label> <prepare-command-or-:> ; prepare runs before each timed build (with the edit number in $i)
run_case() {
  local label=$1 prep=$2 walls=() cpus=()
  for i in $(seq 1 "$N"); do
    restore
    eval "$prep"
    local out
    out=$( { TIMEFORMAT='%R %U %S'; time "${B[@]}" >/dev/null 2>&1 || echo BUILD-FAILED >&2; } 2>&1 )
    if [[ $out == *BUILD-FAILED* ]]; then echo "build failed in case: $label" >&2; exit 1; fi
    read -r r u s <<<"$out"
    walls+=("$(awk -v r="$r" 'BEGIN{printf "%d", r*1000}')")
    cpus+=("$(awk -v u="$u" -v s="$s" 'BEGIN{printf "%d", (u+s)*1000}')")
  done
  local w c
  w=$(printf '%s\n' "${walls[@]}" | sort -n)
  c=$(printf '%s\n' "${cpus[@]}" | sort -n)
  printf '%-52s wall min %5d ms  median %5d ms | cpu median %5d ms\n' "$label" \
    "$(sed -n 1p <<<"$w")" "$(sed -n "$(( (N + 1) / 2 ))p" <<<"$w")" "$(sed -n "$(( (N + 1) / 2 ))p" <<<"$c")"
}

# Cold build: clean out/ then build (once; it is cold only once). Restore is not timed.
rm -rf out
dotnet restore Toy/Toy.csproj -nologo -v:q >/dev/null
out=$( { TIMEFORMAT='%R %U %S'; time "${B[@]}" >/dev/null 2>&1; } 2>&1 )
read -r r u s <<<"$out"
echo "cold build (Ops + Generator + Toy, clean out/, N=1): wall $r s | cpu $(awk -v u="$u" -v s="$s" 'BEGIN{printf "%.2f", u+s}') s"

"${B[@]}" >/dev/null 2>&1
run_case "no-op rebuild" ":"
run_case "edit unrelated Toy file (Program.cs)" 'echo "// edit $i" >> Toy/Program.cs'
run_case "edit model definition (rename one graph node)" 'sed -i "s/l1_act/l1_act$i/g" Toy/ToyModel.cs'
run_case "edit generator source (rebuild Generator + Toy)" 'echo "// edit $i" >> Generator/Graph.cs'
restore
"${B[@]}" >/dev/null 2>&1

echo
echo "generator time inside csc (-p:ReportAnalyzer=true, one forced Toy compile):"
echo "// edit" >> Toy/Program.cs
dotnet build Toy/Toy.csproj -c Release -m:2 --no-restore -nologo -v:d -p:ReportAnalyzer=true 2>&1 | grep -E "ModelGenerator|Total analyzer execution time" | cut -c1-120 || true

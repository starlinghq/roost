#!/bin/bash
# Regenerates docs/m0/r022/evaluate-layers-{drop-attention-gate,drop-z-gate,norm-eps-1e-5}.json (R-022 negative runs).
#
# usage: negative_runs.sh SNAPSHOT RUN_DIR [PERTURBATION...]     (default: all three)
#   SNAPSHOT  the Qwen/Qwen3.5-0.8B snapshot at 2fc06364 (config.json, safetensors, tokenizer files)
#   RUN_DIR   the run artifacts: ref/ (reference-run output with reference.jsonl and layers/), tolerances.json; this writes
#             reference-2cases.jsonl, cand-<perturbation>/ and eval-<perturbation>.json
# Run from the repository root after `dotnet build tests/Starlings.Roost.Models.Qwen35.Tests -c Release`.
# Environment: UV_PROJECT_ENVIRONMENT (default $RUN_DIR/venv, from `uv sync --project evidence/pipeline --locked --extra reference`);
#   HEAVY_LOCK (default $RUN_DIR/../heavy.lock) is the flock file for the checkpoint load and the fp64 forward;
#   CASES (default capital,long-context). Each perturbation takes about 4 minutes for two cases.
# The gate report's "inputs" block hashes reference-2cases.jsonl, tolerances.json and the candidate captures, so a rerun that
# matches the committed report reproduces those hashes too (the reports are deterministic: no timestamps or paths inside).
set -euo pipefail
SNAP=$1 E=$2
shift 2
PERTURBATIONS=("$@")
[ ${#PERTURBATIONS[@]} -gt 0 ] || PERTURBATIONS=(drop-attention-gate drop-z-gate norm-eps-1e-5)
CASES=${CASES:-capital,long-context}
LOCK=${HEAVY_LOCK:-$E/../heavy.lock}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$PWD
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$E/venv}

python3 -I "$HERE/filter_reference.py" "$E/ref/reference.jsonl" "$E/reference-2cases.jsonl" "$CASES"
for p in "${PERTURBATIONS[@]}"; do
  rm -rf "$E/cand-$p"
  mkdir -p "$E/cand-$p"
  flock "$LOCK" -c "ROOST_QWEN35_CHECKPOINT='$SNAP' ROOST_R022_REFERENCE='$E/ref' ROOST_R022_OUT='$E/cand-$p' ROOST_R022_PERTURB='$p' ROOST_R022_CASES='$CASES' dotnet test --project tests/Starlings.Roost.Models.Qwen35.Tests -c Release --no-build -- --filter-method '*TeacherForcedCapture'"
  # The gate exits 1 when it reports "failed", which is the expected outcome of a negative run.
  # The evaluator resolves --reference-captures to the layers/ directory next to reference.jsonl; run from $E so the input names in the report are relative.
  (cd "$E" && uv run --project "$REPO/evidence/pipeline" --locked roost-evidence evaluate-layers "cand-$p" \
      --reference reference-2cases.jsonl --reference-captures ref --tolerances tolerances.json --report "eval-$p.json") || true
done

#!/bin/bash
# Per-layer evaluator (evidence/pipeline at plan/build-plan d6c6fe6, with PR #134's relative metrics) on the GPU captures.
# The tolerance file is measurement-only (every bound loose): the report's measurements are what the proposal is sized from.
R=/home/mark/Projects/roost-evidence/R-014
export UV_PROJECT_ENVIRONMENT=$R/venv
cd "$R/src"
for sm in 86 89; do
  nice -n10 uv run -q --project evidence/pipeline --locked roost-evidence evaluate-layers "$R/gpu-sm$sm" \
    --reference "$R/ref/reference.jsonl" --reference-captures "$R/ref" --tolerances "$R/eval/measure-tolerances.json" \
    --report "$R/eval/layers-sm$sm.json" > "$R/eval/layers-sm$sm.out" 2>&1
  echo "sm$sm rc=$?"
  tail -n 15 "$R/eval/layers-sm$sm.out"
done

#!/bin/bash
R=/home/mark/Projects/roost-evidence/R-014
export UV_PROJECT_ENVIRONMENT=$R/venv
cd "$R/src"
nice -n10 uv run -q --project evidence/pipeline --locked python -I "$R/scripts/analyze.py" "$R" > "$R/eval/summary.txt" 2>&1
echo "rc=$?"
cat "$R/eval/summary.txt"

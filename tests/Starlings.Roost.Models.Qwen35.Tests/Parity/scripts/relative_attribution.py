"""R-022: where a row-relative hidden-state bound would attribute a divergence (an evaluator proposal, not a gate).

For each case: the first forward, then the first stage in model order, whose worst row has ||candidate - reference|| /
||reference|| above the bound, as evaluate_layers attributes with its absolute bound. Test-time only; needs torch and
safetensors.

    python -I relative_attribution.py <reference run dir> <candidate run dir> <bound>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(reference: Path, candidate: Path, bound: float) -> int:
    from safetensors import safe_open

    cases = [
        json.loads(line)["id"] for line in (candidate / "candidate.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    worst_overall = 0.0
    for name in cases:
        with (
            safe_open(str(reference / "layers" / f"{name}.safetensors"), "pt") as ref,
            safe_open(str(candidate / "layers" / f"{name}.safetensors"), "pt") as cand,
        ):
            metadata = ref.metadata()
            stages = json.loads(metadata["stages"])
            first = None
            for forward in range(int(metadata["forwards"])):
                for stage in stages:
                    key = f"forward{forward}/{stage}"
                    r, c = ref.get_tensor(key).double(), cand.get_tensor(key).double()
                    rel = (c - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)
                    worst_overall = max(worst_overall, float(rel.max()))
                    if first is None and float(rel.max()) > bound:
                        position = int((rel > bound).nonzero()[0])
                        first = f"forward {forward} {stage} position {position} ({float(rel[position]):.3g})"
        print(f"{name}: first divergent {first or 'none'}")
    print(f"max row-relative L2 over all stages: {worst_overall:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])))

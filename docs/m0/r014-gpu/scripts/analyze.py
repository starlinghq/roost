"""R-014 GPU analysis: step-0 metrics, greedy agreement and margins, and the per-layer maxima of the evaluator reports.

usage: uv run --project evidence/pipeline python analyze.py <R-014 dir>
(writes <dir>/eval/summary.json and prints a text summary)
"""

import json
import math
import sys
from array import array
from pathlib import Path

from roost_evidence.metrics import compare

R = Path(sys.argv[1])
ref = {}
for line in (R / "ref" / "reference.jsonl").read_text().splitlines():
    if line.strip():
        d = json.loads(line)
        ref[d["id"]] = d


def floats(path: Path) -> list[float]:
    a = array("f")
    a.frombytes(path.read_bytes())
    return a.tolist()


out = {}
for sm in (86, 89):
    gpu = R / f"gpu-sm{sm}"
    cand = {}
    for line in (gpu / "candidate.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            cand[d["id"]] = d
    s = {"step0": {}, "greedy": {}, "layers": {}}
    for cid, r in ref.items():
        rv = floats(R / "ref" / "step0" / f"{cid}.f32")
        cv = floats(gpu / "step0" / f"{cid}.f32")
        st = compare(cv, rv, center=True)
        top5 = sorted(range(len(rv)), key=lambda i: rv[i], reverse=True)[:5]
        s["step0"][cid] = {
            "max_abs": st.max_abs,
            "top5_max_abs": max(abs(cv[i] - rv[i]) for i in top5),
            "rel_l2_centred": st.rel_l2,
            "cosine_centred": st.cosine,
            "max_abs_rel": st.max_abs / st.max_ref,
            "argmax_equal": max(range(len(cv)), key=cv.__getitem__) == top5[0],
        }
        c = cand[cid]
        out_ids = r["output_ids"]
        ref_margin = r["step_margin"]
        flips = [
            {
                "forward": i,
                "reference": out_ids[i],
                "gpu": a,
                "reference_margin": ref_margin[i],
                "gpu_margin": c["margin"][i],
            }
            for i, a in enumerate(c["argmax"])
            if a != out_ids[i]
        ]
        merr = [abs(c["margin"][i] - ref_margin[i]) for i in range(len(out_ids)) if c["argmax"][i] == out_ids[i]]
        s["greedy"][cid] = {
            "forwards": len(out_ids),
            "agree": len(out_ids) - len(flips),
            "flips": flips,
            "max_margin_error_where_equal": max(merr) if merr else None,
            "smallest_reference_margin": min(ref_margin),
            "smallest_reference_margin_forward": ref_margin.index(min(ref_margin)),
        }
    rep = json.loads((R / "eval" / f"layers-sm{sm}.json").read_text())
    for chk in rep["checks"]:
        s["layers"][chk["id"]] = chk["metrics"]
    # Decoder layers versus the final norm (absolute error is dominated by the final norm's scale).
    worst = {"decoder_hidden_max_abs": 0.0, "final_norm_max_abs": 0.0, "decoder_rel_l2": 0.0, "final_norm_rel_l2": 0.0}
    for case in rep["cases"].values():
        for stage, v in case.get("hidden_max_abs", {}).items():
            k = "final_norm_max_abs" if stage == "final_norm" else "decoder_hidden_max_abs"
            worst[k] = max(worst[k], v)
        for stage, v in case.get("hidden_rel_l2", {}).items():
            k = "final_norm_rel_l2" if stage == "final_norm" else "decoder_rel_l2"
            worst[k] = max(worst[k], v)
    s["layers"]["split"] = worst
    out[f"sm{sm}"] = s

(R / "eval" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
for sm, s in out.items():
    print(f"== {sm}")
    for k, v in s["layers"].items():
        print(f"  {k}: {v}")
    print(
        "  step0 max_abs",
        max(v["max_abs"] for v in s["step0"].values()),
        "top5",
        max(v["top5_max_abs"] for v in s["step0"].values()),
        "rel_l2",
        max(v["rel_l2_centred"] for v in s["step0"].values()),
        "cos",
        min(v["cosine_centred"] for v in s["step0"].values()),
        "max_abs_rel",
        max(v["max_abs_rel"] for v in s["step0"].values()),
        "argmax",
        all(v["argmax_equal"] for v in s["step0"].values()),
    )
    for cid, g in s["greedy"].items():
        print(
            f"  {cid}: agree {g['agree']}/{g['forwards']} flips {g['flips']}"
            f" margin_err {g['max_margin_error_where_equal']:.4f}"
            f" min_ref_margin {g['smallest_reference_margin']:.4f}@{g['smallest_reference_margin_forward']}"
        )
    _ = math

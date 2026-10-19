"""R-022: distributions behind the tolerance proposal.

Compares a candidate run (Qwen35CpuParityRun.TeacherForcedCapture) with a transformers fp32 reference run
(roost-evidence reference-run --capture-layers) per stage, per forward and per case, and the step-0 log-probabilities
and the argmax of every forward. Test-time only; needs torch and safetensors (the evidence pipeline's `reference`
extra).

    python -I analyze_parity.py <reference run dir> <candidate run dir> [--json summary.json]

The gate itself is `roost-evidence evaluate-layers`; this script only reports what the bounds are chosen from.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

KINDS = ("recurrent_state", "conv_state")


def _load(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {name: handle.get_tensor(name).double() for name in handle.keys()}  # noqa: SIM118 - safe_open has no items()
    return tensors, metadata


def _f32(path: Path) -> Any:
    import torch

    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32).double()


def compare_case(name: str, reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    ref, meta = _load(reference_dir / "layers" / f"{name}.safetensors")
    cand, _ = _load(candidate_dir / "layers" / f"{name}.safetensors")
    stages = json.loads(meta["stages"])
    forwards = int(meta["forwards"])
    hidden: list[dict[str, Any]] = []
    for forward in range(forwards):
        for stage in stages:
            key = f"forward{forward}/{stage}"
            r, c = ref[key], cand[key]
            diff = (c - r).abs()
            flat = int(diff.argmax())
            row, col = divmod(flat, r.shape[1])
            norms = ((c - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)).max()
            hidden.append(
                {
                    "forward": forward,
                    "stage": stage,
                    "max_abs": float(diff.max()),
                    "ref_max_abs": float(r.abs().max()),
                    "at_position": row,
                    "at_channel": col,
                    "ref_at": float(r[row, col]),
                    "rel_l2": float(norms),
                    "finite": bool(c.isfinite().all()),
                }
            )
    states: list[dict[str, Any]] = []
    for key in sorted(k for k in ref if k.count("/") == 2):
        forward_text, layer_text, kind = key.split("/")
        if key not in cand:
            continue
        r, c = ref[key], cand[key]
        entry = {
            "forward": int(forward_text[7:]),
            "layer": int(layer_text[6:]),
            "kind": kind,
            "max_abs": float((c - r).abs().max()),
            "ref_max_abs": float(r.abs().max()),
            "rel_l2": float((c - r).norm() / r.norm().clamp_min(1e-30)),
        }
        if kind == "conv_state":
            entry["max_abs_kept_columns"] = float((c[:, 1:] - r[:, 1:]).abs().max())
            entry["max_abs_column0"] = float((c[:, 0] - r[:, 0]).abs().max())
        states.append(entry)
    return {"hidden": hidden, "states": states, "stages": stages, "forwards": forwards}


def step0(name: str, reference_dir: Path, candidate_dir: Path, ref_line: dict[str, Any]) -> dict[str, Any]:
    r = _f32(reference_dir / "step0" / f"{name}.f32")
    c = _f32(candidate_dir / "step0" / f"{name}.f32")
    top5 = [int(t) for t, _ in ref_line["step0_top"][:5]]
    finite = r > -1e30
    return {
        "full_max_abs": float((c - r).abs().max()),
        "top5_max_abs": float((c[top5] - r[top5]).abs().max()),
        "top1_abs": float((c[top5[0]] - r[top5[0]]).abs()),
        "argmax_equal": int(c.argmax()) == int(r.argmax()),
        "ref_margin": float(ref_line["step_margin"][0]),
        "where_full": int((c - r).abs().argmax()),
        "ref_logprob_at_full": float(r[int((c - r).abs().argmax())]),
        "finite_ref": bool(finite.all()),
    }


def _fmt(x: float) -> str:
    return f"{x:.3g}" if x != 0 else "0"


def report(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    refs = {
        json.loads(line)["id"]: json.loads(line)
        for line in (reference_dir / "reference.jsonl").read_text(encoding="utf-8").splitlines()
    }
    cands = {
        json.loads(line)["id"]: json.loads(line)
        for line in (candidate_dir / "candidate.jsonl").read_text(encoding="utf-8").splitlines()
    }
    cases = [n for n in refs if n in cands]
    results = {n: compare_case(n, reference_dir, candidate_dir) for n in cases}
    first = results[cases[0]]
    stages = first["stages"]
    out: dict[str, Any] = {"cases": cases, "perturbation": cands[cases[0]].get("perturbation")}
    lines: list[str] = []
    p = lines.append
    p(f"perturbation: {out['perturbation']}; cases: {', '.join(cases)}")
    p("")

    # Hidden states per stage.
    p("## Hidden states per stage (max over cases, forwards and positions)")
    p("")
    header = ["stage", "max abs", "set by (case, forward, position, channel)", "ref value there"]
    p("| " + " | ".join([*header, "stage max abs ref", "max rel L2 (row)"]) + " |")
    p("| --- | --- | --- | --- | --- | --- |")
    per_stage: dict[str, Any] = {}
    for stage in stages:
        rows = [(n, h) for n, r in results.items() for h in r["hidden"] if h["stage"] == stage]
        n, worst = max(rows, key=lambda x: x[1]["max_abs"])
        refmax = max(h["ref_max_abs"] for _, h in rows)
        rel = max(h["rel_l2"] for _, h in rows)
        per_stage[stage] = {"max_abs": worst["max_abs"], "case": n, "forward": worst["forward"], "rel_l2": rel}
        where = f"{n}, {worst['forward']}, {worst['at_position']}, {worst['at_channel']}"
        p(f"| {stage} | {_fmt(worst['max_abs'])} | {where} | {_fmt(worst['ref_at'])} | {_fmt(refmax)} | {_fmt(rel)} |")
    out["hidden_per_stage"] = per_stage
    p("")

    p("## Hidden states per case (max over stages)")
    p("")
    header = ["case", "prompt tokens", "max abs, prefill", "max abs, decode forwards"]
    p("| " + " | ".join([*header, "stage and forward of the max", "max rel L2"]) + " |")
    p("| --- | --- | --- | --- | --- | --- |")
    per_case: dict[str, Any] = {}
    for n, r in results.items():
        pre = max((h for h in r["hidden"] if h["forward"] == 0), key=lambda h: h["max_abs"])
        dec = max((h for h in r["hidden"] if h["forward"] > 0), key=lambda h: h["max_abs"])
        worst = pre if pre["max_abs"] >= dec["max_abs"] else dec
        rel = max(h["rel_l2"] for h in r["hidden"])
        per_case[n] = {"prefill": pre["max_abs"], "decode": dec["max_abs"], "rel_l2": rel}
        p(
            f"| {n} | {len(refs[n]['prompt_ids'])} | {_fmt(pre['max_abs'])} | {_fmt(dec['max_abs'])} | "
            f"{worst['stage']} f{worst['forward']} | {_fmt(rel)} |"
        )
    out["hidden_per_case"] = per_case
    p("")

    forwards = min(r["forwards"] for r in results.values())
    p("## Growth across forwards (max over cases; hidden over all stages, states over all linear layers)")
    p("")
    p("| forward | hidden max abs | hidden, last layer (layer.23) | recurrent state max abs | conv state max abs |")
    p("| --- | --- | --- | --- | --- |")
    growth = []
    for f in range(forwards):
        hid = max(h["max_abs"] for r in results.values() for h in r["hidden"] if h["forward"] == f)
        last = max(
            h["max_abs"]
            for r in results.values()
            for h in r["hidden"]
            if h["forward"] == f and h["stage"] == "layer.23"
        )
        per_kind = {
            kind: max(
                (
                    s["max_abs"]
                    for r in results.values()
                    for s in r["states"]
                    if s["forward"] == f and s["kind"] == kind
                ),
                default=math.nan,
            )
            for kind in KINDS
        }
        growth.append(
            {
                "forward": f,
                "hidden": hid,
                "last_layer": last,
                "recurrent": per_kind["recurrent_state"],
                "conv": per_kind["conv_state"],
            }
        )
        p(f"| {f} | {_fmt(hid)} | {_fmt(last)} | {_fmt(growth[-1]['recurrent'])} | {_fmt(growth[-1]['conv'])} |")
    out["growth"] = growth
    p("")

    for kind in KINDS:
        p(f"## {kind} per linear layer (max over cases and forwards)")
        p("")
        extra = (
            " | max abs, columns 1-3 (held by Roost) | max abs, column 0 (reconstructed)"
            if kind == "conv_state"
            else ""
        )
        p(f"| layer | max abs | set by (case, forward) | max abs ref | max rel L2{extra} |")
        p("| --- | --- | --- | --- | ---" + (" | --- | ---" if extra else "") + " |")
        table: dict[str, Any] = {}
        layers = sorted({s["layer"] for r in results.values() for s in r["states"] if s["kind"] == kind})
        for layer in layers:
            rows = [
                (n, s) for n, r in results.items() for s in r["states"] if s["kind"] == kind and s["layer"] == layer
            ]
            n, worst = max(rows, key=lambda x: x[1]["max_abs"])
            refmax = max(s["ref_max_abs"] for _, s in rows)
            rel = max(s["rel_l2"] for _, s in rows)
            table[str(layer)] = {"max_abs": worst["max_abs"], "case": n, "forward": worst["forward"], "rel_l2": rel}
            cols = ""
            if kind == "conv_state":
                kept = max(s["max_abs_kept_columns"] for _, s in rows)
                col0 = max(s["max_abs_column0"] for _, s in rows)
                table[str(layer)].update({"kept_columns": kept, "column0": col0})
                cols = f" | {_fmt(kept)} | {_fmt(col0)}"
            p(f"| {layer} | {_fmt(worst['max_abs'])} | {n}, {worst['forward']} | {_fmt(refmax)} | {_fmt(rel)}{cols} |")
        out[kind] = table
        p("")

    p("## Step-0 log-probabilities and per-forward argmax")
    p("")
    p(
        "| case | full-vector max abs | ref log-prob at that token | top-5 max abs | top-1 abs | step-0 argmax equal | "
        "reference step-0 margin | max abs margin error, all forwards | smallest reference margin (candidate's) | "
        "forwards whose argmax differs (reference margin) |"
    )
    p("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    s0: dict[str, Any] = {}
    for n in cases:
        s = step0(n, reference_dir, candidate_dir, refs[n])
        ids = refs[n]["output_ids"]
        diffs = [
            f"f{i} ({refs[n]['step_margin'][i]:.3g})"
            for i, (a, b) in enumerate(zip(cands[n]["argmax"], ids, strict=False))
            if a != b
        ]
        s["argmax_differs"] = diffs
        # The greedy near-tie rule is about the top-1/top-2 gap: its error per forward (the runner-ups may differ).
        pairs = list(zip(refs[n]["step_margin"], cands[n]["margin"], strict=False))
        s["margin_error_max"] = max(abs(c - r) for r, c in pairs)
        smallest = min(pairs)
        s["smallest_margin"] = list(smallest)
        s0[n] = s
        p(
            f"| {n} | {_fmt(s['full_max_abs'])} | {s['ref_logprob_at_full']:.3g} | {_fmt(s['top5_max_abs'])} | "
            f"{_fmt(s['top1_abs'])} | {s['argmax_equal']} | {s['ref_margin']:.3g} | {_fmt(s['margin_error_max'])} | "
            f"{smallest[0]:.3g} ({smallest[1]:.3g}) | {', '.join(diffs) or 'none'} |"
        )
    out["step0"] = s0
    p("")
    nonfinite = [
        f"{n} {h['stage']} f{h['forward']}" for n, r in results.items() for h in r["hidden"] if not h["finite"]
    ]
    p(f"non-finite candidate hidden values: {', '.join(nonfinite) or 'none'}")
    out["text"] = "\n".join(lines)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    out = report(args.reference, args.candidate)
    print(out["text"])
    if args.json:
        args.json.write_text(json.dumps({k: v for k, v in out.items() if k != "text"}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

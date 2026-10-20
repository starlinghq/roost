"""Consistency of R-034's Triton AOT inputs, without Triton: the captured specs, the role table of build_gdn.py, and the
launcher in kernels/src/ops (role order and run-time argument lists). A drift between them is a runtime refusal at
context creation; this finds it in CI, where there is no GPU to build the cubins.

Run: python3 -I -m pytest -q kernels/tests
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AOT = ROOT / "kernels" / "aot"
OPS = ROOT / "kernels" / "src" / "ops"


def build_gdn():
    spec = importlib.util.spec_from_file_location("build_gdn", AOT / "build_gdn.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_gdn"] = module
    spec.loader.exec_module(module)
    return module


def runtime_signature(spec: dict) -> str:
    return ",".join(f"{n}:{spec['signature'][n]}" for n in spec["arg_names"] if spec["signature"][n] != "constexpr")


def specs() -> dict:
    return json.loads((AOT / "gdn_specs.json").read_text())


def test_every_role_has_exactly_one_spec():
    module = build_gdn()
    roles = [module.role_of(s) for s in specs()["specs"]]
    assert sorted(roles) == list(range(len(module.ROLES)))


def test_specs_are_the_manifests_pin_and_the_varlen_set():
    doc = specs()
    manifest = tomllib.loads((ROOT / "kernels" / "manifest.toml").read_text())
    fla = next(u for u in manifest["upstreams"] if u["name"] == "flash-linear-attention")
    assert doc["fla_sha"] == fla["sha"]
    assert doc["triton"] == manifest["triton_aot"]["packages"]["triton"]
    assert all(s["constexprs"]["IS_VARLEN"] for s in doc["specs"])
    built = sorted(k["name"] for k in manifest["triton_aot"]["kernels"])
    assert built == sorted({s["kernel"] for s in doc["specs"]})


def test_launcher_argument_lists_match_the_specs():
    module = build_gdn()
    source = (OPS / "gated_delta_rule.cu").read_text()
    block = source.split("kExpectedSignature[roost_gdn::kGdnKernelCount] = {", 1)[1].split("};", 1)[0]
    expected = re.findall(r'"([^"]+)"', block)
    by_role = {module.role_of(s): runtime_signature(s) for s in specs()["specs"]}
    assert expected == [by_role[i] for i in range(len(module.ROLES))]


def test_role_order_matches_the_c_enum():
    module = build_gdn()
    header = (OPS / "gdn_aot.h").read_text()
    enum = header.split("enum GdnRole : int {", 1)[1].split("};", 1)[0]
    names = [n for n in re.findall(r"\b(k[A-Z]\w*)\b", enum) if n != "kGdnKernelCount"]
    snake = {
        "kCumsum": "cumsum",
        "kKktSolve": "kkt_solve",
        "kRecomputeWU": "recompute_w_u",
        "kChunkHInit": "chunk_h_h0",
        "kChunkH": "chunk_h",
        "kChunkO": "chunk_o",
        "kRecurrent": "recurrent",
    }
    assert [snake[n] for n in names] == [role for role, _, _ in module.ROLES]

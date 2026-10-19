"""Tests for the R-004 contract tools: the C5 manifest validator and the C1/C2 golden layout tool.

python3 -m pytest tests/tools
"""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
EXAMPLE = Path(__file__).resolve().parent / "fixtures" / "kernel-manifest.example.toml"


def load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(TOOLS / filename))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


manifest_tool = load("validate_kernel_manifest", "validate-kernel-manifest.py")
layout_tool = load("abi_layout", "abi-layout.py")


@pytest.fixture
def example() -> dict:
    with EXAMPLE.open("rb") as handle:
        return tomllib.load(handle)


def test_example_manifest_is_valid(example):
    assert manifest_tool.check(example) == []


def test_cli_reports_valid_and_missing(capsys):
    assert manifest_tool.main([str(EXAMPLE)]) == 0
    assert manifest_tool.main([str(EXAMPLE.with_name("missing.toml"))]) == 1


def mutate(example: dict, change) -> list[str]:
    manifest = copy.deepcopy(example)
    change(manifest)
    return manifest_tool.check(manifest)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda m: m["upstreams"][0].update(sha="main"), "does not match"),
        (lambda m: m.pop("toolkit"), "missing 'toolkit'"),
        (lambda m: m.update(extra=1), "unknown key 'extra'"),
        (lambda m: m["sm"][1].update(status="works"), "is not one of"),
        (lambda m: m["sm"].append({"arch": 75, "status": "untested"}), "is not one of"),
        (lambda m: m["ops"][0]["variants"][0].update(flags=["fast"]), "is not one of"),
        (lambda m: m["triton_aot"].pop("requires_gpu_build_host"), "missing 'requires_gpu_build_host'"),
        (lambda m: m["ops"][1]["variants"][0]["gemm_algos"][0].pop("library_sha256"), "missing 'library_sha256'"),
    ],
)
def test_schema_rejects(example, change, expected):
    errors = mutate(example, change)
    assert any(expected in e for e in errors), errors


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda m: m["ops"][0].update(op="FlashDecode"), "is not a C3 op"),
        (lambda m: m["ops"][1]["variants"][0].update(id="flashinfer.decode.bf16.h128.nhd"), "already used"),
        (lambda m: m["sm"].pop(1), "is not in the SM list"),
        (lambda m: m["ops"][0]["variants"][0].update(source="nowhere"), "is not an upstream"),
        (lambda m: m["abi"].update(major=2), "ROOST_ABI_MAJOR"),
        (lambda m: m["abi"].update(minor=9), "newer than ROOST_ABI_MINOR"),
        (lambda m: m["ops"][2]["variants"][0].update(triton_kernels=["other"]), "not declared under triton_aot"),
        (lambda m: m["upstreams"][0].update(ops=["Gemm", "Sample"]), "'Sample' is not an op of this manifest"),
    ],
)
def test_semantic_checks_reject(example, change, expected):
    errors = mutate(example, change)
    assert any(expected in e for e in errors), errors


def test_c3_op_names_come_from_the_header():
    ops, major, minor = manifest_tool.header_ops_and_abi()
    assert {"RMSNORM", "PAGEDATTENTION", "GEMMW4A16", "DEQUANTIZEGGUF"} <= ops
    assert (major, minor) == (1, 2)


def test_layout_parser_reads_every_struct():
    structs = {name: fields for _, name, fields in layout_tool.all_structs()}
    assert structs["RoostDeviceSpan"] == ["address", "length", "dtype", "reserved0"]
    assert structs["RoostRmsNormParams"][:2] == ["struct_size", "abi_minor"]
    golden = layout_tool.GOLDEN.read_text(encoding="utf-8")
    assert sorted(structs) == sorted(line.split()[1] for line in golden.splitlines() if line.startswith("struct "))


def test_every_c1_params_struct_starts_with_size_and_minor():
    for header, name, fields in layout_tool.all_structs():
        if header == "roost_kernels.h" and name.endswith("Params"):
            assert fields[:2] == ["struct_size", "abi_minor"], name


def test_no_pointer_types_in_the_shared_memory_header():
    text = layout_tool.COMMENT.sub("", (layout_tool.INCLUDE / "roost_worker_shm.h").read_text(encoding="utf-8"))
    assert "*" not in text.replace("/*", "").replace("*/", "")
    assert "void" not in text


def test_golden_records_a_type_class_per_field():
    golden = layout_tool.GOLDEN.read_text(encoding="utf-8")
    rows = [line.split() for line in golden.splitlines() if line.startswith("  ")]
    assert rows and all(len(row) == 4 for row in rows)
    typed = {name: dict(fields) for _, name, fields in layout_tool.all_typed_structs()}
    assert typed["RoostRingSlot"]["published_ns"] == "u64"
    assert typed["RoostPlanSampling"]["temperature"] == "f32"
    assert typed["RoostOpEntry"]["run"] == "ptr"


def test_no_shared_memory_struct_has_a_pointer_field_in_the_golden_file():
    for header, name, fields in layout_tool.all_typed_structs():
        if header == "roost_worker_shm.h":
            assert all(kind != "ptr" for _, kind in fields), name


def test_type_class_separates_same_size_types_and_rejects_unknown_ones():
    classify = layout_tool.type_class
    assert classify("uint64_t ", set(), set()) == "u64"
    assert classify("void* ", set(), set()) == "ptr"
    assert classify("const int32_t* ", set(), set()) == "ptr"
    assert classify("RoostRunFn ", {"RoostRunFn"}, set()) == "ptr"
    assert classify("float ", set(), set()) == "f32"
    assert classify("uint32_t ", set(), set()) == "u32"
    with pytest.raises(ValueError):
        classify("size_t ", set(), set())


def test_padding_errors_detects_gaps():
    layout = "struct S size=16 align=8\n  a 0 4\n  b 8 8\n"
    assert layout_tool.padding_errors(layout) == ["S.b: 4 bytes of padding before it"]
    assert layout_tool.padding_errors("struct T size=8 align=8\n  a 0 4\n") == ["T: 4 bytes of tail padding"]
    assert layout_tool.padding_errors(layout_tool.GOLDEN.read_text(encoding="utf-8")) == []


def test_program_mentions_every_struct_and_field():
    program = layout_tool.program()
    for _, name, fields in layout_tool.all_typed_structs():
        assert f"S({name});" in program
        for field, kind in fields:
            assert f'F({name}, {field}, "{kind}");' in program


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
def test_golden_matches_gcc():
    assert layout_tool.check(update=False) == 0

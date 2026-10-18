"""Profile manifest schema and validator tests on synthetic manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DATA
from jsonschema import Draft202012Validator
from roost_evidence import cli, manifest


def messages(document: object) -> list[str]:
    return [f"{p.location}: {p.message}" for p in manifest.validate_document(document)]


def has(document: object, location: str, fragment: str = "") -> bool:
    return any(p.location == location and fragment in p.message for p in manifest.validate_document(document))


def test_schema_is_a_valid_draft_2020_12_schema() -> None:
    Draft202012Validator.check_schema(manifest.schema())
    assert manifest.schema()["properties"]["schema_version"] == {"const": 1}


def test_valid_synthetic_manifest_passes(valid: dict) -> None:
    assert messages(valid) == []


def test_valid_manifest_in_json_form_passes(valid: dict) -> None:
    assert manifest.validate_document(manifest.load_text(json.dumps(valid), ".json")) == []


# Each case breaks one acceptance field (R-012) in one way and names where the validator must object.
def drop(*keys: str):
    def apply(doc: dict) -> None:
        target = doc
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]

    return apply


def put(keys: tuple, value: object):
    def apply(doc: dict) -> None:
        target = doc
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return apply


P = "profile"
CASES = {
    "missing kernel manifest": (drop(P, "kernel_manifest"), "$.profile", "kernel_manifest"),
    "branch name as kernel revision": (
        put((P, "kernel_manifest", "revision"), "main"),
        "$.profile.kernel_manifest.revision",
        "",
    ),
    "short digest": (put((P, "kernel_manifest", "sha256"), "abc"), "$.profile.kernel_manifest.sha256", ""),
    "missing selected variants": (drop(P, "selected_variants"), "$.profile", "selected_variants"),
    "variant digest wrong length": (
        put((P, "selected_variants", 0, "digest"), "44" * 31),
        "$.profile.selected_variants[0].digest",
        "",
    ),
    "variant extra property": (
        put((P, "selected_variants", 0, "path"), "/tmp/x.so"),
        "$.profile.selected_variants[0]",
        "path",
    ),
    "tf32 policy as boolean": (put((P, "tf32_policy"), False), "$.profile.tf32_policy", ""),
    "tf32 policy missing": (drop(P, "tf32_policy"), "$.profile", "tf32_policy"),
    "kv dtype as scalar": (put((P, "kv_dtype"), "bf16"), "$.profile.kv_dtype", ""),
    "kv dtype unknown": (put((P, "kv_dtype", 0, "dtype"), "fp64"), "$.profile.kv_dtype[0].dtype", ""),
    "kv group kind unknown": (put((P, "kv_dtype", 0, "kind"), "paged"), "$.profile.kv_dtype[0].kind", ""),
    "kv groups empty": (put((P, "kv_dtype"), []), "$.profile.kv_dtype", ""),
    "attention backend empty": (put((P, "attention_backend"), ""), "$.profile.attention_backend", ""),
    "sm unknown arch": (put((P, "sm", 0, "arch"), "sm_75"), "$.profile.sm[0].arch", ""),
    "sm string instead of entry": (put((P, "sm"), ["sm_86"]), "$.profile.sm[0]", ""),
    "sm status missing": (drop(P, "sm", 1, "status"), "$.profile.sm[1]", "status"),
    "sm supported without gpus": (drop(P, "sm", 0, "gpus"), "$.profile.sm[0]", "gpus"),
    "sm supported without evidence": (drop(P, "sm", 0, "evidence"), "$.profile.sm[0]", "evidence"),
    "driver unquoted float": (put((P, "min_driver"), 580.65), "$.profile.min_driver", ""),
    "driver with text": (put((P, "min_driver"), "r580"), "$.profile.min_driver", ""),
    "toolkit unquoted float": (put((P, "min_toolkit"), 13.0), "$.profile.min_toolkit", ""),
    "toolkit missing": (drop(P, "min_toolkit"), "$.profile", "min_toolkit"),
    "mesh missing": (drop(P, "mesh"), "$.profile", "mesh"),
    "mesh zero ranks": (put((P, "mesh", "ranks"), 0), "$.profile.mesh.ranks", ""),
    "mesh unknown axis": (put((P, "mesh", "sp"), 2), "$.profile.mesh", "sp"),
    "layer precision without evidence": (
        drop(P, "layer_precision", 0, "evidence"),
        "$.profile.layer_precision[0]",
        "evidence",
    ),
    "layer precision negative layer": (
        put((P, "layer_precision", 0, "layer"), -1),
        "$.profile.layer_precision[0].layer",
        "",
    ),
    "layer precision evidence escapes repo": (
        put((P, "layer_precision", 0, "evidence"), "../x.json"),
        "$.profile.layer_precision[0].evidence",
        "",
    ),
    "model revision is a tag": (put(("model", "source", "revision"), "v1.0"), "$.model.source.revision", ""),
    "model extra property": (put(("model", "weights_url"), "https://example.invalid"), "$.model", "weights_url"),
    "supported capability without evidence": (
        drop("capabilities", "text-generation", "evidence"),
        "$.capabilities.text-generation",
        "evidence",
    ),
    "capability status missing": (
        put(("capabilities", "chunked-prefill"), {}),
        "$.capabilities.chunked-prefill",
        "status",
    ),
    "capability status unknown": (
        put(("capabilities", "chunked-prefill", "status"), "maybe"),
        "$.capabilities.chunked-prefill.status",
        "",
    ),
    "schema version 2": (put(("schema_version",), 2), "$.schema_version", ""),
    "id without profile part": (put(("id",), "synthetic"), "$.id", ""),
    "unknown top-level field": (put(("owner",), "x"), "$", "owner"),
    "maturity unknown": (put(("maturity",), "beta"), "$.maturity", ""),
}


@pytest.mark.parametrize(("mutate", "location", "fragment"), CASES.values(), ids=CASES.keys())
def test_invalid_manifest_is_rejected_at_the_right_place(valid: dict, mutate, location: str, fragment: str) -> None:
    mutate(valid)
    assert has(valid, location, fragment), messages(valid)


def test_top_level_must_be_an_object() -> None:
    assert manifest.validate_document([1, 2]) != []


# Maturity rules (AGENTS.md: an SM without hardware evidence is untested; missing never means supported).
def test_proposed_may_have_no_variants_and_only_untested_sm(valid: dict) -> None:
    valid["maturity"] = "proposed"
    valid["profile"]["selected_variants"] = []
    valid["profile"]["sm"] = [{"arch": "sm_86", "status": "untested"}]
    assert messages(valid) == []


def test_experimental_needs_an_sm_with_evidence(valid: dict) -> None:
    valid["profile"]["sm"] = [{"arch": "sm_86", "status": "untested"}]
    valid["profile"]["selected_variants"] = []
    assert has(valid, "$.profile.sm", "does not contain")


def test_preview_needs_a_supported_sm_and_a_selected_variant(valid: dict) -> None:
    valid["maturity"] = "preview"
    valid["profile"]["sm"][0]["status"] = "experimental"
    valid["profile"]["selected_variants"] = []
    found = messages(valid)
    assert any(m.startswith("$.profile.sm:") for m in found)
    assert any(m.startswith("$.profile.selected_variants:") for m in found)


def test_preview_with_supported_sm_and_variant_passes(valid: dict) -> None:
    valid["maturity"] = "preview"
    assert messages(valid) == []


def test_stable_rejects_untested_capability(valid: dict) -> None:
    valid["maturity"] = "stable"
    assert has(valid, "$.capabilities.chunked-prefill.status")
    valid["capabilities"]["chunked-prefill"] = {"status": "unsupported"}
    assert messages(valid) == []


# Semantic checks the schema cannot express.
def test_duplicate_kv_group_name(valid: dict) -> None:
    valid["profile"]["kv_dtype"].append({"group": "attn", "kind": "sliding", "dtype": "fp16"})
    assert has(valid, "$.profile.kv_dtype", "'attn'")


def test_hybrid_model_with_several_groups_is_valid(valid: dict) -> None:
    valid["profile"]["kv_dtype"] = [
        {"group": "attn", "kind": "full", "dtype": "bf16"},
        {"group": "delta", "kind": "recurrent", "dtype": "fp32"},
    ]
    assert messages(valid) == []


def test_duplicate_sm_entry(valid: dict) -> None:
    valid["profile"]["sm"].append({"arch": "sm_86", "status": "untested"})
    assert has(valid, "$.profile.sm", "sm_86")


def test_variant_for_undeclared_sm(valid: dict) -> None:
    valid["profile"]["selected_variants"][0]["sm"] = "sm_90"
    assert has(valid, "$.profile.selected_variants[0].sm", "not declared")


def test_variant_for_untested_sm(valid: dict) -> None:
    valid["profile"]["selected_variants"][0]["sm"] = "sm_89"
    assert has(valid, "$.profile.selected_variants[0].sm", "untested")


def test_duplicate_variant_selection(valid: dict) -> None:
    valid["profile"]["selected_variants"].append(dict(valid["profile"]["selected_variants"][0]))
    assert has(valid, "$.profile.selected_variants", "more than once")


@pytest.mark.parametrize(
    ("mesh", "ok"),
    [
        ({"ranks": 1}, True),
        ({"ranks": 4, "tp": 2, "pp": 2}, True),
        ({"ranks": 8, "tp": 2, "pp": 2, "dp": 2, "ep": 4}, True),
        ({"ranks": 4, "tp": 2}, False),
        ({"ranks": 4, "tp": 2, "pp": 2, "ep": 3}, False),
    ],
)
def test_mesh_arithmetic(valid: dict, mesh: dict, ok: bool) -> None:
    valid["profile"]["mesh"] = mesh
    assert (messages(valid) == []) is ok, messages(valid)


def test_duplicate_layer_precision_index(valid: dict) -> None:
    valid["profile"]["layer_precision"].append(dict(valid["profile"]["layer_precision"][0]))
    assert has(valid, "$.profile.layer_precision", "layer 0")


def test_layer_precision_outside_model(valid: dict) -> None:
    valid["profile"]["layer_precision"][0]["layer"] = 4
    assert has(valid, "$.profile.layer_precision[0].layer", "num_layers")


def test_layer_precision_needs_num_layers(valid: dict) -> None:
    del valid["model"]["num_layers"]
    assert has(valid, "$.model.num_layers")


# Loading.
def test_yaml_duplicate_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate key 'id'"):
        manifest.load_text("id: a/b\nid: c/d\n", ".yaml")


def test_json_duplicate_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        manifest.load_text('{"id": "a/b", "id": "c/d"}', ".json")


def test_yaml_unquoted_toolkit_is_a_float_and_fails() -> None:
    text = (
        (DATA / "valid-synthetic.yaml").read_text(encoding="utf-8").replace('min_toolkit: "13.0"', "min_toolkit: 13.0")
    )
    assert has(manifest.load_text(text, ".yaml"), "$.profile.min_toolkit", "is not of type 'string'")


def test_yaml_date_stays_a_string() -> None:
    assert manifest.load_text("d: 2026-10-18\n", ".yaml") == {"d": "2026-10-18"}


# Files and the command.
def test_validate_file_valid() -> None:
    assert manifest.validate_file(DATA / "valid-synthetic.yaml") == []


def test_registry_mode_rejects_synthetic() -> None:
    problems = manifest.validate_file(DATA / "valid-synthetic.yaml", registry=True)
    assert [p.location for p in problems] == ["$.synthetic"]


def test_unparseable_and_missing_files_are_problems(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed\n", encoding="utf-8")
    assert "cannot parse" in str(manifest.validate_file(bad)[0])
    assert "cannot read" in str(manifest.validate_file(tmp_path / "absent.yaml")[0])


def test_cli_exit_codes_and_output(valid: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(valid), encoding="utf-8")
    assert cli.main(["validate-manifest", str(good)]) == 0
    assert capsys.readouterr().out == ""

    valid["profile"]["tf32_policy"] = "auto"
    broken = tmp_path / "sub" / "broken.json"
    broken.parent.mkdir()
    broken.write_text(json.dumps(valid), encoding="utf-8")
    assert cli.main(["validate-manifest", str(tmp_path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"{broken}: $.profile.tf32_policy: ")


def test_cli_with_an_empty_directory_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate-manifest", str(tmp_path)]) == 0
    assert "no manifest files" in capsys.readouterr().err


def test_yaml_unhashable_key_is_a_reported_problem(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="unhashable mapping key"):
        manifest.load_text("? [a, b]\n: 1\n", ".yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("? [a, b]\n: 1\n", encoding="utf-8")
    assert cli.main(["validate-manifest", str(bad)]) == 1
    assert "unhashable mapping key" in capsys.readouterr().out

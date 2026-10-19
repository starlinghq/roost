"""Tests of the kernel build tools: manifest -> CMake, digest record, and the provenance check (R-016).

Run: python3 -I -m pytest -q kernels/tests. The provenance tests build small git repositories in a temporary directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "kernels" / "tools"
SHA_A = "a" * 40
SHA_B = "b" * 40


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


provenance = load("check_provenance")
digests = load("record_digests")
manifest_cmake = load("manifest_cmake")


def test_repository_manifest_validates_and_is_consistent():
    validator = subprocess.run(
        [sys.executable, "-I", str(ROOT / "tools" / "validate-kernel-manifest.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validator.returncode == 0, validator.stdout
    assert provenance.main([]) == 0


def test_every_upstream_pin_is_a_full_sha_and_the_sm_list_is_complete():
    manifest = tomllib.loads((ROOT / "kernels" / "manifest.toml").read_text(encoding="utf-8"))
    assert all(len(u["sha"]) == 40 for u in manifest["upstreams"])
    assert [s["arch"] for s in manifest["sm"]] == [80, 86, 89, 90, 100, 120]
    # An SM without hardware evidence is untested.
    assert {s["arch"] for s in manifest["sm"] if s["status"] != "untested"} == {86}


def test_cmake_variables_come_from_the_manifest():
    manifest = tomllib.loads((ROOT / "kernels" / "manifest.toml").read_text(encoding="utf-8"))
    text = manifest_cmake.cmake_text(manifest)
    assert 'set(ROOST_MANIFEST_SM_LIST "80;86;89;90;100;120")' in text
    for i, upstream in enumerate(manifest["upstreams"]):
        assert f'set(ROOST_UPSTREAM_{i}_SHA "{upstream["sha"]}")' in text


def make_repo(tmp_path: Path, sha: str, provenance_text: str, notices_text: str) -> Path:
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "manifest.toml").write_text(
        f'[[upstreams]]\nname = "up"\nrepo = "https://example.com/up"\nsha = "{sha}"\n', encoding="utf-8"
    )
    (tmp_path / "PROVENANCE.md").write_text(provenance_text, encoding="utf-8")
    (tmp_path / "THIRD-PARTY-NOTICES.md").write_text(notices_text, encoding="utf-8")
    return tmp_path


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def row(sha: str) -> str:
    return f"| up | https://example.com/up | `{sha}` | x | MIT | None |\n"


def test_sha_bump_without_provenance_files_fails_and_with_them_passes(tmp_path: Path):
    root = make_repo(tmp_path, SHA_A, row(SHA_A), f"https://example.com/up {SHA_A}\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    base = git(root, "rev-parse", "HEAD")

    # Bump the manifest only: the old rows still name the old SHA, so the tree is inconsistent and the range is wrong.
    (root / "kernels" / "manifest.toml").write_text(
        f'[[upstreams]]\nname = "up"\nrepo = "https://example.com/up"\nsha = "{SHA_B}"\n', encoding="utf-8"
    )
    git(root, "commit", "-q", "-am", "bump only")
    manifest = tomllib.loads((root / "kernels" / "manifest.toml").read_text(encoding="utf-8"))
    errors = provenance.consistency_errors(manifest, root) + provenance.diff_errors(manifest, root, base)
    assert any("PROVENANCE.md: no row" in e for e in errors)
    assert any("THIRD-PARTY-NOTICES.md: no section" in e for e in errors)
    assert any("PROVENANCE.md is not changed" in e for e in errors)
    assert any("THIRD-PARTY-NOTICES.md is not changed" in e for e in errors)

    # Touching the two files without naming the new SHA is still refused by the consistency check.
    (root / "PROVENANCE.md").write_text(row(SHA_A) + "\n", encoding="utf-8")
    (root / "THIRD-PARTY-NOTICES.md").write_text(f"https://example.com/up {SHA_A}\n\n", encoding="utf-8")
    git(root, "commit", "-q", "-am", "touch only")
    assert provenance.consistency_errors(manifest, root)

    # The real bump: both files carry the new revision.
    (root / "PROVENANCE.md").write_text(row(SHA_B), encoding="utf-8")
    (root / "THIRD-PARTY-NOTICES.md").write_text(f"https://example.com/up {SHA_B}\n", encoding="utf-8")
    git(root, "commit", "-q", "-am", "bump with provenance")
    assert provenance.consistency_errors(manifest, root) == []
    assert provenance.diff_errors(manifest, root, base) == []


def test_patch_digest_and_unlisted_patch_are_checked(tmp_path: Path):
    root = make_repo(tmp_path, SHA_A, row(SHA_A), f"https://example.com/up {SHA_A}\n")
    patches = root / "kernels" / "upstream" / "up.patches"
    patches.mkdir(parents=True)
    listed = patches / "a.patch"
    listed.write_text("--- a\n+++ b\n", encoding="utf-8")
    (patches / "b.patch").write_text("unlisted\n", encoding="utf-8")
    digest = hashlib.sha256(listed.read_bytes()).hexdigest()
    manifest = {
        "upstreams": [
            {
                "name": "up",
                "repo": "https://example.com/up",
                "sha": SHA_A,
                "patches": [{"path": "kernels/upstream/up.patches/a.patch", "sha256": digest}],
            }
        ]
    }
    errors = provenance.consistency_errors(manifest, root)
    assert errors == ["kernels/upstream/up.patches/b.patch: patch not listed in the manifest"]
    manifest["upstreams"][0]["patches"][0]["sha256"] = "0" * 64
    assert any("SHA-256 differs" in e for e in provenance.consistency_errors(manifest, root))


MANIFEST_WITH_BLOCK = """
[[sm]]
arch = 86
status = "experimental"

[[sm]]
arch = 89
status = "untested"

[[ops]]
op = "Residual"

[[ops.variants]]
id = "v"
source = "roost"
flags = []

[ops.variants.constraints]
sm_min = 80

# BEGIN binaries v
# END binaries v
"""


def test_digest_block_is_rewritten_in_place_and_check_detects_a_difference():
    built = {86: {"sha256": "1" * 64, "bytes": 10}}
    text = digests.rewrite(MANIFEST_WITH_BLOCK, built)
    parsed = tomllib.loads(text)
    assert parsed["ops"][0]["variants"][0]["binaries"] == [{"arch": 86, "sha256": "1" * 64, "bytes": 10}]
    assert digests.check(text, built) == []
    assert digests.check(text, {86: {"sha256": "2" * 64, "bytes": 10}})
    # Another SM is added without dropping the first; an SM not in the manifest's list is not recorded.
    more = digests.rewrite(text, {89: {"sha256": "3" * 64, "bytes": 11}, 120: {"sha256": "4" * 64, "bytes": 1}})
    assert [b["arch"] for b in tomllib.loads(more)["ops"][0]["variants"][0]["binaries"]] == [86, 89]


def test_digest_json_and_library_discovery(tmp_path: Path):
    (tmp_path / "libroost_kernels_sm86.so").write_bytes(b"abc")
    (tmp_path / "libother.so").write_bytes(b"x")
    found = digests.measured(tmp_path)
    assert list(found) == [86]
    assert found[86] == {"sha256": hashlib.sha256(b"abc").hexdigest(), "bytes": 3}
    out = tmp_path / "d.json"
    assert digests.main([str(tmp_path), "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["86"]["bytes"] == 3


@pytest.mark.parametrize("flag", ["--sm-list"])
def test_sm_list_flag(flag, capsys):
    assert manifest_cmake.main([str(ROOT / "kernels" / "manifest.toml"), flag]) == 0
    assert capsys.readouterr().out.strip() == "80;86;89;90;100;120"


def _code(path: Path) -> str:
    """Source text without // comments and string literals' contents, for scans of what a file calls."""
    import re

    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def test_m0_op_sources_do_not_allocate_or_synchronise_in_calls():
    """C1 and R-017: entry points never allocate device memory and never synchronise the host.

    Setup may allocate (roost_context_create builds the cuBLASLt handle; plan() uses host containers inside FlashInfer
    and cuBLASLt), but the op sources themselves never call the allocating or synchronising runtime functions.
    """
    import re

    forbidden = re.compile(
        r"\b(cudaMalloc\w*|cudaFree\w*|cudaHostAlloc|cudaDeviceSynchronize|cudaStreamSynchronize|cudaEventSynchronize"
        r"|cudaMemcpy|cudaMemset|cudaMallocAsync|cudaFreeAsync|malloc|calloc|realloc|free)\s*\("
    )
    for name in ("m0_ops.cu", "paged_attention.cu", "gemm.cu", "op_common.h"):
        hits = forbidden.findall(_code(ROOT / "kernels" / "src" / name))
        assert not hits, f"{name} calls {hits}"
        assert " new " not in _code(ROOT / "kernels" / "src" / name), f"{name} uses new"


def test_registered_m0_variants_are_the_manifests():
    import re

    registry = (ROOT / "kernels" / "src" / "m0_registry.cu").read_text(encoding="utf-8")
    registered = set(re.findall(r'Entry\(ROOST_OP_\w+,\s*\w+(?:\s*\|\s*\w+)*,\s*"([a-z0-9.-]+)"', registry))
    manifest = tomllib.loads((ROOT / "kernels" / "manifest.toml").read_text(encoding="utf-8"))
    declared = {v["id"] for op in manifest["ops"] for v in op["variants"]} - {"residual-stub"}
    assert registered == declared
    # Every M0 variant of the manifest names a source the manifest knows: an upstream, an external library or "roost".
    sources = (
        {u["name"] for u in manifest["upstreams"]} | {x["name"] for x in manifest["external_libraries"]} | {"roost"}
    )
    assert {v["source"] for op in manifest["ops"] for v in op["variants"]} <= sources
    # Pinned Gemm algorithms name the cuBLASLt build they were observed with.
    digest = {x["name"]: x["sha256"] for x in manifest["external_libraries"]}["cublaslt"]
    gemm = next(v for op in manifest["ops"] for v in op["variants"] if v["id"] == "gemm-cublaslt")
    assert gemm["gemm_algos"] and all(a["library_sha256"] == digest for a in gemm["gemm_algos"])

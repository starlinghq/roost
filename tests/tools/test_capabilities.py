"""Shape checks for the capability table, docs/capabilities.md (groundwork for R-026 and R-062).

Every row carries exactly one of the four statuses; every row that is not unsupported names evidence that exists in the
repository; a supported row cites evidence/protected/; and the rows that code also holds agree with that code
(src/Starlings.Generation.Roost/RoostCapabilities.cs and the [[sm]] entries of kernels/manifest.toml).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "docs" / "capabilities.md"
STATUSES = {"supported", "experimental", "untested", "unsupported"}
HEADER = ["Capability", "Status", "Evidence", "Notes"]
CODE = re.compile(r"`([^`]+)`")


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def rows() -> list[dict[str, str]]:
    """Every data row of every table whose header is HEADER, keyed by column name, plus its section."""
    out: list[dict[str, str]] = []
    lines = TABLE.read_text(encoding="utf-8").splitlines()
    section = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            section = line[3:].strip()
        if line.startswith("|") and cells(line) == HEADER:
            i += 2  # header and separator
            while i < len(lines) and lines[i].startswith("|"):
                row = dict(zip(HEADER, cells(lines[i]), strict=True))
                row["section"] = section
                out.append(row)
                i += 1
            continue
        i += 1
    return out


def status_of(row: dict[str, str]) -> str:
    m = re.fullmatch(r"`([a-z]+)`", row["Status"])
    assert m, f"{row['Capability']}: status cell {row['Status']!r} is not one backticked word"
    return m.group(1)


def evidence_paths(row: dict[str, str]) -> list[str]:
    return CODE.findall(row["Evidence"])


def test_table_has_rows():
    assert len(rows()) > 100


def test_every_row_has_one_of_the_four_statuses():
    for row in rows():
        assert status_of(row) in STATUSES, row["Capability"]


def test_capabilities_are_unique_within_a_section():
    seen: set[tuple[str, str]] = set()
    for row in rows():
        key = (row["section"], row["Capability"])
        assert key not in seen, key
        seen.add(key)


def test_every_row_not_unsupported_names_evidence_that_exists():
    for row in rows():
        paths = evidence_paths(row)
        if status_of(row) == "unsupported":
            continue
        assert paths, f"{row['Capability']}: {status_of(row)} without evidence"
        for p in paths:
            assert not Path(p).is_absolute() and ".." not in Path(p).parts, p
            assert (ROOT / p).exists(), f"{row['Capability']}: evidence {p} does not exist"


def test_evidence_of_unsupported_rows_is_empty():
    for row in rows():
        if status_of(row) == "unsupported":
            assert row["Evidence"] == "—", row["Capability"]


def test_supported_needs_protected_evidence():
    for row in rows():
        if status_of(row) == "supported":
            assert any(p.startswith("evidence/protected/") for p in evidence_paths(row)), row["Capability"]


def roost_capabilities() -> dict[str, str]:
    """name -> status from RoostCapabilities.cs: the constants' values and the dictionary initialiser."""
    src = (ROOT / "src" / "Starlings.Generation.Roost" / "RoostCapabilities.cs").read_text(encoding="utf-8")
    consts = dict(re.findall(r'public const string (\w+) = "([^"]+)";', src))
    aliases = dict(re.findall(r"var (\w+) = RoostCapabilityStatus\.(\w+);", src))
    statuses = {}
    for key, value in re.findall(r"\[(\w+)\] = (\w+(?:\.\w+)?),", src):
        status = value.removeprefix("RoostCapabilityStatus.")
        status = aliases.get(status, status)
        statuses[consts[key]] = status.lower()
    assert statuses and set(statuses) == set(consts.values())
    return statuses


def test_roost_chat_client_rows_agree_with_roost_capabilities():
    table = {}
    for row in rows():
        if row["section"].startswith("`RoostChatClient`"):
            m = re.fullmatch(r"`([a-z_]+)`", row["Capability"])
            assert m, row["Capability"]
            table[m.group(1)] = status_of(row)
    assert table == roost_capabilities()


def test_sm_rows_agree_with_kernel_manifest():
    manifest = tomllib.loads((ROOT / "kernels" / "manifest.toml").read_text(encoding="utf-8"))
    expected = {f"SM {sm['arch']}": sm["status"] for sm in manifest["sm"]}
    table = {row["Capability"]: status_of(row) for row in rows() if re.fullmatch(r"SM \d+", row["Capability"])}
    assert table == expected


def test_readme_links_the_table():
    assert "(docs/capabilities.md)" in (ROOT / "README.md").read_text(encoding="utf-8")

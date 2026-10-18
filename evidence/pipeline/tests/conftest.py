"""Shared fixtures: a valid synthetic manifest that each test copies and breaks in one place."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from roost_evidence import manifest

DATA = Path(__file__).parent / "data"


@pytest.fixture
def valid() -> dict:
    """A fresh copy of the valid synthetic manifest."""
    return copy.deepcopy(manifest.load_text((DATA / "valid-synthetic.yaml").read_text(encoding="utf-8"), ".yaml"))

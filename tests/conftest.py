from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = json.loads((FIXTURES.parent / "expected.json").read_text(encoding="utf-8"))["files"]


@pytest.fixture(scope="session")
def results():
    """`check()` of every fixture, computed once (the crop search takes a moment)."""
    from daf_check import check

    cache: dict[str, object] = {}

    def get(name: str):
        if name not in cache:
            cache[name] = check(FIXTURES / name)
        return cache[name]

    return get

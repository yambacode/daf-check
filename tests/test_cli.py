from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import FIXTURES

from daf_check.cli import main

SIGNED = str(FIXTURES / "v1_signed.png")  # the quickest marked fixture to read
UNSIGNED = str(FIXTURES / "unsigned.png")


def test_exit_0_when_a_mark_says_ai(capsys):
    assert main([SIGNED]) == 0
    out = capsys.readouterr().out
    assert "AI-generated" in out and "version 1" in out


def test_exit_1_when_no_mark(capsys):
    assert main([UNSIGNED]) == 1
    out = capsys.readouterr().out
    assert "No Dream Art Fields mark found. That does not prove a person made it" in out


def test_exit_0_when_any_file_has_a_mark(capsys):
    assert main([UNSIGNED, SIGNED]) == 0


def test_exit_2_on_missing_file(capsys, tmp_path):
    assert main([str(tmp_path / "nope.png")]) == 2
    assert "could not read" in capsys.readouterr().err


def test_exit_2_on_usage_error(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["--bogus", SIGNED])
    assert e.value.code == 2


def test_json_shape(capsys, tmp_path):
    missing = str(tmp_path / "nope.png")
    assert main([SIGNED, UNSIGNED, missing, "--json"]) == 2
    doc = json.loads(capsys.readouterr().out)
    assert set(doc) == {"daf_check", "results"} and len(doc["results"]) == 3
    signed, unsigned, err = doc["results"]
    for r in (signed, unsigned):
        assert {"source", "sha256", "mime", "size", "ai_generated", "summary", "watermark",
                "c2pa", "binding_matches", "notes"} <= set(r)  # fmt: skip
        assert {"checked", "found", "note"} <= set(r["watermark"])
        assert {"present", "verified", "intact", "trusted", "ai_generated", "generator",
                "signer", "signed_at", "time_stamped"} <= set(r["c2pa"])  # fmt: skip
    assert signed["ai_generated"] and signed["watermark"]["found"]
    assert signed["watermark"]["version"] == 1 and len(signed["watermark"]["asset_hash"]) == 8
    assert not unsigned["ai_generated"] and not unsigned["watermark"]["found"]
    assert set(err) == {"source", "error"}


def test_module_entry_point():
    proc = subprocess.run([sys.executable, "-m", "daf_check", UNSIGNED, "--json"],
                          capture_output=True, text=True, timeout=60)  # fmt: skip
    assert proc.returncode == 1, proc.stderr
    assert json.loads(proc.stdout)["results"][0]["ai_generated"] is False

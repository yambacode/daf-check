"""The C2PA soft-binding names a reader accepts (SPEC.md section 9)."""

from __future__ import annotations

import pytest

from daf_check.manifest import (
    AUDIO_WATERMARK_ALG,
    WATERMARK_ALG,
    WATERMARK_ALG_LEGACY,
    _summarise,
)


def _report(alg: str) -> dict:
    """A c2pa reader report with one soft-binding for asset hash 0x01020304."""
    binding = {"alg": alg, "blocks": [{"scope": {}, "value": "AQIDBA=="}]}
    return {
        "active_manifest": "m",
        "manifests": {"m": {"assertions": [{"label": "c2pa.soft-binding", "data": binding}]}},
    }


@pytest.mark.parametrize(
    "alg",
    [
        WATERMARK_ALG,  # pictures and video, v2 tile
        AUDIO_WATERMARK_ALG,  # audio
        WATERMARK_ALG_LEGACY,  # manifests from DAF before 2026-11-08
    ],
)
def test_every_dream_art_fields_name_is_read(alg):
    assert _summarise(_report(alg), "Valid").soft_binding == "01020304"


def test_the_names_are_the_versioned_ones():
    assert WATERMARK_ALG == "com.dreamartfields.signature.2"
    assert AUDIO_WATERMARK_ALG == "com.dreamartfields.signature.3"
    assert WATERMARK_ALG_LEGACY == "com.dreamartfields.signature"


@pytest.mark.parametrize(
    "alg", ["com.dreamartfields.signature.9", "com.example.signature", "", "castLabs.1"]
)
def test_other_names_are_ignored(alg):
    assert _summarise(_report(alg), "Valid").soft_binding is None

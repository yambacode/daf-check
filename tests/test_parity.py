"""daf-check gives the same results as Dream Art Fields' own detector on files DAF made.

`tests/expected.json` was recorded with DAF's reference code when the fixtures were made
(`tools/make_fixtures.py`).
"""

from __future__ import annotations

import pytest
from conftest import EXPECTED, FIXTURES

from daf_check import id_hash
from daf_check.manifest import c2pa_available
from daf_check.video import ffmpeg_available
from daf_check.watermark import default_detector, table_digests

# SHA-256 of the key-derived tables (SPEC.md, section 3).
DIGESTS = {
    "v2": "75869c0aa70962b211221a8020315eebafdc0166439bde0c0f670f43ca352a99",
    "v1": "c58ed5fafa0dfc0ae85756ec35ebc9baa2e149c0345a1a7bf4e6b36f23c40b9e",
}

PICTURES = [n for n in EXPECTED if not n.endswith(".mp4")]
VIDEOS = [n for n in EXPECTED if n.endswith(".mp4")]
# v2 video as exported, the same after a 20 % crop and CRF 28, and a clip exported before v2.
VIDEO_FIXTURES = ["video_v2_signed.mp4", "video_v2_crop20_crf28.mp4", "video_v1_signed.mp4"]


def _needs(name: str) -> None:
    if name in VIDEOS and not ffmpeg_available():
        pytest.skip("ffmpeg is not installed")


def test_every_kind_of_fixture_is_there():
    names = set(EXPECTED)
    assert {"v2_signed.png", "v2_crop20_q70.jpg", "v1_signed.png", "unsigned.png",
            "c2pa_signed.png", *VIDEO_FIXTURES} <= names  # fmt: skip
    for name, exp in EXPECTED.items():
        assert (FIXTURES / name).stat().st_size == exp["bytes"] <= 300_000


def test_key_derived_tables_match_the_spec():
    assert table_digests() == DIGESTS
    det = default_detector()
    assert det.v2.role.shape == (31, 31) and det.v1.bit_of.shape == (256, 256)
    assert int((det.v2.role == -1).sum()) == 31 * 31 // 4


@pytest.mark.parametrize("name", list(EXPECTED))
def test_watermark_matches_daf(name, results):
    _needs(name)
    exp, r = EXPECTED[name]["watermark"], results(name)
    if exp is None:
        assert r.watermark is None
        return
    assert r.watermark is not None, f"DAF found {exp}, daf-check found nothing"
    w = r.watermark
    assert (w.version, w.ai, w.id_hash) == (exp["version"], exp["ai"], exp["id_hash"])
    assert abs(w.confidence - exp["confidence"]) <= 0.05


@pytest.mark.parametrize("name", [n for n in EXPECTED if EXPECTED[n]["asset_id"]])
def test_asset_hash_points_back_to_the_library_item(name, results):
    _needs(name)
    exp, r = EXPECTED[name], results(name)
    assert id_hash(exp["asset_id"]) == exp["id_hash"]
    assert r.watermark is not None and r.watermark.matches(exp["asset_id"])


def test_unsigned_gives_nothing(results):
    r = results("unsigned.png")
    assert r.watermark is None and r.watermark_checked
    assert not r.c2pa.present and not r.ai_generated
    assert "does not prove a person made it" in r.summary()


@pytest.mark.parametrize("name", list(EXPECTED))
def test_c2pa_matches_daf(name, results):
    _needs(name)
    exp, c = EXPECTED[name]["c2pa"], results(name).c2pa
    assert c.present == exp["present"]
    if not c2pa_available():
        assert not c.verified
        assert c.ai_generated == exp["ai_generated"]
        return
    if not exp["present"]:
        return
    assert c.verified
    for key in ("state", "intact", "trusted", "ai_generated", "source_type", "generator",
                "issuer", "signed_at", "time_stamped", "system", "provider"):  # fmt: skip
        assert getattr(c, key) == exp[key], key


def test_c2pa_fixture_ties_manifest_and_pixels(results):
    r = results("c2pa_signed.png")
    assert r.ai_generated and r.c2pa.daf_generator
    if c2pa_available():
        assert r.c2pa.intact and not r.c2pa.trusted  # a throwaway local certificate
        assert r.c2pa.soft_binding == f"{EXPECTED['c2pa_signed.png']['id_hash']:08x}"
        assert r.binding_matches is True
        assert r.c2pa.models == ["example/image-model-1"]
        assert "Both marks" in r.summary()
    else:
        assert "not checked" in r.summary()


@pytest.mark.parametrize("name", VIDEO_FIXTURES)
def test_video_matches_daf_exactly(name):
    """Same version, AI flag, asset hash and confidence (2 decimals) as DAF's `detect_video`."""
    _needs(name)
    from daf_check.video import detect_video

    exp = EXPECTED[name]["watermark"]
    assert exp is not None, f"DAF found no mark in {name}"
    d = detect_video(FIXTURES / name)
    assert d is not None, f"DAF found {exp}, daf-check found nothing"
    assert {k: v for k, v in d.as_dict().items() if k in exp} == exp
    assert d.version == (1 if "v1" in name else 2)


def test_v2_video_reads_at_the_exports_grid_and_after_a_crop():
    """The uncropped export reads at its own grid (video_cells); the crop needs the search."""
    _needs(VIDEO_FIXTURES[0])
    import numpy as np
    from PIL import Image

    from daf_check.video import clip_frames, probe

    det = default_detector()
    for name, exact in ((VIDEO_FIXTURES[0], True), (VIDEO_FIXTURES[1], False)):
        info = probe(FIXTURES / name)
        size = (info.width, info.height)
        frames = clip_frames(FIXTURES / name, size, det.p.video_frames, info.duration)
        mean = frames.reshape(frames.shape[0], -1).mean(0, dtype=np.float64)
        im = Image.fromarray(mean.reshape(frames.shape[1:]).astype(np.float32))
        assert (det.v2.detect_exact(im, det.p.video_cells) is not None) == exact, name


def test_v1_video_frames_api_still_works():
    """`Detector.detect_frames` (version 1 video on the 256 x 256 grid) is kept."""
    _needs("video_v1_signed.mp4")
    from daf_check.video import grid_frames

    det = default_detector()
    frames = grid_frames(FIXTURES / "video_v1_signed.mp4", det.p.legacy.grid, det.p.video_frames)
    d = det.detect_frames(frames)
    exp = EXPECTED["video_v1_signed.mp4"]
    assert d is not None and d.version == 1 and d.id_hash == exp["id_hash"]


def test_unmarked_clip_reads_as_no_mark(tmp_path):
    """A clip DAF never touched (ffmpeg's `testsrc`, re-encoded) carries no mark."""
    if not ffmpeg_available():
        pytest.skip("ffmpeg is not installed")
    import subprocess

    from daf_check import check
    from daf_check.video import _tool, detect_video

    clip = tmp_path / "testsrc.mp4"
    subprocess.run([_tool("ffmpeg") or "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc=size=640x360:rate=12:duration=2",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(clip)],
                   check=True, timeout=120)  # fmt: skip
    assert detect_video(clip) is None
    r = check(clip)
    assert r.watermark is None and r.watermark_checked and not r.ai_generated


def test_video_without_ffmpeg_says_so(monkeypatch):
    if not VIDEOS:
        pytest.skip("no video fixture")
    import daf_check.video as video
    from daf_check import check

    monkeypatch.setattr(video, "_tool", lambda name: None)
    r = check(FIXTURES / VIDEOS[0])
    assert r.watermark is None and not r.watermark_checked
    assert "ffmpeg needed" in r.watermark_note and "ffmpeg needed" in r.summary()


def test_bytes_and_path_give_the_same_result(results):
    from daf_check import check

    data = (FIXTURES / "v1_signed.png").read_bytes()
    a, b = check(data), results("v1_signed.png")
    assert a.watermark == b.watermark and a.sha256 == b.sha256 and a.source == "<bytes>"

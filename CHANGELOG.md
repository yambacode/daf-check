# Changelog

## 0.2.0 (2026-11-08)

Reads the version 2 video signature, as Dream Art Fields main does since its video change.

* Video: the v2 tile in every frame (cells of short side / `video_cells` = 128). A check averages
  24 frames at the clip's own size (shrunk to a 1024 px short side above 1536 px), then reads v2
  at the export's grid (exact, or the known period at any offset), the v1 grid (clips exported
  before v2, still read), and the full v2 search with period hints for crops and common rescales.
  A cropped or re-encoded v2 clip is now found; results match DAF's `detect_video` exactly
  (version, asset hash, confidence to 2 decimals).
* Parameters (`daf_check/params.py`, SPEC.md section 2): new `video_cells` 128; `video_strength`
  2.4 and `video_mask` (0.6, 1.0) as in DAF's `branding.yaml`; `VIDEO_CELLS`,
  `VIDEO_SHORT_SIDES`, `NATIVE_UP_TO`.
* API: `Detector.detect_mean(mean, size)`, `TiledDetector.detect_period`, `period_px`, `cells` on
  `cell_px` / `detect_exact`, `hints` on `search` / `detect`; `daf_check.video.probe`,
  `clip_frames`. `detect_video(path, detector=None)` and `Detector.detect_frames` (v1) keep working.
* Picture detection is unchanged.
* SPEC.md section 8 rewritten for v2 video. README limits updated.
* Fixtures: a v2-signed 832 x 480 clip, the same after a 20 % crop and CRF 28, and the 0.1.0 v1
  clip (renamed `video_v1_signed.mp4`, same bytes); an unmarked clip test. `make_fixtures.py
  --keep` re-reads fixtures that are already there instead of making them again.

## 0.1.0 (2026-11-07)

First public version.

* Pixel signature detection, with the same results as Dream Art Fields' own detector: version 2
  (crop-robust tile, exact grid plus the crop and resize search), version 1 (legacy grid) and
  video (24 frames averaged on the version 1 grid, through the `ffmpeg` command-line tool).
* C2PA manifest reading with the optional `c2pa-python` library (presence, validation state,
  trust, AI claim, generator, signer, time, time-stamp, soft binding); without it, a presence
  and AI-claim scan that says the signature was not checked.
* `daf-check FILE [FILE...] [--json]` command and `daf_check.check()` Python API.
* SPEC.md: public specification of both marks.
* Parity tests on fixtures made with Dream Art Fields' reference code.

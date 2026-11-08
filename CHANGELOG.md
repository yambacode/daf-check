# Changelog

## 0.2.2 (2026-11-08)

Reads the versioned C2PA soft-binding names Dream Art Fields writes from now on.

* C2PA: the soft-binding assertion is read when its `alg` is `com.dreamartfields.signature.2`
  (pictures and video, v2 tile), `com.dreamartfields.signature.3` (audio) or, for manifests from
  DAF before 2026-11-08, `com.dreamartfields.signature`. 0.2.1 and older read only the last, so
  they report no binding for new files. Constants: `WATERMARK_ALG` (now the `.2` name),
  `AUDIO_WATERMARK_ALG`, `WATERMARK_ALG_LEGACY`, `WATERMARK_ALGS`.
* SPEC.md 1.1: section 9 names the three values; the header now names the package version.
* Watermark detection is unchanged; fixtures and `tests/expected.json` are unchanged.

## 0.2.1 (2026-11-08)

Follows Dream Art Fields' video reader (DAF PRs #68 and #76); results still match DAF exactly.

* Video: when the export's grid and v1 find nothing, the v2 search now reads the mean frame as it
  is and then, if that fails, the mean **flattened** (high-passed at pixel scale, divided by the
  local activity, times the new `video_flatten` = 16; SPEC.md section 8 step 6). A slow pan leaves
  a still picture that swamps the raw mean; flattened, a 20 % crop of a slow 720p pan at CRF 28
  reads for about 97 in 100 asset ids instead of 55 (DAF's bench). Clips the raw mean already
  read, such as the grainy crop fixture, read as before.
* Search: the period refine step stops when a sharpened period would come out longer than the
  side (zero or negative frequency) instead of raising. Before, such an input (an unmarked
  896 x 504 crop in DAF's false-positive runs) made a check report "no mark" for an error.
* The false-positive bound for one video check is now about 5.0e-5 (two v2 searches; SPEC.md
  section 8).
* Parameters: `video_flatten` (SPEC.md section 2). API: `Detector.flattened(mean)`.
* Fixtures and `tests/expected.json` are unchanged; a new test covers the refine step.

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

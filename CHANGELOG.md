# Changelog

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

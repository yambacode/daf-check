# daf-check

A free, open-source checker for the AI marks of [Dream Art Fields](https://dreamartfields.com), a
local AI image, video and audio app by YambaCode AB (Sweden).

Dream Art Fields marks everything it generates in two machine-readable layers:

1. a signed **C2PA manifest** in the file's metadata, saying the file was made by an AI system
   (IPTC digital source type `trainedAlgorithmicMedia`), by which system and when;
2. an invisible **pixel signature** in pictures and video frames, with a published detection key.

`daf-check` reads both and tells you, in plain words, what they show. It is the free detector that
the EU Code of Practice on marking AI-generated content (Sub-measures 2.1.1-2.1.2, under EU AI Act
Art. 50(2)) asks providers to offer, and the reference implementation of the public specification
in [SPEC.md](SPEC.md).

## Install

Python 3.10 or newer:

```sh
pip install "daf-check[c2pa] @ git+https://github.com/yambacode/daf-check"
```

or, without the C2PA library (the pixel signature is still checked; a C2PA manifest is only
detected, its signature is not verified):

```sh
pip install "git+https://github.com/yambacode/daf-check"
```

Video needs the [ffmpeg](https://ffmpeg.org) command-line tools (`ffmpeg` and `ffprobe`) on your
PATH. No GPU, no network and no account are needed.

## Use

```sh
daf-check picture.png
daf-check *.jpg clip.mp4 --json
```

Example output:

```text
c2pa_signed.png
  verdict         : AI-generated (a mark says so)
  summary         : Both marks say this was made with an AI system (Dream Art Fields).
  pixel signature : found: version 2, AI-generated, asset #0e430a94, confidence 9.93
  C2PA manifest   : signature valid (Valid), signer not on the C2PA trust list
                    AI claim: yes (trainedAlgorithmicMedia)
                    generator: Dream Art Fields 0.1.0
                    system: Dream Art Fields by YambaCode AB
                    signed by: daf-check test fixture (YambaCode AB (test fixture))
                    time: 2026-11-07T18:57:07+00:00 (signer's clock, not time-stamped)
  link            : the manifest names the same asset as the pixel signature

v2_crop20_q70.jpg
  verdict         : AI-generated (a mark says so)
  summary         : The invisible Dream Art Fields signature is in the pixels: it says this was made with an AI system (Dream Art Fields). No signed metadata says so: it is missing or was removed.
  pixel signature : found: version 2, AI-generated, asset #d56fd315, confidence 8.63
  C2PA manifest   : not found

unsigned.png
  verdict         : no Dream Art Fields mark found
  summary         : No Dream Art Fields mark found. That does not prove a person made it: marks can be removed, and other AI systems use other marks.
  pixel signature : not found
  C2PA manifest   : not found
```

Exit status: `0` if a mark in any of the files says AI-generated, `1` if none does, `2` on a usage
error or a file that could not be read.

`--json` prints `{"daf_check": "<version>", "results": [...]}` with one object per file:
`source`, `sha256`, `mime`, `size`, `ai_generated`, `summary`, `watermark` (`checked`, `found`,
`version`, `ai`, `id_hash`, `asset_hash`, `confidence`, `note`), `c2pa` (`present`, `verified`,
`state`, `intact`, `trusted`, `ai_generated`, `generator`, `signer`, `issuer`, `signed_at`,
`time_stamped`, `system`, `provider`, `models`, `soft_binding`, `errors`, ...), `binding_matches`
and `notes`. A file that could not be read gives `{"source": ..., "error": ...}`.

### Python

```python
import daf_check

r = daf_check.check("picture.png")       # a path, or the file's bytes
r.ai_generated                           # True if a mark says AI-generated
r.summary()                              # the plain-language sentence
r.watermark                              # Detection(version, ai, id_hash, confidence) or None
r.watermark.matches("my-asset-id")       # does it point to this library item?
r.c2pa.intact, r.c2pa.ai_generated       # the C2PA manifest's state and claim
r.as_dict()                              # everything, JSON-ready
```

## How to read the result

* **Both marks**: the pixels and the signed metadata agree that Dream Art Fields made the file.
* **Signed C2PA metadata only**: an AI system says it made the file, and the signature and hashes
  check out. "Not on the C2PA trust list" means the signer used its own certificate rather than
  one from a public certificate authority; the manifest is still intact.
* **Pixel signature only**: the pixels say Dream Art Fields made the file; the metadata is
  missing (most social networks and screenshots drop it).
* **C2PA metadata that does not check out**: the file was changed after signing, or the metadata
  is damaged.
* **Nothing found**: that does not mean a person made the file. Marks can be removed or lost, and
  other AI systems use other marks.

The pixel signature's detection key is public on purpose (anyone may check for free), so anyone can
also try to forge or scrub it. Who made a file, and whether it was altered, comes from the C2PA
signature, not from the watermark.

## Privacy

Everything runs on your computer. Files are read locally and nothing is uploaded, stored or
logged. A video is passed to your local `ffmpeg` through a temporary file that is deleted
afterwards. The C2PA library is asked not to fetch remote manifests or certificate status.

## Limits

The pixel signature survives re-encoding (JPEG, WEBP, PNG), resizing and crops. It is not found,
or not reliably, after:

* crops of more than about 30 % per side, or pictures smaller than about 400 px after cropping;
* rotation, mirroring, perspective changes or heavy filtering;
* regeneration through an AI model (img2img, upscalers that redraw detail);
* screenshots of thumbnails or other tiny copies;
* heavy video compression (video uses a simpler, crop-sensitive grid).

The C2PA manifest is lost whenever metadata is stripped: screenshots, most social networks and
messaging apps, many editors and converters.

GIF, animated WEBP and audio carry no pixel signature; audio can carry a C2PA manifest.

## Specification

[SPEC.md](SPEC.md) specifies the payload, key derivation, tile layout, embedding model, detection
steps, thresholds and false-positive bound, the legacy grid, the video variant and the C2PA
manifest fields, so anyone can build an independent detector.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[c2pa,test]"
pytest -q
ruff check .
```

The test fixtures in `tests/fixtures/` were made with Dream Art Fields' own code, and
`tests/expected.json` records what Dream Art Fields' detector reported for each of them;
`tools/make_fixtures.py` shows how. The parity tests check that daf-check gives the same results.

## License

Apache License 2.0. Copyright 2026 YambaCode AB. See [LICENSE](LICENSE).

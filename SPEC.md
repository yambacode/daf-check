# Dream Art Fields AI marks: public specification

Version 1.0 (daf-check 0.1.0), November 2026. Published by YambaCode AB, Sweden.

Dream Art Fields (DAF) is a local AI image, video and audio app. It marks what it generates in two
machine-readable layers, as the EU AI Act (Art. 50(2)) and the EU Code of Practice on marking
AI-generated content ask of providers:

1. a signed **C2PA manifest** in the file's metadata (section 9), and
2. an invisible **pixel signature** in the luminance of pictures and video frames (sections 1-8).

This document specifies both well enough to build an independent detector. `daf-check`
(`daf_check/watermark.py`, `daf_check/manifest.py`) is the reference implementation; where this text
and the reference implementation differ, the reference implementation is normative and this text
will be corrected.

**The detection key is public on purpose.** Anyone may check any file for free, so the keys are
printed below. The consequence is stated plainly: anyone who reads this can also try to forge or
scrub the pixel signature. The signature only says "made with Dream Art Fields, AI-generated,
library item #hash". **Authenticity (who made a file and that it was not altered) comes from the
C2PA manifest's cryptographic signature, not from the watermark.** Neither mark on its own is proof,
and the absence of both does not show that a person made a file.

The key words MUST and SHOULD are used as in RFC 2119.

## 1. Payload

64 bits, transmitted most significant bit first, byte by byte:

| Byte | Content |
|------|---------|
| 0 | magic `0xDA` |
| 1 | `version << 4 \| flags`; version 2 (tile: pictures and video) or 1 (grid: older pictures and video); flag bit 0 = AI-generated |
| 2-5 | asset hash: BLAKE2b with a 4-byte digest of the library asset id (UTF-8), big-endian |
| 6-7 | CRC-16/CCITT-FALSE of bytes 0-5 (poly `0x1021`, init `0xFFFF`, no reflection, no final XOR; check value of `"123456789"` is `0x29B1`), big-endian |

Bit `i` (0..63) is bit `7 - i mod 8` of byte `i div 8`. In the pattern a bit is a sign: 1 is `+1`,
0 is `-1`. The payload holds no personal data: the hash only points back to an item in the user's
own library (Python: `hashlib.blake2b(asset_id.encode(), digest_size=4)`).

DAF always sets the AI flag. A detector MUST accept a payload only if byte 0 is `0xDA`, the high
nibble of byte 1 equals the version being read, and the CRC matches.

## 2. Parameters

The values of the `invisible:` section of DAF's `config/branding.yaml`
(`daf_check/params.py`):

| Name | Value | Used for |
|------|-------|----------|
| `key` | `dream-art-fields/signature/v2` | v2 tables |
| `tile` | 31 | the pattern repeats every 31 x 31 cells |
| `cells` | 256 | cells across the short side (before rounding, section 4) |
| `strength` | 1.8 | embedding: luminance levels (of 255) before texture masking |
| `mask` | (0.7, 1.3) | embedding: texture-masking range |
| `min_side` | 256 | embedding: pictures with a shorter short side are not signed |
| `min_confidence` | 1.5 | detection: mean per-bit \|z\| a read needs (v2) |
| `repair_bits` | 3 | detection: weak bits a v2 read may flip (section 7) |
| `detect_side` | (768, 1024) | detection: short side of the working copy (section 6.3) |
| `legacy.key` | `dream-art-fields/signature/v1` | v1 tables (pictures, and video before v2) |
| `legacy.grid` | 256 | v1 grid (pictures, and video before v2) |
| `legacy.min_confidence` | 1.5 | detection: v1 |
| `video_cells` | 128 | video: cells across the short side (before rounding, section 8) |
| `video_strength` | 2.4 | embedding: video luminance levels |
| `video_mask` | (0.6, 1.0) | embedding: video texture-masking range |
| `video_frames` | 24 | detection: frames a video check averages |

Fixed constants: pictures with a short side under 64 px are not searched; `FLIP_POOL` = 12,
`CANDIDATES` = 3, `SCREEN_BINS` = 2, `SCREEN_MIN_Z` = 4.5, `MAX_BINS_PER_CELL` = 6,
`MAX_ASPECT` = 3, `SEARCH_TRIES` = 2, codec grids (16, 8, 4). Video (section 8):
`NATIVE_UP_TO` = 1.5, `VIDEO_CELLS` = 3, 5, ..., 19, `VIDEO_SHORT_SIDES` = 240, 360, 480, 540,
576, 720, 768, 1080, 1440, 2160.

## 3. Key to tables

```
seed = int.from_bytes(SHA-256(key as UTF-8)[0:8], "big")
rng  = numpy.random.default_rng(seed)        # PCG64 seeded through SeedSequence(seed)
```

**Version 2 tile** (key `dream-art-fields/signature/v2`, T = 31, T*T = 961 cells, numbered row
by row, cell `t = row * 31 + col`):

```
order = rng.permutation(961)
role[order[0:240]]   = PILOT (-1)                 # 961 // 4 = 240 pilot cells
role[order[240:961]] = [0, 1, ..., 720] mod 64    # 721 data cells, each payload bit 11 or 12 times
sign = rng.choice([-1.0, 1.0], 961)               # float32, then reshaped 31 x 31 with role
pilot[t] = sign[t] if role[t] == PILOT else 0
```

**Version 1 grid, also video exported before v2** (key `dream-art-fields/signature/v1`, G = 256):

```
bit_of = rng.permutation(65536) mod 64            # reshaped 256 x 256, row by row
sign   = rng.choice([-1.0, 1.0], 65536)
```

The draws are NumPy's `Generator.permutation` and `Generator.choice` (NumPy >= 1.17). A port to
another language must reproduce them or load the tables, which can be dumped with:

```
python -c "import json; from daf_check.watermark import default_detector as d; x = d(); \
print(json.dumps({'v2_role': x.v2.role.tolist(), 'v2_sign': x.v2.sign.astype(int).tolist(), \
'v1_bit_of': x.v1.bit_of.tolist(), 'v1_sign': x.v1.sign.astype(int).tolist()}))"
```

Check digests: SHA-256 over the row-major tables as signed bytes, role (or bit_of) followed by sign:

| Tables | SHA-256 |
|--------|---------|
| v2 (`role` + `sign`) | `75869c0aa70962b211221a8020315eebafdc0166439bde0c0f670f43ca352a99` |
| v1 (`bit_of` + `sign`) | `c58ed5fafa0dfc0ae85756ec35ebc9baa2e149c0345a1a7bf4e6b36f23c40b9e` |

## 4. Version 2 layout

* **Luminance** `Y` is 8-bit ITU-R 601-2 luma as Pillow computes it
  (`L = R * 299/1000 + G * 587/1000 + B * 114/1000`, of the picture converted to RGB).
* **Cell size** in pixels, from the short side `S`: the smallest odd integer >= `S / 256`, and at
  least 3: `c = max(3, 2 * ceil((S / 256 - 1) / 2) + 1)`. Pictures up to 768 px get 3 px cells,
  up to 1280 px 5 px, and so on.
* **Period** `P = 31 * c` pixels, always odd, so the 4, 8 and 16 px blocks of JPEG and WEBP never
  line up with it.
* **Placement.** Cell `(i, j)` covers rows `i*c .. i*c + c - 1` and columns `j*c .. j*c + c - 1`
  from the picture's top-left corner; its tile position is `t = (i mod 31) * 31 + (j mod 31)`. The
  last row and column of cells may be cut by the picture's edge.
* **Value** of a cell: `v = sign[t] * (+1 if role[t] == PILOT else b[role[t]])`, with `b` the
  payload as signs. A quarter of the cells (the pilot) carry known signs and are used to find the
  grid; the rest carry the payload.

## 5. Embedding model (informative)

Detection does not depend on these details beyond "a texture-masked luminance delta, constant per
cell", but they explain the signal a detector looks for.

1. `y` = `Y` reduced by `c` (the mean of each `c x c` block; partial blocks at the edges average
   what is there), padded by edge replication to `ceil(h/c) x ceil(w/c)`.
2. Texture mask: `m = clip(0.7 + busy / 6, 0.7, 1.3)` with `busy = box2(|y - box1(y)|)`, where
   `boxr` is the mean over a `(2r+1) x (2r+1)` window with edges replicated. Busy areas get more,
   flat areas less.
3. Cell delta: `d = round(v * 1.8 * m)` (round half to even), so 1 or 2 levels of 255.
4. `d` is spread over the cell's pixels and added equally to R, G and B, clipped to 0..255. Alpha
   and the file's other metadata are kept. PSNR stays above 40 dB.

Version 1 (written before v2, read only now): `v1[i, j] = b[bit_of[i, j]] * sign[i, j]` on a
256 x 256 grid, times 1.8 and the mask computed on the picture box-resampled to 256 x 256, resized
to the picture with nearest-neighbour and rounded. The grid is stretched over the whole picture, so
it does not survive crops.

## 6. Detection

A detector SHOULD try, in this order, and stop at the first payload that decodes (section 7):

1. version 2 at the exact grid (6.1), 2. version 1 (6.2), 3. the version 2 search (6.3).

Helpers used below: `boxr` as in section 5 and
`whiten(y) = (y - box1(y)) / (box2(|y - box1(y)|) + 1)`: a high-pass divided by local activity,
so busy picture areas count less.

**Per-bit statistics.** Given readings `x` of data cells with roles `k` and signs `s`:
`corr[k] = sum(x * s)`, `energy[k] = sum(x * x)` over the cells with role `k`, and
`z[k] = corr[k] / sqrt(energy[k] + eps)` (eps = 1e-12 for v2, 1e-9 for v1).

### 6.1 Version 2, exact grid

For a picture that was not cropped or resized since it was signed. If `S < 64`, stop. Compute
`c` from the picture's size (section 4), `x = whiten(Y reduced by c)` (one value per cell, as
float32), take the tile position of every cell from the top-left corner, compute the per-bit
statistics over the data cells and decode as version 2 with up to `repair_bits` = 3 flips.

### 6.2 Version 1

If `S < 64`, stop. `y` = `Y` as float32, box-resampled (Pillow `BOX`) to 256 x 256 regardless of
aspect ratio; `x = whiten(y)`; per-bit statistics with `bit_of` and `sign`; decode as version 1
with no flips.

### 6.3 Version 2, search (crops and resizes)

1. **Working copy.** If the long side exceeds 3 times the short side, cut the centred 3 : 1 part.
   Scale `s = clamp(S, 768, 1024) / S`; resize the float luminance to `(round(w*s), round(h*s))`
   with Pillow `BOX` when shrinking and `BICUBIC` when enlarging. `x = whiten(y)`.
2. **Autocorrelation** `ac = IFFT2(|FFT2(x)|^2)` (circular). The tile puts peaks at every multiple
   of the period.
3. **Period candidates** (working pixels, from `pmin = 2 * 31 = 62`):
   * *Per axis*: on lags `0..n/2` of the first column (rows) and first row (columns) of `ac`,
     normalise by lag 0, subtract the median of lags >= `floor(pmin)/2`, then remove the codec
     grid: for blocks of 16, 8 and 4 file pixels (`g * s` working pixels; skipped if `4s < 3`) take
     the lags at multiples of the block size not already claimed by a larger block, and if at least
     three, subtract their median height (if positive) at those lags with linear weights. Score
     each candidate `p` in `[pmin, lmax / 1.5)` in steps of 0.05 by the matched comb
     `sum_k r(k p) / sqrt(count)` over `k p <= lmax` (linear interpolation).
   * *2-D lattice*: score candidates in steps of 0.1 by the same comb over the half lattice
     `(m p, n p)`, `m >= 0`, `(m, n) != (0, 0)`, within the lag range, with bilinear interpolation
     on the circular `ac` (normalised by `ac[0,0]`, minus its median).
   * From each comb keep the 3 best local maxima that differ by more than 2 %, plus half and a
     third of the best if they are >= `pmin`.
   * Hypotheses: `(p, p)` for every distinct (1 %) value of the 2-D, row and column lists; plus
     `(p_rows, p_cols)` of the two axis bests if they differ by more than 1 % (a stretched picture).
4. **Fold** a hypothesis `(py, px)` with `q` sub-cells per cell (`n = 31 q` bins per tile): working
   pixel row `r` falls in bin `floor((r + 0.5) / py * n)` (columns alike); average the pixels of
   each bin. At every bin, `cell` = mean over the `q x q` bins starting there and `hood` = mean
   over the `3q x 3q` bins starting `q` bins earlier; `h = cell - hood` (0 where no pixels), then
   `h = h / (box_{2q}(|h|) + 1)`. Sum `h` modulo `n x n` in absolute bin phase, correlate the fold
   circularly with the pilot template (`pilot` signs at bins `(a q, b q)`) using FFTs. The peak
   gives where cells start and the offset a crop introduced; `pilot_z = (peak - mean) / std` of the
   correlation. Read every bin where a cell starts, look up its tile position relative to the peak
   and compute the per-bit statistics.
5. **Screen**: fold every hypothesis on a half-size copy (2 x 2 means, periods halved, `q = 2`).
   Take the 2 strongest by `pilot_z`; stop as soon as one has `pilot_z < 4.5`.
6. **Refine** each: `q = clamp(int(min(py, px) / 31 + 0.5), 2, 6)`; fold at full size; then up to
   two rounds: fold the left and right halves (and top and bottom), measure the pilot's drift
   `d` between them in periods (sub-bin peak by a parabola fit, wrapped to `[-0.5, 0.5)`), set
   `p' = 1 / (1/p - d / (half size))` and keep the new fold only if its `pilot_z` is higher.
7. **Decode** the refined fold as version 2 with up to 3 flips; the first success wins.

## 7. Decoding and false positives

1. `confidence = mean(|z|)` over the 64 bits. If below `min_confidence` (1.5), there is no payload.
2. `bits = corr > 0`.
3. With `F` allowed flips (3 for v2, 0 for v1), take the 12 bits with the smallest `|z|`
   and try flipping 0, then 1, ... up to `F` of them (combinations in index order of that list).
   Accept the first candidate whose magic, version and CRC all hold.
4. Report version, AI flag (byte 1 bit 0), asset hash and confidence (rounded to 2 decimals).

**False-positive bound.** In an unmarked picture the 64 bits behave like coin flips, and a
candidate must match 28 check bits (8 magic, 4 version, 16 CRC): `2^-28` per try. A v2 read makes
at most `1 + 12 + 66 + 220 = 299` tries; a picture gets at most 3 v2 reads (exact grid and 2 search
hypotheses) and one v1 read, so the check bits alone bound the false-positive rate at about
`(3 * 299 + 1) * 2^-28 ≈ 3.3e-6` per picture. The confidence gate comes first and cuts this much
further: for unmarked input `z` is roughly standard normal, so `mean |z|` is about 0.80 with a
spread of about 0.075 over 64 bits, and 1.5 sits about 9 standard deviations above it. Real
pictures are not ideal noise, so treat these as estimates, not guarantees.

## 8. Video

**Layout.** Video exports (MP4, MOV, WEBM) carry the **version 2 tile** (section 4: same key,
tables, payload and tile of 31 x 31 cells) in every frame, the same in every frame, anchored at the
frame's top-left corner. Only the cell size differs: for a frame of `W x H` with short side `S`,
`c_v = max(3, 2 * ceil((S / video_cells - 1) / 2) + 1)` with `video_cells` = 128 (5 px at 480p,
7 px at 720p, 9 px at 1080p), so the tile period is `P = 31 * c_v` pixels. Cells are bigger than in
pictures because video codecs and downscales wipe out 3 px cells.

**Embedding (informative).** Each cell's value `v` = +-1 (section 5) times `video_strength` (2.4)
times a mask in `video_mask` (0.6, 1.0), `mask = clamp(0.6 + busy / 6, 0.6, 1.0)`, where `busy` is
the calmest moment of the cell's texture over frames from across the clip; rounded to whole
luminance levels (so 1 or 2), added to luminance (an ffmpeg `grainmerge` blend); chroma is
unchanged. GIF and animated WEBP do not carry it. Clips exported before this change carry the
version 1 field (section 3, legacy key) in every frame instead, and are still read (step 3 below).

**Detection.**

1. **Frames.** Probe the first video stream's `W x H` and duration `D` (ffprobe; stream duration,
   else the container's, else 0). Working size: `W x H` itself if `S <= NATIVE_UP_TO * 1024`
   (1536), else scaled to a short side of 1024 (`(round(W k), round(H k))`, `k = 1024 / S`). Read
   up to `video_frames` = 24 frames spread over the clip as 8-bit gray at the working size
   (reference: `ffmpeg -i FILE -vf fps=24/D,scale=WORK_W:WORK_H:flags=area,format=gray
   -frames:v 24 -f rawvideo OUT`, `fps=1` when `D` is unknown). No frames: no payload.
2. **Mean.** Average the frames per pixel in float64 and keep the mean as float32 (`M`, width
   `w'`). The still tile adds up while the moving picture averages out. `M` is luminance already:
   it is never rounded back to 8 bits in the steps below.
3. **Version 2 at the export's grid.** If the working size is `W x H`, read the exact grid (6.1)
   on `M` with `c_v` in place of `c`. Otherwise read at the known period, any offset: on the
   working copy of `M` (6.3 step 1, scale `s`), fold (6.3 step 4) at
   `(P * h' / H * s, P * w' / W * s)` with `q = clamp(int(min of both / 31 + 0.5), 2, 6)` and decode
   that fold (no screen, no refine).
4. **Version 1.** If that found nothing and `M`'s short side is at least 64, the version 1 read
   (6.2) on `M`.
5. **Version 2 search, with period hints.** The exact grid with the picture cell `c` (6.1), then
   the search (6.3) on `M`, where step 3 also gets, as square hypotheses `(p s, p s)` for every hint
   `p` with `62 <= p s <= max(working h, w) / 1.5`, these **period hints** (pixels of `M`, `k = w' / W`;
   distinct within 1 %, in this order): `31 * c * k` for every odd `c` in `VIDEO_CELLS` (3 to 19:
   a crop, which keeps the export's own cell), then `31 * c_v(S0) * S / S0 * k` for every `S0` in
   `VIDEO_SHORT_SIDES` (a rescale from a common short side, `c_v(S0)` the cell of an `S0 x S0`
   frame). In a mean frame full of still picture the autocorrelation can miss the period; the
   pilot does not. If no fold decodes, read every hint at full size as in step 3 (period
   `(p, p)`), in order; the first success wins.

A frame grab (one picture of a v2 video) can be checked as a picture: DAF's own picture check also
tries 6.1 with `c_v` right after 6.1; this package leaves picture detection as in section 6, so a
frame grab is found by the search (6.3) instead.

**False positives.** A video check makes at most one v2 read in step 3, one v1 read in step 4, and
in step 5 the exact grid, 2 search folds and one read per hint (at most 9 + 10), each v2 read at
most 299 tries:
about `(23 * 299 + 1) * 2^-28 ≈ 2.6e-5` per clip from the check bits alone, before the
confidence gate.

## 9. C2PA manifest

Every generated image, video and audio file DAF stores or exports gets a C2PA manifest in the
formats C2PA can carry (PNG, JPEG, WEBP, MP4, MOV, WAV, ...). Its active manifest holds:

* `claim_generator_info`: `[{"name": "Dream Art Fields", "version": "<app version>"}]`.
* `c2pa.actions.v2` with a `c2pa.created` action: `digitalSourceType`
  `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`, `softwareAgent`
  (name, version), `when` (ISO 8601) and a description. A `c2pa.watermarked` action is added when
  the pixels carry the signature.
* `com.dreamartfields.generator`: `system`, `version`, `provider` (YambaCode AB), `provider_url`,
  `ai_generated: true`, `generated_at` and `models` (up to 16 model ids). Never the prompt, the
  user's name or anything else personal.
* `c2pa.soft-binding` (when the pixels are signed): `{"alg": "com.dreamartfields.signature",
  "blocks": [{"scope": {}, "value": base64(4-byte big-endian asset hash)}]}`. The value equals
  the asset hash in the pixel payload, which ties the two marks together.

**Signing.** ES256 by default, with the install's own certificate (a local CA and a
document-signing certificate made on first use) unless the owner configured one from a public CA.
C2PA readers therefore usually report such manifests as *Valid* (signature and hashes check out)
but not *Trusted* (the signer is not on the C2PA trust list). Exports ask an RFC 3161 time-stamp
authority (only a hash leaves the computer) and fall back to the computer's clock; library saves
use the clock.

**Reading.** A checker SHOULD report: present; intact (validation state *Valid* or *Trusted*);
trusted (*Trusted*); AI claim (a `digitalSourceType` of `trainedAlgorithmicMedia`,
`compositeWithTrainedAlgorithmicMedia`, `algorithmicMedia` or `compositeSynthetic`); generator;
signer; signing time and whether a time-stamp authority vouches for it; and whether the
soft-binding hash equals the pixel signature's. A checker that cannot verify signatures MAY report
the presence of a manifest store (a JUMBF box labelled `c2pa`) and an AI claim inside it, and MUST
say that the signature was not checked.

Metadata is easy to strip (screenshots, most social networks, re-encoding), which is why it is
never the only layer.

## 10. What the marks can and cannot show

* Either mark found: the file says it came from an AI system (DAF for the pixel signature).
* An intact C2PA manifest additionally shows that its signer made the claim and that the file has
  not changed since; a trusted one, that a public CA vouches for the signer.
* Nothing found: no conclusion. Marks can be removed or lost, and other AI systems use other marks.
* The pixel signature survives re-encoding, resizing and crops up to about 30 % per side. It does
  not survive rotation, heavy filtering, tiny thumbnails or screenshots of them, or regeneration
  through a diffusion model.

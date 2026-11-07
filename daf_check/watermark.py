"""Detection of the Dream Art Fields pixel signature (SPEC.md, sections 3 to 7).

Detection only: this package has no code that writes the signature. The algorithm and its
numbers follow Dream Art Fields' own detector exactly (same keys, same random tables, same float
arithmetic), so a file gives the same result here as in the app.

The signature says "made with Dream Art Fields, AI-generated, library item #hash". The detection
key is public on purpose, so anyone can also try to forge or scrub it: authenticity comes from the
signed C2PA manifest, never from this pattern alone.
"""

from __future__ import annotations

import binascii
import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np
from PIL import Image

from .params import (
    BITS,
    CANDIDATES,
    CODEC_GRIDS,
    FLAG_AI,
    FLIP_POOL,
    LEGACY_VERSION,
    MAGIC,
    MAX_ASPECT,
    MAX_BINS_PER_CELL,
    MIN_DETECT_SIDE,
    PARAMS,
    PILOT,
    SCREEN_BINS,
    SCREEN_MIN_Z,
    SEARCH_TRIES,
    VERSION,
    LegacyParams,
    Params,
)


def id_hash(asset_id: str) -> int:
    """The 32-bit asset hash a signature carries: BLAKE2b (4-byte digest) of the UTF-8 id."""
    return int.from_bytes(hashlib.blake2b(asset_id.encode(), digest_size=4).digest(), "big")


def keyed_rng(key: str) -> np.random.Generator:
    """NumPy's default generator (PCG64) seeded with the first 8 bytes of SHA-256(key)."""
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


@dataclass(frozen=True)
class Detection:
    """A signature that was found: its version, AI flag, asset hash and confidence (mean |z|)."""

    version: int
    ai: bool
    id_hash: int
    confidence: float

    @property
    def asset_hash(self) -> str:
        return f"{self.id_hash:08x}"

    def matches(self, asset_id: str) -> bool:
        """True when the signature points to the library item `asset_id`."""
        return self.id_hash == id_hash(asset_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "ai": self.ai,
            "id_hash": self.id_hash,
            "asset_hash": self.asset_hash,
            "confidence": self.confidence,
        }


def decode(
    corr: np.ndarray, z: np.ndarray, version: int, min_conf: float, flips: int = 0
) -> Detection | None:
    """The payload from per-bit correlations; None if confidence, magic, version or CRC fail.

    The confidence gate comes first (unmarked pictures give a mean |z| near 0.8). With `flips`,
    a failed check is repaired by flipping up to that many of the `FLIP_POOL` weakest bits; magic,
    version and CRC (28 check bits) must all hold afterwards."""
    confidence = float(np.mean(np.abs(z)))
    if confidence < min_conf:
        return None
    bits = (corr > 0).astype(np.uint8)
    weak = np.argsort(np.abs(z))[:FLIP_POOL].tolist()
    for n in range(flips + 1):
        for combo in combinations(weak, n):
            b = bits.copy()
            b[list(combo)] ^= 1
            data = np.packbits(b).tobytes()
            if data[0] != MAGIC or data[1] >> 4 != version:
                continue
            if binascii.crc_hqx(data[:6], 0xFFFF) == int.from_bytes(data[6:], "big"):
                ai, h = bool(data[1] & FLAG_AI), int.from_bytes(data[2:6], "big")
                return Detection(version, ai, h, round(confidence, 2))
    return None


# ------------------------------------------------------------------ shared signal helpers
def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, edges repeated."""
    k = 2 * r + 1
    c = np.pad(np.pad(a, r, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _whiten(y: np.ndarray, r: int) -> np.ndarray:
    """High-pass (minus the local mean), divided by local activity."""
    hp = y - _box(y, r)
    return hp / (_box(np.abs(hp), 2) + 1.0)


def _luma(im: Image.Image) -> Image.Image:
    """8-bit luminance (ITU-R 601-2, as Pillow's RGB -> L conversion)."""
    return im.convert("RGB").convert("L")


def grid_luma(im: Image.Image, grid: int) -> np.ndarray:
    """Luminance box-resampled onto a `grid` x `grid` field, as float32."""
    y = _luma(im).convert("F")
    return np.asarray(y.resize((grid, grid), Image.Resampling.BOX), np.float32)


# ------------------------------------------------------------------ version 1 and video
class LegacyDetector:
    """Version 1: a `grid` x `grid` field stretched over the whole picture. Video uses the same
    grid and key, averaged over frames."""

    def __init__(self, p: LegacyParams = PARAMS.legacy):
        self.p = p
        g = p.grid
        rng = keyed_rng(p.key)
        self.bit_of = (rng.permutation(g * g) % BITS).reshape(g, g)
        self.sign = rng.choice(np.array([-1.0, 1.0], np.float32), g * g).reshape(g, g)

    def detect(self, im: Image.Image) -> Detection | None:
        if min(im.size) < MIN_DETECT_SIDE:
            return None
        return self.detect_grid(grid_luma(im, self.p.grid))

    def detect_grid(self, yc: np.ndarray) -> Detection | None:
        """Looks for the signature in luminance already on the grid."""
        hp = _whiten(yc, 1)
        idx = self.bit_of.ravel()
        corr = np.bincount(idx, (hp * self.sign).ravel(), BITS)
        energy = np.bincount(idx, (hp * hp).ravel(), BITS)
        z = corr / np.sqrt(energy + 1e-9)
        return decode(corr, z, LEGACY_VERSION, self.p.min_confidence)

    def detect_frames(self, frames: np.ndarray) -> Detection | None:
        """`frames`: (n, grid, grid) luminance of n video frames. The signature stands still while
        the picture moves, so their mean holds more signature and less picture than one frame."""
        if frames.ndim != 3 or frames.shape[0] == 0:
            return None
        return self.detect_grid(frames.astype(np.float32).mean(axis=0))


# ------------------------------------------------------------------ version 2 search helpers
@dataclass(frozen=True)
class _Fold:
    """One period hypothesis, folded onto a tile of `ny` x `nx` bins and aligned on the pilot."""

    py: float
    px: float
    ny: int
    nx: int
    oy: float  # pilot offset in bins, with sub-bin precision
    ox: float
    pilot_z: float
    corr: np.ndarray
    z: np.ndarray


def _without_codec_grid(r: np.ndarray, s: float, start: float) -> np.ndarray:
    """Subtracts the autocorrelation peaks JPEG/WEBP blocks put at multiples of 4, 8 and 16 file
    pixels (`s` is the working scale), so the period search is not pulled onto them."""
    if 4 * s < 3:
        return r
    r = r.copy()
    n = len(r)
    done = np.zeros(n, bool)
    for g in CODEC_GRIDS:  # largest first; a lag belongs to the largest grid it is on
        step = g * s
        pos = np.arange(np.ceil(start / step), (n - 1) // step + 1) * step
        near = np.minimum(np.rint(pos).astype(np.int64), n - 1)
        pos, near = pos[~done[near]], near[~done[near]]
        done[near] = True
        if len(pos) < 3:
            continue
        level = float(np.median(np.interp(pos, np.arange(n), r)))
        if level > 0:
            i0 = np.floor(pos).astype(np.int64)
            frac = pos - i0
            np.subtract.at(r, i0, level * (1 - frac))
            np.subtract.at(r, np.minimum(i0 + 1, n - 1), level * frac)
    return r


def _distinct(values: list[float], tol: float = 0.01) -> list[float]:
    out: list[float] = []
    for p in values:
        if all(abs(p - q) > tol * q for q in out):
            out.append(p)
    return out


def _peaks(cand: np.ndarray, score: np.ndarray, pmin: float) -> list[float]:
    """The `CANDIDATES` best local maxima of a comb score, plus the half and third of the best."""
    peak = (score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:])
    idx = np.flatnonzero(peak) + 1
    idx = idx[np.argsort(-score[idx])]
    out = _distinct([float(cand[i]) for i in idx[: 4 * CANDIDATES]], 0.02)[:CANDIDATES]
    return _distinct(out + [p / d for p in out[:1] for d in (2, 3) if p / d >= pmin], 0.02)


def _wsums(a: np.ndarray, windows: list[tuple[int, int]]) -> list[np.ndarray]:
    """For each (k, off): at every (i, j), the sum of `a` over rows i+off .. i+off+k-1 and the
    same columns (outside `a` counts as zero)."""
    h, w = a.shape
    p = max(k + abs(off) for k, off in windows)
    c = np.zeros((h + 2 * p + 1, w + 2 * p + 1))
    c[1:, 1:] = np.pad(a, p).cumsum(0).cumsum(1)
    out = []
    for k, off in windows:
        lo, hi = off + p, off + p + k
        out.append(c[hi : hi + h, hi : hi + w] - c[lo : lo + h, hi : hi + w]
                   - c[hi : hi + h, lo : lo + w] + c[lo : lo + h, lo : lo + w])  # fmt: skip
    return out


def _half(y: np.ndarray) -> np.ndarray:
    """2 x 2 block means (an odd last row or column is dropped)."""
    h, w = y.shape[0] // 2 * 2, y.shape[1] // 2 * 2
    return y[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))


def _subpeak(a: np.ndarray, i: int) -> float:
    """Sub-sample position of the peak at `i` of the circular 1-D array `a` (parabola fit)."""
    n = len(a)
    left, mid, right = a[(i - 1) % n], a[i], a[(i + 1) % n]
    den = left - 2 * mid + right
    return i + (0.5 * (left - right) / den if den < 0 else 0.0)


# ------------------------------------------------------------------ version 2
class TiledDetector:
    """Version 2: a key-seeded tile of cells repeated over the picture from its top-left corner."""

    def __init__(self, p: Params = PARAMS):
        self.p = p
        t = p.tile
        rng = keyed_rng(p.key)
        order = rng.permutation(t * t)
        n_pilot = t * t // 4
        role = np.empty(t * t, np.int64)
        role[order[:n_pilot]] = PILOT
        role[order[n_pilot:]] = np.arange(t * t - n_pilot) % BITS
        self.role = role.reshape(t, t)
        self.sign = rng.choice(np.array([-1.0, 1.0], np.float32), t * t).reshape(t, t)
        self.pilot = np.where(self.role == PILOT, self.sign, 0.0).astype(np.float32)
        self._pilot_fft: dict[int, np.ndarray] = {}

    def cell_px(self, size: tuple[int, int]) -> int:
        """Cell size in pixels: the smallest odd number >= short side / cells, at least 3."""
        return max(3, 2 * math.ceil((min(size) / self.p.cells - 1) / 2) + 1)

    def detect_exact(self, im: Image.Image) -> Detection | None:
        """Reads the grid the signer would have used for this exact size, from the top-left
        corner (no search): finds pictures that were not cropped or resized since signing."""
        if min(im.size) < MIN_DETECT_SIDE:
            return None
        c, t = self.cell_px(im.size), self.p.tile
        hp = _whiten(np.asarray(_luma(im).reduce(c), np.float32), 1)
        gh, gw = hp.shape
        ty, tx = (np.arange(gh) % t)[:, None], (np.arange(gw) % t)[None, :]
        role, sign = self.role[ty, tx].ravel(), self.sign[ty, tx].ravel()
        data = role >= 0
        v = hp.ravel()[data]
        corr = np.bincount(role[data], v * sign[data], BITS)
        z = corr / np.sqrt(np.bincount(role[data], v * v, BITS) + 1e-12)
        return decode(corr, z, VERSION, self.p.min_confidence, self.p.repair_bits)

    def search(self, im: Image.Image, tries: int = SEARCH_TRIES) -> Detection | None:
        """The full search for crops and resizes, over the `tries` strongest period hypotheses."""
        for f in self._folds(im, tries):
            found = decode(f.corr, f.z, VERSION, self.p.min_confidence, self.p.repair_bits)
            if found is not None:
                return found
        return None

    def detect(self, im: Image.Image) -> Detection | None:
        """The exact grid first, then the full search."""
        return self.detect_exact(im) or self.search(im)

    # -------------------------------------------------------------- search internals
    def _working_luma(self, im: Image.Image) -> tuple[np.ndarray, float]:
        """Luminance rescaled so the short side lands in `detect_side`, and the scale used. A very
        wide or tall picture is first cut to its centre, at most `MAX_ASPECT` : 1."""
        w, h = im.size
        if max(w, h) > MAX_ASPECT * min(w, h):
            cw, ch = min(w, MAX_ASPECT * h), min(h, MAX_ASPECT * w)
            left, top = (w - cw) // 2, (h - ch) // 2
            im = im.crop((left, top, left + cw, top + ch))
            w, h = im.size
        lo, hi = self.p.detect_side
        s = min(max(min(w, h), lo), hi) / min(w, h)
        y = _luma(im).convert("F")
        if s != 1:
            size = (max(1, round(w * s)), max(1, round(h * s)))
            y = y.resize(size, Image.Resampling.BOX if s < 1 else Image.Resampling.BICUBIC)
        return np.asarray(y, np.float32), s

    def _periods(self, ac: np.ndarray, s: float) -> list[float]:
        """Tile period candidates (working pixels) on one axis from the autocorrelation `ac`
        (lags 0..n/2), scored by a matched comb over the multiples of each candidate."""
        t = self.p.tile
        lmax = len(ac) - 1
        pmin, pmax = 2.0 * t, lmax / 1.5
        if pmax <= pmin:
            return []
        r = ac / (ac[0] + 1e-9)
        r = _without_codec_grid(r - np.median(r[int(pmin) // 2 :]), s, pmin / 2)
        cand = np.arange(pmin, pmax, 0.05)
        k = np.arange(1, int(lmax // pmin) + 1)[:, None]
        lags = k * cand[None, :]
        vals = np.interp(lags, np.arange(lmax + 1), r) * (lags <= lmax)
        score = vals.sum(0) / np.sqrt(np.maximum((lags <= lmax).sum(0), 1))
        return _peaks(cand, score, pmin)

    def _periods_2d(self, ac: np.ndarray) -> list[float]:
        """Square-cell period candidates from the 2-D autocorrelation (a comb over the lattice
        of peaks at (m P, n P))."""
        h, w = ac.shape
        ly, lx = h // 2, w // 2
        pmin, pmax = 2.0 * self.p.tile, min(ly, lx) / 1.5
        if pmax <= pmin:
            return []
        r = ac / (ac[0, 0] + 1e-9)
        r = r - np.median(r)
        km, kn = int(ly // pmin), int(lx // pmin)
        m, n = np.meshgrid(np.arange(km + 1), np.arange(-kn, kn + 1), indexing="ij")
        keep = (m > 0) | (n > 0)  # half the lattice: the autocorrelation is symmetric
        m, n = m[keep][:, None], n[keep][:, None]
        cand = np.arange(pmin, pmax, 0.1)[None, :]
        dy, dx = m * cand, n * cand
        ok = (dy <= ly) & (np.abs(dx) <= lx)
        dy, dx = np.where(ok, dy, 0.0), np.where(ok, dx, 0.0) % w
        y0, x0 = np.floor(dy).astype(np.int64), np.floor(dx).astype(np.int64)
        fy, fx = dy - y0, dx - x0
        y1, x1 = np.minimum(y0 + 1, h - 1), (x0 + 1) % w
        v = (r[y0, x0] * (1 - fy) * (1 - fx) + r[y1, x0] * fy * (1 - fx)
             + r[y0, x1] * (1 - fy) * fx + r[y1, x1] * fy * fx)  # fmt: skip
        score = (v * ok).sum(0) / np.sqrt(np.maximum(ok.sum(0), 1))
        return _peaks(cand[0], score, pmin)

    def _fold(self, y: np.ndarray, py: float, px: float, q: int, y0: int = 0, x0: int = 0) -> _Fold:
        """Tests one period hypothesis on the luminance `y` (top-left at `y0, x0` of the working
        image): sub-cell averages, "cell minus its 3 x 3-cell neighbourhood" at every phase,
        folded modulo the tile, pilot correlation (FFT) for the cell phase and crop offset, then
        the bits read off the unfolded map."""
        t = self.p.tile
        n = t * q
        h, w = y.shape
        jy = np.floor((np.arange(y0, y0 + h) + 0.5) / py * n).astype(np.int64)
        jx = np.floor((np.arange(x0, x0 + w) + 0.5) / px * n).astype(np.int64)
        ay, ax = int(jy[0]), int(jx[0])  # absolute sub-cell of the first row and column
        jy, jx = jy - ay, jx - ax
        gh, gw = int(jy[-1]) + 1, int(jx[-1]) + 1
        idx = (jy[:, None] * gw + jx[None, :]).ravel()
        total = np.bincount(idx, y.ravel(), gh * gw).reshape(gh, gw)
        count = np.bincount(idx, minlength=gh * gw).reshape(gh, gw).astype(np.float64)
        windows = [(q, 0), (3 * q, -q)]  # a cell starting here; it and its 8 neighbours
        s_cell, s_hood = _wsums(total, windows)
        n_cell, n_hood = _wsums(count, windows)
        cell = s_cell / np.maximum(n_cell, 1)
        hp = np.where(n_cell > 0, cell - s_hood / np.maximum(n_hood, 1), 0.0)
        hp = hp / (_box(np.abs(hp), 2 * q) + 1.0)
        # fold modulo the tile, in absolute phase
        top, left = ay % n, ax % n
        fh, fw = -(-(gh + top) // n) * n, -(-(gw + left) // n) * n
        canvas = np.zeros((fh, fw))
        canvas[top : top + gh, left : left + gw] = hp
        fold = canvas.reshape(fh // n, n, fw // n, n).sum((0, 2))
        if q not in self._pilot_fft:
            tpl = np.zeros((n, n))
            tpl[::q, ::q] = self.pilot
            self._pilot_fft[q] = np.conj(np.fft.rfft2(tpl))
        xc = np.fft.irfft2(np.fft.rfft2(fold) * self._pilot_fft[q], s=(n, n))
        oy, ox = np.unravel_index(int(np.argmax(xc)), xc.shape)
        pilot_z = float((xc[oy, ox] - xc.mean()) / (xc.std() + 1e-12))
        # read the cells: rows and columns where a cell starts, and their place in the tile
        r0, c0 = (int(oy) - ay) % q, (int(ox) - ax) % q
        ty = ((np.arange(r0, gh, q) + ay - int(oy)) // q % t)[:, None]
        tx = ((np.arange(c0, gw, q) + ax - int(ox)) // q % t)[None, :]
        v = hp[r0::q, c0::q]
        role, sign = self.role[ty, tx].ravel(), self.sign[ty, tx].ravel()
        data = role >= 0
        v = v.ravel()[data]
        corr = np.bincount(role[data], v * sign[data], BITS)
        energy = np.bincount(role[data], v * v, BITS)
        sy, sx = _subpeak(xc[:, ox], int(oy)), _subpeak(xc[oy, :], int(ox))
        z = corr / np.sqrt(energy + 1e-12)
        return _Fold(py, px, n, n, sy, sx, pilot_z, corr, z)

    def _bins(self, f: _Fold) -> int:
        """Sub-cells per cell for a final read: about one working pixel each."""
        return int(min(max(min(f.py, f.px) / self.p.tile + 0.5, 2), MAX_BINS_PER_CELL))

    def _refine(self, y: np.ndarray, f: _Fold, rounds: int = 2) -> _Fold:
        """Sharpens the period from the pilot's drift between the two halves of the picture;
        kept only if the pilot gets stronger."""
        h, w = y.shape
        q = self._bins(f)
        best = self._fold(y, f.py, f.px, q)
        for _ in range(rounds):
            py, px = best.py, best.px
            left = self._fold(y[:, : w // 2], py, px, q)
            right = self._fold(y[:, w // 2 :], py, px, q, 0, w // 2)
            top = self._fold(y[: h // 2], py, px, q)
            bottom = self._fold(y[h // 2 :], py, px, q, h // 2)
            dx = ((right.ox - left.ox) / left.nx + 0.5) % 1.0 - 0.5  # in periods
            dy = ((bottom.oy - top.oy) / top.ny + 0.5) % 1.0 - 0.5
            g = self._fold(y, 1 / (1 / py - dy / (h / 2)), 1 / (1 / px - dx / (w / 2)), q)
            if g.pilot_z <= best.pilot_z:
                break
            best = g
        return best

    def _folds(self, im: Image.Image, tries: int) -> Iterator[_Fold]:
        """Refined hypotheses, strongest pilot first; only those whose screened pilot stands out
        of the noise (`SCREEN_MIN_Z`) are refined."""
        if min(im.size) < MIN_DETECT_SIDE:
            return
        y, s = self._working_luma(im)
        hp = _whiten(y, 1)
        spec = np.fft.rfft2(hp)
        ac = np.fft.irfft2(spec.real**2 + spec.imag**2, s=hp.shape)
        h, w = hp.shape
        cy, cx = self._periods(ac[: h // 2 + 1, 0], s), self._periods(ac[0, : w // 2 + 1], s)
        # Square cells (any uniform resize): one period for both axes, from every list. A
        # stretched picture: the best of each axis.
        pairs = [(p, p) for p in _distinct(self._periods_2d(ac) + cy + cx)]
        if cy and cx and abs(cy[0] / cx[0] - 1) > 0.01:
            pairs.append((cy[0], cx[0]))
        half = _half(y)  # screen on a half-size copy (cells are still >= 1.5 px there)
        screen = [self._fold(half, py / 2, px / 2, SCREEN_BINS) for py, px in pairs]
        for f in sorted(screen, key=lambda f: -f.pilot_z)[:tries]:
            if f.pilot_z < SCREEN_MIN_Z:
                return
            yield self._refine(y, replace(f, py=f.py * 2, px=f.px * 2))


# ------------------------------------------------------------------ both versions
class Detector:
    """Looks for the signature in a picture: version 2 at its exact grid, then version 1, then
    the version 2 search for crops and resizes (the order Dream Art Fields uses)."""

    def __init__(self, p: Params = PARAMS):
        self.p = p
        self.v2 = TiledDetector(p)
        self.v1 = LegacyDetector(p.legacy)

    def detect(self, im: Image.Image) -> Detection | None:
        return self.v2.detect_exact(im) or self.v1.detect(im) or self.v2.search(im)

    def detect_frames(self, frames: np.ndarray) -> Detection | None:
        """Video: (n, grid, grid) luminance of frames from across the clip, averaged."""
        return self.v1.detect_frames(frames)


@lru_cache
def default_detector() -> Detector:
    return Detector(PARAMS)


def table_digests(det: Detector | None = None) -> dict[str, str]:
    """SHA-256 of the key-derived tables (SPEC.md, section 3), to check a port against."""
    det = det or default_detector()
    v2 = det.v2.role.astype("<i1").tobytes() + det.v2.sign.astype("<i1").tobytes()
    v1 = det.v1.bit_of.astype("<i1").tobytes() + det.v1.sign.astype("<i1").tobytes()
    return {"v2": hashlib.sha256(v2).hexdigest(), "v1": hashlib.sha256(v1).hexdigest()}

"""The published detection parameters of the Dream Art Fields pixel signature.

These are the values of the `invisible:` section of Dream Art Fields' `config/branding.yaml`,
copied verbatim (SPEC.md, section 2 "Parameters"). The keys are public on purpose: anyone may
check a file for free. Change nothing here unless SPEC.md changes with it.

The embedding-only values (`strength`, `mask`, `min_side`, `video_strength`, `video_mask`) are
kept so the whole parameter set is in one place; detection does not use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LegacyParams:
    """Version 1 (one grid stretched over the whole picture); also the video signature."""

    key: str = "dream-art-fields/signature/v1"
    grid: int = 256
    min_confidence: float = 1.5


@dataclass(frozen=True)
class Params:
    """Version 2 (the crop-robust tile) plus the shared values."""

    key: str = "dream-art-fields/signature/v2"
    tile: int = 31  # the pattern repeats every tile x tile cells
    cells: int = 256  # cell = short side / cells, rounded up to an odd pixel count, >= 3
    strength: float = 1.8  # embedding: luminance change in 0-255 levels before texture masking
    mask: tuple[float, float] = (0.7, 1.3)  # embedding: texture-masking range
    min_side: int = 256  # embedding: smaller pictures are not signed
    min_confidence: float = 1.5  # mean per-bit |z| a read needs
    repair_bits: int = 3  # a v2 read may flip up to this many of its weakest bits
    detect_side: tuple[int, int] = (768, 1024)  # short side of the working copy
    legacy: LegacyParams = field(default_factory=LegacyParams)
    video_strength: float = 2.5  # embedding (video): luminance levels
    video_mask: tuple[float, float] = (0.3, 1.5)  # embedding (video): texture-masking range
    video_frames: int = 24  # a video check averages this many frames from across the clip


PARAMS = Params()

# Fixed parts of the algorithm (SPEC.md, sections 1, 5 and 6).
MAGIC = 0xDA
VERSION = 2
LEGACY_VERSION = 1
FLAG_AI = 1
BITS = 64
MIN_DETECT_SIDE = 64  # pictures with a shorter side are not searched
PILOT = -1  # role of a pilot cell in the tile
MAX_BINS_PER_CELL = 6
CANDIDATES = 3  # period candidates kept per comb
SCREEN_BINS = 2  # sub-cells per cell when screening period candidates
SCREEN_MIN_Z = 4.5  # pilot peak (z over all offsets) a candidate needs before it is refined
FLIP_POOL = 12  # the repair only flips among this many weakest bits
CODEC_GRIDS = (16, 8, 4)  # WEBP macroblocks, JPEG blocks, WEBP transforms (file pixels)
MAX_ASPECT = 3  # the search looks at most at a long : short = 3 : 1 centre of the picture
SEARCH_TRIES = 2  # refined period hypotheses read per picture

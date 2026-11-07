"""The pixel signature in video (SPEC.md, section 8): frames from across the clip, read by the
`ffmpeg` command-line tool at a working size (the clip's own size, or a short side of
`detect_side`'s upper end for big clips) and averaged, then read like a picture.

Needs `ffmpeg` and `ffprobe` on PATH (or in the DAF_CHECK_FFMPEG / DAF_CHECK_FFPROBE environment
variables). Without them a video's pixels are not checked, and the result says "ffmpeg needed".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .params import NATIVE_UP_TO
from .watermark import Detection, Detector, default_detector

TIMEOUT = 120.0


class FFmpegMissing(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    """Size (pixels) and duration (seconds, 0 when unknown) of a clip's first video stream."""

    width: int
    height: int
    duration: float


def _tool(name: str) -> str | None:
    env = os.environ.get(f"DAF_CHECK_{name.upper()}")
    if env and Path(env).is_file():
        return env
    return shutil.which(name)


def ffmpeg_available() -> bool:
    return _tool("ffmpeg") is not None and _tool("ffprobe") is not None


def _run(args: list[str], timeout: float = TIMEOUT) -> subprocess.CompletedProcess[bytes]:
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as e:
        raise FFmpegMissing(f"{args[0]} could not be started") from e
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"{Path(args[0]).name} timed out after {timeout:.0f} s") from e
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-3:]
        raise FFmpegError(f"{Path(args[0]).name} failed ({proc.returncode}): {' '.join(tail)}")
    return proc


def probe(path: Path) -> VideoInfo:
    """Size and duration of the first video stream (as Dream Art Fields' `ffmpeg.probe`)."""
    ffprobe = _tool("ffprobe")
    if ffprobe is None:
        raise FFmpegMissing("ffprobe is not installed")
    out = _run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                "-select_streams", "v:0", str(path)]).stdout  # fmt: skip
    data = json.loads(out or b"{}")
    streams = data.get("streams") or []
    if not streams:
        raise FFmpegError("no video stream")
    st = streams[0]
    dur = float(st.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return VideoInfo(int(st.get("width", 0)), int(st.get("height", 0)), dur)


def duration(path: Path) -> float:
    """Duration in seconds of the first video stream (0 when unknown)."""
    return probe(path).duration


def clip_frames(
    path: Path, size: tuple[int, int], n: int, dur: float, timeout: float = TIMEOUT
) -> np.ndarray:
    """(count, h, w) uint8 luminance of up to `n` frames spread evenly over the clip, scaled to
    `size` (w, h) with ffmpeg's area filter; `dur` is the clip's duration in seconds."""
    ff = _tool("ffmpeg")
    if ff is None:
        raise FFmpegMissing("ffmpeg is not installed")
    w, h = size
    fps = n / dur if dur > 0 else 1.0
    vf = f"fps={fps:.6f},scale={w}:{h}:flags=area,format=gray"
    with tempfile.TemporaryDirectory(prefix="daf-check-") as tmp:
        out = Path(tmp) / "frames.gray"
        _run([ff, "-nostdin", "-y", "-loglevel", "error", "-i", str(path), "-vf", vf,
              "-frames:v", str(n), "-f", "rawvideo", str(out)], timeout)  # fmt: skip
        data = np.frombuffer(out.read_bytes(), np.uint8)
    count = data.size // (w * h)
    return data[: count * w * h].reshape(count, h, w)


def grid_frames(path: Path, grid: int, n: int, timeout: float = TIMEOUT) -> np.ndarray:
    """(count, grid, grid) float luminance of up to `n` frames, for `Detector.detect_frames`
    (version 1 video only)."""
    return clip_frames(path, (grid, grid), n, duration(path), timeout).astype(np.float32)


def _short_side(size: tuple[int, int], short: int) -> tuple[int, int]:
    w, h = size
    k = short / min(w, h)
    return max(1, round(w * k)), max(1, round(h * k))


def working_size(size: tuple[int, int], top: int) -> tuple[int, int]:
    """The size frames are averaged at: the clip's own up to a short side of `NATIVE_UP_TO` x
    `top` (`detect_side`'s upper end), else shrunk to a short side of `top`."""
    return size if min(size) <= NATIVE_UP_TO * top else _short_side(size, top)


def detect_video(
    path: str | os.PathLike[str], detector: Detector | None = None
) -> Detection | None:
    """The signature in a video file, from the mean of `video_frames` frames across the clip:
    version 2 (the tile in every frame), or version 1 for clips exported before v2."""
    det = detector or default_detector()
    info = probe(Path(path))
    full = (info.width, info.height)
    if min(full) <= 0:
        return None
    work = working_size(full, det.p.detect_side[1])
    frames = clip_frames(Path(path), work, det.p.video_frames, info.duration)
    if frames.shape[0] == 0:
        return None
    mean = frames.reshape(frames.shape[0], -1).mean(0, dtype=np.float64)
    return det.detect_mean(mean.reshape(frames.shape[1:]).astype(np.float32), full)

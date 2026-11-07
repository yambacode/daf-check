"""The pixel signature in video (SPEC.md, section 7): frames from across the clip, shrunk onto the
version 1 grid by the `ffmpeg` command-line tool and averaged.

Needs `ffmpeg` and `ffprobe` on PATH (or in the DAF_CHECK_FFMPEG / DAF_CHECK_FFPROBE environment
variables). Without them a video's pixels are not checked, and the result says "ffmpeg needed".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .watermark import Detection, Detector, default_detector

TIMEOUT = 120.0


class FFmpegMissing(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    pass


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


def duration(path: Path) -> float:
    """Duration in seconds of the first video stream (0 when unknown)."""
    probe = _tool("ffprobe")
    if probe is None:
        raise FFmpegMissing("ffprobe is not installed")
    out = _run([probe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                "-select_streams", "v:0", str(path)]).stdout  # fmt: skip
    data = json.loads(out or b"{}")
    streams = data.get("streams") or []
    if not streams:
        raise FFmpegError("no video stream")
    return float(streams[0].get("duration") or data.get("format", {}).get("duration") or 0.0)


def grid_frames(path: Path, grid: int, n: int, timeout: float = TIMEOUT) -> np.ndarray:
    """(count, grid, grid) luminance of up to `n` frames spread evenly over the clip."""
    ff = _tool("ffmpeg")
    if ff is None:
        raise FFmpegMissing("ffmpeg is not installed")
    d = duration(path)
    fps = n / d if d > 0 else 1.0
    vf = f"fps={fps:.6f},scale={grid}:{grid}:flags=area,format=gray"
    with tempfile.TemporaryDirectory(prefix="daf-check-") as tmp:
        out = Path(tmp) / "grid.gray"
        _run([ff, "-nostdin", "-y", "-loglevel", "error", "-i", str(path), "-vf", vf,
              "-frames:v", str(n), "-f", "rawvideo", str(out)], timeout)  # fmt: skip
        data = np.frombuffer(out.read_bytes(), np.uint8)
    count = data.size // (grid * grid)
    return data[: count * grid * grid].reshape(count, grid, grid).astype(np.float32)


def detect_video(
    path: str | os.PathLike[str], detector: Detector | None = None
) -> Detection | None:
    """The signature in a video file, from `video_frames` frames averaged on the v1 grid."""
    det = detector or default_detector()
    frames = grid_frames(Path(path), det.p.legacy.grid, det.p.video_frames)
    return det.detect_frames(frames)

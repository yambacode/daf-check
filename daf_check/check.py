"""`check()`: reads both Dream Art Fields marks in a file and says what they show.

Everything happens on this computer: the file is read in memory (videos go through a temporary
file for ffmpeg) and nothing is stored or sent anywhere.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .manifest import ManifestInfo, read_manifest
from .watermark import Detection, Detector, default_detector


def sniff_mime(data: bytes) -> str | None:
    """The media type from the first bytes, for the formats C2PA or the signature can be in."""
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio/wav"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "video/webm"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"avif", b"avis"):
            return "image/avif"
        if brand in (b"heic", b"heix", b"mif1"):
            return "image/heic"
        if brand == b"qt  ":
            return "video/quicktime"
        if brand in (b"M4A ", b"M4B "):
            return "audio/mp4"
        return "video/mp4"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    return None


@dataclass
class Result:
    """What `check` found in one file."""

    source: str
    sha256: str
    mime: str | None
    size: int
    c2pa: ManifestInfo
    watermark: Detection | None
    watermark_checked: bool  # False: not a picture or video, or a video without ffmpeg
    watermark_note: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def c2pa_says_ai(self) -> bool:
        """The manifest claims AI generation and checks out (or could not be checked)."""
        c = self.c2pa
        return c.present and c.ai_generated and (c.intact or not c.verified)

    @property
    def ai_generated(self) -> bool:
        """At least one mark says AI-generated."""
        return self.c2pa_says_ai or bool(self.watermark and self.watermark.ai)

    @property
    def binding_matches(self) -> bool | None:
        """Whether the manifest's soft binding names the asset hash the pixels carry."""
        if self.watermark is None or not self.c2pa.soft_binding:
            return None
        return self.c2pa.soft_binding == self.watermark.asset_hash

    def summary(self) -> str:
        """Plain sentences, careful not to claim more than the marks show."""
        c, mark = self.c2pa, self.watermark is not None and self.watermark.ai
        meta_ok = c.present and c.ai_generated and c.intact
        meta_unchecked = c.present and c.ai_generated and not c.verified
        if mark and meta_ok:
            return "Both marks say this was made with an AI system (Dream Art Fields)."
        if mark and meta_unchecked:
            return (
                "The invisible Dream Art Fields signature says this was made with an AI system. "
                "C2PA metadata says so too, but its signature was not checked."
            )
        if mark:
            tail = (
                "Its C2PA metadata does not check out (changed or damaged)."
                if c.present and c.verified and not c.intact
                else "No signed metadata says so: it is missing or was removed."
            )
            return (
                "The invisible Dream Art Fields signature is in the pixels: it says this was "
                f"made with an AI system (Dream Art Fields). {tail}"
            )
        if meta_ok:
            who = f" ({c.system})" if c.system else ""
            return f"Signed C2PA metadata says this was made with an AI system{who}."
        if meta_unchecked:
            return (
                "C2PA metadata in this file says it was made with an AI system, but its "
                'signature was not checked (install "daf-check[c2pa]" to check it).'
            )
        if self.watermark is not None:
            return (
                "A Dream Art Fields signature is in the pixels, but it does not mark the file "
                "as AI-generated."
            )
        if c.present and c.verified and not c.intact:
            return "The file has C2PA metadata, but it does not check out (changed or damaged)."
        if not self.watermark_checked:
            why = f" ({self.watermark_note})" if self.watermark_note else ""
            return (
                f"No Dream Art Fields mark found in the metadata. The pixel signature was not "
                f"checked{why}. That does not prove a person made it."
            )
        return (
            "No Dream Art Fields mark found. That does not prove a person made it: marks can be "
            "removed, and other AI systems use other marks."
        )

    def as_dict(self) -> dict[str, Any]:
        wm: dict[str, Any] = {
            "checked": self.watermark_checked,
            "found": self.watermark is not None,
            "note": self.watermark_note,
        }
        if self.watermark is not None:
            wm.update(self.watermark.as_dict())
        return {
            "source": self.source,
            "sha256": self.sha256,
            "mime": self.mime,
            "size": self.size,
            "ai_generated": self.ai_generated,
            "summary": self.summary(),
            "watermark": wm,
            "c2pa": self.c2pa.as_dict(),
            "binding_matches": self.binding_matches,
            "notes": list(self.notes),
        }


def _watermark(
    data: bytes, mime: str | None, det: Detector, path: Path | None
) -> tuple[Detection | None, bool, str]:
    """The pixel signature in a picture, or averaged over frames of a video."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.seek(0)
            im = ImageOps.exif_transpose(im)
            return det.detect(im), True, ""
    except Image.DecompressionBombError:
        return None, False, "the picture is too large to check"
    except (UnidentifiedImageError, OSError, ValueError, EOFError):
        pass
    if not (mime or "").startswith("video/"):
        what = "audio" if (mime or "").startswith("audio/") else "not a picture or video"
        return None, False, f"{what}: the pixel signature is only in pictures and video"
    from .video import FFmpegError, FFmpegMissing, detect_video, ffmpeg_available

    if not ffmpeg_available():
        return None, False, "ffmpeg needed to read video frames"
    try:
        if path is not None:
            return detect_video(path, det), True, ""
        with tempfile.TemporaryDirectory(prefix="daf-check-") as tmp:
            src = Path(tmp) / "input"
            src.write_bytes(data)
            return detect_video(src, det), True, ""
    except FFmpegMissing:
        return None, False, "ffmpeg needed to read video frames"
    except (FFmpegError, OSError, ValueError) as e:
        return None, True, f"the video could not be read: {e}"[:200]


def check(
    src: str | os.PathLike[str] | bytes,
    *,
    mime: str | None = None,
    detector: Detector | None = None,
) -> Result:
    """Checks a file (a path or its bytes) for both Dream Art Fields marks.

    Raises OSError when a path cannot be read."""
    path: Path | None = None
    if isinstance(src, (bytes, bytearray, memoryview)):
        data, source = bytes(src), "<bytes>"
    else:
        path = Path(src)
        data, source = path.read_bytes(), str(src)
    mime = mime or sniff_mime(data)
    found, checked, note = _watermark(data, mime, detector or default_detector(), path)
    result = Result(
        source=source,
        sha256=hashlib.sha256(data).hexdigest(),
        mime=mime,
        size=len(data),
        c2pa=read_manifest(data, mime),
        watermark=found,
        watermark_checked=checked,
        watermark_note=note,
    )
    if result.binding_matches is False:
        result.notes.append(
            "The C2PA soft binding names a different asset than the pixel signature carries."
        )
    return result

"""Makes `tests/fixtures/` with Dream Art Fields' own reference code, and records what DAF says.

Not part of the package and not needed to use or test it: the fixtures and `expected.json` are
committed. This script documents how they were made, so anyone with the DAF sources can redo it.

Every marked file is made with DAF's reference `Signer` (`invisible.py`) and, for the C2PA file,
its reference `Provenance` (`provenance.py`) with a throwaway local certificate made in a temporary
folder (only the public certificate ends up in the file). `expected.json` holds what DAF's
`Signer.detect` (pictures), `detect_video` (video) and `read_manifest` (C2PA) report for each
file; the parity tests in `tests/test_parity.py` compare daf-check with it.

Usage (in a Python with numpy, Pillow, pydantic, PyYAML, cryptography and c2pa-python, and the
DAF backend on PYTHONPATH for `daf.branding.config` and `daf.media.ffmpeg`):

    PYTHONPATH=/path/to/dream-art-fields/backend \\
        python tools/make_fixtures.py --reference /path/to/daf-check-reference

`--reference` is a folder with DAF's `invisible.py`, `video.py`, `provenance.py` and
`branding.yaml`. The folder is only read. With `--keep`, fixtures that are already there are not
made again, only read again for `expected.json` (how 0.2.0 kept the 0.1.0 pictures byte for
byte). DAF no longer writes v1 video, so `video_v1_signed.mp4` (made with DAF's pre-v2 video code
for daf-check 0.1.0) is always kept.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import PIL
from PIL import Image, ImageOps

AID_V2 = "daf-check/fixture/v2"
AID_V1 = "daf-check/fixture/v1"
AID_VIDEO = "daf-check/fixture/video"  # the v1 video fixture (made before v2 video)
AID_VIDEO_V2 = "daf-check/fixture/video/v2"
AID_C2PA = "daf-check/fixture/c2pa"
MAX_BYTES = 300_000
VIDEO_W, VIDEO_H, VIDEO_N, VIDEO_FPS, VIDEO_CRF = 832, 480, 24, 12, 22
VIDEO_STEP, VIDEO_GRAIN = 24, 6.0  # pan per frame (px); grain (sigma) in the still: texture
V1_VIDEO_HOW = (
    "36 frames (640 x 360, 12 fps) of a slow pan + the v1-grid Signer.video_frame, libx264 crf 14 "
    "(made by daf-check 0.1.0's make_fixtures.py with DAF's pre-v2 video code; kept as is, since "
    "DAF no longer writes v1 video)"
)


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def natural(w: int, h: int, seed: int, grain: float = 0.0) -> Image.Image:
    """A picture-like test image (as in DAF's tests): smooth gradients and soft shapes. No fine
    grain by default (DAF's tests add sigma 5), so the PNG fixtures stay small."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, w)[None, :, None]
    y = np.linspace(0, 1, h)[:, None, None]
    base = np.concatenate(
        [x * 180 + 30 + 0 * y, y * 160 + 40 + 0 * x, (1 - x) * 120 + 60 + 0 * y], 2
    )
    blobs = rng.normal(0, 40, (h // 32, w // 32, 3))  # DAF's tests: // 8 (busier, bigger PNGs)
    blobs = Image.fromarray(np.clip(blobs + 128, 0, 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.BICUBIC
    )
    arr = base + np.asarray(blobs, float) - 128 + rng.normal(0, grain, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def crop(im: Image.Image, f: float, where: str) -> Image.Image:
    """Removes `f` of the width and `f` of the height (DAF's test helper)."""
    w, h = im.size
    cw, ch = round(w * (1 - f)), round(h * (1 - f))
    x0, y0 = {"tl": (0, 0), "br": (w - cw, h - ch), "c": ((w - cw) // 2, (h - ch) // 2)}[where]
    return im.crop((x0, y0, x0 + cw, y0 + ch))


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def detect_picture(signer, path: Path) -> dict | None:
    """What DAF's checker (`check._watermark`) reports for a picture."""
    with Image.open(path) as im:
        im.seek(0)
        im = ImageOps.exif_transpose(im)
        d = signer.detect(im)
    return None if d is None else _det(d)


def _det(d) -> dict:
    return {"version": d.version, "ai": d.ai, "id_hash": d.id_hash, "confidence": d.confidence}


def make_video(
    signer, path: Path, w: int = VIDEO_W, h: int = VIDEO_H, n: int = VIDEO_N,
    step: int = VIDEO_STEP, grain: float = VIDEO_GRAIN, crf: int = VIDEO_CRF,
) -> None:  # fmt: skip
    """A pan over a picture-like still with DAF's v2 video signature frame (`Signer.video_frame`,
    masked by `Signer.mask_luma` of frames from across the clip, as export does) added to every
    frame's luminance (A + B - 128, as ffmpeg's `grainmerge` blend), encoded with libx264."""
    big = natural(w + step * n, h, seed=5, grain=grain)
    shots = [big.crop((step * i, 0, step * i + w, h)) for i in range(n)]
    mask = np.stack([signer.mask_luma(im) for im in shots[:: max(1, n // signer.cfg.video_frames)]])
    raw = signer.video_frame(w, h, AID_VIDEO_V2, mask)
    delta = np.frombuffer(raw[: w * h], np.uint8).reshape(h, w).astype(np.int16) - 128
    frames = bytearray()
    for im in shots:
        rgb = np.asarray(im, np.int16) + delta[..., None]
        frames += np.clip(rgb, 0, 255).astype(np.uint8).tobytes()
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(VIDEO_FPS), "-i", "-", "-c:v", "libx264",
            "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)]  # fmt: skip
    subprocess.run(args, input=bytes(frames), check=True)


def crop_filter(f: float) -> str:
    """ffmpeg filter removing `f` of the width and height (bottom-right kept, so the tile phase
    moves; even sizes for yuv420p)."""
    k = f"{1 - f:g}"
    return f"crop=trunc(iw*{k}/2)*2:trunc(ih*{k}/2)*2:iw-trunc(iw*{k}/2)*2:ih-trunc(ih*{k}/2)*2"


def crop_video(src: Path, path: Path, vf: str, crf: int) -> None:
    """`src` through the filter `vf`, re-encoded with libx264 at `crf`."""
    args = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-c:v", "libx264",
            "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)]  # fmt: skip
    subprocess.run(args, check=True)


def sign_c2pa(prov: ModuleType, pic: bytes, watermark_id: int) -> bytes:
    """`pic` (PNG) with a C2PA manifest signed by DAF's reference `Provenance`, with a throwaway
    local certificate made in a temporary folder."""
    with tempfile.TemporaryDirectory() as tmp:
        pcfg = prov.ProvenanceConfig(
            system="Dream Art Fields",
            provider="YambaCode AB",
            provider_url="https://dreamartfields.com",
            local_identity={
                "common_name": "daf-check test fixture",
                "organization": "YambaCode AB (test fixture)",
            },
            checker={"name": "daf-check", "url": "https://github.com/yambacode/daf-check"},
        )
        return prov.Provenance(pcfg, Path(tmp)).sign_bytes(
            pic, "image/png", title="c2pa_signed.png", watermark_id=watermark_id,
            models=["example/image-model-1"],
        )  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", type=Path, required=True)
    tests = Path(__file__).resolve().parents[1] / "tests"
    ap.add_argument("--out", type=Path, default=tests / "fixtures")
    ap.add_argument("--expected", type=Path, default=tests / "expected.json")
    ap.add_argument("--keep", action="store_true",
                    help="keep fixtures that are already there (only read them again)")  # fmt: skip
    args = ap.parse_args()
    ref, out = args.reference.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    from daf.branding.config import load_branding_config  # DAF backend, on PYTHONPATH

    inv = load(ref / "invisible.py", "daf_ref_invisible")
    prov = load(ref / "provenance.py", "daf_ref_provenance")
    cfg = load_branding_config(ref / "branding.yaml").invisible
    signer = inv.Signer(cfg)
    expected: dict[str, dict] = {}

    def record(name: str, wm: dict | None, data: bytes, how: str, asset_id: str | None) -> None:
        size = len(data)
        assert size <= MAX_BYTES, f"{name} is {size} bytes"
        c2pa_info = prov.read_manifest(data).as_dict()
        expected[name] = {
            "how": how,
            "asset_id": asset_id,
            "id_hash": None if asset_id is None else inv.id_hash(asset_id),
            "bytes": size,
            "watermark": wm,
            "c2pa": c2pa_info,
        }
        print(f"{name:24} {size:7} B  watermark={wm}  c2pa.present={c2pa_info['present']}")

    def make(path: Path) -> bool:
        """False with `--keep` when `path` is already there: it is then only read again (the
        C2PA fixture, for one, carries its signing time and would change bytes)."""
        return not (args.keep and path.is_file())

    # 1. A v2-signed PNG (DAF's normal output, before C2PA).
    p = out / "v2_signed.png"
    signed = None
    if make(p):
        signed = signer.embed(natural(640, 480, seed=1), AID_V2)
        p.write_bytes(png_bytes(signed))
    record(p.name, detect_picture(signer, p), p.read_bytes(),
           "Signer.embed(natural(640, 480, seed=1)) saved as PNG", AID_V2)  # fmt: skip

    # 2. The same after a 20 % crop (bottom-right kept, so the tile phase moves) and JPEG 70.
    src, p = p, out / "v2_crop20_q70.jpg"
    if make(p):
        if signed is None:
            with Image.open(src) as im:
                signed = im.convert("RGB")
        crop(signed, 0.2, "br").save(p, "JPEG", quality=70)
    record(p.name, detect_picture(signer, p), p.read_bytes(),
           "v2_signed.png cropped by 20 % of width and height (bottom-right kept), JPEG q70",
           AID_V2)  # fmt: skip

    # 3. A v1-signed PNG (pictures signed before v2).
    p = out / "v1_signed.png"
    if make(p):
        p.write_bytes(png_bytes(signer.v1.embed(natural(384, 288, seed=2), AID_V1)))
    record(p.name, detect_picture(signer, p), p.read_bytes(),
           "LegacySignature.embed(natural(384, 288, seed=2)) saved as PNG", AID_V1)  # fmt: skip

    # 4. An unsigned PNG.
    p = out / "unsigned.png"
    if make(p):
        p.write_bytes(png_bytes(natural(384, 288, seed=3)))
    record(p.name, detect_picture(signer, p), p.read_bytes(),
           "natural(384, 288, seed=3), never signed", None)  # fmt: skip

    # 5. A v2-signed PNG with a signed C2PA manifest (throwaway local certificate).
    p = out / "c2pa_signed.png"
    if make(p):
        p.write_bytes(sign_c2pa(prov, png_bytes(signer.embed(natural(384, 288, seed=4), AID_C2PA)),
                                inv.id_hash(AID_C2PA)))  # fmt: skip
    record(p.name, detect_picture(signer, p), p.read_bytes(),
           "Signer.embed(natural(384, 288, seed=4)) as PNG, then Provenance.sign_bytes with a "
           "throwaway local certificate", AID_C2PA)  # fmt: skip

    # 6-8. Short videos (only when ffmpeg is here), read with DAF's `detect_video`.
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        vid = load(ref / "video.py", "daf_ref_video")

        def detect_clip(path: Path) -> dict | None:
            with tempfile.TemporaryDirectory() as tmp:
                d = vid.detect_video(signer, path, Path(tmp))
            return None if d is None else _det(d)

        # 6. A v2-signed clip (DAF's video export since v2 video).
        p = out / "video_v2_signed.mp4"
        if make(p):
            make_video(signer, p)
        record(p.name, detect_clip(p), p.read_bytes(),
               f"{VIDEO_N} frames ({VIDEO_W} x {VIDEO_H}, {VIDEO_FPS} fps) of a pan over "
               f"natural(seed=5, grain={VIDEO_GRAIN:g}), {VIDEO_STEP} px per frame, + "
               f"Signer.video_frame (v2 tile, video_cells), libx264 crf {VIDEO_CRF}",
               AID_VIDEO_V2)  # fmt: skip

        # 7. The same clip cropped by 20 % of width and height and re-encoded at CRF 28.
        p2 = out / "video_v2_crop20_crf28.mp4"
        vf = crop_filter(0.2)
        if make(p2):
            crop_video(p, p2, vf, 28)
        record(p2.name, detect_clip(p2), p2.read_bytes(),
               f"video_v2_signed.mp4 with ffmpeg -vf {vf} (20 % of width and height removed, "
               "bottom-right kept), libx264 crf 28", AID_VIDEO_V2)  # fmt: skip

        # 8. A v1-signed clip (exported before v2 video). DAF no longer writes v1 video, so the
        # 0.1.0 fixture is kept byte for byte (renamed from video_signed.mp4).
        p3, old = out / "video_v1_signed.mp4", out / "video_signed.mp4"
        if old.is_file() and not p3.is_file():
            old.rename(p3)
        if p3.is_file():
            record(p3.name, detect_clip(p3), p3.read_bytes(), V1_VIDEO_HOW, AID_VIDEO)
        else:
            print(f"{p3.name} missing: no v1 video fixture (DAF cannot make one any more)")
    else:
        print("ffmpeg not found: no video fixture")

    meta = {
        "made_with": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "ffmpeg": shutil.which("ffmpeg") is not None,
        },
        "files": expected,
    }
    args.expected.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

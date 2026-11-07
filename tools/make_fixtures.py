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
`branding.yaml`. The folder is only read.
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
AID_VIDEO = "daf-check/fixture/video"
AID_C2PA = "daf-check/fixture/c2pa"
MAX_BYTES = 300_000


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


def make_video(signer, path: Path, w: int = 640, h: int = 360, n: int = 36) -> None:
    """A slow pan with DAF's video signature frame added to every frame's luminance (as DAF's
    test `_signed_frames` does), encoded with libx264."""
    big = natural(w + 4 * n, h, seed=5)
    probe = np.stack([signer.grid_luma(big.crop((4 * i, 0, 4 * i + w, h))) for i in (0, n // 2)])
    raw = signer.video_frame(w, h, AID_VIDEO, probe)
    delta = np.frombuffer(raw[: w * h], np.uint8).reshape(h, w).astype(np.int16) - 128
    frames = bytearray()
    for i in range(n):
        rgb = np.asarray(big.crop((4 * i, 0, 4 * i + w, h)), np.int16)
        frames += np.clip(rgb + delta[..., None], 0, 255).astype(np.uint8).tobytes()
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", "12", "-i", "-", "-c:v", "libx264", "-preset", "slow",
            "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]  # fmt: skip
    subprocess.run(args, input=bytes(frames), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", type=Path, required=True)
    tests = Path(__file__).resolve().parents[1] / "tests"
    ap.add_argument("--out", type=Path, default=tests / "fixtures")
    ap.add_argument("--expected", type=Path, default=tests / "expected.json")
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

    # 1. A v2-signed PNG (DAF's normal output, before C2PA).
    signed = signer.embed(natural(640, 480, seed=1), AID_V2)
    p = out / "v2_signed.png"
    p.write_bytes(png_bytes(signed))
    record(p.name, detect_picture(signer, p), p.read_bytes(),
           "Signer.embed(natural(640, 480, seed=1)) saved as PNG", AID_V2)  # fmt: skip

    # 2. The same after a 20 % crop (bottom-right kept, so the tile phase moves) and JPEG 70.
    p = out / "v2_crop20_q70.jpg"
    crop(signed, 0.2, "br").save(p, "JPEG", quality=70)
    record(p.name, detect_picture(signer, p), p.read_bytes(),
           "v2_signed.png cropped by 20 % of width and height (bottom-right kept), JPEG q70",
           AID_V2)  # fmt: skip

    # 3. A v1-signed PNG (pictures signed before v2).
    p = out / "v1_signed.png"
    p.write_bytes(png_bytes(signer.v1.embed(natural(384, 288, seed=2), AID_V1)))
    record(p.name, detect_picture(signer, p), p.read_bytes(),
           "LegacySignature.embed(natural(384, 288, seed=2)) saved as PNG", AID_V1)  # fmt: skip

    # 4. An unsigned PNG.
    p = out / "unsigned.png"
    p.write_bytes(png_bytes(natural(384, 288, seed=3)))
    record(p.name, detect_picture(signer, p), p.read_bytes(),
           "natural(384, 288, seed=3), never signed", None)  # fmt: skip

    # 5. A v2-signed PNG with a signed C2PA manifest (throwaway local certificate).
    pic = png_bytes(signer.embed(natural(384, 288, seed=4), AID_C2PA))
    with tempfile.TemporaryDirectory() as tmp:
        pcfg = prov.ProvenanceConfig(
            system="Dream Art Fields",
            provider="YambaCode AB",
            provider_url="https://dreamartfields.com",
            local_identity={"common_name": "daf-check test fixture",
                            "organization": "YambaCode AB (test fixture)"},  # fmt: skip
            checker={"name": "daf-check", "url": "https://github.com/cyamba/daf-check"},
        )
        data = prov.Provenance(pcfg, Path(tmp)).sign_bytes(
            pic, "image/png", title="c2pa_signed.png", watermark_id=inv.id_hash(AID_C2PA),
            models=["example/image-model-1"],
        )  # fmt: skip
    p = out / "c2pa_signed.png"
    p.write_bytes(data)
    record(p.name, detect_picture(signer, p), data,
           "Signer.embed(natural(384, 288, seed=4)) as PNG, then Provenance.sign_bytes with a "
           "throwaway local certificate", AID_C2PA)  # fmt: skip

    # 6. A short signed video (only when ffmpeg is here).
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        vid = load(ref / "video.py", "daf_ref_video")
        p = out / "video_signed.mp4"
        make_video(signer, p)
        with tempfile.TemporaryDirectory() as tmp:
            d = vid.detect_video(signer, p, Path(tmp))
        record(p.name, None if d is None else _det(d), p.read_bytes(),
               "36 frames (640 x 360, 12 fps) of a slow pan + Signer.video_frame, libx264 crf 14",
               AID_VIDEO)  # fmt: skip
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

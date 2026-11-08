"""The signed C2PA manifest (SPEC.md, section 8).

With the optional `c2pa-python` library (`pip install "daf-check[c2pa]"`), the active manifest is
read and its signature and hashes are checked. Without it, daf-check only looks for a C2PA
manifest store (JUMBF box) in the file and for an AI claim inside it, and says plainly that the
signature was not checked.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Any

AI_SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
AI_SOURCE_TYPES = (
    AI_SOURCE_TYPE,
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeSynthetic",
)
GENERATOR_LABEL = "com.dreamartfields.generator"
# Soft-binding names (SPEC.md section 9): one per watermark scheme, the suffix the payload's
# version (2: picture and video tile, 3: audio). Manifests from DAF before 2026-11-08 carry the
# unversioned name for every scheme.
WATERMARK_ALG = "com.dreamartfields.signature.2"
AUDIO_WATERMARK_ALG = "com.dreamartfields.signature.3"
WATERMARK_ALG_LEGACY = "com.dreamartfields.signature"
WATERMARK_ALGS = frozenset({WATERMARK_ALG, AUDIO_WATERMARK_ALG, WATERMARK_ALG_LEGACY})
# A JUMBF description box ("jumd") with the C2PA manifest-store UUID and label "c2pa".
C2PA_STORE = b"jumd" + bytes.fromhex("6332706100110010800000aa00389b71")
# Never reach out to the network while reading (remote manifests, certificate status).
_OFFLINE = {"verify": {"remote_manifest_fetch": False, "ocsp_fetch": False}}


@dataclass
class ManifestInfo:
    """What was found about a file's C2PA manifest, in plain fields."""

    present: bool = False
    verified: bool = False  # a C2PA library checked the signature (False: presence scan only)
    state: str = ""  # Valid | Trusted | Invalid (C2PA validation state)
    intact: bool = False  # signature and hashes check out (Valid or Trusted)
    trusted: bool = False  # the signer chains to a CA on the loaded C2PA trust list
    ai_generated: bool = False  # an IPTC digital source type for AI-generated media
    source_type: str = ""
    generator: str = ""  # claim generator (software and version)
    signer: str = ""  # common name of the signing certificate
    issuer: str = ""  # organisation of the signing certificate
    signed_at: str | None = None
    time_stamped: bool = False  # an RFC 3161 time-stamp authority vouches for the time
    system: str = ""  # from the com.dreamartfields.generator assertion
    provider: str = ""
    models: list[str] = field(default_factory=list)
    daf_generator: bool = False  # the com.dreamartfields.generator assertion is there
    soft_binding: str | None = None  # asset hash (8 hex digits) the pixel signature should carry
    errors: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def c2pa_available() -> bool:
    try:
        import c2pa  # noqa: F401
    except Exception:
        return False
    return True


def _supported(c2pa: Any, mime: str) -> bool:
    mimes: set[str] = set()
    for cls in (getattr(c2pa, "Reader", None), getattr(c2pa, "Builder", None)):
        try:
            mimes |= {m.lower() for m in cls.get_supported_mime_types()}
        except Exception:
            continue
    mime = mime.lower()
    return mime in mimes or {"audio/wav": "audio/x-wav"}.get(mime, "") in mimes


def _summarise(report: dict[str, Any], state: str) -> ManifestInfo:
    label = report.get("active_manifest")
    m = (report.get("manifests") or {}).get(label) or {}
    info = ManifestInfo(present=True, verified=True, state=state)
    info.intact = state in ("Valid", "Trusted")
    info.trusted = state == "Trusted"
    gens = m.get("claim_generator_info") or []
    if gens and isinstance(gens, list):
        g = gens[0]
        info.generator = " ".join(str(g.get(k, "")) for k in ("name", "version")).strip()
    else:
        info.generator = str(m.get("claim_generator", ""))
    sig = m.get("signature_info") or {}
    info.issuer = str(sig.get("issuer") or "")
    info.signer = str(sig.get("common_name") or sig.get("issuer") or "")
    info.signed_at = sig.get("time")
    for a in m.get("assertions") or []:
        lbl = str(a.get("label", ""))
        data = a.get("data") or {}
        if lbl.startswith("c2pa.actions"):
            for act in data.get("actions") or []:
                src = act.get("digitalSourceType", "")
                if src in AI_SOURCE_TYPES:
                    info.ai_generated = True
                    info.source_type = src
                    if info.signed_at is None and act.get("when"):
                        info.signed_at = str(act["when"])  # the signer's clock, not a TSA
        elif lbl == GENERATOR_LABEL:
            info.daf_generator = True
            info.system = str(data.get("system", ""))
            info.provider = str(data.get("provider", ""))
            info.models = [str(x) for x in data.get("models") or []][:16]
        elif lbl.startswith("c2pa.soft-binding") and data.get("alg") in WATERMARK_ALGS:
            for block in data.get("blocks") or []:
                try:
                    raw = base64.b64decode(str(block.get("value", "")), validate=True)
                except (binascii.Error, ValueError):
                    continue
                if len(raw) == 4:
                    info.soft_binding = raw.hex()
                    break
    codes = []
    for s in report.get("validation_status") or []:
        code = str(s.get("code", ""))
        codes.append(code)
        if "untrusted" in code.lower():
            continue
        info.errors.append(f"{code}: {s.get('explanation', '')}".strip(": "))
    active = (report.get("validation_results") or {}).get("activeManifest") or {}
    for s in active.get("success") or []:
        code = str(s.get("code", ""))
        if code.startswith("timeStamp.trusted") or code == "timeStamp.validated":
            info.time_stamped = True
    for s in active.get("failure") or []:
        code = str(s.get("code", ""))
        if code == "signingCredential.untrusted" or code in codes:
            continue
        info.errors.append(f"{code}: {s.get('explanation', '')}".strip(": "))
    if info.intact and not info.trusted:
        info.note = "signature valid; the signer is not on the C2PA trust list"
    return info


def _read_with_library(data: bytes, mime: str) -> ManifestInfo:
    import c2pa

    ctx = settings = None
    try:
        settings = c2pa.Settings.from_dict(_OFFLINE)
        ctx = c2pa.Context(settings=settings)
    except Exception:  # an older library without settings: it reads with its defaults
        ctx = None
    try:
        reader = c2pa.Reader.try_create(mime, io.BytesIO(data), None, context=ctx)
        if reader is None:
            return ManifestInfo(present=False, verified=True)
        try:
            report = json.loads(reader.json())
            state = reader.get_validation_state() or ""
            try:
                report.setdefault("validation_results", reader.get_validation_results() or {})
            except Exception:
                pass
        finally:
            reader.close()
    except Exception as e:  # a damaged or foreign manifest: say so, never crash a check
        msg = str(e)
        if "ManifestNotFound" in msg or "no JUMBF" in msg or "JumbfNotFound" in msg:
            return ManifestInfo(present=False, verified=True)
        return ManifestInfo(present=True, verified=True, state="Invalid", errors=[msg[:300]])
    finally:
        for res in (ctx, settings):
            try:
                if res is not None:
                    res.close()
            except Exception:
                pass
    return _summarise(report, state)


def _jpeg_jumbf(data: bytes) -> bytes:
    """The JUMBF boxes of a JPEG, rebuilt from its APP11 segments (ISO 19566-5): each segment
    holds "JP", an instance number, a sequence number and a piece of the box; pieces after the
    first repeat the box header, which is dropped here."""
    out: dict[int, bytearray] = {}
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xD9 or marker == 0xDA:  # end of image / start of scan
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        n = int.from_bytes(data[i + 2 : i + 4], "big")
        seg = data[i + 4 : i + 2 + n]
        if marker == 0xEB and seg[:2] == b"JP" and len(seg) >= 16:
            en, z = int.from_bytes(seg[2:4], "big"), int.from_bytes(seg[4:8], "big")
            box = seg[8:]
            header = 16 if box[4:8] and int.from_bytes(box[:4], "big") == 1 else 8
            out.setdefault(en, bytearray()).extend(box if z <= 1 else box[header:])
        i += 2 + n
    return b"".join(bytes(v) for v in out.values())


def scan(data: bytes, mime: str | None) -> ManifestInfo:
    """Presence of a C2PA manifest store and of an AI claim in it, without checking anything."""
    store = _jpeg_jumbf(data) if mime == "image/jpeg" else data
    at = store.find(C2PA_STORE)
    if at < 0:
        return ManifestInfo(present=False)
    body = store[max(at - 8, 0) :]
    info = ManifestInfo(present=True, verified=False)
    for src in AI_SOURCE_TYPES:
        if src.encode() in body:
            info.ai_generated, info.source_type = True, src
            break
    info.daf_generator = GENERATOR_LABEL.encode() in body
    info.note = (
        "signature not checked: install daf-check[c2pa] (c2pa-python) to check it; "
        "the AI claim was read without verification"
    )
    return info


def read_manifest(data: bytes, mime: str | None) -> ManifestInfo:
    """The file's C2PA manifest: checked with c2pa-python when it is installed and the format is
    one it reads, else a presence scan."""
    if mime and c2pa_available():
        import c2pa

        if _supported(c2pa, mime):
            return _read_with_library(data, mime)
    return scan(data, mime)

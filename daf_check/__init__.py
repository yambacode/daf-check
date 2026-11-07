"""daf-check: a free, open-source checker for Dream Art Fields' AI marks.

    >>> import daf_check
    >>> r = daf_check.check("picture.png")
    >>> r.ai_generated, r.summary()
    >>> r.as_dict()

See SPEC.md for the public specification of both marks.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .check import Result, check, sniff_mime  # noqa: E402
from .manifest import ManifestInfo  # noqa: E402
from .watermark import Detection, Detector, id_hash  # noqa: E402

__all__ = [
    "Detection",
    "Detector",
    "ManifestInfo",
    "Result",
    "__version__",
    "check",
    "id_hash",
    "sniff_mime",
]

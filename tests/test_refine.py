"""The period refine step (SPEC.md 6.3 step 6) on inputs DAF found by measuring false positives."""

from __future__ import annotations

import numpy as np

from daf_check.watermark import default_detector


def test_sharpening_a_period_longer_than_the_picture_does_not_fail():
    """A clip's hinted periods can be longer than its (cropped) side. When the pilot drifts by
    more than h / (2 period) between the halves, the sharpened period came out negative and the
    fold raised, so a check reported "no mark" for an error (DAF PR #76)."""
    y = np.random.default_rng(1).normal(128, 20, (200, 400)).astype(np.float32)
    v2 = default_detector().v2
    refined = v2._refine(y, v2._fold(y, 300.0, 300.0, 2))
    assert refined.py > 0 and refined.px > 0

"""Lightweight heuristic to reject obvious non-mammogram inputs.

The trained classifier has no out-of-distribution awareness: a random
screenshot, photograph, or document fed into the sigmoid head will land at
some arbitrary probability and confidently "diagnose" something that isn't
even tissue. This module is a cheap first line of defence that catches the
two failure modes a reviewer most frequently produces:

  1. UI/text/screenshot inputs — characterised by high edge density (sharp
     character borders) compared to soft mammographic tissue.
  2. Photographs with little dark-background area — mammograms taken in
     standard CC/MLO views always sit on a near-black background; natural
     photos rarely do.

This is NOT a real OOD detector. A real production system needs a dedicated
classifier (e.g. a mammogram-vs-other binary, or density-based novelty
detection in feature space). The trade-off is documented in REPORT.md.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# Thresholds calibrated on a handful of CBIS-DDSM test images vs. screenshots
# and natural photos. Conservative — biased to admit borderline mammograms
# rather than reject them.
_MIN_DARK_FRACTION = 0.15  # fraction of pixels below intensity 0.12
_MAX_EDGE_DENSITY = 0.22  # fraction of pixels flagged by PIL's edge filter
_DARK_INTENSITY_CUTOFF = 0.12


@dataclass(frozen=True)
class OODCheck:
    is_plausible: bool
    reason: str
    dark_fraction: float
    edge_density: float


def is_plausible_mammogram(data: bytes) -> OODCheck:
    """Inspect raw bytes; decode as a small grayscale thumbnail for cheap stats.

    DICOM inputs bypass this check — DICOM is medical imaging by definition,
    and the format itself is strong evidence of intent. The check applies only
    to PNG/JPEG inputs where an arbitrary photo could be uploaded.
    """
    if len(data) >= 132 and data[128:132] == b"DICM":
        return OODCheck(True, "dicom", dark_fraction=float("nan"), edge_density=float("nan"))

    try:
        img = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        # Decoding failure will be surfaced by the preprocessing step; we don't
        # second-guess it here.
        return OODCheck(True, "non-decodable; defer to preprocess", float("nan"), float("nan"))

    # Downsample for cheap stats — preserves coarse intensity & edge structure.
    img.thumbnail((256, 256))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    dark_fraction = float((arr < _DARK_INTENSITY_CUTOFF).mean())

    edges = np.asarray(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    edge_density = float((edges > 40).mean())

    if dark_fraction < _MIN_DARK_FRACTION:
        return OODCheck(
            False,
            (
                f"Input does not look like a mammogram: only {dark_fraction:.0%} "
                "of pixels are near-black background. Standard CC/MLO mammograms "
                f"have ≥{_MIN_DARK_FRACTION:.0%}."
            ),
            dark_fraction,
            edge_density,
        )

    if edge_density > _MAX_EDGE_DENSITY:
        return OODCheck(
            False,
            (
                f"Input does not look like a mammogram: edge density "
                f"{edge_density:.0%} is well above the {_MAX_EDGE_DENSITY:.0%} "
                "threshold typical for soft tissue. Looks more like a screenshot, "
                "diagram, or document."
            ),
            dark_fraction,
            edge_density,
        )

    return OODCheck(True, "ok", dark_fraction, edge_density)
